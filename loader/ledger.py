"""Particle ledger: one JSON line per trapped particle measured by the loader.

This file IS the brightness-vs-size histogram. Entries with an independent size
(`epstein_mass_pg` or `reference_d_m`) are references: they set the scale of the
Mie brightness curve (size_model.estimate_diameter). Add an Epstein mass to an
existing entry later with `load_particle.py ledger-set-mass <id> <pg>`.

Brightness B = counts_per_s / (POZ_INMON / 1000): counts per second per 1000
counts of the z-laser power monitor. POZ_INMON is a raw ADC readback, not mW; it
only has to be proportional to the power at the particle.
"""
import json
import os
import time
import uuid

DEFAULT_PATH = os.path.expanduser(
    "~/Dropbox/Microspheres/MAST-QG/worker1/data/particle_ledger.jsonl")

POZ_UNIT = 1000.0


def brightness(counts_per_s, poz_inmon):
    if poz_inmon is None or not poz_inmon > 0:
        return None
    return counts_per_s / (poz_inmon / POZ_UNIT)


def load(path=DEFAULT_PATH):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append(entry, path=DEFAULT_PATH):
    entry = dict(entry)
    entry.setdefault('id', time.strftime('%y%m%d_%H%M%S_') + uuid.uuid4().hex[:4])
    entry.setdefault('t_unix', time.time())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(entry, default=_json_default) + '\n')
    return entry


def set_field(entry_id, key, value, path=DEFAULT_PATH):
    """Rewrite the ledger with entry[key] = value (atomic replace)."""
    rows = load(path)
    hit = [r for r in rows if r['id'] == entry_id]
    if not hit:
        raise KeyError(entry_id)
    for r in hit:
        r[key] = value
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + '\n')
    os.replace(tmp, path)
    return hit[0]


def references(rows, camera_serial=None, x_laser_on=None, r_ap=None):
    """(B, d_ref_m, id) for rows with an independent size, optionally restricted
    to the same camera, x-laser state and photometry aperture (the reference is
    only valid for the same illumination, imaging and aperture: the aberrated PSF
    has wide wings, so r_ap 50 holds ~half the r_ap 80 flux)."""
    from size_model import diameter_from_mass
    out = []
    for r in rows:
        B = r.get('brightness')
        if B is None:
            continue
        if camera_serial is not None and r.get('camera_serial') != camera_serial:
            continue
        if x_laser_on is not None and r.get('x_laser_on') is not None and r['x_laser_on'] != x_laser_on:
            continue
        if r_ap is not None and r.get('r_ap') is not None and int(r['r_ap']) != int(r_ap):
            continue
        if r.get('reference_d_m'):
            out.append((B, float(r['reference_d_m']), r['id']))
        elif r.get('epstein_mass_pg'):
            out.append((B, diameter_from_mass(float(r['epstein_mass_pg']) * 1e-15), r['id']))
    return out


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(repr(o))
