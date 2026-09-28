#!/usr/bin/env python3
"""Automated particle loading: chirp the dropper, catch a particle on the x cam,
measure its scattered light, estimate its size, reject clusters, repeat.

Run in the `thorcam` conda env on worker1 (camera SDKs); the dropper runs as a
child process in the `moku` env (moku/dropper_server.py). Disconnect the x cam
in dualcam_fast.py first -- a camera can be held by one process only.

Subcommands
  identify-cams   list cameras, save one frame from each (which one is the x cam?)
  measure         photometry + size estimate of the particle trapped NOW -> ledger
  replay          detector validation on recorded frames (real noise, real PSF)
  run             the full loop (default: fake Moku, eject disabled)
  ledger          print the ledger; `ledger --set-mass ID PG` adds an Epstein mass

Safety
  * Refuses to chirp unless the trap is empty (occupancy check at the trap).
  * Dropper deadman: output Off if this process dies or stops pinging.
  * Abort any time: Ctrl-C, 'q' in the preview, or `touch /tmp/abort_particle_loader`.
  * Real Moku output needs --real-moku; electrical eject is disabled (loader/eject.py).
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, 'cameras')):
    if p not in sys.path:
        sys.path.insert(0, p)

import detector as det_mod   # noqa: E402
import eject                 # noqa: E402
import ledger                # noqa: E402
import photometry            # noqa: E402
import size_model            # noqa: E402

ABORT_FILE = '/tmp/abort_particle_loader'
DATA_ROOT = os.path.expanduser('~/Dropbox/Microspheres/MAST-QG/worker1/data')
DEFAULT_CONF = os.path.join(REPO, 'moku', 'conf_sweep.yaml')
X_CAM_SERIAL = '4103314187'     # IDS DCC1545M; x cam per Alma's GUI setup (confirm!)
TRAP_XY = (765.0, 636.0)        # 10 pg reference particle on the x cam, 2026-09-28
DIFF_BELOW_SIGMA_SP = 20.0      # use difference photometry below this annulus significance
DEFAULT_ROI = (0, 500, 1280, 300)   # full width x 300 rows about the trap row (Alma, 2026-09-28)

PVS = {
    'poz_inmon': 'Y1:DMD-LASER_POZ_INMON',
    'pox_inmon': 'Y1:DMD-LASER_POX_INMON',
    'chamber_pressure_mbar': 'Y1:AUX-CHAMBER_PRESSURE',
    'lesx_sum_inmon': 'Y1:DMD-LESX_SUM_INMON',
    'lesz_sum_inmon': 'Y1:DMD-LESZ_SUM_INMON',
}
X_LASER_ON_POX = 200.0   # POX_INMON above this = x laser on (30 with it off, 2026-09-28)


class Abort(Exception):
    pass


# ----------------------------------------------------------------------------- utils

def caget_all(pvs=PVS, timeout=3.0):
    out = {}
    try:
        r = subprocess.run(['caget', '-t', '-w', '1'] + list(pvs.values()),
                           capture_output=True, text=True, timeout=timeout)
        vals = r.stdout.split('\n')
        for (k, _), v in zip(pvs.items(), vals):
            try:
                out[k] = float(v.strip())
            except ValueError:
                out[k] = None
    except Exception:
        out = {k: None for k in pvs}
    return out


def caput(pv, value):
    try:
        subprocess.run(['caput', '-t', pv, str(value)], capture_output=True, timeout=3)
    except Exception:
        pass


def check_abort():
    if os.path.exists(ABORT_FILE):
        raise Abort(f"abort sentinel {ABORT_FILE} present")


class RunLog:
    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, 'events.jsonl')

    def __call__(self, ev, **kw):
        kw['ev'] = ev
        kw['t'] = time.time()
        line = json.dumps(kw, default=ledger._json_default)
        with open(self.path, 'a') as f:
            f.write(line + '\n')
        msg = ' '.join(f"{k}={_fmt(v)}" for k, v in kw.items() if k not in ('ev', 't'))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {ev}: {msg}", flush=True)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def alert(text, telegram=True):
    print('\a' + '=' * 70 + '\n' + text + '\n' + '=' * 70, flush=True)
    if not telegram:
        return
    try:
        sys.path.insert(0, os.path.expanduser('~/.claude/hooks'))
        import telegram_common as tc
        cfg = tc.load_config()
        tc.send_message(cfg['CLAUDE_APPROVAL_BOT_TOKEN'], cfg['CLAUDE_APPROVAL_CHAT_ID'],
                        'particle loader: ' + text)
    except Exception as e:
        print(f"(telegram alert failed: {e})")


class Preview:
    def __init__(self, enabled, center, radius):
        self.enabled = enabled and bool(os.environ.get('DISPLAY'))
        self.center, self.radius = center, radius
        self.lo = self.hi = None
        if self.enabled:
            import cv2
            self.cv2 = cv2
            cv2.namedWindow('particle loader', cv2.WINDOW_NORMAL)

    def show(self, frame, text):
        if not self.enabled:
            return
        cv2 = self.cv2
        if self.lo is None:
            self.lo, self.hi = np.percentile(frame, 1), np.percentile(frame, 99.9) + 1
        img = np.clip((frame - self.lo) / (self.hi - self.lo) * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        c = (int(self.center[0]), int(self.center[1]))
        cv2.circle(img, c, int(self.radius), (0, 200, 255), 2)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('particle loader', img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            raise Abort("'q' pressed in preview")

    def close(self):
        if self.enabled:
            self.cv2.destroyAllWindows()


# ----------------------------------------------------------------------------- measure

def measure_particle(src, center, args, log, reference_mass_pg=None, save_dir=None, baseline=None,
                     search_r=80.0):
    """Exposure ladder + photometry + PVs + size estimate; append to the ledger.

    Annulus photometry is limited by fixed-pattern background (sigma_spatial ~3x
    sigma_temporal, 2026-09-28). When the pre-chirp empty-trap `baseline` is given
    and the particle is faint (annulus net < DIFF_BELOW_SIGMA_SP * sigma_spatial),
    B comes from difference photometry against it at the detection exposure."""
    det_us = src.exposure_us
    # Centre the aperture on the aperture-scale flux maximum within search_r of
    # `center`: the detector reports the ring PSF's bright side (~50 px off),
    # and an off-centre r_ap 50 aperture loses ~20% of the flux.
    h, w = (baseline.shape[1:] if baseline is not None else src.grab(det_us, 1)[0].shape[1:])
    yy, xx = np.indices((h, w))
    near = np.hypot(xx - center[0], yy - center[1]) <= search_r
    best, rungs = photometry.exposure_ladder(
        src.grab, src.exposure_us, min_us=args.min_exposure_us, max_us=args.max_exposure_us,
        n=args.n_measure, center=None, search_mask=near, smooth_r=max(1, args.r_ap // 2),
        r_ap=args.r_ap)
    if best is None:
        raise RuntimeError("every exposure rung saturated -- lower --min-exposure-us")
    method = 'annulus'
    sp = best['noise_floor_spatial_counts']
    sig_ann = best['net_counts'] / sp if math.isfinite(sp) and sp > 0 else float('nan')
    significance = sig_ann
    diff = None
    if baseline is not None and not sig_ann >= DIFF_BELOW_SIGMA_SP:     # NaN -> try diff
        stack, _ = src.grab(det_us, args.n_measure)
        diff = photometry.difference_photometry(stack, baseline, det_us * 1e-6,
                                                src.saturation_level, center=center, r_ap=args.r_ap,
                                                search_r=search_r)
        diff['exposure_us'] = det_us
        if not diff['saturated']:
            best, method, significance = diff, 'difference', diff['significance']
    pv = caget_all()
    B = ledger.brightness(best['counts_per_s'], pv.get('poz_inmon'))
    x_on = (pv['pox_inmon'] > X_LASER_ON_POX) if pv.get('pox_inmon') is not None else None
    entry = {
        'kind': 'measurement', 'camera_serial': src.serial, 'x_laser_on': x_on,
        'brightness': B, 'brightness_unit': 'counts/s per 1000 POZ_INMON counts',
        'counts_per_s': best['counts_per_s'], 'exposure_us': best['exposure_us'],
        'photometry_method': method, 'significance': significance,
        'significance_annulus': sig_ann,
        'photometry': best, 'linearity_ratio': best.get('linearity_ratio'),
        'ladder': [{k: r[k] for k in ('exposure_us', 'counts_per_s', 'peak_raw', 'saturated')}
                   for r in rungs],
        'pvs': pv, 'r_ap': args.r_ap, 'roi': getattr(args, 'roi', None),
    }
    if reference_mass_pg:
        entry['epstein_mass_pg'] = float(reference_mass_pg)
        entry['kind'] = 'reference'
    rows = ledger.load(args.ledger)
    refs = ledger.references(rows, camera_serial=src.serial, x_laser_on=x_on, r_ap=args.r_ap)
    if reference_mass_pg:
        refs = refs + [(B, size_model.diameter_from_mass(float(reference_mass_pg) * 1e-15), 'self')]
    est = None
    if refs and B:
        est = size_model.estimate_diameter(B, [(r[0], r[1]) for r in refs], pols=tuple(args.pol_deg),
                                           na=args.na, size_spread=args.size_spread)
        est['size_spread'] = args.size_spread
    entry['size_estimate'] = est
    entry['n_references'] = len(refs)
    # Mie-free cross-check (INFORMATIONAL, not used in the decision yet): in the
    # magneto-gravitational trap omega is mass-independent, so thermal rms ~ 1/sqrt(m)
    # and m/m_ref ~ (rms_ref/rms)^2. Needs same exposure (blur) and pressure regime.
    ref_rows = [r for r in rows if r.get('kind') == 'reference' and r.get('camera_serial') == src.serial]
    rms = math.hypot(*best['centroid_std_xy_px'])
    if ref_rows and rms > 0:
        rr = ref_rows[-1]
        rms_ref = math.hypot(*rr['photometry']['centroid_std_xy_px'])
        entry['motion_mass_ratio_vs_ref'] = (rms_ref / rms) ** 2
        entry['motion_ref_id'] = rr['id']
    # Limits relative to THIS particle: the detector's (temporal noise, one frame)
    # and the annulus sizing photometry's (fixed-pattern noise, mean frame).
    if best['noise_floor_counts_per_s'] and B:
        r5 = 5 * best['noise_floor_counts_per_s'] / best['counts_per_s']
        entry['min_detectable_ratio_5sigma_frame'] = r5
        entry['min_sizable_ratio_5sigma_annulus'] = (
            5 * best['noise_floor_spatial_counts'] / best['net_counts'])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stack, meta = src.grab(best['exposure_us'], args.n_measure)
        cx, cy = best['center_xy']
        h = args.r_ap + 30
        y0, x0 = max(int(cy) - h, 0), max(int(cx) - h, 0)
        crop = stack[:, y0:int(cy) + h, x0:int(cx) + h]
        path = os.path.join(save_dir, f"particle_{datetime.now().strftime('%H%M%S')}.npz")
        np.savez_compressed(path, frames=crop.astype(np.uint16), origin_xy=(x0, y0),
                            exposure_us=best['exposure_us'])
        entry['snapshot'] = path
    entry = ledger.append(entry, args.ledger)
    log('measured', id=entry['id'], counts_per_s=best['counts_per_s'], brightness=B,
        exposure_us=best['exposure_us'], snr_frame=best['snr_frame'],
        centroid_std=best['centroid_std_xy_px'], saturated=best['saturated'],
        d_est_nm=None if not est else (round(est['d_min_m'] * 1e9), round(est['d_central_m'] * 1e9),
                                       round(est['d_max_m'] * 1e9)),
        n_refs=len(refs), motion_mass_ratio=entry.get('motion_mass_ratio_vs_ref'))
    return entry


def decide(entry, args):
    """'accept' | 'reject' | 'ambiguous' | 'no-calibration', with a reason."""
    est = entry.get('size_estimate')
    sig = entry.get('significance')
    if sig is not None and not math.isfinite(sig):
        return 'ambiguous', 'no noise floor (no source-free aperture fits the ROI)'
    if sig is not None and sig < args.k_occupancy:
        return 'no-particle', f"measured flux is only {sig:.1f} sigma (< {args.k_occupancy})"
    if not est:
        return 'no-calibration', 'no reference particle for this camera / x-laser state'
    if entry['photometry'].get('saturated'):
        return 'ambiguous', 'saturated at the shortest exposure'
    lim = args.accept_max_nm * 1e-9
    band = f"d in [{est['d_min_m'] * 1e9:.0f}, {est['d_max_m'] * 1e9:.0f}] nm"
    if est['d_max_m'] < lim:
        return 'accept', band + f" < {args.accept_max_nm:.0f} nm"
    if est['d_min_m'] > lim:
        return 'reject', band + f" > {args.accept_max_nm:.0f} nm"
    return 'ambiguous', band + f" straddles {args.accept_max_nm:.0f} nm"


# ----------------------------------------------------------------------------- run loop

def baseline_stack(frames_iter, n):
    fr, ts = [], []
    for _ in range(n):
        f, t = next(frames_iter)
        fr.append(f)
        ts.append(t)
    return np.stack(fr), ts


def attempt_load(src, frames_iter, args, log, preview):
    """One chirp. Returns (outcome, where, empty_baseline): outcome 'loaded' |
    'occupied' | 'no-load' (ramp ended) | 'lost'; empty_baseline is the pre-chirp
    stack when the trap was empty (for difference photometry), else None."""
    from dropper_client import DropperClient
    base, _ = baseline_stack(frames_iter, args.n_baseline)
    # The particle does not always trap at the same z (x-cam horizontal): test the
    # nominal trap AND the brightest spot in the trap region.
    for where in (args.trap_xy, None):
        occ, m = det_mod.occupancy(base, where, args.r_ap, src.saturation_level,
                                   src.exposure_us * 1e-6, k_sigma=args.k_occupancy,
                                   search_mask=region_mask(base.shape[1:], args))
        log('occupancy', at=m['center_xy'], occupied=occ, net_counts=m['net_counts'],
            sigma_spatial=m['noise_floor_spatial_counts'])
        if occ:
            return 'occupied', m['center_xy'], None
    det = det_mod.TrapDetector(base, region_center=args.trap_xy, region_radius=args.region_px,
                               k_sigma=args.k_sigma, persist_s=args.persist_s,
                               smooth_r=args.smooth_r, destripe=not args.no_destripe,
                               confirm_s=args.confirm_s)
    log('detector', **det.describe())
    dropper = DropperClient(args.conf, fake=not args.real_moku, log=print)
    log('dropper_ready', **{k: dropper.ready.get(k) for k in
                            ('n_emitted', 'max_time', 'peak_v', 'headroom_ok', 'conf_copy', 'fake')})
    if args.real_moku:
        caput('Y1:AUX-DROPPER_ON', 1)
    stopped_at_candidate = False
    t_candidate = None
    try:
        dropper.start()
        src.on_dropper_start()
        log('chirp_start')
        for f, t in frames_iter:
            check_abort()
            r = det.update(f, t)
            dropper.ping()
            for ev in dropper.drain():
                if ev['ev'] == 'pulse':
                    if args.real_moku:
                        caput('Y1:AUX-DROPPER_AMP', ev['amp'])
                    log('pulse', k=ev['k'], amp=ev['amp'])
            preview.show(f, f"{r['state']} z={r['score']:.1f} "
                            f"{'CHIRP' if dropper.running else 'idle'}")
            if r['state'] in ('candidate', 'confirmed') and dropper.running:
                st = dropper.stop()
                stopped_at_candidate = True
                t_candidate = t
                log('chirp_stop', reason='candidate', score=r['score'], x=r['x'], y=r['y'],
                    moku_latency_s=st.get('latency_s'), client_latency_s=st.get('client_latency_s'),
                    detect_duration_s=r['duration_s'])
            if r['state'] == 'confirmed':
                log('confirmed', x=r['x'], y=r['y'], duration_s=r['duration_s'])
                return 'loaded', (r['x'], r['y']), base
            if stopped_at_candidate and r['state'] in ('empty',):
                log('candidate_lost', after_s=t - t_candidate)
                return 'lost', None, None
            if not dropper.running and not stopped_at_candidate:
                log('chirp_complete_no_particle', stopped=dropper.stopped)
                return 'no-load', None, None
    finally:
        dropper.close()
        if args.real_moku:
            caput('Y1:AUX-DROPPER_ON', 0)
            caput('Y1:AUX-DROPPER_AMP', 0)


def cmd_run(args):
    out = args.out_dir or os.path.join(DATA_ROOT, datetime.now().strftime('%y%m%d'), 'particle_loader',
                                       datetime.now().strftime('%H%M%S'))
    log = RunLog(out)
    if args.replay and args.ledger == ledger.DEFAULT_PATH:
        # Replays never touch the real ledger: seed a scratch one with its references.
        scratch = os.path.join(out, 'ledger_replay.jsonl')
        if not os.path.exists(scratch):
            for r in ledger.load(ledger.DEFAULT_PATH):
                if r.get('kind') == 'reference':
                    ledger.append(r, scratch)
        args.ledger = scratch
    log('start', args=vars(args))
    if os.path.exists(ABORT_FILE):
        os.remove(ABORT_FILE)
    src = make_source(args)
    preview = Preview(not args.no_preview, args.trap_xy, args.region_px)
    frames_iter = src.frames()
    result = None
    try:
        for attempt in range(1, args.max_attempts + 1):
            check_abort()
            log('attempt', n=attempt)
            outcome, where, base = attempt_load(src, frames_iter, args, log, preview)
            log('attempt_result', outcome=outcome, at=where)
            if outcome in ('no-load', 'lost'):
                continue
            entry = measure_particle(src, where, args, log, save_dir=out, baseline=base)
            src.set_exposure_us(args.exposure_us)      # back to the detection exposure
            verdict, why = decide(entry, args)
            ledger.set_field(entry['id'], 'decision', verdict, args.ledger)
            log('decision', verdict=verdict, why=why, id=entry['id'])
            if verdict == 'no-particle':
                continue                # nothing measurable after all: chirp again
            if verdict == 'accept':
                result = f"ACCEPTED particle {entry['id']}: {why}. Loader stopped."
                break
            if verdict == 'reject':
                plan = eject.plan_eject()
                log('eject_plan', **plan)
                if args.replay and args.replay_simulate_eject:
                    src.on_eject()
                    log('eject_simulated')
                    continue
                result = (f"REJECTED particle {entry['id']} ({why}); electrical eject is "
                          f"disabled, so the loader stopped with the particle still trapped.")
                break
            result = f"STOPPED on {verdict} particle {entry['id']}: {why}. Your call."
            break
        else:
            result = f"no accepted particle after {args.max_attempts} attempts"
    except Abort as e:
        result = f"ABORTED: {e}"
    except KeyboardInterrupt:
        result = "ABORTED: Ctrl-C"
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = f"ERROR (loader stopped, dropper deadman turns the output off): {e!r}"
    finally:
        preview.close()
        src.close()
    log('end', result=result)
    alert(result + f"\nlog: {log.path}", telegram=args.telegram)
    return 0 if result and result.startswith('ACCEPTED') else 1


# ----------------------------------------------------------------------------- sources

def make_source(args):
    import sources
    if args.replay:
        import synth
        d = np.load(args.replay)
        stack = d['frames'] if hasattr(d, 'files') else d
        scene, excess, _ = synth.real_scene(stack, args.replay_particle_xy, args.replay_scene_xy,
                                            scene_half=replay_half(args), r_ap=args.r_ap)
        hx, hy = replay_half(args)
        args.trap_xy = (float(hx), float(hy))
        return sources.ReplaySource(scene, excess, args.replay_exposure_us, 255, fps=args.replay_fps,
                                    schedule=args.replay_schedule, delay_s=args.replay_delay_s,
                                    transit_scale=args.replay_transit,
                                    occupied_at_start=args.replay_occupied, serial=args.serial)
    import camera_io
    cam = camera_io.open_camera(args.serial, vendors=tuple(args.vendors))
    return sources.LiveSource(cam, args.exposure_us, roi=args.roi)


def replay_half(args):
    """(half_w, half_h) of the replayed scene crop."""
    return (args.replay_half, args.replay_half_y or args.replay_half)


def region_mask(shape, args):
    """Boolean mask of the trap region (a circle of --region-px about --trap-xy)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return np.hypot(xx - args.trap_xy[0], yy - args.trap_xy[1]) <= args.region_px


# ----------------------------------------------------------------------------- other commands

def cmd_identify(args):
    import camera_io
    import cv2
    out = args.out_dir or os.path.join(DATA_ROOT, datetime.now().strftime('%y%m%d'), 'particle_loader')
    os.makedirs(out, exist_ok=True)
    cams = camera_io.list_cameras(tuple(args.vendors))
    for c in cams:
        print(c)
        if c.in_use:
            print("  in use -- skipped")
            continue
        cam = camera_io.open_camera(c.serial, tuple(args.vendors))
        try:
            cam.set_exposure_us(args.exposure_us)
            st, meta = cam.grab(3, discard=2)
            m = st.mean(0)
            p = os.path.join(out, f"identify_{c.vendor}_{c.serial}.png")
            lo, hi = np.percentile(m, 1), np.percentile(m, 99.9) + 1
            cv2.imwrite(p, np.clip((m - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))
            print(f"  saved {p} (exposure {meta['exposure_us']} us, max {st.max()})")
        finally:
            cam.close()
    return 0


def cmd_measure(args):
    out = args.out_dir or os.path.join(DATA_ROOT, datetime.now().strftime('%y%m%d'), 'particle_loader')
    log = RunLog(out)
    if args.from_npy:
        import sources
        st = np.load(args.from_npy)

        class _Static:
            serial = args.serial
            saturation_level = 255
            exposure_us = args.exposure_us

            def grab(self, exposure_us, n):
                if abs(exposure_us - args.exposure_us) > 1:
                    # a recorded stack has one exposure; scale as a linear camera
                    s = exposure_us / args.exposure_us
                    return np.clip(st[:n] * s, 0, 255), {'exposure_us': exposure_us,
                                                         'saturation_level': 255}
                return st[:n], {'exposure_us': args.exposure_us, 'saturation_level': 255}

            def close(self):
                pass
        src = _Static()
        args.max_exposure_us = args.exposure_us     # no ladder beyond what was recorded
        args.min_exposure_us = args.exposure_us
    else:
        src = make_source(args)
    try:
        entry = measure_particle(src, args.trap_xy, args, log, reference_mass_pg=args.reference_mass_pg,
                                 save_dir=None if args.from_npy else out, search_r=args.region_px)
    finally:
        src.close()
    verdict, why = decide(entry, args)
    print(f"photometry: {entry['photometry_method']}, significance {entry['significance']:.1f} sigma")
    print(f"\nentry {entry['id']}: B = {entry['brightness']:.4g} {entry['brightness_unit']}; "
          f"decision would be {verdict}: {why}")
    r5 = entry.get('min_detectable_ratio_5sigma_frame')
    rs = entry.get('min_sizable_ratio_5sigma_annulus')
    if rs:
        print(f"5-sigma annulus SIZING limit (no empty baseline): B/B_this = {rs:.3g}; the run loop "
              f"sizes faint particles by difference photometry instead (~0.004 at 50 ms, 2026-09-28)")
    if r5:
        print(f"5-sigma single-frame DETECTION floor (temporal noise): B/B_this = {r5:.3g}")
        d_ref = size_model.diameter_from_mass((args.reference_mass_pg or 10.0) * 1e-15)
        for pol in args.pol_deg:
            ds = [d for d in np.geomspace(50e-9, d_ref, 200)
                  if size_model.predicted_ratio(d, d_ref, pol_deg=pol, na=args.na) >= r5]
            print(f"  pol {pol:>4.0f} deg: smallest detectable sphere ~ {ds[0] * 1e9:.0f} nm"
                  if ds else f"  pol {pol}: none")
    return 0


def cmd_replay(args):
    """G3: detector on real scene frames with injected real-PSF particles."""
    import synth
    st = np.load(args.replay)
    st = st['frames'] if hasattr(st, 'files') else st
    scene, excess, tot = synth.real_scene(st, args.replay_particle_xy, args.replay_scene_xy,
                                          scene_half=replay_half(args), r_ap=args.r_ap)
    n = len(scene)
    nb = args.n_baseline
    c = replay_half(args)
    fps = args.replay_fps
    res = {'excess_net_counts': tot, 'n_frames': n, 'cases': []}

    def run_case(name, frames):
        d = det_mod.TrapDetector(frames[:nb], region_center=c, region_radius=args.region_px,
                                 k_sigma=args.k_sigma, persist_s=args.persist_s,
                                 smooth_r=args.smooth_r, destripe=not args.no_destripe,
                                 confirm_s=args.confirm_s)
        first = {}
        for i, f in enumerate(frames[nb:]):
            r = d.update(f, i / fps)
            first.setdefault(r['state'], i / fps)
        return first

    rng = np.random.default_rng(0)
    order = rng.permutation(n)             # decorrelate baseline and live frames
    sc = scene[order]
    ex = excess[order]
    onset = 10
    for scale in args.scales:
        fr = sc.copy()
        fr[nb + onset:] += scale * ex[nb + onset:]
        first = run_case(f'scale {scale}', synth.requantize(fr, 255, rng))
        res['cases'].append({'case': 'trapped', 'scale': scale, 'first': first,
                             'expected': 'confirmed' if scale >= args.expect_min_scale else 'any'})
    first = run_case('empty', synth.requantize(sc, 255, rng))
    res['cases'].append({'case': 'empty', 'first': first, 'expected': 'no candidate'})
    # falling transit: the particle cut-out swept vertically through the scene in ~0.4 s
    fr = sc.copy()
    h = fr.shape[1]
    k = int(0.4 * fps) + 1
    for j in range(k):
        shift = int(-h / 2 - 30 + j * (h + 60) / k)
        e = np.roll(ex[nb + onset + j], shift, axis=0)
        if shift > 0:
            e[:shift] = 0          # no wrap-around: the particle enters from the top
        elif shift < 0:
            e[shift:] = 0
        fr[nb + onset + j] += args.transit_scale * e
    first = run_case('transit', synth.requantize(fr, 255, rng))
    res['cases'].append({'case': 'transit', 'scale': args.transit_scale, 'first': first,
                         'expected': 'no candidate'})
    ok = True
    for cse in res['cases']:
        if cse['expected'] == 'confirmed':
            good = 'confirmed' in cse['first']
        elif cse['expected'] == 'no candidate':
            good = 'candidate' not in cse['first'] and 'confirmed' not in cse['first']
        else:
            good = True
        cse['pass'] = good
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {cse['case']:8s} scale={cse.get('scale', '-')!s:6s} "
              f"first-seen={ {k: round(v, 2) for k, v in cse['first'].items()} }")
    res['all_pass'] = ok
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'replay_report.json'), 'w') as f:
            json.dump(res, f, indent=1, default=ledger._json_default)
    return 0 if ok else 1


def cmd_ledger(args):
    if args.set_mass:
        eid, pg = args.set_mass
        r = ledger.set_field(eid, 'epstein_mass_pg', float(pg), args.ledger)
        print(f"set epstein_mass_pg={pg} on {eid}")
        return 0
    for r in ledger.load(args.ledger):
        est = r.get('size_estimate') or {}
        print(f"{r['id']}  {r.get('kind', ''):11s} B={r.get('brightness') or float('nan'):.4g}  "
              f"exp={r.get('exposure_us')}us  m={r.get('epstein_mass_pg', '-')}pg  "
              f"d=[{(est.get('d_min_m') or 0) * 1e9:.0f},{(est.get('d_max_m') or 0) * 1e9:.0f}]nm  "
              f"{r.get('decision', '')}")
    return 0


# ----------------------------------------------------------------------------- CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--serial', default=X_CAM_SERIAL, help='x-cam serial')
        p.add_argument('--vendors', nargs='+', default=['ids'],
                       help="camera vendors to enumerate; add 'thorlabs' only when dualcam_fast "
                            "does not hold the Thorlabs camera (discovery segfaulted otherwise)")
        p.add_argument('--exposure-us', type=float, default=50000)
        p.add_argument('--min-exposure-us', type=float, default=2000)
        p.add_argument('--max-exposure-us', type=float, default=140000,
                       help='photometry ladder ceiling; the 300-row ROI allows ~156 ms')
        p.add_argument('--n-measure', type=int, default=60, help='frames per photometry rung')
        p.add_argument('--r-ap', type=int, default=photometry.DEFAULT_R_AP)
        p.add_argument('--trap-xy', type=float, nargs=2, default=TRAP_XY,
                       help='nominal trap position, FULL-SENSOR pixels (converted into the ROI)')
        p.add_argument('--roi', type=int, nargs=4, default=DEFAULT_ROI, metavar=('X', 'Y', 'W', 'H'),
                       help='x-cam AOI; full width because the trap z (horizontal) varies. '
                            "'--roi 0 0 0 0' = full sensor")
        p.add_argument('--region-px', type=float, default=600.0,
                       help='trap-region radius: detections must stay inside it (falling '
                            'particles are rejected by the drift/speed test, not by this)')
        p.add_argument('--pol-deg', type=float, nargs='+', default=[0.0, 45.0, 90.0],
                       help='z-laser polarisation vs x-cam axis; all three = unknown')
        p.add_argument('--na', type=float, default=0.1, help='x-cam collection NA (model only)')
        p.add_argument('--accept-max-nm', type=float, default=1000.0,
                       help='accept only if EVERY sphere diameter consistent with the brightness '
                            'is below this; reject if every one is above')
        p.add_argument('--size-spread', type=float, default=0.0,
                       help='log-normal size spread for the Mie curve; 0 = ideal spheres (deepest '
                            'dips, most conservative), 0.1 = irregular clusters')
        p.add_argument('--ledger', default=ledger.DEFAULT_PATH)
        p.add_argument('--out-dir')
        p.add_argument('--telegram', action='store_true', help='also send the final alert to Telegram')
        # replay
        p.add_argument('--replay', help='recorded x-cam stack (.npy) to replay instead of the camera')
        p.add_argument('--replay-particle-xy', type=float, nargs=2, default=TRAP_XY)
        p.add_argument('--replay-scene-xy', type=float, nargs=2, default=(330.0, 330.0),
                       help='centre of a particle-free region used as the empty trap')
        p.add_argument('--replay-half', type=int, default=300)
        p.add_argument('--replay-half-y', type=int, default=None,
                       help='scene half-height if different (a strip ROI recording)')
        p.add_argument('--replay-exposure-us', type=float, default=49996)
        p.add_argument('--replay-fps', type=float, default=13.2)
        p.add_argument('--n-baseline', type=int, default=30)
        p.add_argument('--k-sigma', type=float, default=6.0)
        p.add_argument('--k-occupancy', type=float, default=5.0)
        p.add_argument('--smooth-r', type=int, default=25,
                       help='detector box-smoothing half-width (px). 25 confirmed a 0.3%%-of-'
                            'reference particle in replay (12: 0.5%%) with no false candidate '
                            'in 66 s of empty scene (2026-09-28, lights off, 500 uW)')
        p.add_argument('--no-destripe', action='store_true',
                       help='skip the per-row read-noise removal in the detector')
        p.add_argument('--persist-s', type=float, default=0.5)
        p.add_argument('--confirm-s', type=float, default=3.0)

    p = sub.add_parser('identify-cams')
    common(p)
    p = sub.add_parser('measure')
    common(p)
    p.add_argument('--reference-mass-pg', type=float,
                   help='independent (Epstein) mass of THIS particle: makes the entry a reference')
    p.add_argument('--from-npy', help='measure a recorded stack instead of the live camera')
    p = sub.add_parser('replay')
    common(p)
    p.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.1, 0.05, 0.03, 0.02, 0.01])
    p.add_argument('--expect-min-scale', type=float, default=0.03)
    p.add_argument('--transit-scale', type=float, default=1.0)
    p = sub.add_parser('run')
    common(p)
    p.add_argument('--conf', default=DEFAULT_CONF, help='conf_sweep.yaml for the dropper chirp')
    p.add_argument('--real-moku', action='store_true', help='drive the real piezo (default: fake Moku)')
    p.add_argument('--max-attempts', type=int, default=20)
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--replay-schedule', type=float, nargs='+', default=[1.0, 0.02])
    p.add_argument('--replay-delay-s', type=float, default=1.5)
    p.add_argument('--replay-transit', type=float, default=None)
    p.add_argument('--replay-occupied', action='store_true')
    p.add_argument('--replay-simulate-eject', action='store_true',
                   help='replay only: pretend the rejected particle was ejected, to test the loop')
    p = sub.add_parser('ledger')
    p.add_argument('--ledger', default=ledger.DEFAULT_PATH)
    p.add_argument('--set-mass', nargs=2, metavar=('ID', 'PG'))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, 'trap_xy', None) is not None:
        args.trap_xy = tuple(args.trap_xy)
    if getattr(args, 'roi', None) is not None:
        args.roi = tuple(args.roi) if any(args.roi) else None
        if args.roi and not getattr(args, 'replay', None) and not getattr(args, 'from_npy', None):
            # every frame the loader sees is ROI-sized: move the trap into ROI pixels
            args.trap_xy = (args.trap_xy[0] - args.roi[0], args.trap_xy[1] - args.roi[1])
    if args.cmd == 'run' and args.real_moku and args.replay:
        sys.exit("--real-moku with --replay makes no sense")
    return {'identify-cams': cmd_identify, 'measure': cmd_measure, 'replay': cmd_replay,
            'run': cmd_run, 'ledger': cmd_ledger}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
