#!/var/lib/cds-conda/base/envs/cds-testing/bin/python3
# Run with: /var/lib/cds-conda/base/envs/cds-testing/bin/python3 particle_lo_scan.py <params.yaml>

import sys
import time
import subprocess
import pathlib

import numpy as np
import yaml
import h5py
import nds2

PV_COSGAIN = "Y1:DMD-PARTICLE_LO_COSGAIN"
PV_FREQ    = "Y1:DMD-PARTICLE_LO_FREQ"
PV_TRAMP   = "Y1:DMD-PARTICLE_LO_TRAMP"

GPS_OFFSET = 315964818  # GPS - Unix
ZERO_TRAMP = 1.0        # TRAMP to use when transitioning COSGAIN to or from 0


def caget(pv):
    return subprocess.check_output(["caget", "-t", pv], text=True).strip()


def caput(pv, value):
    subprocess.run(["caput", pv, str(value)], check=True, capture_output=True)


def filename_for(cosgain, freq):
    return f"cosgain_{cosgain:.6g}_freq_{freq:.6g}.hdf5"


def hdf5_key(channel):
    return channel.replace(":", "_")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <params.yaml>")
        sys.exit(1)

    params_path = pathlib.Path(sys.argv[1])
    params_yaml_text = params_path.read_text()
    params = yaml.safe_load(params_yaml_text)

    cosgain_values = np.geomspace(params["cosgain_min"], params["cosgain_max"], params["cosgain_steps"])
    freq_values    = np.geomspace(params["freq_min"],    params["freq_max"],    params["freq_steps"])
    # Outer loop: freq; inner loop: cosgain high→low. Minimizes freq changes; starts at safe high amplitude.
    grid = [(cg, fr) for fr in freq_values for cg in cosgain_values[::-1]]
    n_total = len(grid)

    wait_time_freq    = float(params["wait_time_freq"])
    wait_time_cosgain = float(params["wait_time_cosgain"])
    record_duration   = float(params["record_duration"])
    nds2_server     = params["nds2_server"]
    nds2_port       = int(params["nds2_port"])
    channels        = params["channels"]
    run_label       = params.get("run_label", "scan")
    output_root     = pathlib.Path(params["output_root"]).expanduser()

    # Build output directory
    t_start = time.localtime()
    date_dir   = time.strftime("%y%m%d", t_start)
    ts_dir     = time.strftime("%Y%m%d_%H%M%S", t_start)
    run_dir    = output_root / date_dir / f"{ts_dir}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Output directory: {run_dir}")
    print(f"Grid: {params['cosgain_steps']} COSGAIN × {params['freq_steps']} FREQ = {n_total} points")
    print(f"Timing: wait_freq={wait_time_freq}s  wait_cosgain={wait_time_cosgain}s  record={record_duration}s")
    print(f"Channels: {channels}")
    print()

    # Pre-scan: check no output files exist (abort before touching any PVs)
    for cosgain, freq in grid:
        fp = run_dir / filename_for(cosgain, freq)
        if fp.exists():
            print(f"ERROR: Output file already exists: {fp}")
            sys.exit(1)

    # Save original PV values
    orig_cosgain = caget(PV_COSGAIN)
    orig_freq    = caget(PV_FREQ)
    orig_tramp   = caget(PV_TRAMP)
    print(f"Original values: COSGAIN={orig_cosgain}  FREQ={orig_freq}  TRAMP={orig_tramp}")

    restored = [False]

    def restore_pvs():
        if not restored[0]:
            restored[0] = True
            print("\nRestoring original PV values...")
            # Use fast TRAMP if restoring to/from COSGAIN=0
            restore_tramp = ZERO_TRAMP if (float(orig_cosgain) == 0.0 or prev_cosgain == 0.0) else float(orig_tramp)
            caput(PV_TRAMP,   restore_tramp)
            caput(PV_COSGAIN, orig_cosgain)
            caput(PV_FREQ,    orig_freq)
            caput(PV_TRAMP,   orig_tramp)
            print(f"  COSGAIN={orig_cosgain}  FREQ={orig_freq}  TRAMP={orig_tramp}")

    prev_cosgain = float(orig_cosgain)
    prev_freq    = None
    t_scan_start = time.time()
    files_saved  = 0

    try:
        for step_idx, (cosgain, freq) in enumerate(grid, start=1):
            elapsed = time.time() - t_scan_start
            time_per_step = elapsed / (step_idx - 1) if step_idx > 1 else (wait_time_freq + record_duration + 2)
            remaining_steps = n_total - step_idx + 1
            est_remaining_min = time_per_step * remaining_steps / 60

            freq_changed = (prev_freq is None or freq != prev_freq)
            print(f"[{step_idx}/{n_total}] COSGAIN={cosgain:.6g}  FREQ={freq:.6g}  "
                  f"(est. remaining: {est_remaining_min:.1f} min)")

            # Use a fast TRAMP when transitioning to/from COSGAIN=0 to avoid dwelling at low amplitude;
            # otherwise use half of whichever wait time applies.
            if prev_cosgain == 0.0 or cosgain == 0.0:
                tramp = ZERO_TRAMP
                print(f"  Set TRAMP = {tramp} s (zero-crossing transition)")
            elif freq_changed:
                tramp = wait_time_freq / 2
                print(f"  Set TRAMP = {tramp} s (freq change)")
            else:
                tramp = wait_time_cosgain / 2
                print(f"  Set TRAMP = {tramp} s")
            caput(PV_TRAMP, tramp)

            caput(PV_COSGAIN, cosgain)
            caput(PV_FREQ,    freq)
            prev_cosgain = cosgain
            prev_freq    = freq

            if step_idx == 1:
                this_wait = wait_time_freq * 2
                print(f"  Waiting {this_wait:.0f} s for ramp to settle  (2× initial wait)...")
            elif freq_changed:
                this_wait = wait_time_freq
                print(f"  Waiting {this_wait:.0f} s for ramp to settle  (freq change)...")
            else:
                this_wait = wait_time_cosgain
                print(f"  Waiting {this_wait:.0f} s for ramp to settle...")
            time.sleep(this_wait)

            # Stream live data via iterate() — test point channels are not stored in frames
            # Each stride is 1 second; skip first stride (may be partial), then collect record_duration strides.
            n_strides = int(record_duration)
            print(f"  Streaming {n_strides} s of live NDS2 data...")
            conn = nds2.connection(nds2_server, nds2_port)
            accumulated = {ch: [] for ch in channels}
            gps_start = None
            gps_end   = None
            strides_collected = 0
            for stride_idx, bufs in enumerate(conn.iterate(channels)):
                if stride_idx == 0:
                    # skip first stride — it may already be in progress and thus partial
                    continue
                if gps_start is None:
                    gps_start = bufs[0].gps_seconds
                for buf in bufs:
                    accumulated[buf.channel.name].append(np.array(buf.data, dtype=np.float32))
                gps_end = bufs[0].gps_seconds + 1
                strides_collected += 1
                print(f"    stride {strides_collected}/{n_strides}  GPS {bufs[0].gps_seconds}", flush=True)
                if strides_collected >= n_strides:
                    break

            filepath = run_dir / filename_for(cosgain, freq)
            with h5py.File(filepath, "w") as f:
                f.attrs["cosgain"]     = cosgain
                f.attrs["freq"]        = freq
                f.attrs["gps_start"]   = gps_start
                f.attrs["gps_end"]     = gps_end
                f.attrs["params_yaml"] = params_yaml_text

                for buf in bufs:
                    key = hdf5_key(buf.channel.name)
                    arr = np.concatenate(accumulated[buf.channel.name])
                    ds = f.create_dataset(key, data=arr)
                    ds.attrs["sample_rate"] = buf.channel.sample_rate

            print(f"  Saved: {filepath.name}")
            files_saved += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        restore_pvs()

    total_elapsed = time.time() - t_scan_start
    print(f"\nDone. {files_saved}/{n_total} files saved in {total_elapsed/60:.1f} min.")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
