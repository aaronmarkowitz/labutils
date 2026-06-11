# `scripts/dipole` — dipole / actuation acquisition + deploy

Scripts that talk to the live `Y1:DMD` front end for the magnetically levitated
microdiamond's **sensing** and **actuation** characterization. The heavy *analysis*
lives in a separate repo, the dipole pipeline at
`/home/controls/analysis/mastqg/dipole_pipeline/` (steps 01–08); these labutils
scripts are the **acquisition and EPICS-deploy** half (mirror of the
`step_01 → upload_sense_matrix.py` split).

| File | Purpose |
|---|---|
| `upload_sense_matrix.py` | Upload a sensor-diagonalization (SENSE) matrix to EPICS (consumes pipeline step_01 HDF5). |
| `sense_matrix_config.yml` | SENSE matrix layout for `upload_sense_matrix.py`. |
| `measure_actuator_gain.py` | **Measure** the flat actuator gain of POLES_E1..E4 to particle X/Y/Z via a `diag` SineResponse; extract a complex 3×4 coupling matrix. |
| `measure_actuator_gain_config.yml` | Parameters for `measure_actuator_gain.py`. |
| `abort_actuator_gain.sh` | Emergency abort (touches the sentinel file; wire to an MEDM button). |
| `tests/` | pytest suite (pure-logic) + opt-in `-m loopback` live hardware self-test. |

Run with the **cds-testing** interpreter (`/var/lib/cds-conda/base/envs/cds-testing/bin/python3`)
— it has `nds2`, `dttxml`, `numpy`, `scipy`, `h5py`, `yaml`, `pytest`. The `mastqg`
conda env (used by the analysis pipeline) has `dttxml` but **not** `nds2`/`pytest`.

---

## `measure_actuator_gain.py`

### What it does
Drives all four electrodes (`Y1:DMD-POLES_E{1..4}_EXC`) **simultaneously**, each at
several **distinct** sine tones clustered around the X/Y/Z resonances (dense near
X≈39 / Y≈54 Hz, sparse near Z≈5 Hz). `diag` is used to **inject** the comb (its
injection is exact); we then capture the raw `PARTICLE_X/Y/Z_IN1` + `POLES_E*_EXC`
channels over **NDS2** and compute the transfer functions + coherence **ourselves**
(`TF = S_ER/S_EE`, coherence from Welch averaging). We do NOT use diag's SineResponse
*coefficient* output — it is unreliable for a multi-tone comb in dtt 4.1.x (see
"How diag works" below); diag's result XML is still saved for diaggui/archiving.
Frequency-division multiplexing means each tone is owned by one electrode, so the TF
at that tone is that electrode's coupling into each DOF. Because all electrodes drive
the *same* mechanical mode per DOF, the common plant `H_d(f)` is fit (Lorentzian
f0, Q) and divided out, leaving the per-electrode **relative flat gains** (∝ counts→N).
Output: a complex 3×4 matrix (rows X/Y/Z, cols E1..E4) + per-tone TF/coherence +
fitted f0/Q, in `$MQG_DROPBOX_PATH/worker1/data/YYMMDD/<timestamp>_actgain/`.
Validated on the ACTS_8_8→LOS_IN1 loopback: 0.500 flat at every tone, coherence 1.0.

```
measure_actuator_gain.py <config.yml> [--dry-run] [--premeasure-only] [--label NAME]
                                       [--emit-xml] [--diaggui] [--analyze RAW_CAPTURE_H5]
```
- `--dry-run` — build the tone plan + an example XML, touch **no** hardware (run first).
- `--emit-xml` — write the comb diag XML for you to open in diaggui manually; no hardware.
- `--diaggui` — write the XML and launch `diaggui` on it so you can watch the injection
  live (its multi-tone *coefficient* readout is unreliable — use it to watch, not to read
  gains).
- `--analyze RAW_CAPTURE_H5` — recompute the 3×4 matrix offline from a saved
  `raw_capture.h5` (lets you re-tune `analysis.segment_s`, f0/Q seeds, etc. without
  re-running hardware). The full run saves `raw_capture.h5` when
  `analysis.save_raw_capture` is true.

### Pipeline / control flow
1. Generate the tone plan (guard-band-clean, bin-snapped, distinct; every electrode
   driven near every DOF) and Schroeder-phase each electrode's tones.
2. Snapshot all POLES_E settings; **turn off each module's input switch** (open-loop:
   only the `_EXC` test point drives the DAC).
3. Start the NDS2 trap-loss guard; record a 10–20 Hz band-RMS baseline.
4. Adaptive trim loop (**time-first, then amplitude**): short pre-measurement
   (diag injects, we capture raw NDS2 + compute TFs/coherence) → lengthen the capture
   (more Welch averages, capped for Z) and, only if needed, raise per-tone amplitude
   up to the DAC cap → repeat until X/Y coherence meets target.
5. Final full measurement (inject + capture + compute) → fit plant + gains → write
   HDF5 + report (and diag's result XML alongside).
6. `finally`: ramp excitation to zero, restore all POLES settings (idempotent).

### Safety (read before running on a real particle)
- **Trap-loss guard**: background NDS2 monitor of the **10–20 Hz** band-RMS of
  PARTICLE_X/Y/Z. If it exceeds `guard_monitor.factor` × baseline on any DOF → abort
  + ramp to zero. (z-mode harmonics live at ~10–15 Hz and dominate trap loss.)
- **No excitation tone is ever placed in 10–20 Hz** — a tone there would both
  endanger the trap and pollute the guard (false aborts). Enforced + unit-tested.
- **Manual abort**: Ctrl-C / SIGTERM, OR `abort_actuator_gain.sh` (touches the
  sentinel file the script polls each loop). Wire the latter to a red MEDM
  "shell command" button (see the script header for the `.adl` snippet).
- **POLES restore**: every POLES_E GAIN/OFFSET/TRAMP/switch is snapshotted and
  restored on any exit path.
- v1 assumes **open loop**. A closed-loop variant (inject on top of a live trap loop)
  is future work.

### Deployment note — exercises cymac1 hardware, but runs from worker1
`Y1:DMD` is FEC-11 on **cymac1** (65536 Hz); the DAC/ADC, awg, and test-point
managers live there. A real measurement therefore exercises cymac1 hardware — but
you run the script **from worker1**: `diag`/`diaggui` reach cymac1's awg/tp/nds over
the network (`diag -i`; `NDSSERVER=192.168.1.11:8088`). Fast channels (`_IN1`,
`_EXC`, `_OUT`) are test points, not EPICS PVs, so `caget` cannot see them — read
them via NDS2/diag (the guard monitor streams them over NDS2). Slow monitors
(`_INMON`, `_OUTMON`, `_GAIN`, switches) are normal CA PVs.

---

## How `diag` / AWG / test points work here (hard-won notes for future edits)

- **Excitation & fast readback go through the AWG/test-point/NDS layer, not plain
  CA.** Slow filter-module records (`..._GAIN`, `_OFFSET`, `_TRAMP`, `_SW1R`,
  `_SW2R`) answer `caget`. Fast channels (`_EXC`, `_IN1`, `_OUT16`) do **not** —
  they are test points served by the front end and reached via `diag`/`awg`/`nds2`.
  Don't try to `caget`/`caput` an `_EXC` channel.
- **`diag` is the headless diaggui.** It runs a diaggui measurement XML.
  Invocation used here: `diag -l -f <cmdfile>` (`-l` = local kernel, `-f` = read a
  command script). VERIFIED post-connect verbs (from `diag -l` interactive `help` and
  loopback runs): `restore <xml>` / `run -w` (run + wait) / `save <result.xml>` /
  `quit`. **Paths are UNQUOTED** — diag's help prints `restore 'filename'` but the
  quotes are placeholder notation; passing a quoted path gives "Unable to open input
  file". These live in `DIAG_COMMAND_SEQUENCE` at the top of the script.
  `diag -l` runs a local diagnostics kernel that still connects to the **networked**
  awg/test-point/NDS managers in its config — `diag -i` shows `awg`/`tp` for FEC
  10/11/12 and `nds` all at `192.168.1.11` (cymac1), and the env has
  `NDSSERVER=192.168.1.11:8088`. So `diag` (and `diaggui`) run fine **from worker1**
  and reach the Y1:DMD front end; you do not have to be logged into cymac1. The
  `-m loopback` self-test is gated behind `ACTGAIN_LOOPBACK=1` so it never runs by
  accident (it drives a real DAC).
- **SineResponse XML schema** (`Test` block, `Subtype=SineResponse`): index-aligned
  arrays `StimulusFrequency[i]` / `StimulusAmplitude[i]` / `StimulusOffset[i]` /
  `StimulusPhase[i]` (radians) and `StimulusChannel[i]` / `StimulusActive[i]`;
  `MeasurementChannel[i]` (+`Rate`/`Active`); plus `MeasurementTime` (Dim2),
  `SettlingTime`, `RampUp`/`RampDown`, `Averages`, `AverageType`, `Window`.
  **Multi-tone on one channel = repeat that channel across rows** (verified against a
  real comb result where `StimulusChannel[0..7]` were all the same channel). So
  `#stimulus rows == #tones`. **Gotchas (cost real debugging time):**
  - Must include `<Param Name="FFTResult" Type="boolean">false</Param>` or `run`
    aborts with "Unable to load value from Test.FFTResult".
  - `MeasurementTime` Dim2 = `[min_time_seconds, n_cycles]` (per gds `SweptSine.hh`:
    `SweptSine(... double cycles, double mintime ...)`). The measurement runs for
    `max(min_time, n_cycles/freq)` — seconds floor at low freq, cycles floor at high.
    A tiny first value gives coarse FFT bins. (The user's scratch template "0.1 10"
    = 0.1 s / 10 cycles; a real comb used "10 300".)
  - `SettlingTime` is a RELATIVE fraction of the measurement time, not seconds.
- **DO NOT trust diag's multi-tone coefficient output.** In dtt 4.1.x the SineResponse
  per-tone *excitation* normalization (`sineAnalyze` on the drive channel, divides by
  the diagonal self-term in `sineresponse.cc`) returns wrong amplitudes for most tones
  of a comb — coefficients come back as raw response (≈amp×gain) or NaN, with wrong
  phase. **The captured DATA is perfect** (raw NDS2 of the comb: excitation = commanded
  amplitude and response = gain×amplitude at *every* tone). So inject with diag, then
  **compute TFs ourselves from raw NDS2**:
  ```python
  # capture PARTICLE_*_IN1 + POLES_E*_EXC over NDS2 during the injection, then:
  f, Pee = scipy.signal.csd(exc, exc, fs=fs, nperseg=int(seg_s*fs))
  _, Per = scipy.signal.csd(exc, resp, fs=fs, nperseg=int(seg_s*fs))
  _, Prr = scipy.signal.csd(resp, resp, fs=fs, nperseg=int(seg_s*fs))
  TF  = Per[i]/Pee[i]                                   # response/excitation at tone i
  coh = abs(Per[i])**2 / (abs(Pee[i])*abs(Prr[i]))      # 0..1
  ```
  (`dttxml.DiagAccess(xml).sine_response([chans], freq_idx=k)` → `.FHz`,
  `.coeffs_dict`, `.cohs_dict` still parses the result XML if you want it, but its
  multi-tone coefficients are unreliable per above.)
- **DAC**: 16-bit signed, ±32768 counts; practical safe clip ~32000
  (`amplitude.max_amplitude_counts`). No software limit was found textually in
  `y1dmd.mdl`; the real per-tone safe level is far below clip and is found by the
  trim loop.
- **Schroeder phasing** matters only because each electrode (DAC) carries *multiple*
  tones; it minimizes the summed waveform's crest factor so a given RMS drive uses
  less peak DAC range. Recompute whenever amplitudes change.

### Verified on the loopback (`ACTS_8_8_EXC` → `LOS_IN1`)
All confirmed against the live FE (June 2026):
1. ✅ diag command sequence (`restore`/`run -w`/`save`/`quit`, unquoted paths) runs and
   writes a result; `FFTResult` required; `MeasurementTime` = `[min_time, cycles]`.
2. ✅ index-aligned freq→channel binding (multi-tone via repeated channel rows) injects
   correctly (raw excitation = commanded amplitude at every tone).
3. ✅ our NDS2-capture + `csd`-based TF/coherence gives the loopback gain **0.500 flat**
   at every tone, coherence 1.0.
4. ⚠️ diag's own multi-tone *coefficient* output is unreliable — not used (see above).
5. (manual, operator) the guard trips + ramps to zero on a growing 10–20 Hz inject.

Re-run with `ACTGAIN_LOOPBACK=1 ... -m pytest -m loopback` after any change to the
injection/analysis path.

---

## Testing
```
cd /home/controls/labutils/scripts/dipole
/var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest tests/ -m "not loopback"   # pure logic, no hardware
ACTGAIN_LOOPBACK=1 /var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest tests/ -m loopback   # live FE only
```
The pure-logic suite covers the frequency plan (guard-band exclusion, distinctness,
density, snapping), Schroeder crest-factor reduction, diag-XML round-trip,
`compute_tfs` (recovers known TFs + coherence from synthetic captured data), the
plant+gain fit (recovers known gains from synthetic data), the guard band-RMS math,
and POLES snapshot/restore + amplitude clamping (mocked EPICS).

## dtt version note
The installed dtt is `4.1.5~rc1` from the `bullseye-unstable` apt channel (stable
`bullseye` has `4.1.4`). The multi-tone coefficient behavior is the same across these
(`sineresponse.cc` unchanged since Jan 2024), so our inject-and-compute-ourselves
approach is version-independent. You may still want to pin dtt to stable `4.1.4` for
general hygiene (separate sysadmin step).

## Future work
- Generalize the flat-gain assumption to a frequency-dependent actuation matrix
  (electrode capacitance becomes relevant at higher frequency).
- Closed-loop variant (inject on top of a live trap feedback loop).
- A companion `upload_actuation_matrix.py` (analogue of `upload_sense_matrix.py`) to
  push the measured matrix to EPICS, and/or wire the HDF5 into the pipeline's
  `step_02_actuator_diagonalization.py` (currently a stub).
