# `scripts/dipole` — dipole / actuation acquisition + deploy

Scripts that talk to the live `Y1:DMD` front end for the magnetically levitated
microdiamond's **sensing** and **actuation** characterization. The heavy *analysis*
lives in a separate repo, the dipole pipeline at
`/home/controls/analysis/mastqg/dipole_pipeline/` (steps 01–08); these labutils
scripts are the **acquisition and EPICS-deploy** half (mirror of the
`step_01 → upload_sense_matrix.py` split).

| File | Purpose |
|---|---|
| `upload_sense_matrix.py` | Upload a sensor-diagonalization (SENSE) matrix to EPICS (consumes pipeline step_01 HDF5). Rows read out the particle along **any direction** (axis / xy angle / full-sphere elevation+azimuth / explicit vector). Also **force-enables the input filter modules** feeding each written SENSE column (GAIN=1, in/out switches on) so a zeroed input cannot silently delete a W column. |
| `sense_matrix_config.yml` | SENSE matrix layout + directional row spec for `upload_sense_matrix.py`. |
| `verify_particle_equipartition.py` | Validate a deployed SENSE matrix: reads recorded `PARTICLE_X/Y/Z` from a CSD and checks equipartition `(2πf0)²·Var` across DOFs. PRIMARY FOM = `--baseline <step01_results.h5>` → per-DOF ratio `relT_new/relT_baseline` (geomean-norm), spread is the FOM. Plus per-DOF DHO fit cross-check + inter-DOF coherence. Run after upload on freshly-recorded diagonalized data. |
| `measure_actuator_gain.py` | **Measure** the flat actuator gain of POLES_E1..E4 to active particle DOFs via a `diag` SineResponse; extract a complex K×4 coupling matrix (K = number of active DOFs, typically 2 or 3). |
| `measure_actuator_gain_config.yml` | Parameters for `measure_actuator_gain.py`. |
| `upload_actuation_matrix.py` | **Invert** measured actuator gains and write the ACTS matrix so each ACTS column drives the E-field in a chosen direction. |
| `upload_actuation_matrix_config.yml` | ACTS inversion config (per-column direction/coupling) for `upload_actuation_matrix.py`. |
| `utility.py` | Shared coordinate-system parser (direction → unit vector) used by both the ACTS and SENSE uploaders. |
| `abort_actuator_gain.sh` | Emergency abort (touches the sentinel file; wire to an MEDM button). |
| `tests/` | pytest suite (pure-logic) + opt-in `-m loopback` live hardware self-test. |
| `FUTURE_WORK.md` | Deferred work (matrix-layout auto-extraction, z actuator couplings). |

Run with the **cds-testing** interpreter (`/var/lib/cds-conda/base/envs/cds-testing/bin/python3`)
— it has `nds2`, `dttxml`, `numpy`, `scipy`, `h5py`, `yaml`, `pytest`. The `mastqg`
conda env (used by the analysis pipeline) has `dttxml` but **not** `nds2`/`pytest`.

---

## `upload_sense_matrix.py` — deploy W, and the equipartition calibration it carries

Uploads the step-01 demodulation matrix `W` to the `Y1:DMD-SENSE_{row}_{col}_GAIN`
elements (`value = n·W` per directional row). Two things to understand before trusting
the resulting `PARTICLE_X/Y/Z`:

**1. The SENSE inputs are the LES filter-module OUTPUTS, not the IN1 testpoints.** `W` is
computed against the sensor channels, but the matrix multiplies whatever the filter
modules pass. If a module is gain-zeroed or switched out (observed live: `Y1:DMD-PBSZ_GAIN
= 0`), that `W` column is effectively deleted and the diagonalization breaks — typically
PARTICLE_X and PARTICLE_Y then *both* show the y resonance, because the dropped channel
(PBSZ) was the one separating x from y in dof-space. The uploader now **force-enables every
input module feeding a written column** (sets GAIN=1, enables in/out switches) and prints
the planned changes under `--dry-run`. Always `--dry-run` first.

**2. `W` carries a per-mode displacement calibration set by the step-01 `common_unit_anchor`.**
The pipeline default is **equipartition** (`s ∝ 1/(f0·√(A·Γ))`, equal k_BT per mode) — robust
for an equilibrium (gas-damping-dominated) particle regardless of drag anisotropy, and
correct for the downstream actuator/dipole chain. The legacy **white_force** anchor
(`s ∝ 1/(f0·Γ·√A)`, isotropic force noise) imposes a spurious `1/√Γ` x/y calibration tilt
when per-mode Γ differ (by fluctuation–dissipation `S_FF,i ∝ Γ_i`), which propagates straight
into the actuator coefficients. The two coincide under shared Γ. The applied anchor is
recorded in the h5 `mode_scale_anchor` attribute — check it is `equipartition` for any W that
feeds actuator measurement. (Full derivation: dipole_pipeline README "Common-unit
normalization".)

**Validate after upload** with `verify_particle_equipartition.py <new_CSD.xml> --baseline
<step01_results.h5>`: record a CSD with the diagonalized `PARTICLE_X/Y/Z` channels and confirm
each DOF reproduces the baseline equipartition shape. Equipartition is **pressure-independent
in equilibrium**, so this should hold even at a different gas pressure than `W` was fit on; a
reproducible deviation ≫ fit noise then indicates a real non-equilibrium effect (feedback /
multiple baths / mode-dependent heating), not a calibration error. The script fits a DHO+floor
model per peak and integrates the DHO term (`πAΓ/2`) to reject floor/sensing noise — and
mains-masks + floor-subtracts the empirical band integral (a naive integral lets a 60 Hz line
in the y band or the 1/f wall in the z band inflate the apparent T by up to ~10×). This
estimator is the **single source of truth** in `dipole_pipeline/diagnostics/equipartition.py`,
imported by both this validator and the step-01 self-check (`particle_xyz_from_W.png`,
auto-emitted by step_01), so both report the same rigorous DHO-based relT.

**The apples-to-apples yardstick (read this).** Because the validator **applies `W` to data
and re-fits a DHO+floor** (it does *not* reuse the anchor's `A=1, Γ_fit`), the recovered relT
is **NOT 1.0/1.0/1.0 even on the CSD `W` was fit from** — it is a fixed `(W, estimator,
dataset)` shape (e.g. 1.00/0.83/1.03). Comparing relT to an idealized 1.0 is apples-to-oranges.
Instead pass `--baseline <step01_results.h5>`: the script **reconstructs** the in-sample
baseline relT by applying that run's `W` to its fit CSD (`source_csd_path`, persisted by
step_01) and reducing with the *current* estimator, then reports the per-DOF **ratio**
`relT_new / relT_baseline` (geomean-normalized, so global T/pressure cancels) and its **spread
as the FOM**. The estimator-induced shape divides out, so ratio ≈ 1 means equipartition is
reproduced. Reconstructing (rather than freezing a number) keeps old baselines comparable as
the estimator evolves. **Each ratio carries a physical ±1σ error bar** — the finite-averaging
(χ²) uncertainty `σ(S)=S/√n_avg` propagated through `Var=A·πΓ/2` (via the weighted DHO fit
covariance), combined for both legs in quadrature; the per-mode `W` normalization cancels in
the ratio so it does not enter. There is **no arbitrary tolerance band** — a DOF is consistent
with equipartition if its ratio sits within ±1σ of 1, and the console prints the worst
`|ratio−1|/σ`. The bar shrinks as `1/√n_avg`, so a high-average CSD is judged more tightly.
*Stale-W caveat:* a baseline given as a **CSD xml** is only valid if
recorded **after** the `W` upload — the fit CSD's own `PARTICLE_*` channels are stale (recorded
under the previous SENSE matrix); the robust baseline is the step-01 `results.h5`. The legacy
`--disagreement` band is **deprecated and not the yardstick** (it draws `±exp(disagreement)`,
`disagreement = max|ln(s_white/s_equip)| = ½·spread(lnΓ)`, i.e. only the equip-vs-`white_force`
anchor gap — irrelevant once committed to equipartition).

## `measure_actuator_gain.py`

### What it does
Drives all four electrodes **simultaneously**, each at several **distinct** sine tones
clustered around the X/Y/Z resonances (dense near X≈39 / Y≈54 Hz, sparse near Z≈5 Hz).
Each tone is assigned to its own **distinct physical excitation channel** (`ACTS_N_M_EXC`,
one per tone), which makes diag's SineResponse `sizeA == sizeExc` — enabling its native
per-tone coefficient extraction. We additionally capture the raw `PARTICLE_X/Y/Z_IN1` +
`ACTS_N_M_EXC` channels over **NDS2** and compute the transfer functions + coherence
**ourselves** (`TF = S_ER/S_EE`, coherence from Welch averaging) as the primary analysis
path (validated: 0.500 flat at every tone, coherence 1.0). diag's result XML is still
saved for diaggui/archiving.
Frequency-division multiplexing means each tone is owned by one electrode, so the TF
at that tone is that electrode's coupling into each DOF. Because all electrodes drive
the *same* mechanical mode per DOF, the common plant `H_d(f)` is fit (Lorentzian
f0, Q) and divided out, leaving the per-electrode **relative flat gains** (∝ counts→N).
Output: a complex K×4 matrix (rows = active DOFs, cols E1..E4) + per-tone TF/coherence +
fitted f0/Q, in `$MQG_DROPBOX_PATH/worker1/data/YYMMDD/<timestamp>_actgain/`.
The active DOF list is derived from the step 01 sensor-diagonalization HDF5 (via
`--step01-h5`), or from the `dofs:` section of the config if no HDF5 is provided.
Validated on the ACTS_8_8→LOS_IN1 loopback: 0.500 flat at every tone, coherence 1.0.

```
measure_actuator_gain.py <config.yml> [--step01-h5 PATH] [--dry-run] [--premeasure-only]
                                       [--label NAME] [--emit-xml] [--diaggui]
                                       [--analyze RAW_CAPTURE_H5]
```
- `--step01-h5 PATH` — derive the active DOF list from a step 01 sensor-diagonalization
  HDF5 (reads its `dofs` attribute, e.g. `["x", "y"]`). Only those DOFs are measured
  and fitted. Without this flag, all DOFs defined in the config `dofs:` section are used.
- `--dry-run` — build the tone plan + an example XML, touch **no** hardware (run first).
- `--emit-xml` — write the comb diag XML for you to open in diaggui manually; no hardware.
- `--diaggui` — write the XML and launch `diaggui` on it so you can watch the injection
  live (its multi-tone *coefficient* readout is unreliable — use it to watch, not to read
  gains).
- `--analyze RAW_CAPTURE_H5` — recompute the K×4 matrix offline from a saved
  `raw_capture.h5` (lets you re-tune `analysis.segment_s`, f0/Q seeds, etc. without
  re-running hardware). Respects `--step01-h5` for DOF selection. The full run saves
  `raw_capture.h5` when `analysis.save_raw_capture` is true.

### Typical workflow (2-DOF example)

1. Run step 01 of the dipole pipeline with `dofs: [x, y]`. This produces an HDF5 at
   `<output_dir>/step_01_sensor_diagonalization_results.h5` containing the 2×N W matrix
   and a `dofs` attribute encoding `["x", "y"]`.

2. Upload the new W matrix to the rtcds SENSE filter bank:
   ```
   /var/lib/cds-conda/base/envs/cds-testing/bin/python3 upload_sense_matrix.py \
       sense_matrix_config.yml --step01-h5 <path_to_step01_results.h5>
   ```

3. Measure the actuator gain — dry-run first to verify the tone plan:
   ```
   /var/lib/cds-conda/base/envs/cds-testing/bin/python3 measure_actuator_gain.py \
       measure_actuator_gain_config.yml \
       --step01-h5 <path_to_step01_results.h5> \
       --dry-run
   ```
   Then remove `--dry-run` for the real measurement. Only the X and Y resonances are
   driven and fitted; the output is a 2×4 complex gain matrix.

4. (Future) Feed the result into step 02 of the dipole pipeline
   (`step_02_actuator_diagonalization.py`).

### Pipeline / control flow
1. Generate the tone plan (guard-band-clean, bin-snapped, distinct; every electrode
   driven near every DOF) and Schroeder-phase each electrode's tones.
2. Assign each tone to its own `ACTS_{row}_{col}_EXC` channel (`assign_acts_channels`):
   one column per tone, row = electrode. Snapshot all ACTS element settings; set
   GAIN=1, input switch OFF (blocks ACTS row input; EXC bypasses it), output switch ON
   (EXC must exit the element). Also snapshot POLES_E* and turn off their input switches.
3. Start the NDS2 trap-loss guard; record a 10–20 Hz band-RMS baseline.
4. Adaptive trim loop (**time-first, then amplitude**): short pre-measurement
   (diag injects, we capture raw NDS2 + compute TFs/coherence) → lengthen the capture
   (more Welch averages, capped for Z) and, only if needed, raise per-tone amplitude
   up to the DAC cap → repeat until X/Y coherence meets target.
5. Final full measurement (inject + capture + compute) → fit plant + gains → write
   HDF5 + report (and diag's result XML alongside).
6. `finally`: ramp excitation to zero, restore all POLES and ACTS settings (idempotent).

### Plant fit strategies (`fit_strategy`)

When `fit_plant: true`, the `fit_strategy` per-DOF option controls how f0, Q, and
per-electrode complex gains are extracted from the transfer function data:

| Strategy | Algorithm | When to use |
|----------|-----------|-------------|
| `joint` | Complex fit of f0, Q, G using ALL tones (across all DOFs). | Default when cross-coupling is negligible (X in typical runs). |
| `dof_filtered` | Same complex fit, restricted to tones with `dof_intended` matching this DOF. | General default; eliminates cross-DOF contamination from the fit. |
| `mag_then_linear` | Magnitude-only Lorentzian fit for f0/Q (on dof-intended tones), then weighted linear solve for complex G. | When phase corruption from cross-coupling causes the complex fit to converge to a wrong resonance (common for Y when X is nearby and strongly coupled). |

**Cross-coupling failure mode (motivates `mag_then_linear`):** When nearby DOFs
(e.g. X at 40 Hz, Y at 55 Hz) are measured simultaneously, X excitation produces a
correlated but wrong-phase response in the Y channel. The complex fit can latch onto
this cross-coupling signal and converge to a spurious minimum (e.g. f0=65 Hz for Y
instead of the correct ~55 Hz). The magnitude-only fit is immune because the |TF|
peak at the true resonance is unambiguous regardless of phase corruption.

### Safety (read before running on a real particle)
- **Trap-loss guard**: background NDS2 monitor of the **10–20 Hz** band-RMS of
  PARTICLE_X/Y/Z. If it exceeds `guard_monitor.factor` × baseline on any DOF → abort
  + ramp to zero. (z-mode harmonics live at ~10–15 Hz and dominate trap loss.)
- **No excitation tone is ever placed in 10–20 Hz** — a tone there would both
  endanger the trap and pollute the guard (false aborts). Enforced + unit-tested.
  Z tones can be placed *above* the guard band by setting `dofs.z.f0` above
  `guard_band_hz[1]` (e.g. `f0: 25.0`); the frequency plan pushes tones away from the
  guard in both directions and will not pull them back in.
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
- **Number of tones is not limited by `MAX_NUM_AWG`.** That constant (= 9) limits the
  number of distinct excitation *channels* simultaneously active per DCU, not the number
  of stimulus rows. With 4 POLES_E*_EXC channels we occupy 4 AWG slots regardless of
  how many rows repeat those channels. `MAX_NUM_AWG_COMPONENTS` (= 5000 per slot) is the
  practical ceiling per channel — well above any realistic tone count. So `n_tones` in
  the config can be increased freely.
- **`analysis.n_averages`** controls the NDS2 capture window for the Welch analysis:
  `capture_s = n_averages × segment_s`. Set `n_averages` (integer) *or*
  `measure_capture_s` (seconds), not both. `premeasure_n_averages` / `premeasure_capture_s`
  work the same way for the trim pre-measurement. The trim loop scales `capture_s` by
  1.8× each time-first iteration, so effective averages grow with each trim step.
- **`StimulusReadback` is required for correct multi-tone coefficients** (source-confirmed
  from `sineresponse.cc` / `stdtest.cc` in dtt 4.1.4; set it in `build_sine_response_xml`).
  The coefficient normalization loop in `sineresponse.cc` divides each channel's response
  by `tmp.coeff[a*(sizeA+sizeB)+a]` — the "excitation diagonal" for stimulus `a`. Without
  `StimulusReadback`, that diagonal is populated by an analytic `ampl*exp(i*2π*f*t)` which
  is placed at `resultnum=a` only if `sizeA=sizeExc`. With multiple tones on one channel,
  `sizeA=1` (duplicate flag collapses repeated channels), so `a=1,2,3` index into
  measurement-channel columns → garbage. With `StimulusReadback=<exc_channel>`, diag runs
  `sineAnalyze` on the actual measured excitation at each tone frequency and places it at
  the correct diagonal. Requires **one distinct physical excitation channel per stimulus
  row** — exactly what the real 4-electrode measurement provides (sizeA=4, sizeB=3).
  - Single-channel loopback (`ACTS_8_8_EXC` reused 4×): still broken (sizeA=1, all
    duplicates). Use the **NDS2 workaround** (`compute_tfs`) for loopback validation.
  - Real 4-electrode measurement: readback should fix it. **Not yet tested on hardware.**
  - NDS2 workaround (`compute_tfs`, `inject_and_capture`) works for both cases and is
    the primary analysis path (verified: 0.500 flat, coherence 1.0 on loopback).
  - **ACTS workaround**: assign one distinct `ACTS_{row}_{col}_EXC` channel per tone so
    `sizeA == sizeExc` — diag's native extraction should then work. EXC goes through
    GAIN (confirmed), so setup sets GAIN=1 and output switch ON for each element used.
    Loopback test uses ACTS_8_1..8_4_EXC (not 8_8 alone) to validate this path.
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
4. ⚠️ diag's own multi-tone *coefficient* output was unreliable in the single-channel
   config (sizeA=1); the ACTS distinct-channel approach (sizeA==sizeExc) should fix
   this — **pending hardware validation on ACTS_8_1..8_4** (new loopback test).
5. (manual, operator) the guard trips + ramps to zero on a growing 10–20 Hz inject.

Re-run with `ACTGAIN_LOOPBACK=1 ... -m pytest -m loopback` after any change to the
injection/analysis path.

---

## `upload_actuation_matrix.py` — invert actuator gains → ACTS matrix

Reads one or more `actuator_gain_results.h5` files (from `measure_actuator_gain.py`),
assembles the forward actuation matrix **A** (DOF × electrode), inverts it, and writes
per-electrode coefficients into the **ACTS** matrix so that each *configured ACTS column*
drives the electric field in a chosen direction. The single abstraction — *per column,
say whether it couples to the field and in what direction* — covers every operating mode
(laser-feedback while driving, motion→voltage feedback, fixed-direction lock-in drive,
fixed-magnitude rotating field, and combinations). See the heavily-commented
`upload_actuation_matrix_config.yml` for worked examples of each.

**N files, any layout.** Each result file carries its own `dof_order` + `electrodes`, so
4 single-electrode files, one 4-electrode file, 2×2, or one-DOF-per-file all assemble to
the same A (each cell keyed by `(dof, electrode)`). Duplicates → `error` (default) or
`average`; a missing cell is a named error.

**Physical-field normalization (important).** `measure_actuator_gain` fits a
**peak-normalized** plant (`|H(f0)|=1`), so the stored gain carries the full on-resonance
susceptibility `χ(ω₀)=Q/(mω₀²)`. To drive a *physical field* of equal magnitude in any
direction we divide it out. Gas-dominated damping ⇒ the damping rate `γ=f0/Q` is common
across modes **within a single file** (but drifts between files with pressure), so per
file we pool the per-mode `γ_d` (residual-weighted, with uncertainty σ_γ) into one
`γ_file` and scale each DOF by `s_d = f0_d · γ_file`. Each electrode's column uses **its
own file's γ**, which corrects cross-run pressure drift. Modes: `common_gamma` (default),
`per_mode_q` (`f0²/Q`), `none` (raw response). The dry-run prints per-file f0/Q, γ±σ_γ
(with a spread warning), A, A_field with per-element uncertainties, condition number, and
per-column electrode coefficients with propagated uncertainties.

**Global field anchor (`gain=1` is reproducible).** After χ removal,
`A_field = c·B`, where `B` is the field-per-count matrix (electrode geometry only) and
`c = cal·q/(4π²m)` is a *single global scalar* that changes particle-to-particle (but
**not** with pressure/Q — `A∝χ` and `s_d∝1/χ` cancel). `field_anchor` divides `A_field`
by a degree-1 functional (`frobenius` default, also `sigma_max`/`reference_column`), which
cancels `c` exactly, so the written matrix depends **only on the electrodes** — `gain=1`
produces the same field whenever the electrodes are unchanged. `gain` stays the
after-the-fact rescaling knob. `physical_scale` `P` (default 1.0) is the future V/m hook:
set `P = ‖B‖_F` (from a COMSOL model iterated to match the measured `B` shape) and `gain=1`
→ 1 V/m; realized `|F| = gain·‖B‖_F/P`. `mode: none` restores the old drift-prone behaviour.

**Reductions & safety.** Complex gains → real via **signed magnitude** `|G|·sign(cos φ)`
(warns when φ is near ±90°). Only electrode rows 1–4 of *coupled* columns are written;
laser rows are never touched. Uncoupled columns are left untouched unless `clear: true`
(per column) or `--clear-uncoupled`. Switches are **GAIN-only + warn** by default; pass
`--enable-switches` to take a column live. Always start with `--dry-run`.

```
/var/lib/cds-conda/base/envs/cds-testing/bin/python3 upload_actuation_matrix.py --dry-run
```

xyz-ready: directions are full 3-D; today A is 2×4 (x,y measured) so a z-bearing
direction is projected onto x,y with a warning (`strict_subspace: true` to error). Once
z couplings are measured, add `z` to `dofs:` and A becomes 3×4 with no code change.

---

## Testing
```
cd /home/controls/labutils/scripts/dipole
/var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest tests/ -m "not loopback"   # pure logic, no hardware
ACTGAIN_LOOPBACK=1 /var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest tests/ -m loopback   # live FE only
```
The pure-logic suite covers the frequency plan (guard-band exclusion, distinctness,
density, snapping), Schroeder crest-factor reduction, diag-XML round-trip,
`compute_tfs` (recovers known TFs + coherence + **phase** from synthetic captured data,
coherence scaling with number of Welch averages, `n_averages` config resolution), the
plant+gain fit (recovers known gains from synthetic data), the guard band-RMS math,
and POLES snapshot/restore + amplitude clamping (mocked EPICS). It also covers the
shared coordinate parser (`test_utility.py`), the ACTS inversion
(`test_upload_actuation_matrix.py`: signed-magnitude, N-file forward-matrix assembly,
γ pooling, field normalization, unit-response inversion, rotating field, uncertainty
propagation, coupled/clear/channel-string planning), and the directional SENSE rows
(`test_upload_sense_matrix.py`: n·W math, axis back-compat, full-sphere, subspace
validation, regression vs the real step-01 W).
The loopback hardware suite additionally validates phase recovery and coherence-vs-averages
on the live FE.

## dtt version note
The installed dtt is `4.1.5~rc1` from the `bullseye-unstable` apt channel (stable
`bullseye` has `4.1.4`). The multi-tone coefficient behavior is the same across these
(`sineresponse.cc` unchanged since Jan 2024), so our inject-and-compute-ourselves
approach is version-independent. You may still want to pin dtt to stable `4.1.4` for
general hygiene (separate sysadmin step).

## Future work
See `FUTURE_WORK.md` for the full list (matrix-layout auto-extraction; measuring z
actuator couplings for 3-D field steering). In brief:
- Generalize the flat-gain assumption to a frequency-dependent actuation matrix
  (electrode capacitance becomes relevant at higher frequency).
- Closed-loop variant (inject on top of a live trap feedback loop).
- Measure electrode→PARTICLE_Z couplings so `upload_actuation_matrix.py`'s A becomes
  3×4 and full xyz field directions are realizable (the code already supports it).
- Wire the inverted matrix into the pipeline's `step_02_actuator_diagonalization.py`
  (currently a stub).
