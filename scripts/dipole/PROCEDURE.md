# Dipole-moment measurement procedure (Parts A–D)

Single living runbook for measuring the electric dipole moment of a magnetically
levitated diamond. The electric field driving the particle is calibrated from
**first principles** — particle mass × single-charge-step size on a
position-calibrated sensor — rather than from COMSOL.

This file (in the `labutils` lab-side repo) is the **source of truth**; a summary is
mirrored on the wiki. Deferred/missing pieces are tracked as **GitHub issues**, not
inline TODO lists.

## Two repos, one rule: **hardware access decides**

- **`~/labutils`** — anything that touches EPICS / NDS2 / `diag` / cameras / hardware,
  or otherwise cannot run asynchronously. Runs live in the lab.
- **`~/analysis`** — anything that runs offline on recorded files (mp4 / xml / hdf5),
  even when it is sometimes run live to make a real-time decision (e.g. sensor
  diagonalization, or the video thermal-mass check below).

## Two pressure regimes (a first-class axis of this procedure)

- **Low vacuum, ~5×10⁻² mbar** — Parts A/B calibration. Residual-gas damping
  thermalizes the particle to room temperature, so **equipartition holds** and thermal
  ASDs give absolute calibration and mass.
- **High vacuum, ~1×10⁻⁵ mbar** — Parts C/D. The particle is **not** thermalized to a
  known temperature; thermal equipartition is **invalid**. Absolute position must be
  obtained by **driving** the particle and matching the driven response on the sensor
  to the camera. Never use the thermal video mass method here.

---

## Part A — Diagonalize electrodes → particle DOFs  (once, or occasionally)

Assumes particles trap at approximately the same location relative to the pole pieces.

| # | Step | Where | Tool | Regime | Status |
|---|------|-------|------|--------|--------|
| A1 | Pump down to ~5×10⁻² mbar | lab | manual (pump); CSD recording manual | low-vac | manual → *issue: scripted CSD recorder* |
| A2 | Record CSD of all sensor channels → `.xml` | lab | `diag` / diaggui (manual) | low-vac | manual → *issue: scripted CSD recorder* |
| A3 | Diagonalize sensors → `W`; upload to SENSE | analysis + lab | `dipole_pipeline` step 01 (offline fit) → `upload_sense_matrix.py` | low-vac | implemented (fit robustness ongoing) |
| A4 | Verify equipartition on PARTICLE_XYZ | analysis | `verify_particle_equipartition.py` | low-vac | implemented, robust |
| A5 | Measure actuator gain (comb of tones per resonance) | lab | `measure_actuator_gain.py` | low-vac | implemented |
| A6 | Upload actuator (ACTS) matrix | lab | `upload_actuation_matrix.py` | low-vac | implemented |

Output of Part A: a diagonalized readout (SENSE `W`) and an actuator matrix that can
drive the E-field in an arbitrary direction — but that drive is not yet calibrated in
V/m (Part D).

---

## Part B — Absolute field magnitude / particle mass via camera  (per particle to get mass)

Typically a different particle from Part A. **All at ~5×10⁻² mbar.**

| # | Step | Where | Tool | Regime | Status |
|---|------|-------|------|--------|--------|
| B1 | Pump to ~5×10⁻² mbar; record CSD of all sensor channels | lab | manual | low-vac | manual → *issue: scripted CSD recorder* |
| B2 | Diagonalize sensors; upload to SENSE | analysis + lab | step 01 → `upload_sense_matrix.py` | low-vac | implemented |
| B3 | Verify equipartition | analysis | `verify_particle_equipartition.py` | low-vac | implemented |
| B4 | Record CSD of PARTICLE_XYZ **+ video (z & x cams)** | lab | manual CSD; `run_thorcam.py` / cameras (manual mp4) | low-vac | manual → *issue: headless video acquisition* |
| B5 | Drive along x and y at known freqs (one sine at a time, above/below resonance); record time+ASD of PARTICLE_XYZ + video | lab | `particle_lo_scan.py` (drive) + cameras | low-vac | implemented (drive); video manual |
| B6 | Verify equipartition (again — drive must not have moved the particle) | analysis | `verify_particle_equipartition.py` | low-vac | implemented |
| B7 | Reconstruct particle position (m) from video; match to PARTICLE_XYZ ASD → calibrate channels to meters | analysis | **video → mass:** `dipole_pipeline/video_thermal_asd.py` (thermal); driven-tone: `dipole_pipeline` step 03 | low-vac | implemented |
| B8 | Extract particle **mass** (room-T thermal bath and/or driven response) | analysis | `video_thermal_asd.py` (equipartition) / step 04 | low-vac | implemented |

**B7/B8 detail — the nearest-term build.**
- `video_thermal_asd.py` reconstructs `x(t), y(t)` in meters from a **thermal
  (undriven)** z-cam clip (z-cam images the xy-plane; needs **fps ≳ 120 Hz** to
  Nyquist-sample the ~54 Hz y mode), Welch-ASDs it, fits DHO peaks, and extracts mass
  by equipartition — **no sensor or drive required for the mass** (the video is already
  length-calibrated by `pixel_um`). If a simultaneously recorded sensor CSD is given,
  it also emits a pipeline `adc_to_meters` (thermal cross-cal).
- **Feasibility payoff:** if video reliably yields thermal peaks at ~5×10⁻² mbar, later
  particles can record video at B4 and go straight toward mass without a separate
  CSD/diagonalization/drive-tone step.
- Driven-tone step 03 remains the path when thermal SNR is insufficient, and is the
  **mandatory** path at HV.

---

## Part C — Librational dipole  (per particle; requires mass from Part B)

**At ~1×10⁻⁵ mbar (HV).**

| # | Step | Where | Tool | Regime | Status |
|---|------|-------|------|--------|--------|
| C1 | Pump to ~1×10⁻⁵ mbar | lab | manual | HV | manual |
| C2 | Drive with `PARTICLE_LO_COSGAIN`, E-field rotating in xy at several kHz; set lowest drive so libration freq ≥ 3× y resonance | lab | `particle_lo_scan.py` / spin drive | HV | drive implemented; **libration measurement: model only** (`dipole_sideband_model.py`) → *issue: live libration measurement* |
| C3 | Record librational frequencies | lab | libration measurement script | HV | **missing** → *issue* |
| C4 | Known mass → moment of inertia; with E from Part D → dipole moment | analysis | `dipole_pipeline` step 06 | HV | implemented |
| C5 | (Bonus) repeat at several pressures for drag/precession | lab | — | varies | deferred (needs hardware) → *issue* |

---

## Part D — Calibrate electric-field magnitude  (requires mass from Part B; not every particle)

**At ~1×10⁻⁵ mbar or lower (HV).**

| # | Step | Where | Tool | Regime | Status |
|---|------|-------|------|--------|--------|
| D1 | Pump to ~1×10⁻⁵ mbar or lower | lab | manual | HV | manual |
| D2 | Diagonalize sensors (optional) | analysis + lab | step 01 → `upload_sense_matrix.py` | HV | implemented |
| D3 | Drive with `PARTICLE_LO_COS` along one axis; record time+freq readout + video (z cam) | lab | `particle_lo_scan.py` + cameras | HV | implemented (drive); video manual |
| D4 | Match reconstructed camera motion to sensor response → sensor magnitude calibration | analysis | driven-tone step 03 (HV-valid) | HV | implemented → *issue: driven-tone camera↔sensor cross-cal packaged like 03b* |
| D5 | Neutralize particle (UV / electron bath) toward ~1e charge | lab | `teemController/` UV daemon | HV | UV on/off implemented; **neutralize-to-target loop missing** → *issue* |
| D6 | At ~1e charge, observe single-charge-step size; with q/m known → E in V/m per drive voltage | lab + analysis | charge readout + `dipole_pipeline` step 04 (charge-step) | HV | step 04 implemented; **charge-step size acquisition missing** → *issue* |
| D7 | Cross-check: re-observe camera motion to confirm sensor calibration | lab + analysis | cameras + step 03/03b | HV | implemented |

The drive-voltage → E calibration from D6 is expected to be stable for particles
trapped in a similar location (to leading order, all particles).

---

## Deferred / missing infrastructure (tracked as GitHub issues)

Do **not** maintain a stale FUTURE_WORK list here. Open a `gh issue create` on the
relevant repo when something is deferred. Current known gaps:

- **`labutils`** — scripted fixed-length CSD recorder (→ xml/hdf5 handoff to
  diagonalize/verify); headless video-acquisition script (cameras are GUI-only); live
  libration measurement (only a model today); neutralize-to-target charge loop;
  charge-step-size acquisition; multi-pressure drag/precession hardware.
- **`analysis`** — driven-tone camera↔sensor cross-cal packaged like 03b for HV
  absolute position.
- **Wikis** — reconcile the `labutils` wiki Dipole section and stand up an `analysis`
  wiki with the `dipole_pipeline` overview (incl. 03b and the low-vac↔HV regime split).
