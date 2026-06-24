# Dipole scripts — deferred / future work

Running list of intentionally-deferred work for the dipole ACTS/SENSE tooling.

## 1. Auto-extract ACTS/SENSE matrix layouts from the model

The row/column labels of the ACTS (8×8) and SENSE (5×7) matrices are currently
**hard-coded** in `upload_actuation_matrix_config.yml` and `sense_matrix_config.yml`:

- ACTS columns: `XCTL, YCTL, ZCTL, LOC, LOS, spinC, spinS, RCTL`
- ACTS rows: `V1..V4 = E1..E4`, `LASER_CTLX, LASER_CTLZ`, `DAC6, DAC7`
- SENSE columns: `x_pit, x_yaw, x_sum, z_pit, z_yaw, z_sum, PBSZ`
- SENSE rows: `XERR, YERR, ZERR, charge_in, spins`

A future helper should parse these directly from `/opt/rtcds/userapps/mastqg/y1dmd.mdl`
(block connectivity, reusing `~/labutils/mdl_to_adl.py`) or the labeled `.adl`
screens, so the YAML layouts don't need hand-maintenance when the model is rebuilt.
Until then: if the model changes, update the two YAML files by hand.

## 2. Measure z actuator couplings (3-DOF actuation)

`upload_actuation_matrix.py` is already xyz-general: the direction parser
(`utility.direction_unit_vector`) returns a 3-D vector and the forward matrix `A`
carries one row per measured DOF. Today only x,y are measured, so:

- `A` is 2×4, and any requested field direction with a z component is **projected
  onto x,y with a warning** (or errors under `strict_subspace: true`).

To enable true 3-D field steering: run `measure_actuator_gain.py` over x,y,z
(electrode→PARTICLE_Z couplings), add `"z"` to the `dofs:` list and z-bearing
result files in `upload_actuation_matrix_config.yml`. `A` becomes 3×4 and z field
components are realized with **no code change**.

Caveat (from the experimentalist): z couplings are expected to be smaller than
x,y, so achieving a given z field needs larger electrode voltages — at which point
x,y crosstalk can appear unless the actuation TFs are well characterized.

## 3. Calibrate the ACTS input to physical field units (V/m)

`upload_actuation_matrix.py` already anchors `gain=1` to a *reproducible* field via
`field_anchor` (default `self_norm`/`frobenius`): after χ removal `A_field = c·B`, with
`B` the field-per-count matrix (electrode geometry only) and `c = cal·q/(4π²m)` a single
particle-dependent global scalar that the anchor divides out. So the written matrix is
`B̂ = B/‖B‖_F` — pure electrode geometry, particle-independent.

To make `gain=1` mean **exactly 1 V/m**, set `field_anchor.physical_scale = P = ‖B‖_F`.
The plan to obtain `‖B‖_F`:

1. Iterate the COMSOL trap-geometry model until its field-per-count *shape* `B̂_COMSOL`
   matches the lab-measured `B̂` (up to global scale). The geometric mismodeling that
   plagued the old approach (++-- not producing a pure y-field, +--+ not a pure x-field,
   from imperfect electrode locations) lives entirely in the *shape* `B̂` — which the lab
   actuator-gain measurement now supplies directly. So COMSOL's only remaining job is to
   provide the one magnitude `‖B‖_F` (= `dac_to_volts · comsol_E_per_volt · geometry
   factor`).
2. Evaluate the field along a chosen direction in both COMSOL and the Frobenius-normalized
   matrix, compare, and supply the resulting `P = ‖B‖_F` as `physical_scale` (with
   `mode: physical`). Realized `|F| = gain · ‖B‖_F / P`.

Hardware constants live in `dipole_pipeline/parameters_*.yml`
(`dac_to_volts`, `comsol_E_per_volt`); per-particle `mass_kg`/`adc_to_meters` are in that
pipeline's `results.yml` / step-03/04 HDF5 — none are needed for the anchor itself, only
to interpret `c` if ever desired.

## 4. Possible refinements

- The field-normalization uncertainty currently treats `residual_norm_{dof}` as a
  fit-quality weight proxy, not a rigorous per-mode γ variance, and reports the
  conservative max(internal, external) standard error. With ≥3 DOFs (or repeated
  runs) a proper weighted-least-squares γ with a real covariance could replace it.
- A hardware loopback self-test (DAC6/DAC7 → LOC/LOS cabled inputs) could be added
  as a `-m loopback` pytest, mirroring `test_loopback.py`, to verify the inverted
  LOC/LOS columns end-to-end on real hardware.
