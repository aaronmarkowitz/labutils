---
name: write-sense-row
description: Write one SENSE-matrix row along an arbitrary xyz direction (axis, xy angle, full-sphere elevation/azimuth, or explicit vector) from the latest step-01 diagonalization, or copy one existing SENSE row's live gains into another row. Use when the user wants a custom directional readout on a spare SENSE row (row 4 "charge in" / row 5 "spins", or any row), or wants to duplicate a row's current EPICS gains elsewhere.
---

# Write a SENSE row (arbitrary direction, or row copy)

The MAST-QG `Y1:DMD-SENSE` matrix has 5 output rows (`XERR, YERR, ZERR, charge_in, spins`) x
7 input columns (LES x/y pit/yaw/sum + PBSZ). `XERR/YERR/ZERR` (rows 1-3) are populated by
the `diagonalize-sensors` pipeline (companion skill, lives in `~/analysis`). Rows 4-5 are
spare and can be pointed at **any direction** in DOF space, or set to mirror another row's
live gains.

This skill is pure live-EPICS-write territory (no offline analysis), so it lives in
`~/labutils` per the repo split rule in `~/labutils/CLAUDE.md` ("hardware access decides").

Two distinct operations this skill covers — pick the one the user wants:

1. **Directional row from W** — compute `n·W` for a chosen unit vector `n` and a step-01
   HDF5, write it to a chosen row. Use `~/labutils/scripts/dipole/upload_sense_matrix.py` +
   `sense_matrix_config.yml`.
2. **Row copy** — read one row's current live `_GAIN`/`_SW1R`/`_SW2R` via `caget` and write
   those same values (gains + switch-enable) into another row via `caput`. No HDF5 needed.

---

## What's currently live on SENSE?

After every successful (non-dry-run) upload, `upload_sense_matrix.py` writes
`~/labutils/scripts/dipole/sense_matrix_state.json` (gitignored — it's a live-EPICS mirror,
not committed config): the h5 path, `dofs`, which row labels were written, `TRAMP`, and
per-mode `peak_hz`/`eigenratio`. **Read this file first** to know which diagonalization is
currently deployed before assuming you need a fresh upload — don't rely on memory alone,
since the state file is authoritative (it's written by the script, not hand-maintained) and
memory can drift stale.

```
Read ~/labutils/scripts/dipole/sense_matrix_state.json
```

If the file is missing or its `hdf5_path` no longer exists, treat the live state as unknown
and ask the user, or reconstruct it by `caget`-ing a row's gains and matching them against
candidate step-01 h5 files (see the diamond1/diamond2 disambiguation lesson below).

---

## Operation 1: directional row from the latest W

### Step 1 — Identify the source HDF5 and target row

- Prefer the h5 path in `sense_matrix_state.json` if you're adding a row to the
  already-deployed diagonalization (the common case — e.g. pointing the spare `charge in`
  row along a new direction using the SAME W that's already live on XERR/YERR/ZERR).
- Otherwise find the step-01 results h5 the user means (usually the most recent `FINAL_*` dir
  under `$MQG_DROPBOX_PATH/worker1/data/<date>/.../01_SensorDiagonalization/step_01_sensor_diagonalization_results.h5`).
  If ambiguous, ask.
- Confirm the target row index (4=`charge_in`, 5=`spins`, or re-purposing 1-3) and a
  descriptive label.
- Confirm the direction: axis (`mode: x|y|z`), in-plane angle (`angle_deg`, from +x toward
  +y), full sphere (`elevation_deg`+`azimuth_deg`), or explicit `vector: [x,y,z]`. A 180°
  offset in angle/azimuth flips the sign.

### Step 2 — Edit `sense_matrix_config.yml`

```
Read ~/labutils/scripts/dipole/sense_matrix_config.yml
```

Add (or edit) the row entry under `rows:`, e.g.:

```yaml
  - {index: 4, label: "charge in", mode: "y"}                  # axis shorthand
  - {index: 4, label: "CHARGE_DIAG45", angle_deg: 45}           # or an arbitrary angle
  - {index: 5, label: "CUSTOM_SKEW",   elevation_deg: 45, azimuth_deg: 60}
```

Leave `XERR/YERR/ZERR` and any other row the user is not touching untouched in the file — the
uploader only writes rows present in `rows:`, and only columns present in `cols:` (the col
layout is fixed hardware, don't edit it).

If the requested direction has any component outside the diagonalization's measured DOFs
(e.g. a z-bearing vector when the h5 only has x,y), the uploader **raises** rather than
silently projecting — re-run step 01 over the needed DOFs first, or drop the out-of-plane
component and confirm with the user.

### Step 3 — Dry run, then write

```bash
python3 ~/labutils/scripts/dipole/upload_sense_matrix.py <h5_path> --dry-run
```

Read the printed mapping table — check the target row/column values and the direction label
(`descr`) match what was intended. A useful sanity check: the new row's printed values should
exactly equal an existing axis row's values when the direction is a pure `mode: x|y|z` (e.g.
`charge in` with `mode: y` prints the same numbers as `YERR`). **Always show the user the
dry-run output before writing live.** Then:

```bash
python3 ~/labutils/scripts/dipole/upload_sense_matrix.py <h5_path>
```

Default 5 s `TRAMP`. This also force-enables the input filter module (GAIN=1, switches on)
feeding every column written — necessary or the uploaded value is silently dead (see
`diagonalize-sensors` LESSONS.md, "Recorded PARTICLE channels need sensors on"). On full
success it also (re)writes `sense_matrix_state.json` — check the printed "State recorded:"
line.

**Caveat:** the uploader writes ALL rows listed in the YAML's `rows:` section in one run —
if `XERR/YERR/ZERR` are still listed (the normal committed config), running this will
**re-write those rows too** (with the same W, so a no-op if nothing changed, but it does
touch live EPICS channels for all three). If the user only wants the new row touched, either
temporarily comment out the other rows in the YAML for this run, or confirm re-writing
X/Y/Z with the same W is acceptable.

### Step 4 — Verify with caget

```bash
caget Y1:DMD-SENSE_<row>_1_GAIN Y1:DMD-SENSE_<row>_2_GAIN ... Y1:DMD-SENSE_<row>_7_GAIN
```

Compare against the dry-run table (or against the axis row it should match, if the direction
was a pure axis shorthand).

---

## Operation 2: copy one row's live gains to another row

No script exists for this — do it directly via EPICS, mirroring what `upload_sense_matrix.py`
does for switches (see its `_SW1_INPUT_ON_BIT=4`, `_SW2_OUTPUT_ON_BIT=1024` constants).

### Step 1 — Read the source row (all 7 columns)

```bash
for c in 1 2 3 4 5 6 7; do
  caget -t "Y1:DMD-SENSE_<src_row>_${c}_GAIN"
  caget -t "Y1:DMD-SENSE_<src_row>_${c}_SW1R"
  caget -t "Y1:DMD-SENSE_<src_row>_${c}_SW2R"
done
```

Also read the target row's current switch states the same way, so you only flip switches that
are actually off (SW1/SW2 writes are XOR-toggle bits — writing them when already on flips them
OFF).

### Step 2 — Confirm with the user

Show a table: column, source gain, target's current gain, source/target switch states. Ask
before writing if any target switch is currently off (enabling it changes live signal flow) or
if the target row currently holds nonzero gains that will be overwritten.

### Step 3 — Write

For each column: `caput Y1:DMD-SENSE_<tgt_row>_<col>_TRAMP 5`, then `caput
Y1:DMD-SENSE_<tgt_row>_<col>_GAIN <value>`. Then for any column where the target's SW1R/SW2R
was off but the source's was on, `caput` the toggle bit (`_SW1 4` / `_SW2 1024`) to turn it on
— never toggle a bit that's already in the desired state.

### Step 4 — Verify

Re-`caget` the target row's `_GAIN`/`_SW1R`/`_SW2R` for all 7 columns and confirm they match
the source. Note: a row-copy operation does NOT update `sense_matrix_state.json` (that file
is only written by `upload_sense_matrix.py`'s own successful runs) — if the copy changes what
row 4/5 semantically represents, mention this to the user as a known gap.

---

## Reference

- `~/labutils/scripts/dipole/upload_sense_matrix.py`, `sense_matrix_config.yml` — directional
  row writer (Operation 1).
- `~/labutils/scripts/dipole/sense_matrix_state.json` — script-owned record of what's
  currently live (gitignored; written on every successful non-dry-run upload).
- `~/labutils/scripts/dipole/utility.py` — shared direction-vector parser
  (`direction_unit_vector`, axis/angle/spherical/vector grammar).
- Filter-module switch bit convention: SW1 bit 2 (`4`) = input on, SW2 bit 10 (`1024`) =
  output on (same convention used by the ACTS uploader).
- Companion skill: `diagonalize-sensors` (in `~/analysis`; produces the step-01 h5 this reads
  from).
- `~/labutils/scripts/dipole/FUTURE_WORK.md` #1 — row/col labels are hand-maintained in the
  YAML; if the rtcds model (`y1dmd.mdl`) is rebuilt, update `sense_matrix_config.yml` by hand.
