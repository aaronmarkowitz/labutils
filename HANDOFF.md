# HANDOFF — labutils

## §0 Current session (2026-09-29, branch `feat/particle-loader`)

**Status: auto-ROI (alignment-robust), circular polarisation, x-laser preflight and the
rotating-field eject (plan + dark-run command) are in and tested. Not merged, not pushed.
No real Moku output, no electrode drive, no CDS write. The x cam was opened read-only.**
Waiting on Alma: approval of the eject **dark run**. Also: real chirp, real eject, merge/push/wiki.

### Goal
Automate loading: chirp dropper → detect on x cam → cancel → photometry → size → reject
≥1 µm / accept sub-µm → alert. Eject only on Alma's say-so; real chirp not approved yet.

### Decisions (Alma)
- 09-28: lights off, camera covered, z 500 µW; reject ≥1 µm; `--size-spread 0.1` (clusters).
- 09-29: IDS SN 4103314187 **is** the x cam. z laser **circular** → `--pol-deg circular`
  default (linear available on request). Load with the **x laser off** (`run` refuses if
  POX_INMON > 200 unless `--x-laser-on-ok`). Eject = 1→60 Hz sweep, rotating in x-y,
  |E| normalised by the actuator gains, ≤32000 counts on any DAC.

### Done this pass (09-29)
- **`--roi auto` (default).** Empty trap: detect on the FULL frame. On a particle
  (occupancy or confirm): narrow the AOI and baseline to a 300-row full-width strip about
  the brightest object (`narrow_to_strip`); photometry keeps the validated strip geometry.
  `trap_full_xy` goes in every ledger entry; |dy| > 40 px vs the reference → warning plus
  "CALIBRATION MAY BE STALE" in the decision. `--roi X Y W H` keeps the fixed strip.
  Live timing (read-only): camera 13.9 fps at 1280×1024 / 50 ms; full-frame detector loop
  median 71.4 ms = frame period, p95 85 ms, lag 40 ms that does not accumulate. The strip
  AOI switch plus exposure re-apply works live.
- **`measure --reference-from ID`** (after a realignment, the same particle): copies the
  Epstein size and marks the old row `superseded_by`; `ledger.references` skips it (control test).
- **Circular pol** = mean(0°, 90°) in `size_model` (self-test identity; = pol 45 exactly on
  this cone). Smallest detectable at B/B_ref 0.003: **231 nm** circular; linear-y would give
  207 nm at the same absolute floor (NOT the 160–180 nm stated 09-28, which normalised each
  pol to its own reference). The 10 pg reference is 0.27× as bright under linear-y, so
  switching pol needs a new reference; entries now record `pol_model`.
  decide(), spread 0.1, circular: accept B/B_ref ≤ 0.2, ambiguous ~0.3, reject ≥ 0.5.
- **Eject** (`loader/eject.py`): PARTICLE_SPIN cdsOsc → ACTS cols 6/7. Live plan
  (read-only): COSGAIN 19212, E3 binds at 32000 (quadrature); field min/max 0.99973,
  −360°/cycle; problems []. If quadrature were lost, E3 → 40037 (> DAC clip): the dark
  run checks this. `--dark-run`: spin cells ACTS_{1..4}_{6,7} OUTPUT off, read back, and
  refuse unless all 8 read off. Then the real sequence while NDS2 records
  ACTS_1_{6,7}_IN1 (COS/SIN) and POLES_E*_OUT_DQ (must stay static). Restore only after
  COSGAIN/SINGAIN read 0; otherwise the cells stay OFF. Analysis: amplitude, f, quadrature error.
- `--from-npy` reads `.npz`. G4 replay (fake Moku, r_ap 50, scratch reference): reject →
  eject plan → simulated eject → 1 % particle, chirp stop 0.56 s → ACCEPT [237, 268] nm.
- Tests: pytest 202 passed / 4 skipped; `-m control` 30 passed; size_model self-test PASSED.

### Not verified / caveats
- **Full-frame detection threshold in the dark is unmeasured.** Today the lab was lit
  (12 counts/ms/px, saturated at 50 ms), so `full_50ms.npz` in
  `data/260929/particle_loader/` is a lit scene, useless for thresholds. The 0.003 figure
  is for the strip; the full frame has 3.4× the pixels (look-elsewhere) → re-measure in the dark.
- In the lit frame the brightest spot was at (789, 643), which matches the particle position of 09-28.
- Dark-run unknowns: oscillator quadrature through FREQ steps; whether FREQ ramps over TRAMP;
  whether SINGAIN follows COSGAIN.

### 09-28 results still standing
POZ normalisation holds (B 2.45e5 vs reference 2.424e5, `260928_153227_67f7`). Background
0.07 counts/ms/px in the dark. Detection to B/B_ref 0.003 (smooth_r 25), no false
candidate in 66 s. Difference photometry sizes to ~0.004. A no-particle result stops the loop.
Data: `~/Dropbox/.../data/260928/particle_loader/newconfig_lightsoff_500uW/`.

### Next commands (worker1)
    # eject plan (read-only) / dark run (WRITES CDS -- only with Alma's go):
    /var/lib/cds-conda/base/envs/cds-testing/bin/python3 loader/eject.py
    /var/lib/cds-conda/base/envs/cds-testing/bin/python3 loader/eject.py --dark-run
    # in the dark with the trap EMPTY: fake-Moku run measures full-frame loop timing + threshold
    cd loader && conda run --no-capture-output -n thorcam python load_particle.py run
    # after a beam realignment, same particle trapped:
    conda run --no-capture-output -n thorcam python load_particle.py measure --reference-from 260928_153227_67f7
    # replay (strip): see git log 7fc9744 / tests; needs --r-ap 50 with an r_ap-50 reference

## §1 Previous session (2026-09-10, branch `harness-2026-09`, since merged in 9fb86a6)

**Status: ready for your review. Nothing merged, nothing pushed, no hardware touched.**

### Goal

Apply the "Fix now" items from the 2026-09 harness audit: make `CLAUDE.md`
host-conditional so laptop sessions stop being told they are on worker1, fix the wrong
script paths, cut the file under 150 lines, give bare `pytest` something to discover,
and retire `SESSION_LOG.md`.

### Done and verified

**`CLAUDE.md` — 159 → 149 lines.** Prior version archived verbatim at
`docs/archive/CLAUDE-pre-harness-2026-09.md` (nothing deleted).

- New "Which host am I on? (check first)" table at the top: run `hostname`; on worker1
  hardware is reachable and the safety rules apply; on anything else (the laptop) there
  is no hardware and it is edit/read/test only. Also states that `/var/lib/cds-conda`,
  `/opt/rtcds`, `/home/controls` exist on worker1 only, so a failed path check for them
  from the laptop is expected rather than a bug.
- Safety section kept and *promoted* directly beneath that block: `--dry-run` first,
  never command hardware unsupervised without the viability preconditions, Alma
  orchestrates. Pulled the guard-band pitfall (never put an excitation tone in the
  10–20 Hz trap-loss guard band) up into Safety — it was buried in the diag section.
- **All 13 script paths checked with `ls`. 5 were wrong**, exactly as the audit found:
  | in CLAUDE.md | actual |
  |---|---|
  | `map_y1dmd_state.py` | `scripts/map_y1dmd_state.py` |
  | `engage_intensity_servo_x.sh` | `scripts/engage_intensity_servo_x.sh` |
  | `disengage_intensity_servo_x.sh` | `scripts/disengage_intensity_servo_x.sh` |
  | `run_thorcam.py` | `cameras/run_thorcam.py` |
  | `run_leybold_turbolab.py` | `epics/run_leybold_turbolab.py` |
  The other 8, plus the five `epics/` snapshot files and `scripts/dipole/README.md`,
  resolved correctly. Also added `scripts/particle_lo_scan.py`, which exists but was
  undocumented. The `/opt/rtcds/…` and `/home/controls/…` paths were **not** verified —
  they are worker1-only (see below).
- Removed: the "Autonomic layer (future)" section and the Phase-4 forward pointer inside
  Safety (future work, not instruction); the restated branch-first convention, now one
  pointer line to `~/.claude/CLAUDE.md`.
- The equipartition / `common_unit_anchor` rule appeared **in full twice** (the
  `upload_sense_matrix.py` and `verify_particle_equipartition.py` table rows). Both are
  now short, and the semantics are a single pointer to
  `analysis/.claude/skills/verify-equipartition/SKILL.md` — confirmed to exist.
- The worker1 interpreter/env table is now a real table marked canonical, since
  `lab-context/conventions.md:30` points here for it.

**`pytest.ini` (new, repo root).** There was no `pytest.ini`, `setup.cfg`,
`pyproject.toml`, or `tox.ini` anywhere in the repo, so bare `pytest` discovery depended
on cwd. It sets `testpaths = scripts/dipole/tests` (the 14 existing safety test files),
`pythonpath = scripts/dipole`, `--strict-markers`, and registers two markers: the
pre-existing `loopback` (previously declared only in conftest's `pytest_configure`, which
`--strict-markers` would have rejected) and the new `control`.

**`scripts/dipole/tests/test_control_safety.py` (new).** One `@pytest.mark.control` test,
parametrized over two out-of-range guard bands (inverted `lo > hi`, collapsed `lo == hi`),
requiring `validate_config` to raise `ValueError` — and asserting in the same test that
the untouched fixture still passes, so the check is shown to discriminate rather than
raise on everything.

**Test run (this laptop, with a caveat — read "Not verified"):**

```
conda run -n mastqg python -m pytest -q      # PYTHONPATH=<scratchpad stub for nds2>
139 passed, 5 skipped, 1 warning in 40.48s
conda run -n mastqg python -m pytest -m control -q
2 passed, 142 deselected
```
The 5 skips are self-declared: 4 in `test_loopback.py` ("set ACTGAIN_LOOPBACK=1 on a
host that reaches the Y1:DMD FE"), 1 in `test_upload_sense_matrix.py` ("real step-01 file
not present"). I also confirmed a validator neutered to `return None` accepts the
inverted band, i.e. the new control test would really fail if the check regressed.

**`SESSION_LOG.md`** — appended a "RETIRED 2026-09-10 … superseded by `HANDOFF.md`" block
at the bottom. File kept, history intact, no further entries.

### Not verified — needs worker1

1. **The test run above is not clean.** `scripts/dipole/measure_actuator_gain.py:64`
   imports `nds2` at module top, and `nds2` is not in `mastqg`, so *every* test in the
   suite failed to import on the laptop. To verify that `pytest.ini` discovers the right
   files I put a two-line `nds2` stub on `PYTHONPATH` **from the session scratchpad — it
   is not in the repo and not committed**. So: discovery, marker wiring, and the pure-logic
   assertions are verified; anything that would exercise real `nds2` is not.
   **Re-run on worker1 with the documented interpreter, which is the only clean run:**
   `/var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest -q`
2. **`/opt/rtcds/…`, `/home/controls/…`, `/var/lib/cds-conda/…` paths in `CLAUDE.md`** —
   copied forward unchanged and unverifiable from macOS. Worth one `ls` pass on worker1.
3. **`cameras/test_*.py`** (2 files) — excluded from `testpaths`; needs the `thorcam` env
   on worker1. Run `pytest cameras/` there to confirm they still pass.
4. No hardware, EPICS, `caput`, or device script was run at any point, per the brief.

### Decisions (made conservatively rather than asking)

- **Repo path.** The brief said `~/Moore/labutils`, which does not exist. Used
  `~/Moore/MastQG/MagneticTrap/labutils`, confirmed by `lab-context/projects.yml:34`.
- **14 vs 16 test files.** `find` returns 16; 14 are in `scripts/dipole/tests` and 2 are
  `cameras/test_*.py`. Read the audit's "14" as the dipole suite and scoped `testpaths`
  to it, since the camera tests need SDKs that only exist in one env on worker1.
- **Interpreter.** The documented `cds-testing` python does not exist on this laptop
  (`ls` confirms), so fell back to `conda run -n mastqg` as instructed.
- **`mastqg` does have pytest** (9.0.3), contradicting the old `CLAUDE.md` claim of
  "`dttxml` but not `nds2`/`pytest`". `nds2` genuinely is not there. Corrected that row
  to "`dttxml`, `pytest`, but no `nds2` — cannot run this repo's tests". Verified on the
  laptop only; worth a re-check on worker1, where the env may differ.
- **Did not touch the `nds2` top-level import.** Making it lazy would let the suite run
  on the laptop unstubbed, but that edits hardware analysis code on a doc/test cleanup
  branch. Left alone deliberately — see "Suggested next".
- **Deleted nothing.** Old `CLAUDE.md` → `docs/archive/`; `SESSION_LOG.md` kept.

### Exact next commands for Alma

```bash
cd ~/Moore/MastQG/MagneticTrap/labutils

# 1. review — 4 commits, main untouched
git log --oneline main..harness-2026-09
git diff main..harness-2026-09 -- CLAUDE.md          # the substantive one
git diff main..harness-2026-09 -- pytest.ini scripts/dipole/tests/test_control_safety.py

# 2. on worker1, the only clean test run:
/var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest -q
/var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest -m control -q   # expect 2 passed
conda run -n thorcam python -m pytest cameras/ -q                              # camera tests

# 3. merge when happy (I did not push; nothing is on origin)
git checkout main && git merge --no-ff harness-2026-09
```

### Suggested next (not done)

- Make `nds2`/`dttxml` lazy imports in `measure_actuator_gain.py` so the pure-logic suite
  runs on the laptop without a stub. This is the single biggest laptop-productivity win
  and is a real (if small) code change, so it wants its own branch.
- `lab-context/conventions.md:73` still describes the `SESSION_LOG.md`-at-repo-root
  convention. Now stale for this repo; update it when you next touch lab-context.
