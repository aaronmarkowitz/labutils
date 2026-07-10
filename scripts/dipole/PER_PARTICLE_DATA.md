# Per-particle data checklist

Order matters. Details: `PROCEDURE.md`.

## Low vac (3e-2 – 8e-2 mbar; avoid <1e-2)

1. 7-ch noExc diag CSD (BW 0.2 Hz, ≳200 avg). Check FM gains first.
2. Step 01 fit → `upload_sense_matrix.py`.
3. Verify equip vs new baseline — must PASS before continuing.
4. Undriven video, both cams, ≥3 min, inside CSD window + simultaneous noExc CSD.
   Post-W-upload only.
5. Driven tones, x and y: one off-res tone each (above-res not needed) + both cams
   + videoSimul XML. On-res optional for SNR.
6. Equip re-check (noExc CSD + verify).

## HV (~1e-5 mbar)

7. (opt) Re-diag; finer BW if peaks < df.
8. Driven tone + zCam video + XML → step 03 cross-cal (thermal invalid here).
9. Charge steps — every particle. UV toward ~1e, record staircase → q + field cal
   (04b). Log charge state on all subsequent data.
10. Translational dipole drives (step 05).
11. Neutralize → libration (step 06). ≥2 charge states incl. near-neutral if time
    (charge→dipole check).
12. (bonus) Drag / precession vs pressure.

## Standing cals (check, don't retake)

- `zpixel_um` auto-staleness-checked; `xpixel_um` STILL UNMEASURED (do once, any pressure).
- ACTS matrix + field cal: reuse unless trap location moved.
