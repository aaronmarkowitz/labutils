"""Plant + gain fit: recover known gains from synthetic data (noiseless + noisy)."""
import numpy as np

import measure_actuator_gain as mag


def test_recover_gains_noiseless(synthetic_records):
    records, true_G, electrodes, dof_params = synthetic_records
    for dof, (f0, Q) in dof_params.items():
        fit_plant = dof in ("x", "y")
        fit = mag.fit_dof(records, dof, electrodes, f0, Q, fit_plant)
        # gains recovered up to negligible error (relative within the DOF)
        np.testing.assert_allclose(fit.gains, true_G[dof], rtol=1e-3, atol=1e-3)
        if fit_plant:
            assert abs(fit.f0 - f0) < 0.5
            assert abs(fit.Q - Q) < 5.0


def test_recover_gains_with_noise(synthetic_records):
    records, true_G, electrodes, dof_params = synthetic_records
    rng = np.random.default_rng(1)
    noisy = []
    for r in records:
        rr = {"electrode": r["electrode"], "dof_intended": r["dof_intended"],
              "freq": r["freq"], "tf": {}, "coh": {}}
        for d in ("x", "y", "z"):
            scale = max(abs(r["tf"][d]), 1e-3)
            n = (rng.normal(0, 0.02 * scale) + 1j * rng.normal(0, 0.02 * scale))
            rr["tf"][d] = r["tf"][d] + n
            rr["coh"][d] = 0.97
        noisy.append(rr)
    fit = mag.fit_dof(noisy, "x", electrodes, 39.0, 30.0, True)
    # within a few percent of the truth despite noise
    np.testing.assert_allclose(np.abs(fit.gains), np.abs(true_G["x"]), rtol=0.15, atol=0.05)


def test_cross_coupling_offdiagonal_recovered(synthetic_records):
    # true_G["x"][1] = 0.6+0.2j is a cross term (E2 into X); recover its magnitude.
    records, true_G, electrodes, _ = synthetic_records
    fit = mag.fit_dof(records, "x", electrodes, 39.0, 30.0, True)
    assert abs(abs(fit.gains[1]) - abs(true_G["x"][1])) < 1e-2


def test_plant_lorentzian_peak_normalized():
    f0, Q = 40.0, 25.0
    assert abs(abs(mag.plant_lorentzian(f0, f0, Q)) - 1.0) < 1e-9
