"""Plant + gain fit: recover known gains from synthetic data (noiseless + noisy)."""
import numpy as np
import pytest

import measure_actuator_gain as mag


STRATEGIES = ("joint", "dof_filtered", "mag_then_linear")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_recover_gains_noiseless(synthetic_records, strategy):
    records, true_G, electrodes, dof_params = synthetic_records
    for dof, (f0, Q) in dof_params.items():
        fit_plant = dof in ("x", "y")
        fit = mag.fit_dof(records, dof, electrodes, f0, Q, fit_plant,
                          fit_strategy=strategy)
        np.testing.assert_allclose(fit.gains, true_G[dof], rtol=1e-2, atol=1e-2)
        if fit_plant:
            assert abs(fit.f0 - f0) < 0.5
            assert abs(fit.Q - Q) < 5.0


@pytest.mark.parametrize("strategy", ("joint", "dof_filtered"))
def test_recover_gains_with_noise(synthetic_records, strategy):
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
    fit = mag.fit_dof(noisy, "x", electrodes, 39.0, 30.0, True,
                      fit_strategy=strategy)
    np.testing.assert_allclose(np.abs(fit.gains), np.abs(true_G["x"]), rtol=0.15, atol=0.05)


def test_mag_then_linear_noisy_single_electrode(synthetic_records):
    """mag_then_linear needs enough tones per electrode to constrain Q.

    With a single electrode and many tones (the typical real scenario), it
    recovers gains accurately even with noise.
    """
    records, true_G, electrodes, dof_params = synthetic_records
    rng = np.random.default_rng(42)
    # Build single-electrode records: all X tones on E1
    single_e_records = []
    f0_x, Q_x = 39.0, 30.0
    freqs = f0_x + (np.arange(8) - 3.5) * 1.5
    for f in freqs:
        tf = {}
        coh = {}
        for d, (f0d, Qd) in dof_params.items():
            val = true_G[d][0] * mag.plant_lorentzian(f, f0d, Qd)
            noise = rng.normal(0, 0.02 * max(abs(val), 1e-3)) + \
                    1j * rng.normal(0, 0.02 * max(abs(val), 1e-3))
            tf[d] = complex(val + noise)
            coh[d] = 0.95
        single_e_records.append({"electrode": "E1", "dof_intended": "x",
                                 "freq": float(f), "tf": tf, "coh": coh})
    fit = mag.fit_dof(single_e_records, "x", ["E1"], 39.0, 30.0, True,
                      fit_strategy="mag_then_linear")
    assert abs(fit.f0 - 39.0) < 1.0
    np.testing.assert_allclose(np.abs(fit.gains), [1.0], rtol=0.15)


def test_cross_coupling_offdiagonal_recovered(synthetic_records):
    # true_G["x"][1] = 0.6+0.2j is a cross term (E2 into X); recover its magnitude.
    records, true_G, electrodes, _ = synthetic_records
    fit = mag.fit_dof(records, "x", electrodes, 39.0, 30.0, True)
    assert abs(abs(fit.gains[1]) - abs(true_G["x"][1])) < 1e-2


def test_plant_lorentzian_peak_normalized():
    f0, Q = 40.0, 25.0
    assert abs(abs(mag.plant_lorentzian(f0, f0, Q)) - 1.0) < 1e-9


def test_y_fit_with_x_crosstalk(synthetic_records):
    """Verify dof_filtered/mag_then_linear are robust to cross-coupling contamination.

    Injects strong X resonance signal into the Y measurement of X-intended tones
    (simulating the real failure mode where X→Y cross-coupling has high coherence
    but wrong phase for a Y Lorentzian).
    """
    records, true_G, electrodes, dof_params = synthetic_records
    corrupted = []
    for r in records:
        rr = {"electrode": r["electrode"], "dof_intended": r["dof_intended"],
              "freq": r["freq"], "tf": dict(r["tf"]), "coh": dict(r["coh"])}
        if r["dof_intended"] == "x":
            # X-intended tones: make Y measurement look like X plant (wrong phase)
            rr["tf"]["y"] = r["tf"]["x"] * 0.5
            rr["coh"]["y"] = 0.95
        corrupted.append(rr)

    # dof_filtered and mag_then_linear should find correct Y resonance
    for strategy in ("dof_filtered", "mag_then_linear"):
        fit = mag.fit_dof(corrupted, "y", electrodes, 54.0, 30.0, True,
                          fit_strategy=strategy)
        assert abs(fit.f0 - 54.0) < 1.0, (
            f"{strategy}: f0={fit.f0:.1f}, expected ~54")
        np.testing.assert_allclose(
            np.abs(fit.gains), np.abs(true_G["y"]), rtol=0.1, atol=0.05)


def test_invalid_strategy_raises():
    """fit_dof rejects unknown fit_strategy values."""
    with pytest.raises(ValueError, match="fit_strategy"):
        mag.fit_dof([], "x", ["E1"], 39.0, 30.0, True, fit_strategy="bogus")
