import pytest

import size_model as sm


def test_self_test_checks_pass():
    bad = [(n, d) for n, ok, d in sm._checks() if not ok]
    assert not bad


@pytest.mark.control
def test_control_golden_rejects_wrong_index():
    name, ok, detail = sm._checks(m_golden=1.50)[0]
    assert not ok, detail


def test_reference_diameter_10pg():
    assert sm.diameter_from_mass(10e-15) == pytest.approx(1.758e-6, rel=2e-3)


def test_estimate_accepts_tiny_and_rejects_reference():
    refs = [(1.0e6, 1.758e-6)]
    small = sm.estimate_diameter(1.0e6 * sm.predicted_ratio(150e-9, 1.758e-6, pol_deg=45), refs)
    assert small['d_max_m'] < 400e-9
    same = sm.estimate_diameter(1.0e6, refs)
    assert same['d_min_m'] > 400e-9
