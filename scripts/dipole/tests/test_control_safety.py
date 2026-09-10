"""Negative control for the config safety checks.

Every other test in this suite asserts that a *good* input is accepted. That
leaves one failure mode uncovered: a validator that has been broken into
returning None unconditionally would keep all of them green while silently
letting an unsafe config reach the hardware.

This test closes that gap. It feeds `validate_config` an out-of-range guard
band and requires it to raise, and in the same test asserts the untouched
config still passes -- so the check is proven to *discriminate*, not merely to
raise on everything.

The guard band is the trap-loss abort band (10-20 Hz for the particle). An
inverted or collapsed band means `in_guard()` matches nothing, the tone
placement in `generate_frequency_plan` stops excluding anything, and an
excitation tone can land inside the band the guard monitor is watching --
which shows up as a false abort mid-measurement at best.

Run just this one:  pytest -m control
"""
import copy

import pytest

import measure_actuator_gain as mag


@pytest.mark.control
@pytest.mark.parametrize(
    "guard_band, why",
    [
        ([20.0, 10.0], "inverted: lo > hi"),
        ([15.0, 15.0], "collapsed to zero width: lo == hi"),
    ],
)
def test_validate_config_rejects_out_of_range_guard_band(base_cfg, guard_band, why):
    # Control arm: the unmodified fixture must PASS, so a raise below is
    # attributable to the bad value and not to an incomplete fixture.
    mag.validate_config(base_cfg)

    bad = copy.deepcopy(base_cfg)
    bad["frequency_plan"]["guard_band_hz"] = guard_band

    with pytest.raises(ValueError, match="guard_band_hz"):
        mag.validate_config(bad)
