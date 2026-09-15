"""Output-range headroom check for the Moku sweep drive (`sweep.py`).

`sweep.py` ramps the sine amplitude multiplicatively for the whole run, so a DC
offset that is harmless on pulse 1 can push the output past the rail hundreds of
pulses later -- mid-run, unattended. Moku amplitude is peak-to-peak, so the
instantaneous output spans `offset +- amplitude/2`: the excursion that matters is
`|offset| + amplitude/2`, not the offset or the amplitude alone.

This lives in its own module (and imports nothing beyond the stdlib) because
`sweep.py` connects to the instrument at import time and therefore cannot be
imported by a test. Run the checks with no hardware and no conda env:

    python3 sweep_limits.py --self-test

The self-test includes a negative control: the checker must PASS a config known
to fit inside the rail and FAIL the same config with an offset added, so it is
shown to discriminate rather than to warn about everything.
"""

PEAK_LIMIT_V = 5.0
"""Moku:Go output range, +-5 V peak. A Moku:Pro rail is higher -- pass
`peak_limit_v` explicitly if this module is ever reused for one."""

AMP_MAX_VPP = 10.0
"""Upper bound of the `amplitude` argument of `generate_waveform`
(moku/instruments/_waveformgenerator.py:112, documented range 4e-3 V to 10 V
peak-to-peak). Exceeding it is rejected by the instrument, not clipped."""

OFFSET_LIMIT_V = 5.0
"""Upper bound of the `offset` argument of `generate_waveform`
(_waveformgenerator.py:118, documented range -5 V to +5 V)."""


def peak_excursion(amp_vpp, dc_offset_v):
    """Most extreme instantaneous output voltage of a sine of peak-to-peak
    amplitude `amp_vpp` sitting on `dc_offset_v`."""
    return abs(dc_offset_v) + abs(amp_vpp) / 2.0


def pulse_amplitudes(start_amp, amp_incr, n_pulses):
    """The amplitude of each pulse the sweep loop will emit.

    Mirrors `c_amp = c_amp * amp_incr` in the `sweep.py` drive loop, including
    the fact that the loop is bounded by elapsed time and not by `end amp`, so
    the last pulses can overshoot the configured `end amp`.
    """
    return [start_amp * amp_incr ** k for k in range(n_pulses)]


def n_pulses_emitted(sweep_duration, max_time):
    """Number of iterations the `sweep.py` drive loop actually performs.

    The loop is `while time_elapsed <= max_time` stepping by `sweep_duration`
    from 0, so it fires once more than `max_time / sweep_duration` -- one extra
    pulse beyond the `no_pulses` reported to the user. Modelled here rather than
    corrected so the headroom check covers every pulse that really goes out.
    """
    return int(max_time / sweep_duration + 1e-9) + 1


def check_headroom(amplitudes, dc_offset_v, peak_limit_v=PEAK_LIMIT_V,
                   amp_max_vpp=AMP_MAX_VPP, offset_limit_v=OFFSET_LIMIT_V):
    """Check a whole planned amplitude ramp against the output range.

    Returns a dict with:
      `messages`        list of human-readable warnings; empty means the ramp fits
      `first_unsafe`    1-based index of the first pulse whose excursion exceeds
                        `peak_limit_v` (matches the pulse numbering `sweep.py`
                        prints), or None
      `peak_v`          largest excursion over the ramp
      `first_over_amp`  1-based index of the first pulse whose amplitude alone
                        exceeds `amp_max_vpp`, or None
    """
    messages = []
    excursions = [peak_excursion(a, dc_offset_v) for a in amplitudes]
    peak_v = max(excursions) if excursions else 0.0

    first_unsafe = next((k + 1 for k, v in enumerate(excursions)
                         if v > peak_limit_v), None)
    first_over_amp = next((k + 1 for k, a in enumerate(amplitudes)
                           if abs(a) > amp_max_vpp), None)

    if abs(dc_offset_v) > offset_limit_v:
        messages.append(
            "dc offset {:+g} V is outside the +-{:g} V range of the offset "
            "argument; generate_waveform will reject it on the first pulse."
            .format(dc_offset_v, offset_limit_v))

    if first_unsafe is not None:
        messages.append(
            "output range: pulse {} of {} is the first to exceed +-{:g} V "
            "(amplitude {:.3f} Vpp on a {:+g} V offset reaches {:.3f} V); the "
            "ramp peaks at {:.3f} V. Beyond that point the Moku either rejects "
            "the call or clips."
            .format(first_unsafe, len(amplitudes), peak_limit_v,
                    amplitudes[first_unsafe - 1], dc_offset_v,
                    excursions[first_unsafe - 1], peak_v))

    if first_over_amp is not None:
        messages.append(
            "amplitude range: pulse {} of {} is the first to exceed the {:g} Vpp "
            "amplitude limit ({:.3f} Vpp); the ramp ends at {:.3f} Vpp."
            .format(first_over_amp, len(amplitudes), amp_max_vpp,
                    amplitudes[first_over_amp - 1], amplitudes[-1]))

    return {"messages": messages, "first_unsafe": first_unsafe,
            "peak_v": peak_v, "first_over_amp": first_over_amp}


def _self_test():
    # --- unit checks on the excursion formula -------------------------------
    # A 2 Vpp sine on no offset spans -1..+1 V.
    assert peak_excursion(2.0, 0.0) == 1.0
    # Sign of the offset cannot matter: the excursion is symmetric.
    assert peak_excursion(2.0, 4.0) == peak_excursion(2.0, -4.0) == 5.0
    # Limiting case: no amplitude at all leaves just the DC level.
    assert peak_excursion(0.0, 3.3) == 3.3

    # --- boundary: exactly at the rail is allowed, a hair over is not -------
    # 10 Vpp on 0 V offset reaches exactly +-5 V.
    at_rail = check_headroom([10.0], 0.0)
    assert at_rail["first_unsafe"] is None, at_rail
    assert at_rail["peak_v"] == 5.0
    over = check_headroom([10.0], 0.01)
    assert over["first_unsafe"] == 1, over

    # --- golden value on the real config shape ------------------------------
    # amp 0.3 Vpp ramping by 1.02 with a +0.5 V offset: unsafe once the
    # amplitude passes 2*(5 - 0.5) = 9 Vpp. 0.3*1.02**171 = 8.867 Vpp (excursion
    # 4.933 V, safe), 0.3*1.02**172 = 9.044 Vpp (excursion 5.022 V, over), so
    # the first unsafe pulse is 1-based 173.
    ramp = pulse_amplitudes(0.3, 1.02, 200)
    assert abs(ramp[171] - 8.8665) < 0.001, ramp[171]
    assert abs(ramp[172] - 9.0438) < 0.001, ramp[172]
    assert abs(peak_excursion(ramp[171], 0.5) - 4.9332) < 0.001
    assert abs(peak_excursion(ramp[172], 0.5) - 5.0219) < 0.001
    offset_run = check_headroom(ramp, 0.5)
    assert offset_run["first_unsafe"] == 173, offset_run

    # --- negative control ---------------------------------------------------
    # The SAME ramp with no offset must pass. Without this arm, a checker that
    # flagged every config would still satisfy every assertion above.
    control = check_headroom(pulse_amplitudes(0.3, 1.02, 173), 0.0)
    assert control["messages"] == [], control
    assert control["first_unsafe"] is None, control
    # and the offset is the only thing that changed between the two.
    flagged = check_headroom(pulse_amplitudes(0.3, 1.02, 173), 0.5)
    assert flagged["messages"], flagged
    assert "pulse 173" in flagged["messages"][0], flagged["messages"]

    # --- an offset the API itself would reject ------------------------------
    rejected = check_headroom([1.0], 7.0)
    assert any("outside the +-5 V range" in m for m in rejected["messages"]), rejected

    # --- amplitude-limit arm, independent of the offset ---------------------
    # 12 Vpp exceeds the 10 Vpp amplitude bound even with zero offset.
    amp_over = check_headroom([1.0, 12.0], 0.0)
    assert amp_over["first_over_amp"] == 2, amp_over
    assert check_headroom([1.0, 9.9], 0.0)["first_over_amp"] is None

    # --- loop-length model matches sweep.py's `while time_elapsed <= max_time`
    # T=10, max_time=30 fires at t=0,10,20,30 -> 4 pulses, not 3.
    assert n_pulses_emitted(10, 30) == 4
    assert n_pulses_emitted(0.5, 5.0) == 11

    print("sweep_limits self-test: all checks passed "
          "(excursion formula, rail boundary, golden ramp, negative control, "
          "offset/amplitude API limits, loop length).")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        sys.exit("nothing to do; run with --self-test")
