"""Eject a rejected particle -- PLANNING ONLY for now.

Agreed 2026-09-28: the electrical knock-out is not run until the rest of the
loader is validated. `plan_eject` describes the drive that would be applied and
checks it against the standing rules; `execute` refuses. When the procedure is
agreed, implement `execute` with measure_actuator_gain.py's pattern (snapshot
POLES/ACTS settings, inject through diag/awg, restore on exit, abort sentinel).
"""
GUARD_BAND_HZ = (10.0, 20.0)   # trap-loss guard band: never excite here (CLAUDE.md)
DAC_SAFE_COUNTS = 32000


class EjectDisabled(RuntimeError):
    pass


def plan_eject(freq_hz=39.0, electrode='Y1:DMD-POLES_E1_EXC', amp_start_counts=100.0,
               amp_max_counts=3000.0, ramp_factor=2.0, dwell_s=5.0):
    """Return the planned drive and the list of rule violations (empty = OK)."""
    problems = []
    if GUARD_BAND_HZ[0] <= freq_hz <= GUARD_BAND_HZ[1]:
        problems.append(f"{freq_hz} Hz is inside the {GUARD_BAND_HZ} Hz trap-loss guard band")
    if amp_max_counts > DAC_SAFE_COUNTS:
        problems.append(f"amp_max {amp_max_counts} exceeds the DAC safe clip {DAC_SAFE_COUNTS}")
    steps = []
    a = amp_start_counts
    while a <= amp_max_counts:
        steps.append(round(a, 1))
        a *= ramp_factor
    return {'electrode_exc': electrode, 'freq_hz': freq_hz, 'amplitudes_counts': steps,
            'dwell_s': dwell_s,
            'stop_when': 'x-cam occupancy check reports the trap empty',
            'restore': 'snapshot/restore POLES & ACTS settings as measure_actuator_gain.py',
            'problems': problems}


def execute(plan, allow=False):
    raise EjectDisabled("electrical eject is disabled until the procedure is agreed "
                        "(2026-09-28); planned drive: " + repr(plan))
