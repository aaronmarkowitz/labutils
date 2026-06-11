"""Trap-loss guard: 10-20 Hz band-RMS math and trip logic."""
import numpy as np

import measure_actuator_gain as mag


def _tone(freq, fs, dur, amp=1.0):
    t = np.arange(int(fs * dur)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def test_inband_tone_dominates_outofband():
    fs, dur = 1024.0, 4.0
    band = (10.0, 20.0)
    inband = mag.band_rms(_tone(15.0, fs, dur), fs, band)
    outband = mag.band_rms(_tone(40.0, fs, dur), fs, band)
    assert inband > 10 * outband
    # 15 Hz sine of amplitude 1 -> band RMS ~ 1/sqrt(2)
    assert abs(inband - 1 / np.sqrt(2)) < 0.05


def test_trip_factor_logic():
    fs, dur = 1024.0, 4.0
    band = (10.0, 20.0)
    baseline = mag.band_rms(_tone(15.0, fs, dur, amp=0.1), fs, band)
    grown = mag.band_rms(_tone(15.0, fs, dur, amp=0.3), fs, band)
    factor = 2.0
    assert grown > factor * baseline       # 3x growth trips a 2x threshold
    mild = mag.band_rms(_tone(15.0, fs, dur, amp=0.15), fs, band)
    assert not (mild > factor * baseline)   # 1.5x growth does not trip
