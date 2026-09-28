"""Stand-in for `moku.instruments.WaveformGenerator` that records every call.

Used by the tests and by `dropper_server.py --fake` so the dropper path (ramp,
stop-on-detect, output off, relinquish) can be exercised with no instrument and
no `moku` package. `call_latency_s` imitates the HTTP round trip of a real call,
so cancel latency measured against the fake is not trivially zero.
"""
import threading
import time


class FakeWaveformGenerator:
    def __init__(self, ip=None, force_connect=False, call_latency_s=0.0, **_):
        self.ip = ip
        self.force_connect = force_connect
        self.call_latency_s = float(call_latency_s)
        self.calls = []            # (t_monotonic, name, kwargs)
        self.output_on = {}        # channel -> bool
        self._lock = threading.Lock()

    def _rec(self, name, **kw):
        if self.call_latency_s:
            time.sleep(self.call_latency_s)
        with self._lock:
            self.calls.append((time.monotonic(), name, kw))

    def generate_waveform(self, channel, type, amplitude=1, frequency=10000,
                          offset=0, **kw):
        self._rec('generate_waveform', channel=channel, type=type,
                  amplitude=amplitude, frequency=frequency, offset=offset, **kw)
        self.output_on[channel] = (type != 'Off')

    def set_sweep_mode(self, channel, source, stop_frequency=30000000,
                       sweep_time=1, trigger_level=0, **kw):
        self._rec('set_sweep_mode', channel=channel, source=source,
                  stop_frequency=stop_frequency, sweep_time=sweep_time,
                  trigger_level=trigger_level)

    def set_defaults(self):
        self._rec('set_defaults')
        self.output_on = {c: False for c in self.output_on}

    def summary(self):
        self._rec('summary')
        return 'FakeWaveformGenerator summary: outputs ' + repr(self.output_on)

    def relinquish_ownership(self):
        self._rec('relinquish_ownership')

    def names(self):
        return [c[1] for c in self.calls]
