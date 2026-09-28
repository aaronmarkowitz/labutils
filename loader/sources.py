"""Frame sources for the loader: the live x cam, or a replayed real scene.

Both expose:
    frames()                   iterator of (frame float array, t_unix)
    grab(exposure_us, n)       (stack, meta) -- used by the exposure ladder
    exposure_us, saturation_level, serial
    on_dropper_start()/on_eject()  hooks (no-ops live; drive the replay scenario)
"""
import time

import numpy as np

import synth


class LiveSource:
    def __init__(self, cam, exposure_us, roi=None):
        self.cam = cam
        self.serial = cam.serial
        self.saturation_level = cam.saturation_level
        if roi is not None:
            cam.set_roi(*roi)
        self.exposure_us = cam.set_exposure_us(exposure_us)

    def frames(self):
        for f, t in self.cam.stream():
            yield f.astype(float), t

    def grab(self, exposure_us, n):
        got = self.cam.set_exposure_us(exposure_us)
        stack, meta = self.cam.grab(n, discard=2)
        return stack, meta

    def set_exposure_us(self, us):
        self.exposure_us = self.cam.set_exposure_us(us)
        return self.exposure_us

    def on_dropper_start(self):
        pass

    def on_eject(self):
        pass

    def close(self):
        self.cam.close()


class ReplaySource:
    """Real particle-free scene frames (looped) with a scripted particle.

    `excess` is a real particle cut-out cube aligned to the scene (synth.real_scene).
    `schedule`: list of brightness scales, one per particle; the k-th particle
    appears `delay_s` after the k-th dropper start and stays until on_eject().
    A scale of 1.0 reproduces the recorded particle. Frames are paced at `fps`
    (real time) so dropper latency is measured against a realistic clock.
    `occupied_at_start`: a particle (the first scale) is already there.
    """

    def __init__(self, scene, excess, exposure_us, saturation_level, fps=13.0,
                 schedule=(1.0,), delay_s=1.5, transit_scale=None, realtime=True,
                 occupied_at_start=False, serial='replay'):
        self.scene = np.asarray(scene, dtype=float)
        self.excess = np.asarray(excess, dtype=float)
        self.exposure_us = exposure_us
        self.saturation_level = saturation_level
        self.fps = fps
        self.schedule = list(schedule)
        self.delay_s = delay_s
        self.transit_scale = transit_scale
        self.realtime = realtime
        self.serial = serial
        self._i = 0
        self._rng = np.random.default_rng(1)
        self._k = 0                       # next particle index
        self._present = None              # scale of the particle now in the trap
        self._appear_at = None
        self._transit_t0 = None
        if occupied_at_start and self.schedule:
            self._present = self.schedule[0]
            self._k = 1
        self.log = []

    def _next(self):
        j = self._i % len(self.scene)
        f = self.scene[j].copy()
        t = time.time()
        if self._appear_at is not None and t >= self._appear_at:
            self._present, self._appear_at = self._pending, None
            self.log.append(('appear', t, self._present))
        if self._present is not None:
            f += self._present * self.excess[j]
        if self._transit_t0 is not None:
            # a falling particle crossing the scene top->bottom in ~0.4 s
            dt = t - self._transit_t0
            h = f.shape[0]
            y = -30 + dt * (h + 60) / 0.4
            if y > h + 30:
                self._transit_t0 = None
            else:
                shift = int(round(y - h / 2))
                e = np.roll(self.excess[j], shift, axis=0)
                if shift > 0:
                    e[:shift] = 0
                elif shift < 0:
                    e[shift:] = 0
                f += self.transit_scale * e
        self._i += 1
        if self.realtime:
            time.sleep(1.0 / self.fps)
        return synth.requantize(f, self.saturation_level, self._rng), t

    def frames(self):
        while True:
            yield self._next()

    def grab(self, exposure_us, n):
        # Replay has one exposure: brightness scales with exposure as a linear camera.
        s = exposure_us / self.exposure_us
        stack = []
        ts = []
        for _ in range(n):
            f, t = self._next()
            stack.append(np.clip(f * s, 0, self.saturation_level))
            ts.append(t)
        return np.stack(stack), {'exposure_us': exposure_us,
                                 'saturation_level': self.saturation_level, 't_unix': ts,
                                 'serial': self.serial}

    def set_exposure_us(self, us):
        return self.exposure_us

    def on_dropper_start(self):
        if self.transit_scale is not None:
            self._transit_t0 = time.time() + 0.3 * self.delay_s
        if self._present is None and self._k < len(self.schedule):
            self._pending = self.schedule[self._k]
            self._k += 1
            self._appear_at = time.time() + self.delay_s

    def on_eject(self):
        self.log.append(('eject', time.time(), self._present))
        self._present = None

    def close(self):
        pass
