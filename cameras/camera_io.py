"""Headless camera access for scripts (no Qt GUI): raw counts, not display-scaled.

`dualcam_fast.py` min-max normalises every frame for display and recording, so
absolute brightness never leaves it. Scripts that need photometry (the particle
loader) open a camera here instead. A camera can be held by only one process:
disconnect it in dualcam_fast first.

    list_cameras() -> [CameraInfo]
    cam = open_camera(serial)            # Thorlabs or IDS, chosen by serial
    cam.set_exposure_us(20000)
    cam.set_roi(x, y, w, h)              # explicit ROI only (CS165MU: never None)
    stack, meta = cam.grab(n)            # uint16 (n, h, w) raw counts + metadata
    cam.close()

Runs in the `thorcam` conda env. The IDS adapter is reused from dualcam_fast
(imported, not copied, so there is one implementation).
"""
import ctypes
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@dataclass
class CameraInfo:
    vendor: str         # 'thorlabs' | 'ids'
    serial: str
    model: str
    in_use: bool
    device_id: int = None   # IDS only


_tl_sdk = None


def _thorlabs_sdk():
    global _tl_sdk
    if _tl_sdk is None:
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        _tl_sdk = TLCameraSDK()
    return _tl_sdk


def list_cameras(vendors=('ids', 'thorlabs')):
    """Enumerate cameras. IDS first: its list reports in-use and is harmless.
    Thorlabs discovery from a second process while dualcam_fast holds the
    Thorlabs camera has segfaulted (2026-09-28), so pass vendors=('ids',) when
    the GUI is running with a Thorlabs camera connected."""
    out = []
    if 'ids' in vendors:
        out.extend(_list_ids())
    if 'thorlabs' in vendors:
        try:
            sdk = _thorlabs_sdk()
            for serial in sdk.discover_available_cameras():
                # The Thorlabs SDK does not report "in use"; opening fails if it is.
                out.append(CameraInfo('thorlabs', str(serial), 'Thorlabs', False))
        except Exception as e:
            print(f"camera_io: Thorlabs discovery failed: {e}")
    return out


def _list_ids():
    out = []
    try:
        from pyueye import ueye
        n = ctypes.c_int(0)
        if ueye.is_GetNumberOfCameras(n) == ueye.IS_SUCCESS and n.value > 0:
            cl = ueye.UEYE_CAMERA_LIST(ueye.UEYE_CAMERA_INFO * n.value)
            cl.dwCount = n.value
            if ueye.is_GetCameraList(cl) == ueye.IS_SUCCESS:
                for i in range(n.value):
                    info = cl.uci[i]
                    name = (info.FullModelName or info.Model).decode('ascii', 'replace').rstrip('\x00')
                    out.append(CameraInfo('ids', info.SerNo.decode('ascii', 'replace').rstrip('\x00'),
                                          name, bool(info.dwInUse), int(info.dwDeviceID)))
    except Exception as e:
        print(f"camera_io: IDS discovery failed: {e}")
    return out


class RawCamera:
    """Common interface. Subclasses fill `_open`, `_frame`, exposure/ROI."""
    vendor = None
    bit_depth = None
    serial = None
    model = None

    @property
    def saturation_level(self):
        return (1 << self.bit_depth) - 1

    def grab(self, n, timeout_s=5.0, discard=1):
        """Return (stack uint16 (n,h,w), meta). `discard` frames are dropped first
        so a frame exposed under previous settings is not returned."""
        frames, ts = [], []
        t_end = time.monotonic() + timeout_s + n * max(self.exposure_us, 1) * 1e-6 * 2
        while len(frames) < n + discard:
            if time.monotonic() > t_end:
                raise TimeoutError(f"{self.serial}: got {len(frames)} of {n + discard} frames")
            f = self._frame()
            if f is None:
                continue
            frames.append(f)
            ts.append(time.time())
        stack = np.stack(frames[discard:]).astype(np.uint16)
        meta = {'vendor': self.vendor, 'serial': self.serial, 'model': self.model,
                'bit_depth': self.bit_depth, 'saturation_level': self.saturation_level,
                'exposure_us': self.exposure_us, 'roi': self.roi,
                't_unix': ts[discard:]}
        return stack, meta

    def stream(self):
        """Yield (frame uint16 (h,w), t_unix) forever."""
        while True:
            f = self._frame()
            if f is not None:
                yield f.astype(np.uint16), time.time()


class ThorlabsRaw(RawCamera):
    vendor = 'thorlabs'

    def __init__(self, serial, n_buffers=30):
        sdk = _thorlabs_sdk()
        self.cam = sdk.open_camera(serial)
        self.serial = str(serial)
        self.model = getattr(self.cam, 'model', 'Thorlabs')
        self.cam.frames_per_trigger_zero_for_unlimited = 0
        self.cam.image_poll_timeout_ms = 200
        self.bit_depth = int(self.cam.bit_depth)
        self.sensor_w = int(self.cam.sensor_width_pixels)
        self.sensor_h = int(self.cam.sensor_height_pixels)
        self.exposure_us = int(self.cam.exposure_time_us)
        self.roi = (0, 0, self.sensor_w, self.sensor_h)
        self._n_buffers = n_buffers
        self._arm()

    def _arm(self):
        self.cam.arm(self._n_buffers)
        time.sleep(0.2)
        self.cam.issue_software_trigger()

    def _frame(self):
        fr = self.cam.get_pending_frame_or_null()
        if fr is None:
            return None
        h = fr.image_buffer_size_pixels_vertical
        w = fr.image_buffer_size_pixels_horizontal
        return np.array(fr.image_buffer, dtype=np.uint16, copy=True).reshape(h, w)

    def set_exposure_us(self, us):
        self.cam.exposure_time_us = int(us)
        self.exposure_us = int(self.cam.exposure_time_us)
        return self.exposure_us

    def set_roi(self, x, y, w, h):
        from thorlabs_tsi_sdk.tl_camera import ROI
        self.cam.disarm()
        # Explicit ROI always: roi=None crashes the CS165MU SDK (memory note).
        self.cam.roi = ROI(int(x), int(y), int(x + w - 1), int(y + h - 1))
        r = self.cam.roi
        self.roi = (r.upper_left_x_pixels, r.upper_left_y_pixels,
                    r.lower_right_x_pixels - r.upper_left_x_pixels + 1,
                    r.lower_right_y_pixels - r.upper_left_y_pixels + 1)
        self._arm()
        return self.roi

    def reset_roi(self):
        return self.set_roi(0, 0, self.sensor_w, self.sensor_h)

    def close(self):
        try:
            self.cam.disarm()
        finally:
            self.cam.dispose()


class IDSRaw(RawCamera):
    vendor = 'ids'

    def __init__(self, device_id, serial=None):
        from dualcam_fast import IDSCamera   # single implementation, reused
        self.cam = IDSCamera(device_id)
        self.serial = serial or self.cam.serial_number
        self.model = self.cam.model
        self.bit_depth = 8
        self.sensor_w = self.cam.sensor_width_pixels
        self.sensor_h = self.cam.sensor_height_pixels
        self.roi = (0, 0, self.sensor_w, self.sensor_h)
        self.exposure_us = 10000
        self.cam.arm()
        self.set_exposure_us(self.exposure_us)

    def _frame(self):
        fr = self.cam.wait_for_frame(timeout_ms=200)
        if fr is None:
            return None
        return fr.image_buffer.reshape(fr.image_buffer_size_pixels_vertical,
                                       fr.image_buffer_size_pixels_horizontal)

    def set_exposure_us(self, us):
        # Exposure is capped by the frame period; slow the frame rate to fit.
        lo, hi = self.cam.get_frame_rate_range()
        if hi > 0:
            fps = min(hi, 0.9e6 / max(us, 1))
            self.cam.set_frame_rate(max(fps, lo))
        self.cam.exposure_time_us = int(us)
        exp = ctypes.c_double()
        ue = self.cam._ue
        ue.is_Exposure(self.cam._hCam, ue.EXPOSURE_CMD.IS_EXPOSURE_CMD_GET_EXPOSURE,
                       exp, ctypes.sizeof(exp))
        self.exposure_us = int(round(exp.value * 1000))
        return self.exposure_us

    def set_roi(self, x, y, w, h):
        self.cam.set_aoi(int(x), int(y), int(w), int(h))
        self.roi = (int(x), int(y), int(w), int(h))
        return self.roi

    def reset_roi(self):
        self.cam.reset_aoi()
        self.roi = (0, 0, self.sensor_w, self.sensor_h)
        return self.roi

    def close(self):
        try:
            self.cam.disarm()
        finally:
            self.cam.dispose()


def open_camera(serial, vendors=('ids', 'thorlabs')):
    for info in list_cameras(vendors):
        if info.serial == str(serial):
            if info.vendor == 'thorlabs':
                return ThorlabsRaw(info.serial)
            if info.in_use:
                raise RuntimeError(f"IDS camera {serial} is in use (disconnect it in dualcam_fast)")
            return IDSRaw(info.device_id, info.serial)
    raise LookupError(f"no camera with serial {serial}; found "
                      f"{[(c.vendor, c.serial) for c in list_cameras(vendors)]}")


def close_sdk():
    global _tl_sdk
    if _tl_sdk is not None:
        try:
            _tl_sdk.dispose()
        finally:
            _tl_sdk = None
