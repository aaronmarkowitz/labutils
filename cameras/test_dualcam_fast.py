#!/usr/bin/env python3
"""
Unit tests for dualcam_fast.py pure helper functions.
No camera hardware, no PyQt5, no real SDK required.
Run with: conda run -n thorcam python3 -m pytest test_dualcam_fast.py -v
"""
import sys
import types
import time
import ctypes
import threading
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Minimal SDK stubs — injected before importing dualcam_fast so top-level
# try/except ImportError blocks succeed without real hardware SDKs present.
# ---------------------------------------------------------------------------

# Stub pyueye
_ueye_stub = types.SimpleNamespace(
    IS_SUCCESS=0,
    IS_TIMED_OUT=1,
    IS_TRIGGER_ACTIVATED=2,
    IS_SET_EVENT_FRAME=2,
    IS_SET_TRIGGER_OFF=0,
    IS_SET_TRIGGER_SOFTWARE=8,
    IS_DONT_WAIT=0,
    IS_FORCE_VIDEO_STOP=1,
    IS_CM_MONO8=6,
    IS_AOI_IMAGE_SET_AOI=1,
    IS_PIXELCLOCK_CMD_GET_NUMBER=0,
    IS_PIXELCLOCK_CMD_GET_LIST=1,
    IS_PIXELCLOCK_CMD_GET=2,
    IS_PIXELCLOCK_CMD_SET=3,
    IS_USE_DEVICE_ID=0x8000,
    EXPOSURE_CMD=types.SimpleNamespace(IS_EXPOSURE_CMD_SET_EXPOSURE=12),
)

# IS_RECT ctypes struct used by build_ids_rect
class _IS_RECT(ctypes.Structure):
    _fields_ = [
        ('s32X',      ctypes.c_int32),
        ('s32Y',      ctypes.c_int32),
        ('s32Width',  ctypes.c_int32),
        ('s32Height', ctypes.c_int32),
    ]

_ueye_stub.IS_RECT = _IS_RECT
_ueye_stub.c_mem_p = ctypes.c_char_p

_pyueye_mod = types.ModuleType("pyueye")
_pyueye_mod.ueye = _ueye_stub
sys.modules.setdefault("pyueye", _pyueye_mod)
sys.modules.setdefault("pyueye.ueye", _ueye_stub)

# Stub thorlabs_tsi_sdk
_tsi_mod      = types.ModuleType("thorlabs_tsi_sdk")
_tsi_cam_mod  = types.ModuleType("thorlabs_tsi_sdk.tl_camera")
_tsi_enum_mod = types.ModuleType("thorlabs_tsi_sdk.tl_camera_enums")
_tsi_cam_mod.TLCameraSDK = object
_tsi_cam_mod.TLCamera    = object

import collections
_ROI = collections.namedtuple(
    'ROI', ['upper_left_x_pixels', 'upper_left_y_pixels',
            'lower_right_x_pixels', 'lower_right_y_pixels'])
_tsi_cam_mod.ROI = _ROI

class _FakeSensorType:
    MONOCHROME = 0
_tsi_enum_mod.SENSOR_TYPE = _FakeSensorType
_tsi_mod.tl_camera  = _tsi_cam_mod
_tsi_mod.tl_camera_enums = _tsi_enum_mod
sys.modules.setdefault("thorlabs_tsi_sdk",              _tsi_mod)
sys.modules.setdefault("thorlabs_tsi_sdk.tl_camera",    _tsi_cam_mod)
sys.modules.setdefault("thorlabs_tsi_sdk.tl_camera_enums", _tsi_enum_mod)

# Stub cv2 and PyQt5 so the module-level imports don't fail
import unittest.mock as mock
sys.modules.setdefault("cv2", mock.MagicMock())
for _mod in ("PyQt5", "PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.QtGui",
             "PyQt5.Qt", "PyQt5.QtCore"):
    sys.modules.setdefault(_mod, mock.MagicMock())

# Now import the pure functions from dualcam_fast
sys.path.insert(0, '/home/controls/labutils/cameras')
from dualcam_fast import clamp_roi, compute_pacing_delay, reshape_ids_frame, build_ids_rect


# ---------------------------------------------------------------------------
# TestClampRoi
# ---------------------------------------------------------------------------

class TestClampRoi(unittest.TestCase):

    def test_full_sensor(self):
        self.assertEqual(clamp_roi(0, 0, 1280, 1024, 1280, 1024), (0, 0, 1280, 1024))

    def test_inside(self):
        self.assertEqual(clamp_roi(100, 100, 500, 400, 1280, 1024), (100, 100, 500, 400))

    def test_right_edge_clamp(self):
        # x=1200, w=200 → clamped to w=80
        x, y, w, h = clamp_roi(1200, 0, 200, 1024, 1280, 1024)
        self.assertEqual(x, 1200)
        self.assertEqual(w, 80)
        self.assertEqual(h, 1024)

    def test_bottom_edge_clamp(self):
        x, y, w, h = clamp_roi(0, 900, 1280, 200, 1280, 1024)
        self.assertEqual(y, 900)
        self.assertEqual(h, 124)
        self.assertEqual(w, 1280)

    def test_outside_raises(self):
        with self.assertRaises(ValueError):
            clamp_roi(1280, 0, 10, 10, 1280, 1024)

    def test_ids_step_x_floor(self):
        # x=5 should floor to 4 with step_x=4
        x, y, w, h = clamp_roi(5, 0, 100, 100, 1280, 1024, step_x=4)
        self.assertEqual(x, 4)

    def test_ids_step_y_floor(self):
        # y=3 should floor to 2 with step_y=2
        x, y, w, h = clamp_roi(0, 3, 100, 100, 1280, 1024, step_y=2)
        self.assertEqual(y, 2)

    def test_width_alignment(self):
        # w=101 should floor to 100 with step_x=4? No: 101//4=25*4=100
        x, y, w, h = clamp_roi(0, 0, 101, 100, 1280, 1024, step_x=4)
        self.assertEqual(w, 100)

    def test_negative_x_clamped_to_zero(self):
        x, y, w, h = clamp_roi(-10, 0, 100, 100, 1280, 1024)
        self.assertEqual(x, 0)
        self.assertEqual(w, 100)

    def test_zero_width_raises(self):
        with self.assertRaises(ValueError):
            clamp_roi(0, 0, 0, 100, 1280, 1024)

    def test_min_lrx_expands_width(self):
        # CS165MU lower_right_x_min=79: ROI(0,0,40,100) must expand to lrx >= 79
        x, y, w, h = clamp_roi(0, 0, 40, 100, 1440, 1080, min_lrx=79)
        self.assertGreaterEqual(x + w - 1, 79)

    def test_min_lry_expands_height(self):
        # CS165MU lower_right_y_min=3: ROI(0,0,100,2) must expand to lry >= 3
        x, y, w, h = clamp_roi(0, 0, 100, 2, 1440, 1080, min_lry=3)
        self.assertGreaterEqual(y + h - 1, 3)


# ---------------------------------------------------------------------------
# TestComputePacingDelay
# ---------------------------------------------------------------------------

class TestComputePacingDelay(unittest.TestCase):

    def test_future_target_positive(self):
        t = time.monotonic() + 0.5
        delay = compute_pacing_delay(t)
        self.assertGreater(delay, 0.0)
        self.assertLessEqual(delay, 0.5 + 0.01)  # allow tiny slop

    def test_past_target_zero(self):
        t = time.monotonic() - 1.0
        self.assertEqual(compute_pacing_delay(t), 0.0)

    def test_now_target_near_zero(self):
        t = time.monotonic()
        delay = compute_pacing_delay(t)
        self.assertGreaterEqual(delay, 0.0)
        self.assertLess(delay, 0.01)


# ---------------------------------------------------------------------------
# TestReshapeIdsFrame
# ---------------------------------------------------------------------------

class TestReshapeIdsFrame(unittest.TestCase):

    def test_no_padding(self):
        h, w, pitch = 4, 4, 4
        raw = np.arange(16, dtype=np.uint8)
        result = reshape_ids_frame(raw.tobytes(), w, h, pitch)
        self.assertEqual(result.shape, (4, 4))
        np.testing.assert_array_equal(result.ravel(), raw)

    def test_with_padding(self):
        h, w, pitch = 4, 4, 8
        # 4 rows × 8 bytes pitch; first 4 bytes are pixel data, last 4 padding
        raw = np.zeros(32, dtype=np.uint8)
        raw[0:4]   = [10, 11, 12, 13]   # row 0 pixels
        raw[8:12]  = [20, 21, 22, 23]   # row 1 pixels
        raw[16:20] = [30, 31, 32, 33]   # row 2 pixels
        raw[24:28] = [40, 41, 42, 43]   # row 3 pixels
        result = reshape_ids_frame(raw.tobytes(), w, h, pitch)
        self.assertEqual(result.shape, (4, 4))
        np.testing.assert_array_equal(result[0], [10, 11, 12, 13])
        np.testing.assert_array_equal(result[3], [40, 41, 42, 43])

    def test_wrong_size_raises(self):
        with self.assertRaises(ValueError):
            reshape_ids_frame(np.zeros(12, dtype=np.uint8).tobytes(), 4, 4, 4)

    def test_returns_copy(self):
        raw = np.zeros(16, dtype=np.uint8)
        result1 = reshape_ids_frame(raw.tobytes(), 4, 4, 4)
        result2 = reshape_ids_frame(raw.tobytes(), 4, 4, 4)
        result1[0, 0] = 99
        self.assertEqual(result2[0, 0], 0)  # independent copies


# ---------------------------------------------------------------------------
# TestBuildIdsRect
# ---------------------------------------------------------------------------

class TestBuildIdsRect(unittest.TestCase):

    def test_fields_set(self):
        rect = build_ids_rect(10, 20, 640, 480)
        self.assertEqual(rect.s32X,      10)
        self.assertEqual(rect.s32Y,      20)
        self.assertEqual(rect.s32Width,  640)
        self.assertEqual(rect.s32Height, 480)

    def test_zero_origin(self):
        rect = build_ids_rect(0, 0, 1280, 1024)
        self.assertEqual(rect.s32X,      0)
        self.assertEqual(rect.s32Y,      0)
        self.assertEqual(rect.s32Width,  1280)
        self.assertEqual(rect.s32Height, 1024)


# ---------------------------------------------------------------------------
# Integration smoke: acquisition loop exits when camera is None
# ---------------------------------------------------------------------------

class TestAcquisitionLoopExitsOnNoneCamera(unittest.TestCase):

    def test_loop_exits_when_camera_none(self):
        """_acquisition_loop must exit immediately when cam_instance.camera is None."""
        import dualcam_fast as _df
        from dualcam_fast import CameraInstance

        # Retrieve the real unbound function by walking the MRO of the class dict.
        # (ThorlabsCameraApp may be a Mock subclass due to mocked Qt; use __dict__ directly.)
        loop_fn = None
        for klass in type.mro(type(_df.ThorlabsCameraApp)):
            if '_acquisition_loop' in klass.__dict__:
                loop_fn = klass.__dict__['_acquisition_loop']
                break
        if loop_fn is None:
            self.skipTest("Cannot extract _acquisition_loop from mocked Qt environment")

        cam = CameraInstance("TestCam")
        cam.cam_type = 'thorlabs'
        cam.camera   = None          # loop exits on first iteration
        cam.acq_stop_event = threading.Event()
        cam.acq_frame_count = 0
        cam.fps = 30

        class StubApp:
            thorlabs_sdk_lock = threading.Lock()
            cameras = {"cam1": cam}

        app = StubApp()
        thread = threading.Thread(target=loop_fn, args=(app, "cam1"))
        thread.start()
        thread.join(timeout=0.5)
        self.assertFalse(thread.is_alive(), "Acquisition loop did not exit within 0.5s")


if __name__ == '__main__':
    unittest.main(verbosity=2)
