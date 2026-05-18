#!/usr/bin/env python3
"""
Simplified IDS camera viewer - single camera, single process.
Runs independently to avoid corrupting Thorlabs SDK in shared process.
"""
import sys
import time
import threading
import ctypes
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSlider, QComboBox
from PyQt5.QtCore import Qt
import numpy as np
import cv2

from simple_cam_base import SimpleCameraViewer

# Import IDS SDK
from pyueye import ueye


class IDSFrame:
    """Simple frame wrapper for IDS camera data (matches Thorlabs API)."""
    def __init__(self, buffer, width, height):
        self.image_buffer = buffer
        self.width = width
        self.height = height


class IDSCamera:
    """IDS camera wrapper with proven freerun + queue mode (from run_thorcam.py)."""

    IS_USE_DEVICE_ID = 0x8000

    def __init__(self, device_id: int):
        self._ue = ueye
        self._pending_frame = None
        self._last_seq_num = -1

        # Open camera
        self._hCam = ctypes.c_uint(device_id | self.IS_USE_DEVICE_ID)
        ret = ueye.is_InitCamera(self._hCam, None)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_InitCamera failed (device {device_id}): {ret}")

        # Get sensor info
        sinfo = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(self._hCam, sinfo)
        self.sensor_width_pixels = int(sinfo.nMaxWidth)
        self.sensor_height_pixels = int(sinfo.nMaxHeight)
        self.model = sinfo.strSensorName.decode('ascii', errors='replace').rstrip('\x00')

        # Get serial number
        binfo = ueye.BOARDINFO()
        if ueye.is_GetCameraInfo(self._hCam, binfo) == ueye.IS_SUCCESS:
            self.serial_number = binfo.SerNo.decode('ascii', errors='replace').rstrip('\x00')
        else:
            self.serial_number = "unknown"

        self.firmware_version = "N/A"
        self.bit_depth = 8

        # Set mono8 mode
        ret = ueye.is_SetColorMode(self._hCam, ueye.IS_CM_MONO8)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_SetColorMode(MONO8) failed: {ret}")

        # Allocate 3-buffer queue for non-blocking polling
        self._mem_ptrs = []
        self._mem_ids = []
        for i in range(3):
            mem_ptr = ueye.c_mem_p()
            mem_id = ctypes.c_int()
            ret = ueye.is_AllocImageMem(
                self._hCam,
                self.sensor_width_pixels, self.sensor_height_pixels,
                8, mem_ptr, mem_id)
            if ret != ueye.IS_SUCCESS:
                raise RuntimeError(f"is_AllocImageMem failed on buffer {i}: {ret}")

            ret = ueye.is_AddToSequence(self._hCam, mem_ptr, mem_id)
            if ret != ueye.IS_SUCCESS:
                raise RuntimeError(f"is_AddToSequence failed on buffer {i}: {ret}")

            self._mem_ptrs.append(mem_ptr)
            self._mem_ids.append(mem_id)

        # Keep first buffer reference
        self._mem_ptr = self._mem_ptrs[0]
        self._mem_id = self._mem_ids[0]

        # Cache pitch
        pitch = ctypes.c_int()
        ueye.is_GetImageMemPitch(self._hCam, pitch)
        self._pitch = pitch.value

        # Enable frame-ready event for event-driven acquisition (no polling!)
        ret = ueye.is_EnableEvent(self._hCam, ueye.IS_SET_EVENT_FRAME)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_EnableEvent(FRAME) failed: {ret}")

        print(f"IDS camera initialized: {self.model} (SN: {self.serial_number}), event mode enabled")

    @property
    def exposure_time_us(self):
        return None

    @exposure_time_us.setter
    def exposure_time_us(self, value_us: int):
        exp_ms = ctypes.c_double(value_us / 1000.0)
        self._ue.is_Exposure(
            self._hCam,
            self._ue.EXPOSURE_CMD.IS_EXPOSURE_CMD_SET_EXPOSURE,
            exp_ms,
            ctypes.sizeof(exp_ms))

    def arm(self, n_buffers=2):
        """Start live video capture in freerun mode."""
        # Set freerun mode (no external trigger)
        ret = self._ue.is_SetExternalTrigger(self._hCam, self._ue.IS_SET_TRIGGER_OFF)
        if ret != self._ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetExternalTrigger(OFF) failed: {ret}")

        # Start live video capture
        ret = self._ue.is_CaptureVideo(self._hCam, self._ue.IS_DONT_WAIT)
        if ret != self._ue.IS_SUCCESS:
            raise RuntimeError(f"is_CaptureVideo failed: {ret}")

        print("IDS camera armed in freerun mode")

    def disarm(self):
        """Stop live video capture."""
        self._ue.is_StopLiveVideo(self._hCam, self._ue.IS_FORCE_VIDEO_STOP)

    def wait_for_frame(self, timeout_ms=1000):
        """Block until frame ready (event-driven, eliminates USB polling)."""
        ret = self._ue.is_WaitEvent(self._hCam, self._ue.IS_SET_EVENT_FRAME, timeout_ms)

        if ret == self._ue.IS_TIMED_OUT:
            return None  # No frame within timeout
        elif ret != self._ue.IS_SUCCESS:
            print(f"IDS is_WaitEvent error: {ret}")
            return None

        # Frame is ready, get it from queue
        nNum = ctypes.c_int()
        pcMem = self._ue.c_mem_p()
        pcMemLast = self._ue.c_mem_p()
        ret = self._ue.is_GetActSeqBuf(self._hCam, nNum, pcMem, pcMemLast)

        if ret != self._ue.IS_SUCCESS:
            return None

        # Check sequence number to avoid duplicate frames
        seq_num = nNum.value
        if seq_num == self._last_seq_num:
            return None

        self._last_seq_num = seq_num

        # Copy frame data
        w, h = self.sensor_width_pixels, self.sensor_height_pixels
        raw = self._ue.get_data(pcMem, w, h, 8, self._pitch, copy=True)
        frame_np = raw.reshape(h, self._pitch)[:, :w].copy()
        frame = IDSFrame(frame_np.ravel(), w, h)

        # Unlock buffer for reuse
        self._ue.is_UnlockSeqBuf(self._hCam, nNum, pcMemLast)

        return frame

    def issue_software_trigger(self):
        """Poll for next frame from image queue (non-blocking)."""
        nNum = ctypes.c_int()
        pcMem = self._ue.c_mem_p()
        pcMemLast = self._ue.c_mem_p()
        ret = self._ue.is_GetActSeqBuf(self._hCam, nNum, pcMem, pcMemLast)

        if ret != self._ue.IS_SUCCESS:
            self._pending_frame = None
            return

        # Check sequence number to avoid duplicate frames
        seq_num = nNum.value
        if seq_num == self._last_seq_num:
            self._pending_frame = None
            return

        self._last_seq_num = seq_num

        # Copy frame data
        w, h = self.sensor_width_pixels, self.sensor_height_pixels
        raw = self._ue.get_data(pcMem, w, h, 8, self._pitch, copy=True)
        frame_np = raw.reshape(h, self._pitch)[:, :w].copy()
        self._pending_frame = IDSFrame(frame_np.ravel(), w, h)

        # Unlock buffer for reuse
        self._ue.is_UnlockSeqBuf(self._hCam, nNum, pcMemLast)

    def get_pending_frame_or_null(self):
        """Return pending frame or None."""
        f = self._pending_frame
        self._pending_frame = None
        return f

    def get_pixel_clock_list(self):
        """Get available pixel clock frequencies."""
        nNumber = ctypes.c_uint()
        ret = ueye.is_PixelClock(
            self._hCam,
            ueye.IS_PIXELCLOCK_CMD_GET_NUMBER,
            nNumber,
            ctypes.sizeof(nNumber))
        if ret != ueye.IS_SUCCESS:
            return []

        count = nNumber.value
        clocks = (ctypes.c_uint * count)()
        ret = ueye.is_PixelClock(
            self._hCam,
            ueye.IS_PIXELCLOCK_CMD_GET_LIST,
            clocks,
            count * ctypes.sizeof(ctypes.c_uint))

        if ret == ueye.IS_SUCCESS:
            return list(clocks)
        return []

    def get_pixel_clock(self):
        """Get current pixel clock frequency."""
        clock = ctypes.c_uint()
        ret = ueye.is_PixelClock(
            self._hCam,
            ueye.IS_PIXELCLOCK_CMD_GET,
            clock,
            ctypes.sizeof(clock))
        if ret == ueye.IS_SUCCESS:
            return clock.value
        return 0

    def set_pixel_clock(self, mhz: int):
        """Set pixel clock frequency."""
        clock = ctypes.c_uint(mhz)
        ret = ueye.is_PixelClock(
            self._hCam,
            ueye.IS_PIXELCLOCK_CMD_SET,
            clock,
            ctypes.sizeof(clock))
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_PixelClock(SET, {mhz}) failed: {ret}")

    def dispose(self):
        """Clean up IDS camera resources."""
        # Disable frame event
        try:
            ueye.is_DisableEvent(self._hCam, ueye.IS_SET_EVENT_FRAME)
        except Exception as e:
            print(f"DisableEvent error: {e}")

        # Free image memory
        for mem_ptr, mem_id in zip(self._mem_ptrs, self._mem_ids):
            try:
                ueye.is_FreeImageMem(self._hCam, mem_ptr, mem_id)
            except Exception as e:
                print(f"FreeImageMem error: {e}")

        # Close camera
        try:
            ueye.is_ExitCamera(self._hCam)
        except Exception as e:
            print(f"ExitCamera error: {e}")


class IDSViewer(SimpleCameraViewer):
    """Simplified viewer for IDS cameras only."""

    def __init__(self):
        self.exposure_slider = None
        self.exposure_label = None
        self.pixel_clock_combo = None
        self.pixel_clock_label = None
        self.sensor_width = 0
        self.sensor_height = 0
        super().__init__("IDS")

    def create_controls(self):
        """Add IDS-specific controls."""
        # Exposure control
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (ms):"))

        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(1)
        self.exposure_slider.setMaximum(200)
        self.exposure_slider.setValue(50)
        self.exposure_slider.setEnabled(False)
        self.exposure_slider.valueChanged.connect(self.set_exposure)
        exp_layout.addWidget(self.exposure_slider)

        self.exposure_label = QLabel("50")
        self.exposure_label.setMinimumWidth(40)
        exp_layout.addWidget(self.exposure_label)

        exp_layout.addStretch()
        self.controls_layout.addLayout(exp_layout)

        # Pixel clock control
        clock_layout = QHBoxLayout()
        self.pixel_clock_label = QLabel("Pixel Clock (MHz):")
        self.pixel_clock_label.setVisible(False)
        clock_layout.addWidget(self.pixel_clock_label)

        self.pixel_clock_combo = QComboBox()
        self.pixel_clock_combo.setVisible(False)
        self.pixel_clock_combo.setEnabled(False)
        self.pixel_clock_combo.currentTextChanged.connect(self.set_pixel_clock)
        clock_layout.addWidget(self.pixel_clock_combo)

        clock_layout.addStretch()
        self.controls_layout.addLayout(clock_layout)

    def refresh_cameras(self):
        """Discover IDS cameras."""
        print("Discovering IDS cameras...")

        try:
            # Enumerate cameras
            num_cameras = ctypes.c_int()
            ret = ueye.is_GetNumberOfCameras(num_cameras)
            if ret != ueye.IS_SUCCESS:
                print(f"is_GetNumberOfCameras failed: {ret}")
                self.status_label.setText("Status: Failed to enumerate cameras")
                return

            count = num_cameras.value
            print(f"Found {count} IDS camera(s)")

            if count == 0:
                self.camera_combo.clear()
                self.status_label.setText("Status: No cameras found")
                return

            # Get camera list
            camera_list = (ueye.UEYE_CAMERA_INFO * count)()
            ret = ueye.is_GetCameraList(camera_list)
            if ret != ueye.IS_SUCCESS:
                print(f"is_GetCameraList failed: {ret}")
                self.status_label.setText("Status: Failed to get camera list")
                return

            # Populate dropdown
            self.camera_combo.clear()
            for i in range(count):
                cam_info = camera_list[i]
                device_id = cam_info.dwDeviceID
                model = cam_info.Model.decode('ascii', errors='replace').rstrip('\x00')
                serial = cam_info.SerNo.decode('ascii', errors='replace').rstrip('\x00')
                label = f"[IDS] {model} (SN:{serial})"
                self.camera_combo.addItem(label, device_id)
                print(f"  {label}, device_id={device_id}")

            self.status_label.setText(f"Status: Found {count} camera(s)")

        except Exception as e:
            print(f"Error discovering cameras: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText("Status: Discovery failed")

    def connect_camera(self):
        """Connect to selected IDS camera."""
        if self.camera_combo.count() == 0:
            self.status_label.setText("Status: No cameras available")
            return

        device_id = self.camera_combo.currentData()
        if device_id is None:
            self.status_label.setText("Status: No camera selected")
            return

        try:
            print(f"Connecting to IDS camera {device_id}...")
            self.camera = IDSCamera(device_id)
            self.camera_id = device_id

            # Set initial exposure
            exp_ms = self.exposure_slider.value()
            self.camera.exposure_time_us = int(exp_ms * 1000)
            print(f"Exposure set to {exp_ms} ms")

            # Cache sensor dimensions
            self.sensor_width = self.camera.sensor_width_pixels
            self.sensor_height = self.camera.sensor_height_pixels
            print(f"Sensor: {self.sensor_width}x{self.sensor_height}")

            # Setup pixel clock control
            try:
                clocks = self.camera.get_pixel_clock_list()
                current = self.camera.get_pixel_clock()
                self.pixel_clock_combo.blockSignals(True)
                self.pixel_clock_combo.clear()
                for c in clocks:
                    self.pixel_clock_combo.addItem(str(c))
                self.pixel_clock_combo.setCurrentText(str(current))
                self.pixel_clock_combo.blockSignals(False)
                self.pixel_clock_label.setVisible(True)
                self.pixel_clock_combo.setVisible(True)
                self.pixel_clock_combo.setEnabled(True)
                print(f"Pixel clock: {current} MHz (available: {clocks})")
            except Exception as e:
                print(f"Could not read pixel clock info: {e}")

            # Arm camera
            self.camera.arm(2)
            print("IDS camera armed")

            # Start acquisition and display
            self.start_acquisition_thread()
            self.start_display_timer()

            # Update UI
            self.connect_btn.setText("Disconnect")
            self.exposure_slider.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.status_label.setText(f"Status: Connected to device {device_id}")
            print("IDS camera connected successfully")

        except Exception as e:
            print(f"Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Status: Connection failed - {e}")
            if self.camera:
                try:
                    self.camera.dispose()
                except:
                    pass
                self.camera = None

    def dispose_camera(self):
        """Dispose IDS camera."""
        if self.camera:
            try:
                self.camera.disarm()
            except Exception as e:
                print(f"Disarm error: {e}")

            try:
                self.camera.dispose()
            except Exception as e:
                print(f"Dispose error: {e}")

    def _acquisition_loop(self):
        """IDS-specific acquisition loop."""
        print("Starting IDS acquisition loop")
        poll_count = 0
        frame_count = 0
        last_report = time.time()

        while not self.acq_stop_event.is_set():
            if self.camera is None:
                break

            try:
                # EVENT-DRIVEN: Block until frame ready (no USB polling!)
                frame = self.camera.wait_for_frame(timeout_ms=1000)
                poll_count += 1  # Count wait attempts

                if self.acq_stop_event.is_set():
                    break

                if frame is None:
                    continue  # Timeout, retry immediately

                frame_count += 1

                # Store for display
                with self.frame_lock:
                    self.pending_frame = frame
                    self.new_frame_available = True

                # Record if active
                with self.video_lock:
                    if self.recording and self.video_writer:
                        w, h = frame.width, frame.height
                        img = np.frombuffer(frame.image_buffer, dtype=np.uint8).reshape(h, w)
                        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        self.video_writer.write(bgr)
                        self.recorded_frame_count += 1

                # Debug report every 5 seconds
                if time.time() - last_report >= 5.0:
                    elapsed = time.time() - last_report
                    poll_rate = poll_count / elapsed
                    frame_rate = frame_count / elapsed
                    print(f"IDS: {poll_count} polls ({poll_rate:.0f}/s), {frame_count} frames ({frame_rate:.1f} fps) in {elapsed:.1f}s")
                    poll_count = 0
                    frame_count = 0
                    last_report = time.time()

            except Exception as e:
                if self.acq_stop_event.is_set():
                    break
                print(f"Acquisition error: {e}")
                time.sleep(0.1)

        print("IDS acquisition loop exiting")

    def get_frame_dimensions(self, frame):
        """Get IDS frame dimensions."""
        return frame.width, frame.height

    def set_exposure(self, value_ms):
        """Update camera exposure."""
        if self.camera:
            try:
                self.camera.exposure_time_us = int(value_ms * 1000)
                self.exposure_label.setText(str(value_ms))
            except Exception as e:
                print(f"Failed to set exposure: {e}")

    def set_pixel_clock(self, value_str):
        """Update pixel clock frequency."""
        if self.camera and value_str:
            try:
                mhz = int(value_str)
                self.camera.set_pixel_clock(mhz)
                print(f"Pixel clock set to {mhz} MHz")
            except Exception as e:
                print(f"Failed to set pixel clock: {e}")


def main():
    app = QApplication(sys.argv)
    viewer = IDSViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
