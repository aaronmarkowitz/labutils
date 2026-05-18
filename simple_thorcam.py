#!/usr/bin/env python3
"""
Simplified Thorlabs camera viewer - single camera, single process.
Runs independently to avoid SDK corruption from IDS camera.
"""
import sys
import time
import threading
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
import numpy as np
import cv2

from simple_cam_base import SimpleCameraViewer

# Import Thorlabs SDK
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera


class ThorlabsViewer(SimpleCameraViewer):
    """Simplified viewer for Thorlabs cameras only."""

    def __init__(self):
        self.sdk = None
        self.exposure_slider = None
        self.exposure_label = None
        self.sensor_width = 0
        self.sensor_height = 0
        self.bit_depth = 8
        super().__init__("Thorlabs")

    def create_controls(self):
        """Add Thorlabs-specific controls."""
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

    def refresh_cameras(self):
        """Discover Thorlabs cameras."""
        print("Discovering Thorlabs cameras...")

        # Initialize SDK if needed
        if self.sdk is None:
            try:
                self.sdk = TLCameraSDK()
                print("Thorlabs SDK initialized")
            except Exception as e:
                print(f"Failed to initialize Thorlabs SDK: {e}")
                self.status_label.setText("Status: SDK initialization failed")
                return

        # Discover cameras
        try:
            cameras = self.sdk.discover_available_cameras()
            print(f"Found {len(cameras)} Thorlabs camera(s): {cameras}")

            self.camera_combo.clear()
            for serial in cameras:
                self.camera_combo.addItem(f"Thorlabs Camera {serial}", serial)

            if len(cameras) == 0:
                self.status_label.setText("Status: No cameras found")
            else:
                self.status_label.setText(f"Status: Found {len(cameras)} camera(s)")
        except Exception as e:
            print(f"Error discovering cameras: {e}")
            self.status_label.setText("Status: Discovery failed")

    def connect_camera(self):
        """Connect to selected Thorlabs camera."""
        if self.camera_combo.count() == 0:
            self.status_label.setText("Status: No cameras available")
            return

        serial = self.camera_combo.currentData()
        if not serial:
            self.status_label.setText("Status: No camera selected")
            return

        try:
            print(f"Connecting to Thorlabs camera {serial}...")
            self.camera = self.sdk.open_camera(serial)
            self.camera_id = serial

            print(f"Camera model: {self.camera.model}")
            print(f"Firmware: {self.camera.firmware_version}")

            # Configure for continuous mode with BLOCKING poll (eliminates USB polling!)
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.image_poll_timeout_ms = 60000  # 60 second timeout - blocks until frame ready
            print("Configured for continuous mode with blocking poll (event-driven)")

            # Set initial exposure
            exp_ms = self.exposure_slider.value()
            self.camera.exposure_time_us = int(exp_ms * 1000)
            print(f"Exposure set to {exp_ms} ms")

            # Cache sensor dimensions and bit depth
            self.sensor_width = self.camera.sensor_width_pixels
            self.sensor_height = self.camera.sensor_height_pixels
            self.bit_depth = self.camera.bit_depth
            print(f"Sensor: {self.sensor_width}x{self.sensor_height}, {self.bit_depth}-bit")

            # Arm camera with many buffers
            num_buffers = 30
            self.camera.arm(num_buffers)
            print(f"Armed with {num_buffers} buffers")

            # Issue initial trigger for continuous mode
            time.sleep(0.2)
            self.camera.issue_software_trigger()
            print("Initial trigger issued")

            # Start acquisition and display
            self.start_acquisition_thread()
            self.start_display_timer()

            # Update UI
            self.connect_btn.setText("Disconnect")
            self.exposure_slider.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.status_label.setText(f"Status: Connected to {serial}")
            print("Thorlabs camera connected successfully")

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
        """Dispose Thorlabs camera."""
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
        """Thorlabs-specific acquisition loop."""
        print("Starting Thorlabs acquisition loop")
        poll_count = 0
        frame_count = 0
        last_report = time.time()

        while not self.acq_stop_event.is_set():
            if self.camera is None:
                break

            try:
                # BLOCKING POLL: Waits until frame ready (eliminates USB polling!)
                # With 60s timeout, this blocks until frame available
                poll_count += 1
                frame = self.camera.get_pending_frame_or_null()

                if self.acq_stop_event.is_set():
                    break

                if frame is None:
                    # Timeout or error - shouldn't happen often with blocking mode
                    continue

                frame_count += 1

                # Store for display
                with self.frame_lock:
                    self.pending_frame = frame
                    self.new_frame_available = True

                # Record if active
                with self.video_lock:
                    if self.recording and self.video_writer:
                        w = frame.image_buffer_size_pixels_horizontal
                        h = frame.image_buffer_size_pixels_vertical
                        raw = frame.image_buffer
                        # Handle bit depth
                        if self.bit_depth <= 8:
                            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
                        else:
                            img16 = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
                            img = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        self.video_writer.write(bgr)
                        self.recorded_frame_count += 1

                # Debug report every 5 seconds
                if time.time() - last_report >= 5.0:
                    elapsed = time.time() - last_report
                    poll_rate = poll_count / elapsed
                    frame_rate = frame_count / elapsed
                    print(f"Thorlabs: {poll_count} polls ({poll_rate:.0f}/s), {frame_count} frames ({frame_rate:.1f} fps) in {elapsed:.1f}s")
                    poll_count = 0
                    frame_count = 0
                    last_report = time.time()

            except Exception as e:
                if self.acq_stop_event.is_set():
                    break
                print(f"Acquisition error: {e}")
                time.sleep(0.1)

        print("Thorlabs acquisition loop exiting")

    def get_frame_dimensions(self, frame):
        """Get Thorlabs frame dimensions."""
        return self.sensor_width, self.sensor_height

    def get_bit_depth(self):
        """Get Thorlabs bit depth."""
        return self.bit_depth

    def set_exposure(self, value_ms):
        """Update camera exposure."""
        if self.camera:
            try:
                self.camera.exposure_time_us = int(value_ms * 1000)
                self.exposure_label.setText(str(value_ms))
            except Exception as e:
                print(f"Failed to set exposure: {e}")

    def closeEvent(self, event):
        """Clean up SDK on close."""
        super().closeEvent(event)
        if self.sdk:
            try:
                self.sdk.dispose()
                print("Thorlabs SDK disposed")
            except Exception as e:
                print(f"SDK dispose error: {e}")
        event.accept()


def main():
    app = QApplication(sys.argv)
    viewer = ThorlabsViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
