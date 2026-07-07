#!/usr/bin/env python3
"""
Base class for simplified single-camera viewers.
Provides common GUI components and acquisition/display pattern.
"""
import sys
import time
import threading
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QComboBox, QSlider, QFileDialog)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2

try:
    from video_metadata import write_sidecar
except ImportError:  # when imported as a package
    from cameras.video_metadata import write_sidecar


class SimpleCameraViewer(QMainWindow):
    """Base class for single-camera viewers with process isolation."""

    def __init__(self, camera_type: str):
        super().__init__()
        self.camera_type = camera_type

        # Camera state
        self.camera = None
        self.camera_id = None

        # Acquisition thread
        self.acq_thread = None
        self.acq_stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.pending_frame = None
        self.new_frame_available = False

        # FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Recording state
        self.recording = False
        self.video_lock = threading.Lock()
        self.video_writer = None
        self.recorded_frame_count = 0
        self.recording_start_time = 0

        # Display timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)

        self.setup_ui()

    def setup_ui(self):
        """Create the GUI layout."""
        self.setWindowTitle(f"{self.camera_type} Camera Viewer")
        self.setGeometry(100, 100, 800, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("No camera connected")
        self.video_label.setStyleSheet("QLabel { background-color: black; color: gray; }")
        layout.addWidget(self.video_label)

        # Camera selection
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(300)
        cam_layout.addWidget(self.camera_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        cam_layout.addWidget(self.refresh_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connect)
        cam_layout.addWidget(self.connect_btn)

        cam_layout.addStretch()
        layout.addLayout(cam_layout)

        # Camera-specific controls (subclass implements)
        self.controls_layout = QVBoxLayout()
        self.create_controls()
        layout.addLayout(self.controls_layout)

        # FPS display
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_label = QLabel("--")
        fps_layout.addWidget(self.fps_label)
        fps_layout.addStretch()
        layout.addLayout(fps_layout)

        # Recording controls
        rec_layout = QHBoxLayout()
        self.record_btn = QPushButton("⚫ Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        rec_layout.addWidget(self.record_btn)

        self.status_label = QLabel("Status: Ready")
        rec_layout.addWidget(self.status_label)
        rec_layout.addStretch()
        layout.addLayout(rec_layout)

        # Initial camera discovery
        self.refresh_cameras()

    def create_controls(self):
        """Override in subclass to add camera-specific controls."""
        pass

    def refresh_cameras(self):
        """Override in subclass for SDK-specific camera discovery."""
        pass

    def toggle_connect(self):
        """Connect or disconnect camera."""
        if self.camera is None:
            self.connect_camera()
        else:
            self.disconnect_camera()

    def connect_camera(self):
        """Override in subclass for SDK-specific connection."""
        pass

    def disconnect_camera(self):
        """Disconnect camera and clean up."""
        if self.camera is None:
            return

        print("Disconnecting camera...")

        # Stop recording if active
        if self.recording:
            self.toggle_recording()

        # Stop acquisition thread
        self.acq_stop_event.set()
        if self.acq_thread and self.acq_thread.is_alive():
            self.acq_thread.join(timeout=2.0)
        self.acq_thread = None

        # Stop display timer
        self.timer.stop()

        # Dispose camera (subclass implements)
        try:
            self.dispose_camera()
        except Exception as e:
            print(f"Error disposing camera: {e}")

        self.camera = None
        self.camera_id = None
        self.connect_btn.setText("Connect")
        self.record_btn.setEnabled(False)
        self.video_label.setText("No camera connected")
        self.status_label.setText("Status: Disconnected")
        print("Camera disconnected")

    def dispose_camera(self):
        """Override in subclass to dispose SDK resources."""
        pass

    def start_acquisition_thread(self):
        """Start the acquisition thread."""
        self.acq_stop_event.clear()
        self.acq_thread = threading.Thread(
            target=self._acquisition_loop,
            daemon=True,
            name="acquisition")
        self.acq_thread.start()
        print("Acquisition thread started")

    def _acquisition_loop(self):
        """Override in subclass for SDK-specific frame capture."""
        pass

    def start_display_timer(self):
        """Start the display update timer at 30 fps."""
        if not self.timer.isActive():
            self.timer.start(33)  # ~30 fps
            print("Display timer started")

    def update_display(self):
        """Update display with latest frame."""
        with self.frame_lock:
            if not self.new_frame_available or self.pending_frame is None:
                return
            frame = self.pending_frame
            self.new_frame_available = False

        # Get frame dimensions
        try:
            w = frame.image_buffer_size_pixels_horizontal
            h = frame.image_buffer_size_pixels_vertical
        except AttributeError:
            # IDS frame doesn't have these attributes
            w, h = self.get_frame_dimensions(frame)

        # Convert to numpy array - use bit_depth from camera
        bit_depth = self.get_bit_depth()
        raw = frame.image_buffer

        if bit_depth <= 8:
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        else:
            img16 = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
            # Normalize to 8-bit for display
            img = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert to QPixmap
        height, width = img.shape
        bytes_per_line = width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit display
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.last_fps_time
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"{fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = time.time()

    def get_frame_dimensions(self, frame):
        """Override in subclass if frame dimensions need special handling."""
        return 0, 0

    def get_bit_depth(self):
        """Override in subclass to return camera bit depth."""
        return 8

    def toggle_recording(self):
        """Start or stop recording."""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Video",
                f"{self.camera_type.lower()}_{timestamp}.mp4",
                "Video Files (*.mp4)")

            if not filename:
                return

            # Get frame dimensions from current frame
            with self.frame_lock:
                if self.pending_frame is None:
                    self.status_label.setText("Status: No frame to record")
                    return
                try:
                    w = self.pending_frame.image_buffer_size_pixels_horizontal
                    h = self.pending_frame.image_buffer_size_pixels_vertical
                except AttributeError:
                    w, h = self.get_frame_dimensions(self.pending_frame)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h), False)

            if not self.video_writer.isOpened():
                self.status_label.setText("Status: Failed to create video file")
                self.video_writer = None
                return

            self.recording = True
            self.recorded_frame_count = 0
            self.recording_start_time = time.time()
            self._recording_filename = filename       # for the metadata sidecar at stop
            self._recording_dims = (w, h)
            self.record_btn.setText("⬛ Stop")
            self.status_label.setText("Status: Recording...")
            print(f"Recording to {filename}")
        else:
            # Stop recording
            self.recording = False
            with self.video_lock:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

            stop_time = time.time()
            duration = stop_time - self.recording_start_time
            self.record_btn.setText("⚫ Record")
            self.status_label.setText(f"Status: Saved {self.recorded_frame_count} frames ({duration:.1f}s)")
            print(f"Recording stopped: {self.recorded_frame_count} frames in {duration:.1f}s")

            # Best-effort: write the true measured rate to a JSON sidecar so analysis
            # can recover the framerate (the mp4 container fps is a fixed nominal).
            # write_sidecar never raises, so this cannot affect recording/streaming.
            _dims = getattr(self, "_recording_dims", (None, None))
            write_sidecar(
                getattr(self, "_recording_filename", None),
                self.recorded_frame_count, self.recording_start_time, stop_time,
                camera=getattr(self, "camera_type", None),
                width=_dims[0], height=_dims[1], nominal_fps=20.0,
            )

    def closeEvent(self, event):
        """Clean up on window close."""
        print("Application closing...")
        self.disconnect_camera()
        event.accept()
