#!/usr/bin/env python3

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import traceback
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                            QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                            QSpinBox, QDoubleSpinBox, QFileDialog, QGroupBox,
                            QComboBox, QTabWidget, QMessageBox, QCheckBox,
                            QTableWidget, QTableWidgetItem,
                            QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Import Thorlabs SDK
try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
    print("Successfully imported Thorlabs SDK modules")
except ImportError as e:
    print(f"Error: Thorlabs TSI SDK not found: {e}")
    print("Please install it using:")
    print("pip install thorlabs_tsi_sdk")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error importing Thorlabs SDK: {e}")
    traceback.print_exc()
    sys.exit(1)

class CameraLabel(QLabel):
    """QLabel subclass that supports click-and-drag repositioning of markup overlays."""

    def __init__(self, cam_id, app, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.app = app
        self._drag_idx = -1
        self.setMouseTracking(True)

    @property
    def _cam(self):
        return self.app.cameras[self.cam_id]

    def _scale_info(self):
        """Return (scale, x_offset, y_offset) mapping image pixels → label pixels."""
        cam = self._cam
        iw = getattr(cam, 'image_width', None)
        ih = getattr(cam, 'image_height', None)
        if not iw or not ih:
            return None
        lw, lh = self.width(), self.height()
        scale = min(lw / iw, lh / ih)
        x_off = (lw - iw * scale) / 2
        y_off = (lh - ih * scale) / 2
        return scale, x_off, y_off

    def _to_image(self, lx, ly):
        """Convert label pixel coords to image pixel coords."""
        si = self._scale_info()
        if si is None:
            return None, None
        scale, x_off, y_off = si
        return round((lx - x_off) / scale), round((ly - y_off) / scale)

    def _find_overlay(self, lx, ly, threshold=10):
        """Return index of the closest overlay within *threshold* label-pixels, or -1."""
        si = self._scale_info()
        if si is None:
            return -1
        scale, x_off, y_off = si
        best, best_dist = -1, threshold
        for i, ov in enumerate(self._cam.overlays):
            if ov['type'] == 'hline':
                d = abs(ly - (ov['pos'] * scale + y_off))
            elif ov['type'] == 'vline':
                d = abs(lx - (ov['pos'] * scale + x_off))
            elif ov['type'] == 'circle':
                cx_l = ov['center'][0] * scale + x_off
                cy_l = ov['center'][1] * scale + y_off
                r_l  = ov['radius'] * scale
                d = abs(((lx - cx_l) ** 2 + (ly - cy_l) ** 2) ** 0.5 - r_l)
            else:
                continue
            if d < best_dist:
                best_dist, best = d, i
        return best

    def _set_hover_cursor(self, lx, ly):
        idx = self._find_overlay(lx, ly)
        if idx < 0:
            self.setCursor(Qt.CrossCursor)
            return
        ov = self._cam.overlays[idx]
        if ov['type'] == 'hline':
            self.setCursor(Qt.SizeVerCursor)
        elif ov['type'] == 'vline':
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_idx = self._find_overlay(event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_idx >= 0 and (event.buttons() & Qt.LeftButton):
            ix, iy = self._to_image(event.x(), event.y())
            if ix is None:
                return
            cam = self._cam
            ov = cam.overlays[self._drag_idx]
            if ov['type'] == 'hline':
                ov['pos'] = max(0, iy)
            elif ov['type'] == 'vline':
                ov['pos'] = max(0, ix)
            elif ov['type'] == 'circle':
                ov['center'] = (max(0, ix), max(0, iy))
            self.app.sync_overlay_to_table_row(cam, self._drag_idx)
        else:
            self._set_hover_cursor(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_idx = -1
        super().mouseReleaseEvent(event)


# BGR color tuples for markup overlays
MARKUP_COLORS = {
    "White":   (255, 255, 255),
    "Red":     (0,   0,   255),
    "Green":   (0,   255, 0),
    "Blue":    (255, 0,   0),
    "Yellow":  (0,   255, 255),
    "Cyan":    (255, 255, 0),
    "Magenta": (255, 0,   255),
}

class CameraInstance:
    """Class to store state and controls for each camera"""
    def __init__(self, name="Camera"):
        self.camera = None
        self.camera_id = ""
        self.usb_port = ""
        self.name = name
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 30
        self.exposure_ms = 10.0
        self.record_duration_limit = 0
        self.record_frame_limit = 0
        self.recorded_frame_count = 0
        self.recording_start_time = 0
        self.last_frame = None
        self.image_width = None   # set from live frames; used for drag coordinate mapping
        self.image_height = None
        # Markup overlays: list of dicts with keys 'type', 'pos'/'center'/'radius', 'color'
        self.overlays = []
        # Add locks for thread safety
        self.camera_lock = threading.Lock()

class ThorlabsCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sdk = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.sdk_lock = threading.Lock()  # Add lock for SDK access
        
        # Create two camera instances
        self.cameras = {
            "cam1": CameraInstance("Camera 1"),
            "cam2": CameraInstance("Camera 2")
        }
        
        # Flag to track if we're currently refreshing cameras
        self.refreshing_cameras = False
        
        self.init_ui()
        # Delay SDK initialization to prevent segfaults during startup
        QTimer.singleShot(500, self.init_sdk)
        
    def init_sdk(self):
        """Initialize the SDK and discover available cameras"""
        # Make sure any previous SDK instance is disposed
        with self.sdk_lock:
            if self.sdk is not None:
                try:
                    print("Disposing of existing SDK instance")
                    self.sdk.dispose()
                    self.sdk = None
                    # Small delay to ensure SDK is fully disposed
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error disposing SDK: {e}")
                    traceback.print_exc()
        
        try:
            print("Initializing SDK...")
            with self.sdk_lock:
                self.sdk = TLCameraSDK()
                print(f"SDK initialized: {self.sdk}")
                
                # Add a small delay before discovering cameras to prevent race conditions
                time.sleep(0.2)
                
                available_cameras = self.sdk.discover_available_cameras()
                # Filter out invalid camera IDs to prevent segfaults
                valid_cameras = []
                for cam_id in available_cameras:
                    # Check if ID is valid for processing
                    if self.is_valid_camera_id(cam_id):
                        valid_cameras.append(cam_id)
                    else:
                        print(f"Skipping invalid camera ID: {repr(cam_id)}")
                
                # Check if we have multiple cameras and display a warning
                if len(valid_cameras) > 1:
                    print("NOTICE: Multiple cameras detected. Some Thorlabs camera models may")
                    print("not support simultaneous operation. If you encounter issues, try")
                    print("disconnecting one camera before connecting another.")
                
                print(f"Valid cameras: {[repr(c) for c in valid_cameras]}")
                
                # Store camera count for later use
                self.available_camera_count = len(valid_cameras)
            
            if not valid_cameras:
                print("No valid cameras found during discovery")
                self.show_error("No cameras found!", "Make sure cameras are connected and powered on.")
                self.statusBar().showMessage("No cameras found!")
                return
            
            # Populate the camera selection dropdown
            self.camera_selector.clear()
            
            for cam_id in valid_cameras:
                try:
                    # Safely handle camera info retrieval
                    with self.sdk_lock:
                        # Open camera briefly to get info - use repr for safer printing
                        print(f"Getting info for camera {repr(cam_id)}")
                        temp_camera = self.sdk.open_camera(cam_id)
                        
                        # Safely get camera name and port info, handle potential encoding issues
                        try:
                            camera_name = str(temp_camera.name)
                            print(f"Camera name: {camera_name}")
                        except (UnicodeDecodeError, AttributeError):
                            # Fallback: use part of ID as name
                            try:
                                if isinstance(cam_id, str):
                                    camera_name = f"Camera {cam_id[-6:]}"
                                else:
                                    camera_name = f"Camera {cam_id}"
                            except:
                                camera_name = "Unknown Camera"
                            print(f"Using fallback name: {camera_name}")
                            
                        usb_port = self.get_camera_usb_port(temp_camera)
                        print(f"USB port: {usb_port}")
                        
                        display_text = f"{camera_name} ({usb_port})"
                        # Store camera ID with USB port info for later use
                        self.camera_selector.addItem(display_text, (cam_id, usb_port))
                        
                        temp_camera.dispose()
                        print(f"Successfully added camera to selector: {display_text}")
                        
                        # Add a small delay between camera operations
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error getting info for camera {repr(cam_id)}: {e}")
                    traceback.print_exc()
                    # Still try to add the camera with minimal information
                    try:
                        id_part = str(cam_id)[-6:] if isinstance(cam_id, str) and len(str(cam_id)) >= 6 else "unknown"
                        self.camera_selector.addItem(f"Camera {id_part}", (cam_id, "Unknown"))
                    except Exception as e2:
                        print(f"Failed to add camera to selector: {e2}")
            
            msg = f"Found {len(valid_cameras)} valid camera(s)"
            print(msg)
            self.statusBar().showMessage(msg)
            
        except Exception as e:
            print(f"SDK Initialization Error: {e}")
            traceback.print_exc()
            self.show_error("SDK Initialization Error", str(e))
            self.statusBar().showMessage(f"Error initializing SDK: {str(e)}")
    
    def is_valid_camera_id(self, camera_id):
        """Check if the camera ID is valid and safe to use"""
        try:
            # Check if camera_id is a string and can be safely printed
            if not isinstance(camera_id, str):
                print(f"Non-string camera ID: {repr(camera_id)}")
                return False
                
            # Check if the string contains only printable characters
            if not all(c.isprintable() for c in camera_id):
                print(f"Camera ID contains non-printable characters: {repr(camera_id)}")
                return False
                
            # Check if the string is empty
            if not camera_id.strip():
                print("Empty camera ID")
                return False
                
            # If it's a single control character, it's likely invalid
            if len(camera_id) == 1 and ord(camera_id) < 32:
                print(f"Camera ID is a control character: {repr(camera_id)}")
                return False
                
            return True
        except Exception as e:
            print(f"Error validating camera ID: {e}")
            return False
            
    def get_camera_usb_port(self, camera):
        """Attempts to get USB port information for the camera"""
        try:
            # Try several approaches to get meaningful camera identifier info
            if hasattr(camera, "usb_port") and camera.usb_port:
                return str(camera.usb_port)
            elif hasattr(camera, "serial_number") and camera.serial_number:
                return str(camera.serial_number)
            elif hasattr(camera, "model") and camera.model:
                return str(camera.model)
            else:
                # Create a unique identifier from the camera id
                try:
                    camera_id = str(camera.camera_id) if hasattr(camera, "camera_id") else "Unknown"
                    # Make sure we only use printable characters
                    camera_id = ''.join(c for c in camera_id if c.isprintable())
                    return f"ID-{camera_id[-6:] if len(camera_id) >= 6 else camera_id}"
                except:
                    return "Unknown ID"
        except Exception as e:
            print(f"Error getting camera USB port: {e}")
            return "Unknown port"
            
    def init_ui(self):
        # Main window setup
        self.setWindowTitle("Thorlabs Dual Camera Control")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Camera connection controls
        connection_layout = QHBoxLayout()
        main_layout.addLayout(connection_layout)
        
        # Camera selector dropdown
        connection_layout.addWidget(QLabel("Available Cameras:"))
        self.camera_selector = QComboBox()
        connection_layout.addWidget(self.camera_selector)
        
        # Connect buttons for each camera
        self.connect_cam1_btn = QPushButton("Connect to Camera 1")
        self.connect_cam1_btn.clicked.connect(lambda: self.connect_camera("cam1"))
        connection_layout.addWidget(self.connect_cam1_btn)
        
        self.connect_cam2_btn = QPushButton("Connect to Camera 2")
        self.connect_cam2_btn.clicked.connect(lambda: self.connect_camera("cam2"))
        connection_layout.addWidget(self.connect_cam2_btn)
        
        # Refresh camera list button
        self.refresh_btn = QPushButton("Refresh Camera List")
        self.refresh_btn.clicked.connect(self.refresh_camera_list)
        connection_layout.addWidget(self.refresh_btn)
        
        # Debug mode checkbox
        self.debug_checkbox = QCheckBox("Debug Mode")
        connection_layout.addWidget(self.debug_checkbox)
        
        # Camera displays - create a tab widget with a tab for each camera
        self.camera_tabs = QTabWidget()
        main_layout.addWidget(self.camera_tabs)
        
        # Create tabs for each camera
        for cam_id, cam_instance in self.cameras.items():
            cam_widget = QWidget()
            cam_layout = QVBoxLayout()
            cam_widget.setLayout(cam_layout)
            
            # Camera display
            cam_instance.image_label = CameraLabel(cam_id, self)
            cam_instance.image_label.setAlignment(Qt.AlignCenter)
            cam_instance.image_label.setMinimumSize(800, 600)
            cam_instance.image_label.setText(f"Connect to {cam_instance.name} to view feed")
            cam_layout.addWidget(cam_instance.image_label)
            
            # Controls for this camera
            controls_layout = QHBoxLayout()
            
            # --- Exposure Controls ---
            exposure_group = QGroupBox("Exposure Control")
            exposure_layout = QVBoxLayout()
            exposure_group.setLayout(exposure_layout)
            
            exposure_header = QHBoxLayout()
            exposure_layout.addLayout(exposure_header)
            
            exposure_header.addWidget(QLabel("Exposure Time (ms):"))
            cam_instance.exposure_value = QDoubleSpinBox()
            cam_instance.exposure_value.setRange(0.1, 1000.0)
            cam_instance.exposure_value.setValue(10.0)
            cam_instance.exposure_value.setSingleStep(1.0)
            cam_instance.exposure_value.valueChanged.connect(lambda value, c=cam_id: self.set_exposure(c, value))
            exposure_header.addWidget(cam_instance.exposure_value)
            
            cam_instance.exposure_slider = QSlider(Qt.Horizontal)
            cam_instance.exposure_slider.setRange(1, 10000)
            cam_instance.exposure_slider.setValue(100)
            cam_instance.exposure_slider.valueChanged.connect(lambda value, c=cam_id: self.exposure_slider_changed(c, value))
            exposure_layout.addWidget(cam_instance.exposure_slider)
            
            # --- Framerate Controls ---
            framerate_group = QGroupBox("Framerate Control")
            framerate_layout = QVBoxLayout()
            framerate_group.setLayout(framerate_layout)
            
            framerate_header = QHBoxLayout()
            framerate_layout.addLayout(framerate_header)
            
            framerate_header.addWidget(QLabel("Frame Rate (FPS):"))
            cam_instance.framerate_value = QSpinBox()
            cam_instance.framerate_value.setRange(1, 100)
            cam_instance.framerate_value.setValue(30)
            cam_instance.framerate_value.valueChanged.connect(lambda value, c=cam_id: self.set_framerate(c, value))
            framerate_header.addWidget(cam_instance.framerate_value)
            
            cam_instance.framerate_slider = QSlider(Qt.Horizontal)
            cam_instance.framerate_slider.setRange(1, 100)
            cam_instance.framerate_slider.setValue(30)
            cam_instance.framerate_slider.valueChanged.connect(lambda value, c=cam_id: self.framerate_slider_changed(c, value))
            framerate_layout.addWidget(cam_instance.framerate_slider)
            
            # Actual FPS display
            fps_layout = QHBoxLayout()
            fps_layout.addWidget(QLabel("Actual FPS:"))
            cam_instance.fps_label = QLabel("0")
            fps_layout.addWidget(cam_instance.fps_label)
            framerate_layout.addLayout(fps_layout)
            
            # --- Recording Controls ---
            recording_group = QGroupBox("Recording")
            recording_layout = QVBoxLayout()
            recording_group.setLayout(recording_layout)
            
            cam_instance.record_button = QPushButton("Start Recording")
            cam_instance.record_button.clicked.connect(lambda _, c=cam_id: self.toggle_recording(c))
            recording_layout.addWidget(cam_instance.record_button)
            
            cam_instance.recording_label = QLabel("Not Recording")
            recording_layout.addWidget(cam_instance.recording_label)
            
            # Controls for recording limits
            duration_layout = QHBoxLayout()
            duration_layout.addWidget(QLabel("Duration (s):"))
            cam_instance.duration_spinbox = QDoubleSpinBox()
            cam_instance.duration_spinbox.setRange(0, 3600)
            cam_instance.duration_spinbox.setValue(0)
            duration_layout.addWidget(cam_instance.duration_spinbox)
            recording_layout.addLayout(duration_layout)
            
            frames_layout = QHBoxLayout()
            frames_layout.addWidget(QLabel("Frame Count:"))
            cam_instance.framecount_spinbox = QSpinBox()
            cam_instance.framecount_spinbox.setRange(0, 1000000)
            cam_instance.framecount_spinbox.setValue(0)
            frames_layout.addWidget(cam_instance.framecount_spinbox)
            recording_layout.addLayout(frames_layout)
            
            # --- Markup Overlay Controls ---
            markup_group = QGroupBox("Markup Overlays")
            markup_layout = QVBoxLayout()
            markup_group.setLayout(markup_layout)

            # Table: Type | X/CX | Y/CY | Radius | Thickness | Color
            cam_instance.markup_table = QTableWidget(0, 6)
            cam_instance.markup_table.setHorizontalHeaderLabels(
                ["Type", "X / CX", "Y / CY", "Radius", "Thickness", "Color"])
            cam_instance.markup_table.setMaximumHeight(110)
            cam_instance.markup_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            cam_instance.markup_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            cam_instance.markup_table.verticalHeader().setVisible(False)
            cam_instance.markup_table.cellChanged.connect(
                lambda row, col, c=cam_id: self._on_markup_cell_changed(c, row, col))
            markup_layout.addWidget(cam_instance.markup_table)

            color_row = QHBoxLayout()
            color_row.addWidget(QLabel("Color:"))
            cam_instance.markup_color = QComboBox()
            for color_name in MARKUP_COLORS:
                cam_instance.markup_color.addItem(color_name)
            color_row.addWidget(cam_instance.markup_color)
            markup_layout.addLayout(color_row)

            markup_btn_row = QHBoxLayout()
            add_hline_btn = QPushButton("Add H Line")
            add_hline_btn.clicked.connect(lambda _, c=cam_id: self.add_hline_overlay(c))
            markup_btn_row.addWidget(add_hline_btn)
            add_vline_btn = QPushButton("Add V Line")
            add_vline_btn.clicked.connect(lambda _, c=cam_id: self.add_vline_overlay(c))
            markup_btn_row.addWidget(add_vline_btn)
            add_circle_btn = QPushButton("Add Circle")
            add_circle_btn.clicked.connect(lambda _, c=cam_id: self.add_circle_overlay(c))
            markup_btn_row.addWidget(add_circle_btn)
            markup_layout.addLayout(markup_btn_row)

            remove_markup_btn = QPushButton("Remove Selected")
            remove_markup_btn.clicked.connect(lambda _, c=cam_id: self.remove_selected_overlay(c))
            markup_layout.addWidget(remove_markup_btn)

            # Add control groups to main control layout
            controls_layout.addWidget(exposure_group)
            controls_layout.addWidget(framerate_group)
            controls_layout.addWidget(recording_group)
            controls_layout.addWidget(markup_group)
            
            # Add controls to camera layout
            cam_layout.addLayout(controls_layout)
            
            # Status display for this camera
            cam_instance.status_label = QLabel(f"{cam_instance.name} not connected")
            cam_layout.addWidget(cam_instance.status_label)
            
            # Debug info area
            cam_instance.debug_label = QLabel("Debug info: None")
            cam_layout.addWidget(cam_instance.debug_label)
            cam_instance.debug_label.setVisible(False)
            
            # Add the tab for this camera
            self.camera_tabs.addTab(cam_widget, cam_instance.name)
        
        # Status bar for global information
        self.statusBar().showMessage("Ready. Please connect cameras.")
        
        # Connect debug checkbox to update debug visibility
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_mode)
    
    def toggle_debug_mode(self, state):
        """Toggle visibility of debug information"""
        is_visible = state == Qt.Checked
        for cam_id, cam_instance in self.cameras.items():
            cam_instance.debug_label.setVisible(is_visible)
            
    # ------------------------------------------------------------------ #
    # Markup overlay helpers                                               #
    # ------------------------------------------------------------------ #

    def _markup_color(self, cam_instance):
        return MARKUP_COLORS.get(cam_instance.markup_color.currentText(), (255, 255, 255))

    # --- table helpers ---

    def _make_cell(self, text, editable=True):
        item = QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _add_overlay_table_row(self, cam_instance, ov, color_name):
        """Append one row to the markup table for the given overlay dict."""
        table = cam_instance.markup_table
        table.blockSignals(True)
        row = table.rowCount()
        table.insertRow(row)
        labels = {'hline': 'H Line', 'vline': 'V Line', 'circle': 'Circle'}
        table.setItem(row, 0, self._make_cell(labels[ov['type']], editable=False))
        if ov['type'] == 'hline':
            table.setItem(row, 1, self._make_cell("—", editable=False))
            table.setItem(row, 2, self._make_cell(str(ov['pos'])))
            table.setItem(row, 3, self._make_cell("—", editable=False))
        elif ov['type'] == 'vline':
            table.setItem(row, 1, self._make_cell(str(ov['pos'])))
            table.setItem(row, 2, self._make_cell("—", editable=False))
            table.setItem(row, 3, self._make_cell("—", editable=False))
        elif ov['type'] == 'circle':
            table.setItem(row, 1, self._make_cell(str(ov['center'][0])))
            table.setItem(row, 2, self._make_cell(str(ov['center'][1])))
            table.setItem(row, 3, self._make_cell(str(ov['radius'])))
        table.setItem(row, 4, self._make_cell(str(ov.get('thickness', 1))))
        table.blockSignals(False)
        # Color column: always-visible QComboBox; not affected by blockSignals
        color_combo = QComboBox()
        for cname in MARKUP_COLORS:
            color_combo.addItem(cname)
        color_combo.setCurrentText(color_name)
        color_combo.currentTextChanged.connect(
            lambda _, ci=cam_instance, cb=color_combo: self._on_color_combo_changed(ci, cb))
        table.setCellWidget(row, 5, color_combo)

    def _on_color_combo_changed(self, cam_instance, combo):
        """Update overlay color when the user changes the color combobox."""
        table = cam_instance.markup_table
        for row in range(table.rowCount()):
            if table.cellWidget(row, 5) is combo:
                if row < len(cam_instance.overlays):
                    cam_instance.overlays[row]['color'] = MARKUP_COLORS.get(
                        combo.currentText(), (255, 255, 255))
                break

    def sync_overlay_to_table_row(self, cam_instance, row):
        """Push overlay dict values back into the table row (blocks signals to avoid loops)."""
        table = cam_instance.markup_table
        ov = cam_instance.overlays[row]
        table.blockSignals(True)
        if ov['type'] == 'hline':
            table.item(row, 2).setText(str(ov['pos']))
        elif ov['type'] == 'vline':
            table.item(row, 1).setText(str(ov['pos']))
        elif ov['type'] == 'circle':
            table.item(row, 1).setText(str(ov['center'][0]))
            table.item(row, 2).setText(str(ov['center'][1]))
            table.item(row, 3).setText(str(ov['radius']))
        table.item(row, 4).setText(str(ov.get('thickness', 1)))
        table.blockSignals(False)
        combo = table.cellWidget(row, 5)
        if combo:
            combo.blockSignals(True)
            color_name = next((n for n, c in MARKUP_COLORS.items() if c == ov['color']), 'White')
            combo.setCurrentText(color_name)
            combo.blockSignals(False)

    def _on_markup_cell_changed(self, cam_id, row, col):
        """Update overlay dict when the user edits a table cell."""
        cam_instance = self.cameras[cam_id]
        if row >= len(cam_instance.overlays):
            return
        ov = cam_instance.overlays[row]
        item = cam_instance.markup_table.item(row, col)
        if item is None:
            return
        try:
            val = int(item.text())
        except ValueError:
            self.sync_overlay_to_table_row(cam_instance, row)
            return
        if ov['type'] == 'hline' and col == 2:
            ov['pos'] = max(0, val)
        elif ov['type'] == 'vline' and col == 1:
            ov['pos'] = max(0, val)
        elif ov['type'] == 'circle':
            cx, cy = ov['center']
            if col == 1:
                ov['center'] = (max(0, val), cy)
            elif col == 2:
                ov['center'] = (cx, max(0, val))
            elif col == 3:
                ov['radius'] = max(1, val)
        if col == 4:
            ov['thickness'] = max(1, val)
        # Re-sync to clamp any out-of-range values the user may have typed
        self.sync_overlay_to_table_row(cam_instance, row)

    # --- add / remove ---

    def add_hline_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        y = (cam_instance.image_height // 2) if cam_instance.image_height else 0
        color_name = cam_instance.markup_color.currentText()
        ov = {'type': 'hline', 'pos': y, 'color': self._markup_color(cam_instance), 'thickness': 1}
        cam_instance.overlays.append(ov)
        self._add_overlay_table_row(cam_instance, ov, color_name)

    def add_vline_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        x = (cam_instance.image_width // 2) if cam_instance.image_width else 0
        color_name = cam_instance.markup_color.currentText()
        ov = {'type': 'vline', 'pos': x, 'color': self._markup_color(cam_instance), 'thickness': 1}
        cam_instance.overlays.append(ov)
        self._add_overlay_table_row(cam_instance, ov, color_name)

    def add_circle_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        cx = (cam_instance.image_width  // 2) if cam_instance.image_width  else 0
        cy = (cam_instance.image_height // 2) if cam_instance.image_height else 0
        r  = (min(cam_instance.image_width, cam_instance.image_height) // 8
              if cam_instance.image_width and cam_instance.image_height else 50)
        color_name = cam_instance.markup_color.currentText()
        ov = {'type': 'circle', 'center': (cx, cy), 'radius': r,
              'color': self._markup_color(cam_instance), 'thickness': 1}
        cam_instance.overlays.append(ov)
        self._add_overlay_table_row(cam_instance, ov, color_name)

    def remove_selected_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        table = cam_instance.markup_table
        rows = sorted({idx.row() for idx in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)
            cam_instance.overlays.pop(row)

    def apply_overlays(self, cam_instance, image):
        """Draw markup overlays onto a grayscale image.

        Returns a BGR numpy array with overlays drawn, or None if no overlays
        are defined (so the caller can skip the conversion cost).
        """
        if not cam_instance.overlays:
            return None
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = out.shape[:2]
        for ov in cam_instance.overlays:
            color = ov['color']
            t = ov.get('thickness', 1)
            if ov['type'] == 'hline':
                cv2.line(out, (0, ov['pos']), (w - 1, ov['pos']), color, t)
            elif ov['type'] == 'vline':
                cv2.line(out, (ov['pos'], 0), (ov['pos'], h - 1), color, t)
            elif ov['type'] == 'circle':
                cv2.circle(out, ov['center'], ov['radius'], color, t)
        return out

    def refresh_camera_list(self):
        """Safely refresh the camera list"""
        if self.refreshing_cameras:
            print("Camera refresh already in progress")
            return
            
        self.refreshing_cameras = True
        self.refresh_btn.setEnabled(False)
        self.statusBar().showMessage("Refreshing camera list...")
        
        # Dispose current cameras first
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.camera:
                self.disconnect_camera(cam_id)
        
        # Run SDK reinitialization after a brief delay
        QTimer.singleShot(500, self.delayed_refresh)
    
    def delayed_refresh(self):
        """Second part of refresh after cameras are disconnected"""
        try:
            self.init_sdk()
        finally:
            self.refreshing_cameras = False
            self.refresh_btn.setEnabled(True)
    
    def connect_camera(self, cam_id):
        """Connect to selected camera and assign to the specified camera slot"""
        if not self.sdk:
            self.show_error("SDK Not Initialized", "Camera SDK is not initialized.")
            return
        
        if self.camera_selector.count() == 0:
            self.show_error("No Cameras Available", "No cameras were detected. Please check connections and refresh.")
            return
            
        # Get the selected camera ID and USB port
        selected_idx = self.camera_selector.currentIndex()
        if selected_idx < 0:
            self.show_error("No Camera Selected", "Please select a camera from the dropdown.")
            return
            
        camera_info = self.camera_selector.itemData(selected_idx)
        if not camera_info:
            self.show_error("Invalid Camera Selection", "Could not retrieve camera information.")
            return
            
        cam_instance = self.cameras[cam_id]
        device_id, usb_port = camera_info
        
        # Check if another camera instance is already using this camera
        for other_id, other_cam in self.cameras.items():
            if other_id != cam_id and other_cam.camera_id == device_id:
                self.show_error("Camera Already In Use", 
                                f"This camera is already connected as {other_cam.name}.")
                return
        
        # Get count of currently connected cameras
        connected_count = sum(1 for c in self.cameras.values() if c.camera is not None)
        
        # Check if we're trying to connect multiple cameras
        if connected_count > 0:
            # Show warning if connecting second camera
            if getattr(self, 'available_camera_count', 0) > 1:
                msg = "You are connecting to a second camera while another is active.\n\n"
                msg += "Some Thorlabs camera models may not support simultaneous operation.\n"
                msg += "If this fails, try disconnecting the first camera before connecting another."
                
                warning_box = QMessageBox()
                warning_box.setIcon(QMessageBox.Warning)
                warning_box.setWindowTitle("Multiple Camera Warning")
                warning_box.setText(msg)
                warning_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                
                if warning_box.exec_() == QMessageBox.Cancel:
                    return
        
        # Clean up existing camera if there is one
        if cam_instance.camera:
            try:
                with cam_instance.camera_lock:
                    cam_instance.camera.disarm()
                    cam_instance.camera.dispose()
                    # Add delay after disposing camera
                    time.sleep(0.2)
            except Exception as e:
                error_msg = f"Error disconnecting from {cam_instance.name}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                cam_instance.debug_label.setText(f"Error: {error_msg}")
                self.show_error(f"Error disconnecting from {cam_instance.name}", str(e))
        
        # Stop the timer temporarily while connecting to camera
        timer_was_active = self.timer.isActive()
        if timer_was_active:
            self.timer.stop()
            
        # Use a separate try-except block for each major step to provide better error reporting
        try:
            # Connect to the camera
            print(f"Connecting to camera {device_id} for {cam_id}")
            self.statusBar().showMessage(f"Connecting to camera {device_id}...")
            
            with self.sdk_lock:
                # Ensure SDK is still valid
                if not self.sdk:
                    raise RuntimeError("SDK is no longer valid")
                    
                # Add additional delay to ensure system stability
                time.sleep(0.5)  
                
                # Try to connect with retry mechanism
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        cam_instance.camera = self.sdk.open_camera(device_id)
                        cam_instance.camera_id = device_id
                        cam_instance.usb_port = usb_port
                        break  # Successfully connected
                    except Exception as e:
                        last_error = e
                        retry_count += 1
                        print(f"Connection attempt {retry_count} failed: {e}")
                        time.sleep(1.0)  # Wait before retrying
                
                if retry_count == max_retries:
                    raise RuntimeError(f"Failed to connect after {max_retries} attempts: {last_error}")
        
            # Configure camera step
            try:
                # Configure camera with lock to ensure thread safety
                with cam_instance.camera_lock:
                    print(f"Configuring camera {device_id}")
                    cam_instance.camera.frames_per_trigger_zero_for_unlimited = 0  # Continuous acquisition
                    cam_instance.camera.exposure_time_us = int(cam_instance.exposure_ms * 1000)  # Convert ms to μs
                    cam_instance.camera.image_poll_timeout_ms = 1000  # 1 second timeout
            except Exception as e:
                error_msg = f"Error configuring camera: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                # Clean up the camera connection
                with self.sdk_lock:
                    try:
                        if cam_instance.camera:
                            cam_instance.camera.dispose()
                            cam_instance.camera = None
                    except:
                        pass
                raise RuntimeError(error_msg)
            
            # Arm camera step
            try:
                # Start the camera
                with cam_instance.camera_lock:
                    print(f"Arming camera {device_id}")
                    cam_instance.camera.arm(2)  # 2 buffers for frame acquisition
                    time.sleep(0.2)  # Short delay after arming
                    print(f"Triggering camera {device_id}")
                    cam_instance.camera.issue_software_trigger()
            except Exception as e:
                error_msg = f"Error arming camera: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                # Clean up the camera connection
                with self.sdk_lock:
                    try:
                        if cam_instance.camera:
                            cam_instance.camera.dispose()
                            cam_instance.camera = None
                    except:
                        pass
                raise RuntimeError(error_msg)
            
            # Update UI
            if cam_id == "cam1":
                self.connect_cam1_btn.setText(f"Disconnect {cam_instance.name}")
                self.connect_cam1_btn.clicked.disconnect()
                self.connect_cam1_btn.clicked.connect(lambda: self.disconnect_camera("cam1"))
            else:
                self.connect_cam2_btn.setText(f"Disconnect {cam_instance.name}")
                self.connect_cam2_btn.clicked.disconnect()
                self.connect_cam2_btn.clicked.connect(lambda: self.disconnect_camera("cam2"))
            
            cam_instance.status_label.setText(f"Connected to {cam_instance.name} ({usb_port})")
            
            # Update debug info
            try:
                camera_info = f"Model: {cam_instance.camera.model}, "
                camera_info += f"SN: {cam_instance.camera.serial_number}, "
                camera_info += f"Firmware: {cam_instance.camera.firmware_version}"
                cam_instance.debug_label.setText(camera_info)
            except Exception as e:
                cam_instance.debug_label.setText(f"Camera info error: {str(e)}")
            
            # Start the timer if it's not already running
            interval = int(1000 / max(self.cameras["cam1"].fps if self.cameras["cam1"].camera else 1, 
                                     self.cameras["cam2"].fps if self.cameras["cam2"].camera else 1))
            print(f"Starting timer with interval {interval}ms")
            self.timer.start(interval)
            
            msg = f"{cam_instance.name} connected successfully"
            print(msg)
            self.statusBar().showMessage(msg)
            
        except Exception as e:
            error_msg = f"Failed to connect to camera: {str(e)}"
            print(f"Error connecting to {cam_instance.name}: {error_msg}")
            traceback.print_exc()
            cam_instance.debug_label.setText(f"Connection error: {error_msg}")
            self.show_error(f"Error connecting to {cam_instance.name}", error_msg)
            self.statusBar().showMessage(f"Error connecting to {cam_instance.name}: {error_msg}")
            
            # Restore timer if it was active
            if timer_was_active and not self.timer.isActive():
                self.timer.start(int(1000 / max(self.cameras["cam1"].fps if self.cameras["cam1"].camera else 30, 
                                              self.cameras["cam2"].fps if self.cameras["cam2"].camera else 30)))
                
    def safe_camera_operation(self, func, *args, **kwargs):
        """Execute a function with proper locking to ensure thread safety"""
        with self.sdk_lock:
            return func(*args, **kwargs)

    def disconnect_camera(self, cam_id):
        """Disconnect the specified camera"""
        cam_instance = self.cameras[cam_id]
        
        if cam_instance.recording:
            self.toggle_recording(cam_id)  # Stop recording if active
        
        try:
            if cam_instance.camera:
                print(f"Disconnecting camera {cam_id}")
                with cam_instance.camera_lock:
                    cam_instance.camera.disarm()
                    cam_instance.camera.dispose()
                cam_instance.camera = None
                cam_instance.camera_id = ""
                
                # Update UI
                if cam_id == "cam1":
                    self.connect_cam1_btn.setText(f"Connect to {cam_instance.name}")
                    self.connect_cam1_btn.clicked.disconnect()
                    self.connect_cam1_btn.clicked.connect(lambda: self.connect_camera("cam1"))
                else:
                    self.connect_cam2_btn.setText(f"Connect to {cam_instance.name}")
                    self.connect_cam2_btn.clicked.disconnect()
                    self.connect_cam2_btn.clicked.connect(lambda: self.connect_camera("cam2"))
                
                cam_instance.status_label.setText(f"{cam_instance.name} not connected")
                cam_instance.image_label.setText(f"Connect to {cam_instance.name} to view feed")
                cam_instance.image_label.setPixmap(QPixmap())  # Clear the image
                cam_instance.debug_label.setText("Debug info: Disconnected")
                
                msg = f"{cam_instance.name} disconnected"
                print(msg)
                self.statusBar().showMessage(msg)
                
                # Stop the timer if no cameras are connected
                if not self.cameras["cam1"].camera and not self.cameras["cam2"].camera:
                    print("Stopping timer - no cameras connected")
                    self.timer.stop()
                    
        except Exception as e:
            error_msg = f"Error disconnecting {cam_instance.name}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            cam_instance.debug_label.setText(f"Disconnect error: {error_msg}")
            self.show_error(f"Error disconnecting {cam_instance.name}", str(e))
    
    def update_frames(self):
        """Update frames from all connected cameras"""
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.camera:
                self.update_camera_frame(cam_id)
    
    def update_camera_frame(self, cam_id):
        """Update frame for a specific camera"""
        cam_instance = self.cameras[cam_id]
        
        try:
            # Use lock to ensure thread safety
            with cam_instance.camera_lock:
                if not cam_instance.camera:  # Double-check camera is still valid
                    return
                
                # Don't try to get frames from all cameras at exactly the same time
                # Add a tiny delay between cameras to avoid resource conflicts
                if cam_id == "cam2":
                    time.sleep(0.001)  # 1ms stagger between cameras
                    
                # Trigger acquisition of the next frame for live display
                cam_instance.camera.issue_software_trigger()
                # Get frame from camera
                frame = cam_instance.camera.get_pending_frame_or_null()
                
            if frame is None:
                return
            
            # Determine frame dimensions, fallback to camera sensor size if necessary
            try:
                width = frame.image_buffer_size_pixels_horizontal
                height = frame.image_buffer_size_pixels_vertical
            except AttributeError:
                with cam_instance.camera_lock:
                    width = cam_instance.camera.sensor_width_pixels
                    height = cam_instance.camera.sensor_height_pixels
            
            with cam_instance.camera_lock:
                bit_depth = cam_instance.camera.bit_depth
            
            # Convert frame to numpy array
            image_data = frame.image_buffer
            
            # Create numpy array from image data
            if bit_depth <= 8:
                image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width)
            else:
                # For 16-bit images, we need to rescale to 8-bit for display
                image = np.frombuffer(image_data, dtype=np.uint16).reshape(height, width)
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Store the latest frame and dimensions (used for drag coordinate mapping)
            cam_instance.last_frame = image
            cam_instance.image_width = width
            cam_instance.image_height = height
            
            # Apply markup overlays (returns BGR image, or None when no overlays)
            overlay_image = self.apply_overlays(cam_instance, image)

            # Record video if needed
            if cam_instance.recording and cam_instance.video_writer:
                # Write overlaid frame (BGR) or plain grayscale-to-BGR
                if overlay_image is not None:
                    cam_instance.video_writer.write(overlay_image)
                else:
                    cam_instance.video_writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                # Update recording duration and frame count
                duration = time.time() - cam_instance.recording_start_time
                cam_instance.recorded_frame_count += 1
                cam_instance.recording_label.setText(f"Recording: {duration:.1f}s, Frames: {cam_instance.recorded_frame_count}")
                # Auto-stop if limits reached
                if ((cam_instance.record_duration_limit > 0 and duration >= cam_instance.record_duration_limit) or
                    (cam_instance.record_frame_limit > 0 and cam_instance.recorded_frame_count >= cam_instance.record_frame_limit)):
                    self.toggle_recording(cam_id)
                    return

            # Display the image
            if overlay_image is not None:
                display_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(display_image.data, width, height, width * 3, QImage.Format_RGB888)
            else:
                q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            cam_instance.image_label.setPixmap(pixmap.scaled(
                cam_instance.image_label.width(), 
                cam_instance.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            
            # Calculate and show actual FPS
            cam_instance.frame_count += 1
            elapsed = time.time() - cam_instance.last_frame_time
            if elapsed >= 1.0:  # Update FPS display every second
                actual_fps = cam_instance.frame_count / elapsed
                cam_instance.fps_label.setText(f"{actual_fps:.1f}")
                cam_instance.frame_count = 0
                cam_instance.last_frame_time = time.time()
                
        except Exception as e:
            error_msg = f"Error acquiring frame for {cam_instance.name}: {str(e)}"
            self.statusBar().showMessage(error_msg)
            print(error_msg)
            if self.debug_checkbox.isChecked():
                traceback.print_exc()
                cam_instance.debug_label.setText(f"Frame error: {error_msg}")
    
    def exposure_slider_changed(self, cam_id, value):
        """Handle exposure slider change for a specific camera"""
        cam_instance = self.cameras[cam_id]
        # Convert slider value (which is integer) to actual exposure in ms
        exposure_ms = value / 10.0
        cam_instance.exposure_value.blockSignals(True)
        cam_instance.exposure_value.setValue(exposure_ms)
        cam_instance.exposure_value.blockSignals(False)
        self.set_exposure(cam_id, exposure_ms)
    
    def set_exposure(self, cam_id, value_ms):
        """Set exposure for a specific camera"""
        cam_instance = self.cameras[cam_id]
        cam_instance.exposure_ms = value_ms
        
        if cam_instance.camera:
            try:
                # Convert from ms to μs for the camera
                with cam_instance.camera_lock:
                    cam_instance.camera.exposure_time_us = int(value_ms * 1000)
                cam_instance.status_label.setText(f"Exposure set to {value_ms} ms")
                # Update slider if value was changed directly
                slider_value = int(value_ms * 10)
                if cam_instance.exposure_slider.value() != slider_value:
                    cam_instance.exposure_slider.blockSignals(True)
                    cam_instance.exposure_slider.setValue(slider_value)
                    cam_instance.exposure_slider.blockSignals(False)
            except Exception as e:
                error_msg = f"Error setting exposure: {str(e)}"
                print(error_msg)
                if self.debug_checkbox.isChecked():
                    traceback.print_exc()
                    cam_instance.debug_label.setText(f"Exposure error: {error_msg}")
    
    def framerate_slider_changed(self, cam_id, value):
        """Handle framerate slider change for a specific camera"""
        cam_instance = self.cameras[cam_id]
        fps = value
        cam_instance.framerate_value.blockSignals(True)
        cam_instance.framerate_value.setValue(fps)
        cam_instance.framerate_value.blockSignals(False)
        self.set_framerate(cam_id, fps)
    
    def set_framerate(self, cam_id, fps):
        """Set framerate for a specific camera"""
        cam_instance = self.cameras[cam_id]
        cam_instance.fps = fps
        
        # Update the timer to use the faster of the two cameras' framerates
        if self.cameras["cam1"].camera or self.cameras["cam2"].camera:
            max_fps = max(
                self.cameras["cam1"].fps if self.cameras["cam1"].camera else 0,
                self.cameras["cam2"].fps if self.cameras["cam2"].camera else 0
            )
            if max_fps > 0:
                if self.timer.isActive():
                    self.timer.stop()
                self.timer.start(int(1000 / max_fps))
        
        cam_instance.status_label.setText(f"Frame rate set to {fps} FPS")
        # Update slider if value was changed directly
        if cam_instance.framerate_slider.value() != fps:
            cam_instance.framerate_slider.blockSignals(True)
            cam_instance.framerate_slider.setValue(fps)
            cam_instance.framerate_slider.blockSignals(False)
    
    def toggle_recording(self, cam_id):
        """Toggle recording for a specific camera"""
        cam_instance = self.cameras[cam_id]
        
        if not cam_instance.camera:
            self.show_error("Camera Not Connected", f"{cam_instance.name} is not connected.")
            return
            
        if not cam_instance.recording:
            # Start recording
            try:
                filename, _ = QFileDialog.getSaveFileName(
                    self, f"Save Video - {cam_instance.name}", 
                    f"{cam_instance.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    "Video Files (*.mp4)"
                )
                if filename:
                    # Use sensor dimensions for resolution to avoid missing frame attributes
                    with cam_instance.camera_lock:
                        width = cam_instance.camera.sensor_width_pixels
                        height = cam_instance.camera.sensor_height_pixels
                    
                    # Initialize video writer with MJPG codec which has good compatibility with MP4
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    # Ensure filename has .mp4 extension
                    if not filename.lower().endswith('.mp4'):
                        filename = filename + '.mp4'
                    
                    cam_instance.video_writer = cv2.VideoWriter(
                        filename, fourcc, cam_instance.fps, (width, height)
                    )
                    
                    if cam_instance.video_writer.isOpened():
                        cam_instance.recording = True
                        cam_instance.recording_start_time = time.time()
                        # Initialize recording limits
                        cam_instance.record_duration_limit = cam_instance.duration_spinbox.value()
                        cam_instance.record_frame_limit = cam_instance.framecount_spinbox.value()
                        cam_instance.recorded_frame_count = 0
                        cam_instance.record_button.setText("Stop Recording")
                        cam_instance.recording_label.setText("Recording started")
                        cam_instance.status_label.setText(f"Recording to {os.path.basename(filename)}")
                    else:
                        self.show_error("Recording Error", f"Failed to create video writer for {cam_instance.name}")
                        print(f"Failed to open video writer for {filename}")
            except Exception as e:
                error_msg = f"Recording error for {cam_instance.name}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                cam_instance.debug_label.setText(f"Recording error: {error_msg}")
                self.show_error(f"Recording Error - {cam_instance.name}", str(e))
                cam_instance.video_writer = None
        else:
            # Stop recording
            if cam_instance.video_writer:
                try:
                    cam_instance.video_writer.release()
                except Exception as e:
                    print(f"Error releasing video writer: {e}")
                cam_instance.video_writer = None
            
            cam_instance.recording = False
            cam_instance.record_button.setText("Start Recording")
            cam_instance.recording_label.setText("Not Recording")
            cam_instance.status_label.setText(f"Recording stopped - {cam_instance.name}")
    
    def show_error(self, title, message):
        """Display an error dialog with the given title and message"""
        print(f"ERROR: {title} - {message}")
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()
    
    def closeEvent(self, event):
        # Cleanup when application is closed
        print("Application closing, cleaning up...")
        
        # Stop the timer first
        if self.timer.isActive():
            self.timer.stop()
        
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.recording and cam_instance.video_writer:
                try:
                    print(f"Releasing video writer for {cam_id}")
                    cam_instance.video_writer.release()
                except Exception as e:
                    print(f"Error releasing video writer for {cam_id}: {e}")
            
            if cam_instance.camera:
                try:
                    print(f"Disposing camera {cam_id}")
                    cam_instance.camera.disarm()
                    cam_instance.camera.dispose()
                except Exception as e:
                    print(f"Error disposing camera {cam_id}: {e}")
        
        # Add a small delay to ensure cameras are properly disposed
        time.sleep(0.5)
        
        if self.sdk:
            try:
                print("Disposing SDK")
                self.sdk.dispose()
            except Exception as e:
                print(f"Error disposing SDK: {e}")
        
        print("Cleanup complete")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThorlabsCameraApp()
    window.show()
    sys.exit(app.exec_())