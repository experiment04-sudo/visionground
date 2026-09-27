"""
Main GUI Application Module
---------------------------
Production-grade PyQt6 interface with real-time 30 FPS video rendering pipeline,
aspect-ratio-preserved spatial grounding overlays, non-blocking inference,
and camera permission troubleshooting dialogs.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from PyQt6.QtCore import (
    Qt, QTimer, QThreadPool, QRect, QPoint, QSize
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, QIcon
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog,
    QMessageBox, QFrame, QSizePolicy, QStatusBar, QDialog, QTextEdit
)

from config import CONFIG
from camera_manager import CameraThread
from vision_engine import VisionInferenceWorker, DetectionBox

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s"
)
logger = logging.getLogger("MainWindow")


class VideoCanvasWidget(QWidget):
    """
    High-performance custom video rendering canvas.
    Handles aspect-ratio-preserving letterbox/pillarbox projection
    and overlays spatial grounding bounding boxes directly in pixel space.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #0b0f19; border-radius: 8px;")

        self._current_frame: Optional[np.ndarray] = None
        self._detections: List[DetectionBox] = []
        self._is_disconnected: bool = False
        self._is_analyzing: bool = False
        self._status_overlay: str = "Waiting for camera stream..."

    def update_frame(self, frame_bgr: np.ndarray):
        """Update active video frame and trigger a repaint."""
        self._current_frame = frame_bgr
        self._is_disconnected = False
        self.update()

    def set_detections(self, detections: List[DetectionBox]):
        """Update active object detection boxes."""
        self._detections = detections
        self.update()

    def clear_detections(self):
        """Clear all active bounding boxes."""
        self._detections = []
        self.update()

    def set_disconnected(self, disconnected: bool, message: str = "Camera Disconnected"):
        """Display disconnected warning overlay."""
        self._is_disconnected = disconnected
        self._status_overlay = message
        self.update()

    def set_analyzing(self, analyzing: bool):
        """Toggle analyzing pulse indicator."""
        self._is_analyzing = analyzing
        self.update()

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """
        Renders the current frame with all active bounding boxes and labels
        burned directly onto the full-resolution image matrix (for snapshot export).
        """
        if self._current_frame is None:
            return None

        # Work on a copy of the high-res frame
        annotated = self._current_frame.copy()
        img_h, img_w = annotated.shape[:2]

        for det in self._detections:
            # Convert 0-1000 coordinates to full-resolution frame pixels
            norm = CONFIG.COORD_SCALE
            ymin = int((det.ymin / norm) * img_h)
            xmin = int((det.xmin / norm) * img_w)
            ymax = int((det.ymax / norm) * img_h)
            xmax = int((det.xmax / norm) * img_w)

            # Draw rectangle (BGR)
            bgr_color = (det.color_rgb[2], det.color_rgb[1], det.color_rgb[0])
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), bgr_color, 2)

            # Label text
            label_text = det.label
            if det.confidence is not None:
                label_text += f" {int(det.confidence * 100)}%"

            # Calculate text size for badge background
            font_scale = 0.6
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw badge background
            badge_ymin = max(0, ymin - text_h - 10)
            badge_ymax = ymin
            cv2.rectangle(
                annotated,
                (xmin, badge_ymin),
                (xmin + text_w + 12, badge_ymax),
                bgr_color,
                -1
            )
            # Draw contrast text
            cv2.putText(
                annotated,
                label_text,
                (xmin + 6, badge_ymax - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return annotated

    def paintEvent(self, event):
        """PyQt paint event handling aspect ratio letterboxing and bounding box drawing."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        widget_w = self.width()
        widget_h = self.height()

        # Fill widget canvas background
        painter.fillRect(0, 0, widget_w, widget_h, QColor("#090d16"))

        if self._current_frame is None or self._is_disconnected:
            # Render empty / disconnected status screen
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont(CONFIG.LABEL_FONT_FAMILY, 14, QFont.Weight.Medium))
            rect = QRect(0, 0, widget_w, widget_h)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                self._status_overlay if self._is_disconnected else "Camera stream offline. Connect device to start."
            )
            return

        # ----------------------------------------------------------------------
        # Edge Case 2: Coordinate Normalization & Aspect Ratio Projection
        # ----------------------------------------------------------------------
        frame_h, frame_w = self._current_frame.shape[:2]
        
        # Calculate optimal scale to fit without distortion
        scale = min(widget_w / frame_w, widget_h / frame_h)
        rendered_w = int(frame_w * scale)
        rendered_h = int(frame_h * scale)
        
        # Centering offsets (Letterboxing / Pillarboxing)
        offset_x = (widget_w - rendered_w) // 2
        offset_y = (widget_h - rendered_h) // 2

        # Convert OpenCV BGR to RGB QImage
        rgb_frame = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * frame_w
        qimg = QImage(rgb_frame.data, frame_w, frame_h, bytes_per_line, QImage.Format.Format_RGB888)

        # Draw video frame onto canvas
        target_rect = QRect(offset_x, offset_y, rendered_w, rendered_h)
        painter.drawImage(target_rect, qimg)

        # ----------------------------------------------------------------------
        # Overlay Spatial Grounding Bounding Boxes
        # ----------------------------------------------------------------------
        font = QFont(CONFIG.LABEL_FONT_FAMILY, CONFIG.LABEL_FONT_SIZE, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        for det in self._detections:
            bx, by, bw, bh = det.to_pixel_box(rendered_w, rendered_h, offset_x, offset_y)
            r, g, b = det.color_rgb

            # 1. Semi-transparent box interior fill
            fill_color = QColor(r, g, b, CONFIG.BOX_FILL_ALPHA)
            painter.setBrush(QBrush(fill_color))
            
            # 2. Solid bounding border
            pen = QPen(QColor(r, g, b), CONFIG.BOX_BORDER_WIDTH)
            painter.setPen(pen)
            painter.drawRoundedRect(bx, by, bw, bh, CONFIG.LABEL_CORNER_RADIUS, CONFIG.LABEL_CORNER_RADIUS)

            # 3. Label Badge Calculation
            label_str = det.label
            if det.confidence is not None:
                label_str += f" {int(det.confidence * 100)}%"

            text_width = fm.horizontalAdvance(label_str)
            text_height = fm.height()
            badge_w = text_width + (CONFIG.LABEL_PADDING_H * 2)
            badge_h = text_height + (CONFIG.LABEL_PADDING_V * 2)

            # Position badge above box; if too close to top edge, flip inside
            badge_x = bx
            badge_y = by - badge_h - 2
            if badge_y < offset_y:
                badge_y = by + 2

            # Draw badge background
            badge_rect = QRect(badge_x, badge_y, badge_w, badge_h)
            painter.setBrush(QBrush(QColor(r, g, b)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_rect, CONFIG.LABEL_CORNER_RADIUS, CONFIG.LABEL_CORNER_RADIUS)

            # Draw high-contrast text (dark text on bright saturated badges)
            painter.setPen(QColor("#000000"))
            text_rect = QRect(
                badge_x + CONFIG.LABEL_PADDING_H,
                badge_y + CONFIG.LABEL_PADDING_V,
                text_width,
                text_height
            )
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, label_str)

        # 4. Analyzing status badge indicator
        if self._is_analyzing:
            badge_rect = QRect(widget_w - 180, 16, 160, 32)
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            painter.setPen(QPen(QColor(0, 217, 255), 1))
            painter.drawRoundedRect(badge_rect, 16, 16)

            painter.setPen(QColor(0, 217, 255))
            painter.setFont(QFont(CONFIG.LABEL_FONT_FAMILY, 10, QFont.Weight.Bold))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "⚡ Gemini Analyzing...")


class PermissionTroubleshootDialog(QDialog):
    """Explicit modal dialog providing troubleshooting steps for camera access failures."""

    def __init__(self, title: str, error_details: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(520, 380)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTextEdit {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QLabel("Camera Permission or Hardware Issue")
        header.setFont(QFont(CONFIG.LABEL_FONT_FAMILY, 15, QFont.Weight.Bold))
        header.setStyleSheet("color: #ef4444;")
        layout.addWidget(header)

        desc = QLabel(
            "The application could not access your webcam device. "
            "Please follow the operating system instructions below to resolve:"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(error_details)
        layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("Dismiss")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)


class MainWindow(QMainWindow):
    """Main Application Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionGround - Real-Time Gemini Spatial Grounding")
        self.setMinimumSize(1080, 720)
        self._apply_dark_theme()

        # State variables
        self._camera_thread: Optional[CameraThread] = None
        self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(4)
        
        self._is_inference_running = False
        self._continuous_active = False
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self._latest_raw_frame: Optional[np.ndarray] = None

        # Build UI
        self._init_ui()

        # Timers
        self._continuous_timer = QTimer(self)
        self._continuous_timer.setInterval(CONFIG.CONTINUOUS_INFERENCE_INTERVAL_MS)
        self._continuous_timer.timeout.connect(self._on_continuous_timer_tick)

        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._calculate_fps)
        self._fps_timer.start()

        # Start Camera Thread
        self._start_camera(CONFIG.DEFAULT_CAMERA_INDEX)

    def _apply_dark_theme(self):
        """Applies refined, high-contrast dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f19;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QFrame#controlPanel {
                background-color: #111827;
                border-top: 1px solid #1f2937;
                padding: 12px 18px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton#detectBtn {
                background-color: #0284c7;
                border-color: #38bdf8;
                color: #ffffff;
            }
            QPushButton#detectBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#detectBtn:disabled {
                background-color: #1e293b;
                border-color: #334155;
                color: #64748b;
            }
            QCheckBox {
                font-size: 13px;
                font-weight: 600;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #475569;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #10b981;
                border-color: #34d399;
            }
            QComboBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                min-width: 140px;
            }
            QStatusBar {
                background-color: #080c14;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
            }
        """)

    def _init_ui(self):
        """Constructs layout and widgets."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Bar
        header_layout = QHBoxLayout()
        title_label = QLabel("VisionGround")
        title_label.setFont(QFont(CONFIG.LABEL_FONT_FAMILY, 16, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #38bdf8;")
        header_layout.addWidget(title_label)

        sub_label = QLabel("Real-Time Spatial Grounding with Gemini 2.0 Flash")
        sub_label.setFont(QFont(CONFIG.LABEL_FONT_FAMILY, 12))
        sub_label.setStyleSheet("color: #64748b; margin-left: 8px;")
        header_layout.addWidget(sub_label)
        header_layout.addStretch()

        # Device selector
        header_layout.addWidget(QLabel("Camera Device:"))
        self.camera_selector = QComboBox()
        self.camera_selector.addItem(f"Camera {CONFIG.DEFAULT_CAMERA_INDEX}", CONFIG.DEFAULT_CAMERA_INDEX)
        self.camera_selector.currentIndexChanged.connect(self._on_camera_selected)
        header_layout.addWidget(self.camera_selector)

        btn_rescan = QPushButton("↻ Scan")
        btn_rescan.setToolTip("Rescan available video devices")
        btn_rescan.clicked.connect(self._rescan_cameras)
        header_layout.addWidget(btn_rescan)

        main_layout.addLayout(header_layout)

        # 2. Main Video Canvas
        self.video_canvas = VideoCanvasWidget(self)
        main_layout.addWidget(self.video_canvas, stretch=1)

        # 3. Control Action Bar
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(14)

        # Detect Objects Button
        self.btn_detect = QPushButton("⚡ Detect Objects")
        self.btn_detect.setObjectName("detectBtn")
        self.btn_detect.setToolTip("Trigger one-shot spatial object detection")
        self.btn_detect.clicked.connect(self._trigger_manual_detection)
        control_layout.addWidget(self.btn_detect)

        # Continuous Analysis Toggle
        self.chk_continuous = QCheckBox("Continuous Stream Analysis (1.5s)")
        self.chk_continuous.setToolTip("Automatically run vision grounding every 1.5 seconds")
        self.chk_continuous.stateChanged.connect(self._toggle_continuous_analysis)
        control_layout.addWidget(self.chk_continuous)

        control_layout.addStretch()

        # Clear Detections
        self.btn_clear = QPushButton("Clear Boxes")
        self.btn_clear.clicked.connect(self.video_canvas.clear_detections)
        control_layout.addWidget(self.btn_clear)

        # Snapshot & Export
        self.btn_snapshot = QPushButton("📷 Snapshot & Export")
        self.btn_snapshot.setToolTip("Save the current frame with all bounding boxes to disk as PNG")
        self.btn_snapshot.clicked.connect(self._export_snapshot)
        control_layout.addWidget(self.btn_snapshot)

        main_layout.addWidget(control_panel)

        # 4. Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.fps_badge = QLabel("FPS: 0.0")
        self.fps_badge.setStyleSheet("color: #38bdf8; font-weight: bold; margin-right: 16px;")
        self.status_bar.addPermanentWidget(self.fps_badge)

        self.latency_badge = QLabel("Latency: --")
        self.latency_badge.setStyleSheet("color: #34d399; font-weight: bold; margin-right: 16px;")
        self.status_bar.addPermanentWidget(self.latency_badge)

        self.objects_badge = QLabel("Objects: 0")
        self.objects_badge.setStyleSheet("color: #f59e0b; font-weight: bold; margin-right: 12px;")
        self.status_bar.addPermanentWidget(self.objects_badge)

        self.status_bar.showMessage("Ready. Select device or trigger detection.")

    def _start_camera(self, camera_index: int):
        """Spawns and connects camera capture thread."""
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread = None

        self._camera_thread = CameraThread(camera_index, parent=self)
        self._camera_thread.frame_received.connect(self._on_frame_received)
        self._camera_thread.camera_error.connect(self._on_camera_error)
        self._camera_thread.camera_disconnected.connect(self._on_camera_disconnected)
        self._camera_thread.status_changed.connect(self._on_camera_status_changed)
        self._camera_thread.start()

    def _on_frame_received(self, frame: np.ndarray):
        """Slot for incoming 30 FPS video frames."""
        self._frame_count += 1
        self._latest_raw_frame = frame
        self.video_canvas.update_frame(frame)

    def _on_camera_error(self, title: str, message: str, is_permission_issue: bool):
        """Handle initial camera access failures with troubleshooting modal."""
        self.video_canvas.set_disconnected(True, "Camera Offline")
        self.status_bar.showMessage(f"Error: {title}")
        if is_permission_issue:
            dialog = PermissionTroubleshootDialog(title, message, self)
            dialog.exec()
        else:
            QMessageBox.critical(self, title, message)

    def _on_camera_disconnected(self, reason: str):
        """Edge Case 4: Graceful mid-stream disconnection handler."""
        logger.warning(f"Handling camera disconnection: {reason}")
        self.video_canvas.set_disconnected(True, "Camera Disconnected")
        self.status_bar.showMessage("Camera was disconnected. Please check physical connection or switch device.")

    def _on_camera_status_changed(self, connected: bool, text: str):
        """Updates status indicators when connection status changes."""
        self.status_bar.showMessage(text)

    def _calculate_fps(self):
        """Updates current display FPS."""
        now = time.time()
        elapsed = now - self._fps_start_time
        if elapsed > 0:
            self._current_fps = self._frame_count / elapsed
            self.fps_badge.setText(f"FPS: {self._current_fps:.1f}")
        self._frame_count = 0
        self._fps_start_time = now

    def _on_camera_selected(self, index: int):
        """Slot when user changes camera device in combobox."""
        cam_idx = self.camera_selector.currentData()
        if cam_idx is not None:
            self._start_camera(cam_idx)

    def _rescan_cameras(self):
        """Rescans system for available webcams."""
        self.status_bar.showMessage("Scanning for video devices...")
        available = CameraThread.enumerate_cameras(max_tested=4)
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for idx in available:
            self.camera_selector.addItem(f"Camera {idx}", idx)
        self.camera_selector.blockSignals(False)

        if available:
            self.status_bar.showMessage(f"Found {len(available)} camera device(s).")
            self._start_camera(available[0])
        else:
            self.status_bar.showMessage("No camera devices detected.")
            dialog = PermissionTroubleshootDialog(
                "No Video Devices Detected",
                CONFIG.PERMISSION_TROUBLESHOOTING_GUIDE,
                self
            )
            dialog.exec()

    def _trigger_manual_detection(self):
        """Manually trigger Gemini spatial inference."""
        self._run_inference()

    def _toggle_continuous_analysis(self, state: int):
        """Toggle continuous 1.5s inference timer."""
        self._continuous_active = bool(state == Qt.CheckState.Checked.value)
        if self._continuous_active:
            self._continuous_timer.start()
            self.status_bar.showMessage("Continuous stream analysis active (1.5s cadence).")
            # Trigger immediate first run
            self._run_inference()
        else:
            self._continuous_timer.stop()
            self.video_canvas.set_analyzing(False)
            self.status_bar.showMessage("Continuous analysis paused.")

    def _on_continuous_timer_tick(self):
        """Cadence trigger every 1.5 seconds."""
        if self._continuous_active and not self._is_inference_running:
            self._run_inference()

    def _run_inference(self):
        """
        Dispatches Gemini inference worker to QThreadPool.
        Guarantees non-blocking GUI execution.
        """
        if self._is_inference_running:
            logger.debug("Inference already in-flight, skipping duplicate dispatch.")
            return

        if self._latest_raw_frame is None:
            self.status_bar.showMessage("No camera frame available for analysis.")
            return

        api_key = CONFIG.GEMINI_API_KEY
        if not api_key:
            QMessageBox.warning(
                self,
                "API Key Missing",
                "Please configure the GEMINI_API_KEY environment variable in your .env file."
            )
            self.chk_continuous.setChecked(False)
            return

        self._is_inference_running = True
        self.btn_detect.setEnabled(False)
        self.video_canvas.set_analyzing(True)

        worker = VisionInferenceWorker(
            frame_bgr=self._latest_raw_frame,
            api_key=api_key,
            model_name=CONFIG.MODEL_NAME
        )
        worker.signals.finished.connect(self._on_inference_finished)
        worker.signals.error.connect(self._on_inference_error)
        self._thread_pool.start(worker)

    def _on_inference_finished(self, detections: List[DetectionBox], latency: float):
        """Slot called when vision worker finishes."""
        self._is_inference_running = False
        self.btn_detect.setEnabled(True)
        self.video_canvas.set_analyzing(False)

        # Update detections on canvas
        self.video_canvas.set_detections(detections)
        
        # Update metrics
        self.latency_badge.setText(f"Latency: {latency * 1000:.0f} ms")
        self.objects_badge.setText(f"Objects: {len(detections)}")
        self.status_bar.showMessage(
            f"Detected {len(detections)} objects in {latency:.2f}s using {CONFIG.MODEL_NAME}."
        )

    def _on_inference_error(self, err_msg: str):
        """Slot called if vision worker encounters an error."""
        self._is_inference_running = False
        self.btn_detect.setEnabled(True)
        self.video_canvas.set_analyzing(False)
        logger.error(f"Inference failed: {err_msg}")
        self.status_bar.showMessage(f"AI Error: {err_msg}")

    def _export_snapshot(self):
        """Saves current frame with drawn bounding boxes to disk as PNG."""
        annotated = self.video_canvas.get_annotated_frame()
        if annotated is None:
            QMessageBox.information(self, "Snapshot", "No active frame to export.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"visionground_snapshot_{timestamp}.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Snapshot",
            default_filename,
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )

        if file_path:
            success = cv2.imwrite(file_path, annotated)
            if success:
                self.status_bar.showMessage(f"Snapshot exported successfully to: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Export Failed", f"Could not write file to {file_path}.")

    def closeEvent(self, event):
        """Gracefully release camera hardware and threads on application exit."""
        logger.info("Application closing. Terminating threads...")
        self._continuous_timer.stop()
        self._fps_timer.stop()
        if self._camera_thread is not None:
            self._camera_thread.stop()
        self._thread_pool.waitForDone(1500)
        event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
