"""
Camera Manager Module
---------------------
Threaded OpenCV video capture with hardware permission verification,
graceful disconnection handling, and clean frame delivery for PyQt6.
"""

import time
import logging
from typing import List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from config import CONFIG

logger = logging.getLogger("CameraManager")


class CameraThread(QThread):
    """
    Dedicated QThread for continuous OpenCV frame acquisition.
    Decouples frame polling from the PyQt GUI event loop to ensure
    smooth 30 FPS playback and zero UI stuttering.
    """

    # Signals
    frame_received = pyqtSignal(np.ndarray)               # Raw BGR frame
    camera_error = pyqtSignal(str, str, bool)             # (title, message, is_permission_issue)
    camera_disconnected = pyqtSignal(str)                 # Reason
    status_changed = pyqtSignal(bool, str)                # (is_connected, status_text)

    def __init__(self, camera_index: int = CONFIG.DEFAULT_CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._is_running = False
        self._mutex = QMutex()
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0

    def set_camera_index(self, index: int):
        """Update active camera device index."""
        with QMutexLocker(self._mutex):
            self.camera_index = index

    def stop(self):
        """Thread-safe request to stop camera capture and release hardware."""
        with QMutexLocker(self._mutex):
            self._is_running = False
        self.wait(2000)

    @staticmethod
    def enumerate_cameras(max_tested: int = 5) -> List[int]:
        """
        Scan and detect available video devices.
        Returns a list of integer camera indices that open successfully.
        """
        available: List[int] = []
        for index in range(max_tested):
            # Try opening camera with fast timeout
            cap = cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(index)
                cap.release()
        return available

    def _open_capture(self) -> bool:
        """
        Attempts to open cv2.VideoCapture with requested resolution.
        Catches device errors and emits detailed troubleshooting guidance.
        """
        logger.info(f"Opening camera index {self.camera_index}...")
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap or not self._cap.isOpened():
            logger.error(f"Failed to open camera device at index {self.camera_index}")
            self.camera_error.emit(
                "Camera Access Failure",
                f"Unable to access camera index {self.camera_index}.\n\n"
                f"{CONFIG.PERMISSION_TROUBLESHOOTING_GUIDE}",
                True
            )
            return False

        # Request desired resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.REQUESTED_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.REQUESTED_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, CONFIG.TARGET_FPS)

        # Verify initial frame read
        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.error("Camera opened but cannot read initial frame (access denied or busy).")
            self._cap.release()
            self._cap = None
            self.camera_error.emit(
                "Camera Stream Error",
                f"Device {self.camera_index} was detected, but failed to return video frames.\n"
                "The camera may be exclusively locked by another software (Zoom, Teams, etc.).",
                True
            )
            return False

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened successfully: {actual_w}x{actual_h} @ {CONFIG.TARGET_FPS} FPS")
        self.status_changed.emit(True, f"Connected: {actual_w}x{actual_h} ({self.camera_index})")
        return True

    def run(self):
        """Continuous frame polling loop running at ~30 FPS."""
        self._is_running = True
        self._consecutive_failures = 0

        if not self._open_capture():
            self._is_running = False
            self.status_changed.emit(False, "Camera not accessible")
            return

        frame_duration = 1.0 / CONFIG.TARGET_FPS

        try:
            while True:
                with QMutexLocker(self._mutex):
                    if not self._is_running:
                        break

                start_time = time.perf_counter()

                if self._cap is None or not self._cap.isOpened():
                    self._consecutive_failures += 1
                else:
                    ret, frame = self._cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self._consecutive_failures = 0
                        # Emit frame to GUI thread
                        self.frame_received.emit(frame)
                    else:
                        self._consecutive_failures += 1

                # Edge Case 4: Device Unavailability & Mid-stream Disconnection
                if self._consecutive_failures >= CONFIG.MAX_CONSECUTIVE_READ_FAILURES:
                    logger.warning(f"Device disconnected: {self._consecutive_failures} consecutive frame read drops.")
                    self.camera_disconnected.emit(
                        f"Camera {self.camera_index} was disconnected or became unavailable."
                    )
                    self.status_changed.emit(False, "Camera Disconnected")
                    break

                # Sleep precisely to maintain TARGET_FPS without burning CPU
                elapsed = time.perf_counter() - start_time
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            logger.exception("Unexpected error in CameraThread loop")
            self.camera_error.emit("Stream Exception", str(exc), False)
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._is_running = False
            logger.info("CameraThread stopped and resources released.")
