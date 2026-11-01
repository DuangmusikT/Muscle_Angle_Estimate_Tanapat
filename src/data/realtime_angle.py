"""
Real-time hand angle extraction from a webcam using the existing
HandAngleDetector (MediaPipe). Only the simplified wrist/palm angle is used.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .angle_detection import HandAngleDetector


class RealTimeAngleSource:
    """Webcam-based angle source producing (timestamp, angle) samples."""

    def __init__(self, camera_index: int = 0, fps: int = 30):
        self.camera_index = camera_index
        self.fps = fps
        self.detector = HandAngleDetector(static_image_mode=False, max_hands=1)
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_angle: Optional[float] = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read_frame(self) -> Tuple[float, Optional[float], Optional[np.ndarray]]:
        """Capture one frame and return (timestamp, angle, frame).

        ``angle`` is None when no hand is detected.
        """
        if self.cap is None:
            raise RuntimeError("Angle source not started")
        ts = time.perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            return ts, None, None
        result = self.detector.process_frame(frame)
        angle = None
        if result is not None and "angles" in result:
            angle = result["angles"].get("wrist_angle", None)
            if angle is not None:
                self._last_angle = angle
        # Carry forward last known angle to avoid gaps
        if angle is None:
            angle = self._last_angle
        return ts, angle, frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.detector.close()
