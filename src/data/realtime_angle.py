"""Real-time hand angle extraction from a webcam using HandAngleDetector."""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .angle_detection import HandAngleDetector


class RealTimeAngleSource:
    """Webcam-based angle source producing (timestamp, angle, frame, landmarks)."""

    def __init__(self, camera_index: int = 0, fps: int = 30):
        self.camera_index = camera_index
        self.fps = fps
        self.detector = HandAngleDetector(static_image_mode=False, max_hands=1)
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_angle: Optional[float] = None
        self._last_landmarks: Optional[np.ndarray] = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read_frame(self) -> Tuple[float, Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.cap is None:
            raise RuntimeError("Angle source not started")
        ts = time.perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            return ts, None, None, None
        result = self.detector.process_frame(frame)
        angle = None
        landmarks = None
        if result is not None and "angles" in result:
            angle = result["angles"].get("wrist_angle", None)
            landmarks = result.get("landmarks", None)
            if angle is not None:
                self._last_angle = angle
                self._last_landmarks = landmarks
        if angle is None:
            angle = self._last_angle
            landmarks = self._last_landmarks
        return ts, angle, frame, landmarks

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.detector.close()
