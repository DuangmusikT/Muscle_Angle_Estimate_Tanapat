"""
Hand Angle Detection Module

Uses OpenCV and MediaPipe to detect hand landmarks and calculate joint angles.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict


class HandAngleDetector:
    """Detect hand joint angles from video using MediaPipe."""

    def __init__(self, static_image_mode: bool = False, max_hands: int = 1):
        """
        Initialize hand angle detector.

        Args:
            static_image_mode: Whether to treat input as static images
            max_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single video frame to detect hand landmarks.

        Args:
            frame: Video frame (BGR format)

        Returns:
            Dictionary containing landmarks and angles, or None if no hand detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks is None:
            return None

        # Get first hand (assuming single hand)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract landmark coordinates
        landmarks = self._extract_landmarks(hand_landmarks)

        # Calculate joint angles
        angles = self._calculate_all_angles(landmarks)

        return {
            'landmarks': landmarks,
            'angles': angles
        }

    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Extract landmark coordinates from MediaPipe result.

        Args:
            hand_landmarks: MediaPipe hand landmarks

        Returns:
            Array of shape (21, 3) containing (x, y, z) coordinates
        """
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks)

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle between three points (p1-p2-p3).

        Args:
            p1: First point
            p2: Vertex point
            p3: Third point

        Returns:
            Angle in degrees
        """
        # Vectors from vertex
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def _calculate_all_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate all relevant finger joint angles.

        Args:
            landmarks: Hand landmark coordinates (21, 3)

        Returns:
            Dictionary of joint angles in degrees
        """
        angles = {}

        # MediaPipe landmark indices:
        # 0: Wrist
        # 1-4: Thumb (CMC, MCP, IP, TIP)
        # 5-8: Index finger (MCP, PIP, DIP, TIP)
        # 9-12: Middle finger (MCP, PIP, DIP, TIP)
        # 13-16: Ring finger (MCP, PIP, DIP, TIP)
        # 17-20: Pinky (MCP, PIP, DIP, TIP)

        # Thumb angles
        angles['thumb_cmc'] = self._calculate_angle(
            landmarks[0], landmarks[1], landmarks[2]
        )
        angles['thumb_mcp'] = self._calculate_angle(
            landmarks[1], landmarks[2], landmarks[3]
        )
        angles['thumb_ip'] = self._calculate_angle(
            landmarks[2], landmarks[3], landmarks[4]
        )

        # Index finger angles
        angles['index_mcp'] = self._calculate_angle(
            landmarks[0], landmarks[5], landmarks[6]
        )
        angles['index_pip'] = self._calculate_angle(
            landmarks[5], landmarks[6], landmarks[7]
        )
        angles['index_dip'] = self._calculate_angle(
            landmarks[6], landmarks[7], landmarks[8]
        )

        # Middle finger angles
        angles['middle_mcp'] = self._calculate_angle(
            landmarks[0], landmarks[9], landmarks[10]
        )
        angles['middle_pip'] = self._calculate_angle(
            landmarks[9], landmarks[10], landmarks[11]
        )
        angles['middle_dip'] = self._calculate_angle(
            landmarks[10], landmarks[11], landmarks[12]
        )

        # Ring finger angles
        angles['ring_mcp'] = self._calculate_angle(
            landmarks[0], landmarks[13], landmarks[14]
        )
        angles['ring_pip'] = self._calculate_angle(
            landmarks[13], landmarks[14], landmarks[15]
        )
        angles['ring_dip'] = self._calculate_angle(
            landmarks[14], landmarks[15], landmarks[16]
        )

        # Pinky angles
        angles['pinky_mcp'] = self._calculate_angle(
            landmarks[0], landmarks[17], landmarks[18]
        )
        angles['pinky_pip'] = self._calculate_angle(
            landmarks[17], landmarks[18], landmarks[19]
        )
        angles['pinky_dip'] = self._calculate_angle(
            landmarks[18], landmarks[19], landmarks[20]
        )

        # Wrist flexion/extension (simplified)
        angles['wrist_angle'] = self._calculate_angle(
            landmarks[0], landmarks[5], landmarks[17]
        )

        return angles

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process entire video to extract angles over time.

        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video

        Returns:
            angles_array: Array of shape (n_frames, n_angles)
            angle_names: List of angle names
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        angles_list = []
        angle_names = None

        # Setup video writer if output path provided
        writer = None
        if output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            if result is not None:
                angles_list.append(result['angles'])
                if angle_names is None:
                    angle_names = list(result['angles'].keys())

                # Draw landmarks on frame
                if writer is not None:
                    self._draw_landmarks(frame, result['landmarks'])
                    writer.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{n_frames} frames")

        cap.release()
        if writer is not None:
            writer.release()

        angles_array = np.array(angles_list)

        return angles_array, angle_names

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray):
        """
        Draw hand landmarks on frame for visualization.

        Args:
            frame: Video frame
            landmarks: Hand landmark coordinates
        """
        h, w = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        points = []
        for lm in landmarks:
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            points.append((x, y))

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        for start, end in connections:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        # Draw points
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def close(self):
        """Release resources."""
        self.hands.close()
