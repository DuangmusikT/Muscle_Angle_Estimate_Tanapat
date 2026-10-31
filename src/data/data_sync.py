"""
Data Synchronization Module

Aligns sEMG sensor data with OpenCV-detected angles temporally.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal


class DataSynchronizer:
    """Synchronize sEMG signals with angle measurements."""

    def __init__(self, emg_sampling_rate: int = 1000, video_fps: int = 30):
        """
        Initialize synchronizer.

        Args:
            emg_sampling_rate: sEMG sampling rate in Hz
            video_fps: Video frame rate in Hz
        """
        self.emg_sampling_rate = emg_sampling_rate
        self.video_fps = video_fps
        self.time_ratio = emg_sampling_rate / video_fps

    def sync_data(
        self,
        emg_data: np.ndarray,
        angles: np.ndarray,
        emg_timestamps: Optional[np.ndarray] = None,
        angle_timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronize EMG data with angle data.

        Args:
            emg_data: EMG data array (n_emg_samples, n_channels)
            angles: Angle data array (n_angle_samples, n_angles)
            emg_timestamps: Optional EMG timestamps
            angle_timestamps: Optional angle timestamps

        Returns:
            synced_emg: Synchronized EMG data
            synced_angles: Synchronized angle data
        """
        if emg_timestamps is None:
            emg_timestamps = np.arange(len(emg_data)) / self.emg_sampling_rate

        if angle_timestamps is None:
            angle_timestamps = np.arange(len(angles)) / self.video_fps

        # Interpolate angles to match EMG timestamps
        synced_angles = self._interpolate_angles(
            angle_timestamps,
            angles,
            emg_timestamps
        )

        # Filter out NaN values (outside angle measurement range)
        valid_mask = ~np.isnan(synced_angles).any(axis=1)
        synced_emg = emg_data[valid_mask]
        synced_angles = synced_angles[valid_mask]

        return synced_emg, synced_angles

    def _interpolate_angles(
        self,
        source_timestamps: np.ndarray,
        source_data: np.ndarray,
        target_timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate angle data to target timestamps.

        Args:
            source_timestamps: Original timestamps
            source_data: Original angle data
            target_timestamps: Target timestamps to interpolate to

        Returns:
            Interpolated angle data
        """
        interpolated = np.zeros((len(target_timestamps), source_data.shape[1]))

        for i in range(source_data.shape[1]):
            interpolated[:, i] = np.interp(
                target_timestamps,
                source_timestamps,
                source_data[:, i],
                left=np.nan,
                right=np.nan
            )

        return interpolated

    def create_windows(
        self,
        emg_data: np.ndarray,
        angles: np.ndarray,
        window_size: int = 200,
        overlap: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping windows from synchronized data.

        Args:
            emg_data: Synchronized EMG data
            angles: Synchronized angle data
            window_size: Window size in samples
            overlap: Overlap ratio (0-1)

        Returns:
            windows_emg: Windowed EMG data (n_windows, window_size, n_channels)
            windows_angles: Windowed angle data (n_windows, n_angles)
        """
        step = int(window_size * (1 - overlap))
        n_samples = emg_data.shape[0]

        windows_emg = []
        windows_angles = []

        for i in range(0, n_samples - window_size + 1, step):
            windows_emg.append(emg_data[i:i + window_size, :])
            # Use angle at the center of the window
            center_idx = i + window_size // 2
            windows_angles.append(angles[center_idx])

        windows_emg = np.array(windows_emg)
        windows_angles = np.array(windows_angles)

        return windows_emg, windows_angles

    def cross_correlation_sync(
        self,
        emg_data: np.ndarray,
        angles: np.ndarray,
        max_lag: int = 100
    ) -> int:
        """
        Find optimal lag between EMG and angle data using cross-correlation.

        Args:
            emg_data: EMG data (use one channel or summed channels)
            angles: Angle data (use one angle or averaged)
            max_lag: Maximum lag to search

        Returns:
            Optimal lag in samples
        """
        # Use first channel and first angle for correlation
        emg_signal = emg_data[:, 0]
        angle_signal = angles[:, 0]

        # Normalize signals
        emg_signal = (emg_signal - np.mean(emg_signal)) / np.std(emg_signal)
        angle_signal = (angle_signal - np.mean(angle_signal)) / np.std(angle_signal)

        # Compute cross-correlation
        correlation = signal.correlate(emg_signal, angle_signal, mode='valid')

        # Find peak correlation
        lag = np.argmax(correlation) - len(angle_signal) + 1

        # Clip to max lag
        lag = np.clip(lag, -max_lag, max_lag)

        return lag

    def apply_lag(
        self,
        emg_data: np.ndarray,
        angles: np.ndarray,
        lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply lag to align EMG and angle data.

        Args:
            emg_data: EMG data
            angles: Angle data
            lag: Lag in samples (positive = EMG leads, negative = EMG lags)

        Returns:
            Aligned EMG and angle data
        """
        if lag > 0:
            # EMG leads, trim start of EMG and end of angles
            aligned_emg = emg_data[lag:]
            aligned_angles = angles[:-lag] if lag < len(angles) else angles[:0]
        elif lag < 0:
            # EMG lags, trim end of EMG and start of angles
            aligned_emg = emg_data[:lag]
            aligned_angles = angles[-lag:]
        else:
            aligned_emg = emg_data
            aligned_angles = angles

        return aligned_emg, aligned_angles
