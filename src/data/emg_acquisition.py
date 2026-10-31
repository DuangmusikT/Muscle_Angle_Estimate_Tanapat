"""
sEMG Data Acquisition Module

Handles acquisition and preprocessing of surface electromyography signals.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


class EMGAcquisition:
    """sEMG signal acquisition and preprocessing."""

    def __init__(
        self,
        sampling_rate: int = 1000,
        channels: int = 8,
        notch_freq: float = 50.0,
        bandpass_low: float = 20.0,
        bandpass_high: float = 450.0,
    ):
        """
        Initialize EMG acquisition parameters.

        Args:
            sampling_rate: Sampling rate in Hz
            channels: Number of EMG channels
            notch_freq: Power line frequency for notch filter (Hz)
            bandpass_low: Low cutoff frequency for bandpass filter (Hz)
            bandpass_high: High cutoff frequency for bandpass filter (Hz)
        """
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.notch_freq = notch_freq
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high

    def load_emg_data(self, filepath: str) -> np.ndarray:
        """
        Load raw EMG data from file.

        Args:
            filepath: Path to EMG data file

        Returns:
            EMG data array of shape (n_samples, n_channels)
        """
        # TODO: Implement based on your specific data format
        # This is a placeholder - adapt to your actual data format
        data = np.load(filepath)
        return data

    def apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.

        Args:
            data: Raw EMG data

        Returns:
            Filtered EMG data
        """
        nyquist = self.sampling_rate / 2
        notch_width = 2.0  # Hz

        # Design notch filter
        low = (self.notch_freq - notch_width) / nyquist
        high = (self.notch_freq + notch_width) / nyquist
        b, a = signal.iirnotch(self.notch_freq / nyquist, Q=30)

        filtered = signal.filtfilt(b, a, data, axis=0)
        return filtered

    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to extract EMG frequency range.

        Args:
            data: EMG data

        Returns:
            Bandpass filtered EMG data
        """
        nyquist = self.sampling_rate / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist

        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=0)
        return filtered

    def rectify(self, data: np.ndarray) -> np.ndarray:
        """
        Full-wave rectification of EMG signal.

        Args:
            data: Filtered EMG data

        Returns:
            Rectified EMG data
        """
        return np.abs(data)

    def normalize(self, data: np.ndarray, method: str = 'max') -> np.ndarray:
        """
        Normalize EMG signal.

        Args:
            data: EMG data
            method: Normalization method ('max', 'zscore', 'minmax')

        Returns:
            Normalized EMG data
        """
        if method == 'max':
            # Normalize by maximum value per channel
            max_vals = np.max(data, axis=0, keepdims=True)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            return data / max_vals
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            std[std == 0] = 1
            return (data - mean) / std
        elif method == 'minmax':
            # Min-max normalization
            min_vals = np.min(data, axis=0, keepdims=True)
            max_vals = np.max(data, axis=0, keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            return (data - min_vals) / range_vals
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def extract_features(
        self,
        data: np.ndarray,
        window_size: int = 200,
        overlap: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time-domain features from EMG signal.

        Args:
            data: Preprocessed EMG data (n_samples, n_channels)
            window_size: Window size in samples
            overlap: Overlap ratio between windows (0-1)

        Returns:
            features: Feature array (n_windows, n_channels * n_features)
            timestamps: Center timestamp for each window
        """
        step = int(window_size * (1 - overlap))
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        features_list = []
        timestamps = []

        for i in range(0, n_samples - window_size + 1, step):
            window = data[i:i + window_size, :]

            # Time-domain features
            mav = np.mean(np.abs(window), axis=0)  # Mean Absolute Value
            rms = np.sqrt(np.mean(window ** 2, axis=0))  # Root Mean Square
            wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)  # Waveform Length
            zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)  # Zero Crossings
            ssc = np.sum(np.diff(np.diff(window), axis=0) != 0, axis=0)  # Slope Sign Changes

            # Combine features
            window_features = np.concatenate([
                mav, rms, wl, zc, ssc
            ])
            features_list.append(window_features)

            timestamps.append(i + window_size // 2)

        features = np.array(features_list)
        timestamps = np.array(timestamps)

        return features, timestamps

    def preprocess_pipeline(
        self,
        raw_data: np.ndarray,
        normalize_method: str = 'max'
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to raw EMG data.

        Args:
            raw_data: Raw EMG data
            normalize_method: Normalization method

        Returns:
            Preprocessed EMG data
        """
        # Apply filters
        data = self.apply_notch_filter(raw_data)
        data = self.apply_bandpass_filter(data)

        # Rectify and normalize
        data = self.rectify(data)
        data = self.normalize(data, method=normalize_method)

        return data
