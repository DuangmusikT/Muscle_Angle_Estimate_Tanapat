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
        sampling_rate: int = 1200,
        channels: int = 8,
        notch_freq: float = 50.0,
        lpf_cutoff: float = 488.652,
        gain: float = 816.0,
        ref_voltage: float = 0.0,
    ):
        """
        Initialize EMG acquisition parameters.

        Args:
            sampling_rate: Sampling rate in Hz (default: 1200 Hz for sync with 30/60 fps video)
            channels: Number of EMG channels
            notch_freq: Power line frequency for notch filter (Hz)
            lpf_cutoff: Low-pass filter cutoff frequency (Hz)
            gain: Amplifier gain for voltage conversion
            ref_voltage: Reference voltage for ADC
        """
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.notch_freq = notch_freq
        self.lpf_cutoff = lpf_cutoff
        self.gain = gain
        self.ref_voltage = ref_voltage

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

    def convert_to_voltage(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Convert raw ADC values to voltage using gain.

        Args:
            raw_data: Raw ADC data (n_samples, n_channels)

        Returns:
            Voltage data in microvolts (µV)
        """
        # Convert to voltage: (raw - ref) * 1e6 / gain
        voltage_data = (raw_data - self.ref_voltage) * 1e6 / self.gain
        return voltage_data

    def apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line interference at 50 Hz.

        Args:
            data: Raw EMG data

        Returns:
            Filtered EMG data
        """
        nyquist = self.sampling_rate / 2
        notch_freq_normalized = self.notch_freq / nyquist

        # Design notch filter with Q=30 for sharp rejection
        b, a = signal.iirnotch(notch_freq_normalized, Q=30)

        filtered = signal.filtfilt(b, a, data, axis=0)
        return filtered

    def apply_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter to extract EMG frequency range.

        Args:
            data: EMG data

        Returns:
            Low-pass filtered EMG data
        """
        nyquist = self.sampling_rate / 2
        cutoff = self.lpf_cutoff / nyquist

        b, a = signal.butter(4, cutoff, btype='low')
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
        normalize_method: str = 'max',
        apply_voltage_conversion: bool = True
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to raw EMG data.

        Args:
            raw_data: Raw EMG data
            normalize_method: Normalization method
            apply_voltage_conversion: Whether to convert ADC to voltage

        Returns:
            Preprocessed EMG data
        """
        # Convert to voltage if needed
        if apply_voltage_conversion:
            data = self.convert_to_voltage(raw_data)
        else:
            data = raw_data

        # Apply filters
        data = self.apply_notch_filter(data)
        data = self.apply_lowpass_filter(data)

        # Rectify and normalize
        data = self.rectify(data)
        data = self.normalize(data, method=normalize_method)

        return data
