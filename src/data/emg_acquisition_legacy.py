"""
sEMG Data Acquisition Module (Legacy format support)

Supports TDMS file format and filtering pipeline from previous projects.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, List
from nptdms import TdmsFile


class EMGAcquisitionLegacy:
    """sEMG signal acquisition supporting TDMS format with legacy filtering."""

    def __init__(
        self,
        sampling_rate: int = 2000,
        notch_freqs: List[float] = [50.0, 100.0],
        lowpass_cutoff: float = 500.0,
        quality_factor: int = 30,
        lowpass_order: int = 4
    ):
        """
        Initialize EMG acquisition with legacy parameters.

        Args:
            sampling_rate: Sampling rate in Hz
            notch_freqs: List of frequencies for notch filters (Hz)
            lowpass_cutoff: Low-pass filter cutoff frequency (Hz)
            quality_factor: Q factor for notch filters
            lowpass_order: Order for low-pass Butterworth filter
        """
        self.sampling_rate = sampling_rate
        self.notch_freqs = notch_freqs
        self.lowpass_cutoff = lowpass_cutoff
        self.quality_factor = quality_factor
        self.lowpass_order = lowpass_order

        # Pre-compute filter coefficients
        self.notch_sos_list = [
            self._design_notch_filter(freq, sampling_rate, quality_factor)
            for freq in notch_freqs
        ]
        self.lowpass_sos = self._design_lowpass_filter(lowpass_cutoff, sampling_rate, lowpass_order)

    def _design_notch_filter(self, notch_freq: float, fs: float, quality_factor: int = 30):
        """Design a notch filter to remove power line interference."""
        nyquist = fs / 2
        notch_freq_normalized = notch_freq / nyquist
        b, a = signal.iirnotch(notch_freq_normalized, quality_factor)
        return signal.tf2sos(b, a)

    def _design_lowpass_filter(self, cutoff_freq: float, fs: float, order: int = 4):
        """Design a Butterworth low-pass filter."""
        nyquist = fs / 2
        cutoff_freq_normalized = cutoff_freq / nyquist
        sos = signal.butter(order, cutoff_freq_normalized, btype='low', output='sos')
        return sos

    def load_tdms(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load EMG data from TDMS file.

        Args:
            filepath: Path to TDMS file

        Returns:
            Tuple of (data_array, channel_names)
            data_array shape: (n_samples, n_channels)
        """
        tdms_file = TdmsFile.read(filepath)

        # Collect all channels from all groups
        all_channels = []
        for group in tdms_file.groups():
            all_channels.extend(group.channels())

        # Extract data from all channels
        data_list = []
        channel_names = []

        for channel in all_channels:
            data = channel[:]
            data_list.append(data)
            channel_names.append(channel.path)

        # Stack as (n_samples, n_channels)
        data_array = np.column_stack(data_list)

        return data_array, channel_names

    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filtering cascade: notch filters -> low-pass filter.

        Args:
            data: Raw EMG data (n_samples, n_channels)

        Returns:
            Filtered EMG data
        """
        filtered = data.copy()

        # Apply notch filters sequentially
        for notch_sos in self.notch_sos_list:
            filtered = signal.sosfiltfilt(notch_sos, filtered, axis=0)

        # Apply low-pass filter
        filtered = signal.sosfiltfilt(self.lowpass_sos, filtered, axis=0)

        return filtered

    def calculate_rms(
        self,
        data: np.ndarray,
        window_ms: int = 200,
        step_ms: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate RMS with sliding window.

        Args:
            data: Filtered EMG data (n_samples, n_channels)
            window_ms: Window size in milliseconds
            step_ms: Step size in milliseconds

        Returns:
            Tuple of (rms_values, timestamps)
            rms_values shape: (n_windows, n_channels)
        """
        window_samples = int(window_ms / 1000 * self.sampling_rate)
        step_samples = int(step_ms / 1000 * self.sampling_rate)

        rms_values = []
        timestamps = []

        for i in range(0, len(data) - window_samples + 1, step_samples):
            window = data[i:i + window_samples, :]
            rms = np.sqrt(np.mean(window ** 2, axis=0))
            rms_values.append(rms)
            timestamps.append((i + window_samples / 2) / self.sampling_rate)

        return np.array(rms_values), np.array(timestamps)

    def calculate_ema(self, data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Calculate Exponential Moving Average.

        Args:
            data: Filtered EMG data (n_samples, n_channels)
            alpha: EMA smoothing factor

        Returns:
            EMA values (n_samples, n_channels)
        """
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_mav(
        self,
        data: np.ndarray,
        window_ms: int = 200,
        step_ms: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Mean Absolute Value with sliding window.

        Args:
            data: Filtered EMG data (n_samples, n_channels)
            window_ms: Window size in milliseconds
            step_ms: Step size in milliseconds

        Returns:
            Tuple of (mav_values, timestamps)
        """
        window_samples = int(window_ms / 1000 * self.sampling_rate)
        step_samples = int(step_ms / 1000 * self.sampling_rate)

        mav_values = []
        timestamps = []

        for i in range(0, len(data) - window_samples + 1, step_samples):
            window = data[i:i + window_samples, :]
            mav = np.mean(np.abs(window), axis=0)
            mav_values.append(mav)
            timestamps.append((i + window_samples / 2) / self.sampling_rate)

        return np.array(mav_values), np.array(timestamps)

    def preprocess_pipeline(
        self,
        raw_data: np.ndarray,
        feature_type: str = 'rms'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full preprocessing pipeline.

        Args:
            raw_data: Raw EMG data
            feature_type: Type of feature extraction ('rms', 'ema', 'mav', 'raw')

        Returns:
            Tuple of (features, timestamps)
        """
        # Apply filters
        filtered = self.apply_filters(raw_data)

        if feature_type == 'raw':
            timestamps = np.arange(len(filtered)) / self.sampling_rate
            return filtered, timestamps
        elif feature_type == 'rms':
            return self.calculate_rms(filtered)
        elif feature_type == 'ema':
            ema = self.calculate_ema(filtered)
            timestamps = np.arange(len(ema)) / self.sampling_rate
            return ema, timestamps
        elif feature_type == 'mav':
            return self.calculate_mav(filtered)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")


class EMGAcquisitionCSV:
    """sEMG signal acquisition for CSV format (from Senkouka project)."""

    def __init__(
        self,
        sampling_rate: int = 10000,
        gain: float = 883.97,
        mid_vol: float = 0.0
    ):
        """
        Initialize EMG acquisition for CSV format.

        Args:
            sampling_rate: Sampling rate in Hz
            gain: Amplifier gain
            mid_vol: Reference voltage
        """
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.mid_vol = mid_vol

        # Pre-compute filter coefficients (from SignalFilterAndDisplay.py)
        self.b_dc = [1.0, -1.0]
        self.a_dc = [1.0, -0.995]

        self.b_bessel = [0.046034, 0.092067, 0.046034]
        self.a_bessel = [1.0, -1.36170, 0.545836]

        self.b_notch = [0.9994767612707923, -1.9979671596465018, 0.9994767612707923]
        self.a_notch = [1.0, -1.9979671596465018, 0.9989535225415846]

    def load_csv(self, filepath: str, ref_vol: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EMG data from CSV file.

        Args:
            filepath: Path to CSV file
            ref_vol: Reference voltage

        Returns:
            Tuple of (time_array, data_array)
            data_array shape: (n_samples, n_channels)
        """
        import csv

        with open(filepath, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)

            # Skip 3 rows
            for _ in range(3):
                next(reader)

            # Get sample count
            row = next(reader)
            samp_count = int(row[0].split(": ")[1])

            # Skip 1 row (Start Time)
            next(reader)

            # Get sampling frequency
            row = next(reader)
            fs = int(row[0].split(": ")[1])

            # Skip 1 more row (header row)
            next(reader)

            # Extract data columns
            time_col, out1_col, out2_col, out3_col = [], [], [], []

            for row in reader:
                if len(row) >= 5:
                    time_col.append(float(row[0]))
                    out1_col.append((float(row[2]) - ref_vol) * 1000000 / self.gain)
                    out2_col.append((float(row[3]) - ref_vol) * 1000000 / self.gain)
                    out3_col.append((float(row[4]) - ref_vol) * 1000000 / self.gain)
                else:
                    time_col.append(float(row[0]))
                    out1_col.append(0.0)
                    out2_col.append(0.0)
                    out3_col.append(0.0)

        time_array = np.array(time_col)
        data_array = np.column_stack([
            np.array(out1_col),
            np.array(out2_col),
            np.array(out3_col)
        ])

        return time_array, data_array

    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply 3-stage filtering cascade (DC blocker → Bessel → Notch).

        Args:
            data: Raw EMG data (n_samples, n_channels)

        Returns:
            Filtered EMG data
        """
        # DC blocker
        data_dc = signal.lfilter(self.b_dc, self.a_dc, data, axis=0)
        # Bessel filter
        data_bessel = signal.lfilter(self.b_bessel, self.a_bessel, data_dc, axis=0)
        # Notch filter
        final_output = signal.lfilter(self.b_notch, self.a_notch, data_bessel, axis=0)
        return final_output
