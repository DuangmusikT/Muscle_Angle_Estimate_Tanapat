"""
Streaming (causal) preprocessing for real-time EMG inference.

Uses SOS filters (causal) so that each new sample can be processed without
waiting for future samples, at the cost of non-zero-phase filtering. The same
SOS coefficients are used by ``EMGAcquisition`` for offline training, keeping
the live and training representations consistent.
"""

import numpy as np
from collections import deque
from scipy import signal

from .emg_acquisition import EMGAcquisition


class RealtimePreprocessor:
    """Stateful causal preprocessing for a 3-channel EMG stream."""

    def __init__(
        self,
        sampling_rate: float = 1200.0,
        notch_freq: float = 50.0,
        lpf_cutoff: float = 488.652,
        gain: float = 816.0,
        ref_voltage: float = 0.0,
        norm_params: dict = None,
    ):
        self.fs = sampling_rate
        self.gain = gain
        self.ref_voltage = ref_voltage
        self.norm_params = norm_params or {}

        nyq = sampling_rate / 2.0
        # Notch
        b, a = signal.iirnotch(notch_freq / nyq, Q=30)
        self.sos_notch = signal.tf2sos(b, a)
        # Lowpass
        self.sos_lpf = signal.butter(4, lpf_cutoff / nyq, btype="low", output="sos")

        self._zi_notch = None
        self._zi_lpf = None
        self._initialized = False

        # Rolling preprocessed buffer
        self.buffer = deque(maxlen=400)

    def _init_state(self, first_sample: np.ndarray):
        # first_sample shape: (n_channels,)
        self._zi_notch = signal.sosfilt_zi(self.sos_notch)[:, :, None] * first_sample[None, None, :]
        # Lowpass state initialized after notch output is known; placeholder
        self._zi_lpf = None
        self._initialized = True

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of raw EMG (n_samples, n_channels).

        Returns the preprocessed (filtered, rectified, normalized) samples
        for this chunk. Internal buffer is also updated.
        """
        if chunk.size == 0:
            return np.zeros((0, chunk.shape[1]))

        # Voltage conversion (ADC -> uV). If data already in volts, set gain=1.
        data = (chunk - self.ref_voltage) * 1e6 / self.gain

        if not self._initialized:
            self._init_state(data[0])

        # Notch (causal)
        out_notch, self._zi_notch = signal.sosfilt(
            self.sos_notch, data, axis=0, zi=self._zi_notch
        )
        if self._zi_lpf is None:
            self._zi_lpf = signal.sosfilt_zi(self.sos_lpf)[:, :, None] * out_notch[0][None, None, :]

        # Lowpass (causal)
        out_lpf, self._zi_lpf = signal.sosfilt(
            self.sos_lpf, out_notch, axis=0, zi=self._zi_lpf
        )

        # Rectify
        rect = np.abs(out_lpf)

        # Normalize using saved params
        norm = self._normalize(rect)

        for row in norm:
            self.buffer.append(row)
        return norm

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        params = self.norm_params
        if not params:
            # Fallback: per-chunk max (not recommended for live)
            mx = np.max(data, axis=0, keepdims=True)
            mx[mx == 0] = 1
            return data / mx
        if "min" in params and "range" in params:
            mn = np.asarray(params["min"]).reshape(-1)
            rng = np.asarray(params["range"]).reshape(-1)
            rng[rng == 0] = 1
            return (data - mn) / rng
        if "max" in params:
            mx = np.asarray(params["max"]).reshape(-1)
            mx[mx == 0] = 1
            return data / mx
        return data

    def get_window(self, window_size: int = 200) -> np.ndarray:
        """Return the last ``window_size`` preprocessed samples, or None."""
        if len(self.buffer) < window_size:
            return None
        arr = np.asarray(self.buffer)
        return arr[-window_size:, :]

    def reset(self):
        self._zi_notch = None
        self._zi_lpf = None
        self._initialized = False
        self.buffer.clear()
