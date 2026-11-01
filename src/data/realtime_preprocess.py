"""Streaming (causal) preprocessing + feature stacking for real-time EMG inference."""

import numpy as np
from collections import deque
from scipy import signal


class RealtimePreprocessor:
    """Stateful causal preprocessing for an EMG stream.

    Produces 32-dim stacked feature vectors (4 features x n_channels x 4 windows)
    matching the offline training pipeline (EMGAcquisition.extract_stacked_features).
    """

    def __init__(
        self,
        sampling_rate: float = 1200.0,
        notch_freq: float = 50.0,
        lpf_cutoff: float = 488.652,
        gain: float = 816.0,
        ref_voltage: float = 0.0,
        norm_params: dict = None,
        window_size: int = 320,
        hop: int = 40,
        stack: int = 4,
    ):
        self.fs = sampling_rate
        self.gain = gain
        self.ref_voltage = ref_voltage
        self.norm_params = norm_params or {}
        self.window_size = window_size
        self.hop = hop
        self.stack = stack

        nyq = sampling_rate / 2.0
        b, a = signal.iirnotch(notch_freq / nyq, Q=30)
        self.sos_notch = signal.tf2sos(b, a)
        self.sos_lpf = signal.butter(4, lpf_cutoff / nyq, btype="low", output="sos")

        self._zi_notch = None
        self._zi_lpf = None
        self._initialized = False

        self.buffer = deque(maxlen=window_size + hop * stack + 64)
        self._prev_rms = None
        self._feature_buffer = deque(maxlen=stack)
        self._samples_since_feature = 0
        self._latest_stacked = None
        self._last_feat_count = 0

    def _init_state(self, first_sample):
        self._zi_notch = signal.sosfilt_zi(self.sos_notch)[:, :, None] * first_sample[None, None, :]
        self._zi_lpf = None
        self._initialized = True

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros((0, chunk.shape[1]))

        data = (chunk - self.ref_voltage) * 1e6 / self.gain

        if not self._initialized:
            self._init_state(data[0])

        out_notch, self._zi_notch = signal.sosfilt(
            self.sos_notch, data, axis=0, zi=self._zi_notch
        )
        if self._zi_lpf is None:
            self._zi_lpf = signal.sosfilt_zi(self.sos_lpf)[:, :, None] * out_notch[0][None, None, :]
        out_lpf, self._zi_lpf = signal.sosfilt(
            self.sos_lpf, out_notch, axis=0, zi=self._zi_lpf
        )
        rect = np.abs(out_lpf)
        norm = self._normalize(rect)

        for row in norm:
            self.buffer.append(row)
            self._samples_since_feature += 1
            if self._samples_since_feature >= self.hop and len(self.buffer) >= self.window_size:
                window = np.asarray(list(self.buffer))[-self.window_size:]
                mav = np.mean(np.abs(window), axis=0)
                rms = np.sqrt(np.mean(window ** 2, axis=0))
                wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
                if self._prev_rms is None:
                    drms = np.zeros_like(rms)
                else:
                    drms = rms - self._prev_rms
                self._prev_rms = rms.copy()
                feat = np.concatenate([rms, mav, wl, drms])
                self._feature_buffer.append(feat)
                self._samples_since_feature = 0
                if len(self._feature_buffer) >= self.stack:
                    stacked = np.concatenate([self._feature_buffer[-1 - j] for j in range(self.stack)])
                    self._latest_stacked = stacked
                    self._last_feat_count = len(self._feature_buffer)
        return norm

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        params = self.norm_params
        if not params:
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

    def get_feature_vector(self):
        """Return the latest 32-dim stacked feature vector, or None."""
        if self._latest_stacked is None:
            return None
        if len(self._feature_buffer) <= self._last_feat_count:
            # no new stack since last call
            pass
        # Return a copy only when new features arrived
        if self._last_feat_count > 0:
            v = self._latest_stacked.copy()
            self._last_feat_count = 0  # consume
            return v
        return None

    def get_window(self, window_size: int = 200):
        if len(self.buffer) < window_size:
            return None
        arr = np.asarray(self.buffer)
        return arr[-window_size:, :]

    def reset(self):
        self._zi_notch = None
        self._zi_lpf = None
        self._initialized = False
        self.buffer.clear()
        self._prev_rms = None
        self._feature_buffer.clear()
        self._samples_since_feature = 0
        self._latest_stacked = None
        self._last_feat_count = 0
