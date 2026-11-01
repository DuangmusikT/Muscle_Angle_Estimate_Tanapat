"""
Real-time sEMG acquisition helpers.

Provides a small, source-agnostic interface for collecting EMG samples in real
time, plus an implementation for the Digilent (Measurement Computing)
USB-1608GX using the ``uldaq`` library, and a simulated source for dry runs.
"""

import time
import threading
from typing import Optional, Tuple

import numpy as np

try:
    from uldaq import (
        get_daq_device_inventory,
        InterfaceType,
        DaqDevice,
        AiInputMode,
        Range,
        ScanOption,
        AInScanFlag,
        ScanStatus,
    )

    _ULDAQ_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _ULDAQ_AVAILABLE = False


class RealTimeEMGSource:
    """Base interface for real-time EMG sources."""

    def start(self) -> None:
        raise NotImplementedError

    def read_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (timestamps, data) for samples collected since the last call.

        ``timestamps`` shape: (n_samples,) in seconds since ``start()``.
        ``data`` shape: (n_samples, n_channels).
        Returns (None, None) when no new samples are available.
        """
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError

    @property
    def n_channels(self) -> int:
        raise NotImplementedError


class SimulatedEMGSource(RealTimeEMGSource):
    """Synthetic EMG source for testing the collection pipeline without hardware."""

    def __init__(
        self,
        sampling_rate: float = 1200.0,
        n_channels: int = 3,
        seed: Optional[int] = None,
    ):
        self._fs = sampling_rate
        self._channels = n_channels
        self._rng = np.random.default_rng(seed)
        self._t0: Optional[float] = None
        self._next_sample = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._next_sample = 0

    def read_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._t0 is None:
            raise RuntimeError("Source not started")
        now = time.perf_counter() - self._t0
        target = int(now * self._fs)
        n_new = target - self._next_sample
        if n_new <= 0:
            return None, None
        idx = np.arange(self._next_sample, self._next_sample + n_new)
        t = idx / self._fs
        data = self._rng.standard_normal((n_new, self._channels))
        for c in range(self._channels):
            freq = 10 + c * 5
            data[:, c] += 0.5 * np.sin(2 * np.pi * freq * t)
        timestamps = idx / self._fs
        with self._lock:
            self._next_sample += n_new
        return timestamps, data

    def stop(self) -> None:
        self._t0 = None

    @property
    def sampling_rate(self) -> float:
        return self._fs

    @property
    def n_channels(self) -> int:
        return self._channels
class USB1608GXSource(RealTimeEMGSource):
    """Real-time acquisition from a Digilent (MCC) USB-1608GX via ``uldaq``."""

    def __init__(
        self,
        sampling_rate: float = 1200.0,
        n_channels: int = 3,
        duration: float = 60.0,
        gain: float = 816.0,
        device_index: int = 0,
    ):
        if not _ULDAQ_AVAILABLE:
            raise RuntimeError(
                "uldaq is not installed. Install libuldaq + uldaq first."
            )
        self._fs = sampling_rate
        self._requested_channels = n_channels
        self._duration = duration
        self._gain = gain
        self._device_index = device_index

        self._daq = None
        self._ai = None
        self._buffer = None
        self._channels = n_channels
        self._samples_per_channel = 0
        self._start_time: Optional[float] = None
        self._next_sample = 0
        self._started = False

    def start(self) -> None:
        devices = get_daq_device_inventory(InterfaceType.USB)
        if not devices:
            raise RuntimeError(
                "No USB DAQ device detected. Is the USB-1608GX connected?"
            )
        if self._device_index >= len(devices):
            raise RuntimeError(
                f"Device index {self._device_index} out of range ({len(devices)} found)"
            )

        desc = devices[self._device_index]
        self._daq = DaqDevice(desc)
        self._daq.connect()
        self._ai = self._daq.get_ai_device()
        info = self._ai.get_info()

        max_chans = info.get_num_chans()
        self._channels = min(self._requested_channels, max_chans)
        if self._channels < 1:
            raise RuntimeError("Device reports 0 analog input channels")

        # Input mode
        input_mode = AiInputMode.DIFFERENTIAL
        try:
            modes = info.get_chan_types() if hasattr(info, "get_chan_types") else []
            if modes and input_mode not in modes:
                input_mode = modes[0]
        except Exception:
            pass

        # Range
        ranges = info.get_ranges(input_mode)
        analog_range = None
        for r in ranges:
            if r == Range.BIP2VOLTS:
                analog_range = r
                break
        if analog_range is None:
            analog_range = ranges[0] if ranges else Range.BIP2VOLTS

        if not info.has_pacer():
            raise RuntimeError("Device does not support hardware-paced scanning")

        self._samples_per_channel = int((self._duration + 5.0) * self._fs)
        total = self._channels * self._samples_per_channel
        from ctypes import c_double
        self._buffer_ctypes = (c_double * total)()
        self._buffer = np.ctypeslib.as_array(self._buffer_ctypes)

        options = ScanOption.CONTINUOUS | ScanOption.DEFAULTIO
        flags = AInScanFlag.DEFAULT

        self._ai.a_in_scan(
            low_channel=0,
            high_channel=self._channels - 1,
            input_mode=input_mode,
            analog_range=analog_range,
            samples_per_channel=self._samples_per_channel,
            rate=self._fs,
            options=options,
            flags=flags,
            data=self._buffer_ctypes,
        )

        self._start_time = time.perf_counter()
        self._next_sample = 0
        self._started = True

    def read_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._started:
            raise RuntimeError("Source not started")
        status, transfer = self._ai.get_scan_status()
        n_total = int(transfer.current_total_count // self._channels)
        n_new = n_total - self._next_sample
        if n_new <= 0:
            return None, None

        start = self._next_sample
        end = start + n_new
        chunks = []
        for ch in range(self._channels):
            off = ch * self._samples_per_channel
            chunks.append(self._buffer[off + start : off + end])
        data = np.column_stack(chunks)

        timestamps = np.arange(start, end) / self._fs
        self._next_sample = end
        return timestamps, data

    def stop(self) -> None:
        if self._started and self._ai is not None:
            try:
                self._ai.scan_stop()
            except Exception:
                pass
        if self._daq is not None:
            try:
                self._daq.disconnect()
            except Exception:
                pass
            try:
                self._daq.release()
            except Exception:
                pass
        self._started = False
        self._daq = None
        self._ai = None

    @property
    def sampling_rate(self) -> float:
        return self._fs

    @property
    def n_channels(self) -> int:
        return self._channels

