"""
Utility functions for the project.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List
import json


def ensure_dir(path: str):
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, filepath: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def split_data(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: Input data
        labels: Label data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle data before splitting

    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n_samples = data.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices]
    val_labels = labels[val_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def normalize_signals(
    signals: np.ndarray,
    method: str = 'minmax',
    axis: int = 0
) -> Tuple[np.ndarray, dict]:
    """
    Normalize signals using specified method.

    Args:
        signals: Input signals
        method: Normalization method ('minmax', 'zscore', 'max')
        axis: Axis along which to normalize

    Returns:
        Tuple of (normalized_signals, normalization_params)
    """
    if method == 'minmax':
        min_vals = np.min(signals, axis=axis, keepdims=True)
        max_vals = np.max(signals, axis=axis, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        normalized = (signals - min_vals) / range_vals
        params = {'min': min_vals, 'max': max_vals, 'range': range_vals}

    elif method == 'zscore':
        mean_vals = np.mean(signals, axis=axis, keepdims=True)
        std_vals = np.std(signals, axis=axis, keepdims=True)
        std_vals[std_vals == 0] = 1

        normalized = (signals - mean_vals) / std_vals
        params = {'mean': mean_vals, 'std': std_vals}

    elif method == 'max':
        max_vals = np.max(signals, axis=axis, keepdims=True)
        max_vals[max_vals == 0] = 1

        normalized = signals / max_vals
        params = {'max': max_vals}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def apply_normalization(
    signals: np.ndarray,
    params: dict,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Apply previously computed normalization parameters.

    Args:
        signals: Input signals
        params: Normalization parameters
        method: Normalization method used

    Returns:
        Normalized signals
    """
    if method == 'minmax':
        min_vals = params['min']
        range_vals = params['range']
        normalized = (signals - min_vals) / range_vals

    elif method == 'zscore':
        mean_vals = params['mean']
        std_vals = params['std']
        normalized = (signals - mean_vals) / std_vals

    elif method == 'max':
        max_vals = params['max']
        normalized = signals / max_vals

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def create_time_windows(
    data: np.ndarray,
    window_size: int,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Create overlapping time windows from data.

    Args:
        data: Input data (n_samples, ...)
        window_size: Window size in samples
        overlap: Overlap ratio (0-1)

    Returns:
        Windowed data (n_windows, window_size, ...)
    """
    step = int(window_size * (1 - overlap))
    n_samples = data.shape[0]

    windows = []
    for i in range(0, n_samples - window_size + 1, step):
        windows.append(data[i:i + window_size])

    return np.array(windows)


def compute_fft(signal: np.ndarray, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of signal.

    Args:
        signal: Input signal
        sampling_rate: Sampling rate in Hz

    Returns:
        Tuple of (frequencies, magnitudes)
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)
    magnitudes = np.abs(fft_result)

    # Return only positive frequencies
    positive_freq_idx = frequencies >= 0
    return frequencies[positive_freq_idx], magnitudes[positive_freq_idx]
