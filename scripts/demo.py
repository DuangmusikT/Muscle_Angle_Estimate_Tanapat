"""
Demo script showing basic usage of the sEMG to angle estimation pipeline.
This generates synthetic data for demonstration purposes.
"""

import sys
sys.path.append('..')

import numpy as np
from src.data import EMGAcquisition, HandAngleDetector, DataSynchronizer
from src.utils import split_data, normalize_signals, apply_normalization
from src.visualization import plot_signal_comparison, plot_prediction_comparison


def generate_synthetic_emg(n_samples: int, n_channels: int = 8) -> np.ndarray:
    """Generate synthetic EMG-like signals."""
    # Create random noise with muscle-like frequency content
    emg = np.random.randn(n_samples, n_channels)

    # Add some periodic components to simulate muscle activation
    t = np.linspace(0, n_samples / 1000, n_samples)
    for i in range(n_channels):
        freq = 10 + i * 5  # Different frequencies per channel
        emg[:, i] += 0.5 * np.sin(2 * np.pi * freq * t)

    return emg


def generate_synthetic_angles(n_samples: int, n_angles: int = 1) -> np.ndarray:
    """Generate synthetic angle trajectories."""
    angles = np.zeros((n_samples, n_angles))

    # Create smooth angle trajectories
    t = np.linspace(0, n_samples / 30, n_samples)
    for i in range(n_angles):
        freq = 0.5 + i * 0.1
        angles[:, i] = 30 * np.sin(2 * np.pi * freq * t) + 45  # Angles between 15-75 degrees

    return angles


def main():
    print("="*60)
    print("sEMG to Hand Angle Estimation - Demo")
    print("="*60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    n_samples = 10000  # 10 seconds at 1000 Hz
    n_channels = 8
    n_angles = 1

    raw_emg = generate_synthetic_emg(n_samples, n_channels)
    angles = generate_synthetic_angles(n_samples, n_angles)

    print(f"Generated EMG data: shape = {raw_emg.shape}")
    print(f"Generated angle data: shape = {angles.shape}")

    # Preprocess EMG
    print("\n2. Preprocessing EMG data...")
    emg_acq = EMGAcquisition(sampling_rate=1000, channels=n_channels)
    processed_emg = emg_acq.preprocess_pipeline(raw_emg)
    print(f"Preprocessed EMG data: shape = {processed_emg.shape}")

    # Synchronize (with synthetic data, they're already synced)
    print("\n3. Synchronizing data...")
    sync = DataSynchronizer(emg_sampling_rate=1000, video_fps=30)
    synced_emg, synced_angles = sync.sync_data(processed_emg, angles)
    print(f"Synchronized data: EMG shape = {synced_emg.shape}, Angles shape = {synced_angles.shape}")

    # Create windows
    print("\n4. Creating training windows...")
    window_size = 200
    overlap = 0.5
    windows_emg, windows_angles = sync.create_windows(
        synced_emg,
        synced_angles,
        window_size=window_size,
        overlap=overlap
    )
    print(f"Created {len(windows_emg)} windows")

    # Split data
    print("\n5. Splitting into train/val/test...")
    train_emg, train_angles, val_emg, val_angles, test_emg, test_angles = split_data(
        windows_emg,
        windows_angles,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True
    )
    print(f"Train: {len(train_emg)}, Val: {len(val_emg)}, Test: {len(test_emg)}")

    # Normalize
    print("\n6. Normalizing data...")
    train_emg_norm, norm_params = normalize_signals(train_emg, method='minmax')
    val_emg_norm = apply_normalization(val_emg, norm_params, method='minmax')
    test_emg_norm = apply_normalization(test_emg, norm_params, method='minmax')

    # Visualize
    print("\n7. Visualizing data...")
    timestamps = np.arange(len(synced_emg)) / 1000
    plot_signal_comparison(
        synced_emg[:, :4],
        synced_angles[:, :1],
        timestamps,
        save_path='results/demo_signal_comparison.png'
    )

    print("\nDemo complete!")
    print("\nGenerated synthetic data is ready for experimentation.")
    print("Visualization saved to results/demo_signal_comparison.png")


if __name__ == '__main__':
    main()
