"""
Data Preparation Script

This script processes raw sEMG and video data to create training datasets.
Supports both TDMS format (from sEMG_Robot_Demo) and CSV format (from Senkouka).
"""

import sys
sys.path.append('..')

import numpy as np
from pathlib import Path
from src.data import EMGAcquisitionLegacy, EMGAcquisitionCSV, HandAngleDetector, DataSynchronizer
from src.utils import split_data, normalize_signals, ensure_dir, save_json


def prepare_tdms_data(tdms_path: str, video_path: str, output_dir: str):
    """Prepare data from TDMS format."""
    print("\n=== TDMS Format Processing ===")

    # Configuration
    emg_sampling_rate = 2000  # From your TDMS project
    video_fps = 30
    window_size = 200
    overlap = 0.5

    print("\n1. Loading TDMS file...")
    emg_acq = EMGAcquisitionLegacy(
        sampling_rate=emg_sampling_rate,
        notch_freqs=[50.0, 100.0],
        lowpass_cutoff=500.0
    )

    raw_emg, channel_names = emg_acq.load_tdms(tdms_path)
    print(f"Loaded EMG data: shape = {raw_emg.shape}")
    print(f"Channels: {channel_names}")

    print("\n2. Preprocessing EMG data...")
    processed_emg, timestamps = emg_acq.preprocess_pipeline(raw_emg, feature_type='rms')
    print(f"Preprocessed EMG data: shape = {processed_emg.shape}")

    print("\n3. Extracting hand angles from video...")
    detector = HandAngleDetector(static_image_mode=False, max_hands=1)

    angles, angle_names = detector.process_video(
        video_path,
        output_path=f'{output_dir}/annotated_video.mp4'
    )
    print(f"Extracted {len(angle_names)} angles from {len(angles)} frames")
    print("Angle names:", angle_names)

    print("\n4. Synchronizing sEMG and angle data...")
    sync = DataSynchronizer(emg_sampling_rate=emg_sampling_rate, video_fps=video_fps)

    synced_emg, synced_angles = sync.sync_data(processed_emg, angles, timestamps)
    print(f"Synchronized data: EMG shape = {synced_emg.shape}, Angles shape = {synced_angles.shape}")

    print("\n5. Creating training windows...")
    windows_emg, windows_angles = sync.create_windows(
        synced_emg,
        synced_angles,
        window_size=window_size,
        overlap=overlap
    )
    print(f"Created {len(windows_emg)} windows")

    return windows_emg, windows_angles, angle_names, channel_names, emg_sampling_rate, video_fps, window_size, overlap


def prepare_csv_data(csv_path: str, video_path: str, output_dir: str):
    """Prepare data from CSV format."""
    print("\n=== CSV Format Processing ===")

    # Configuration
    emg_sampling_rate = 10000  # From your Senkouka project
    video_fps = 30
    window_size = 200
    overlap = 0.5

    print("\n1. Loading CSV file...")
    emg_acq = EMGAcquisitionCSV(
        sampling_rate=emg_sampling_rate,
        gain=883.97,
        mid_vol=0.0
    )

    time_array, raw_emg = emg_acq.load_csv(csv_path)
    print(f"Loaded EMG data: shape = {raw_emg.shape}")
    print(f"Duration: {time_array[-1]:.2f} seconds")

    print("\n2. Preprocessing EMG data...")
    processed_emg = emg_acq.apply_filters(raw_emg)
    print(f"Preprocessed EMG data: shape = {processed_emg.shape}")

    print("\n3. Extracting hand angles from video...")
    detector = HandAngleDetector(static_image_mode=False, max_hands=1)

    angles, angle_names = detector.process_video(
        video_path,
        output_path=f'{output_dir}/annotated_video.mp4'
    )
    print(f"Extracted {len(angle_names)} angles from {len(angles)} frames")
    print("Angle names:", angle_names)

    print("\n4. Synchronizing sEMG and angle data...")
    sync = DataSynchronizer(emg_sampling_rate=emg_sampling_rate, video_fps=video_fps)

    synced_emg, synced_angles = sync.sync_data(processed_emg, angles, time_array)
    print(f"Synchronized data: EMG shape = {synced_emg.shape}, Angles shape = {synced_angles.shape}")

    print("\n5. Creating training windows...")
    windows_emg, windows_angles = sync.create_windows(
        synced_emg,
        synced_angles,
        window_size=window_size,
        overlap=overlap
    )
    print(f"Created {len(windows_emg)} windows")

    return windows_emg, windows_angles, angle_names, ['Ch1', 'Ch2', 'Ch3'], emg_sampling_rate, video_fps, window_size, overlap


def main():
    print("="*60)
    print("sEMG to Hand Angle Estimation - Data Preparation")
    print("="*60)

    # Choose data format
    data_format = input("\nChoose data format (tdms/csv): ").strip().lower()

    if data_format == 'tdms':
        tdms_path = input("Enter path to TDMS file: ").strip()
        video_path = input("Enter path to video file: ").strip()
        windows_emg, windows_angles, angle_names, channel_names, fs, fps, window_size, overlap = prepare_tdms_data(
            tdms_path, video_path, 'data/processed'
        )
    elif data_format == 'csv':
        csv_path = input("Enter path to CSV file: ").strip()
        video_path = input("Enter path to video file: ").strip()
        windows_emg, windows_angles, angle_names, channel_names, fs, fps, window_size, overlap = prepare_csv_data(
            csv_path, video_path, 'data/processed'
        )
    else:
        print("Invalid format. Please choose 'tdms' or 'csv'.")
        return

    print("\n6. Splitting into train/val/test...")
    train_emg, train_angles, val_emg, val_angles, test_emg, test_angles = split_data(
        windows_emg,
        windows_angles,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True
    )
    print(f"Train: {len(train_emg)}, Val: {len(val_emg)}, Test: {len(test_emg)}")

    print("\n7. Normalizing data...")
    train_emg_norm, norm_params = normalize_signals(train_emg, method='minmax')
    val_emg_norm = normalize_signals(val_emg, params=norm_params, method='minmax')
    test_emg_norm = normalize_signals(test_emg, params=norm_params, method='minmax')

    print("\n8. Saving processed data...")
    output_dir = 'data/processed'
    ensure_dir(output_dir)

    # Save data
    np.save(f'{output_dir}/train_emg.npy', train_emg_norm)
    np.save(f'{output_dir}/train_angles.npy', train_angles)
    np.save(f'{output_dir}/val_emg.npy', val_emg_norm)
    np.save(f'{output_dir}/val_angles.npy', val_angles)
    np.save(f'{output_dir}/test_emg.npy', test_emg_norm)
    np.save(f'{output_dir}/test_angles.npy', test_angles)

    # Save metadata
    metadata = {
        'data_format': data_format,
        'window_size': window_size,
        'overlap': overlap,
        'emg_sampling_rate': fs,
        'video_fps': fps,
        'angle_names': angle_names,
        'channel_names': channel_names,
        'num_channels': train_emg.shape[2],
        'num_angles': train_angles.shape[1],
        'norm_params': {
            'min': norm_params['min'].tolist(),
            'max': norm_params['max'].tolist(),
            'range': norm_params['range'].tolist()
        }
    }
    save_json(metadata, f'{output_dir}/metadata.json')

    print("\nData preparation complete!")
    print("\nNext steps:")
    print("1. Update config file with num_channels and num_angles from metadata")
    print("2. Train model: python src/models/train.py --config configs/default_config.yaml --data_dir data/processed")
    print("3. Evaluate model: python src/models/evaluate.py --checkpoint data/models/best_model.pth --test_data data/processed")


if __name__ == '__main__':
    main()
