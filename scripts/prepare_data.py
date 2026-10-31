"""
Data Preparation Script

This script processes raw sEMG and video data to create training datasets.
Adapt this script to your specific data format and pipeline.
"""

import sys
sys.path.append('..')

import numpy as np
from pathlib import Path
from src.data import EMGAcquisition, HandAngleDetector, DataSynchronizer
from src.utils import split_data, normalize_signals, ensure_dir, save_json


def main():
    print("="*60)
    print("sEMG to Hand Angle Estimation - Data Preparation")
    print("="*60)

    # Configuration
    emg_sampling_rate = 1000
    video_fps = 30
    window_size = 200
    overlap = 0.5

    # Paths - ADAPT THESE TO YOUR DATA
    raw_emg_path = 'data/raw/emg_data.npy'
    video_path = 'data/raw/hand_video.mp4'
    output_dir = 'data/processed'

    print("\n1. Loading and preprocessing sEMG data...")
    emg_acq = EMGAcquisition(
        sampling_rate=emg_sampling_rate,
        channels=8,
        notch_freq=50.0
    )

    # Load raw EMG data
    # TODO: Implement based on your data format
    # raw_emg = emg_acq.load_emg_data(raw_emg_path)
    # print(f"Loaded EMG data: shape = {raw_emg.shape}")

    # Preprocess
    # processed_emg = emg_acq.preprocess_pipeline(raw_emg)
    # print(f"Preprocessed EMG data: shape = {processed_emg.shape}")

    print("\n2. Extracting hand angles from video...")
    detector = HandAngleDetector(static_image_mode=False, max_hands=1)

    # Process video
    # angles, angle_names = detector.process_video(
    #     video_path,
    #     output_path=f'{output_dir}/annotated_video.mp4'
    # )
    # print(f"Extracted {len(angle_names)} angles from {len(angles)} frames")
    # print("Angle names:", angle_names)

    print("\n3. Synchronizing sEMG and angle data...")
    sync = DataSynchronizer(emg_sampling_rate=emg_sampling_rate, video_fps=video_fps)

    # Synchronize
    # synced_emg, synced_angles = sync.sync_data(processed_emg, angles)
    # print(f"Synchronized data: EMG shape = {synced_emg.shape}, Angles shape = {synced_angles.shape}")

    print("\n4. Creating training windows...")
    # windows_emg, windows_angles = sync.create_windows(
    #     synced_emg,
    #     synced_angles,
    #     window_size=window_size,
    #     overlap=overlap
    # )
    # print(f"Created {len(windows_emg)} windows")

    print("\n5. Splitting into train/val/test...")
    # train_emg, train_angles, val_emg, val_angles, test_emg, test_angles = split_data(
    #     windows_emg,
    #     windows_angles,
    #     train_ratio=0.7,
    #     val_ratio=0.15,
    #     test_ratio=0.15,
    #     shuffle=True
    # )
    # print(f"Train: {len(train_emg)}, Val: {len(val_emg)}, Test: {len(test_emg)}")

    print("\n6. Normalizing data...")
    # train_emg_norm, norm_params = normalize_signals(train_emg, method='minmax')
    # val_emg_norm = normalize_signals(val_emg, params=norm_params, method='minmax')
    # test_emg_norm = normalize_signals(test_emg, params=norm_params, method='minmax')

    print("\n7. Saving processed data...")
    ensure_dir(output_dir)

    # Save data
    # np.save(f'{output_dir}/train_emg.npy', train_emg_norm)
    # np.save(f'{output_dir}/train_angles.npy', train_angles)
    # np.save(f'{output_dir}/val_emg.npy', val_emg_norm)
    # np.save(f'{output_dir}/val_angles.npy', val_angles)
    # np.save(f'{output_dir}/test_emg.npy', test_emg_norm)
    # np.save(f'{output_dir}/test_angles.npy', test_angles)

    # Save metadata
    # metadata = {
    #     'window_size': window_size,
    #     'overlap': overlap,
    #     'emg_sampling_rate': emg_sampling_rate,
    #     'video_fps': video_fps,
    #     'angle_names': angle_names,
    #     'num_channels': train_emg.shape[2],
    #     'num_angles': train_angles.shape[1],
    #     'norm_params': {
    #         'min': norm_params['min'].tolist(),
    #         'max': norm_params['max'].tolist(),
    #         'range': norm_params['range'].tolist()
    #     }
    # }
    # save_json(metadata, f'{output_dir}/metadata.json')

    print("\nData preparation complete!")
    print("\nNext steps:")
    print("1. Train model: python src/models/train.py --config configs/default_config.yaml --data_dir data/processed")
    print("2. Evaluate model: python src/models/evaluate.py --checkpoint data/models/best_model.pth --test_data data/processed")


if __name__ == '__main__':
    main()
