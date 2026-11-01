"""Reprocess a saved raw_realtime session into data/processed/ without re-collecting."""

import sys
import argparse
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.emg_acquisition import EMGAcquisition
from src.data.data_sync import DataSynchronizer
from src.utils import ensure_dir, save_json, split_data, normalize_signals, apply_normalization


def main():
    parser = argparse.ArgumentParser(description='Reprocess a raw_realtime session')
    parser.add_argument('--session_dir', type=str, required=True, help='Path to session_ directory')
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise SystemExit(f'Session dir not found: {session_dir}')

    raw_emg = np.load(session_dir / 'raw_emg.npy')
    raw_emg = raw_emg[:, :2]
    emg_ts = np.load(session_dir / 'emg_timestamps.npy')
    vid_ts = np.load(session_dir / 'video_timestamps.npy')
    vid_ang = np.load(session_dir / 'video_angles.npy')

    meta = {}
    meta_path = session_dir / 'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    fs = float(meta.get('sampling_rate', 1200.0))
    video_fps = int(meta.get('video_fps', 30))
    n_channels = raw_emg.shape[1]

    print(f'Session: {session_dir}')
    print(f'Raw EMG: {raw_emg.shape}  fs={fs}  channels={n_channels}')
    print(f'Video angles: {vid_ang.shape}')

    # --- Preprocess EMG (SOS causal, no voltage conversion - DAQ returns volts) ---
    acq = EMGAcquisition(sampling_rate=int(fs), channels=n_channels, gain=816.0, ref_voltage=0.0)
    processed_emg = acq.preprocess_pipeline(raw_emg, normalize_method='max', apply_voltage_conversion=False)
    print(f'Processed EMG: {processed_emg.shape}')

    # --- Interpolate angles to EMG timestamps ---
    sync = DataSynchronizer(emg_sampling_rate=int(fs), video_fps=video_fps)
    synced_angles = sync._interpolate_angles(vid_ts, vid_ang.reshape(-1, 1), emg_ts)  # (N,1)

    # --- EMD shift: pair EMG with a future vision angle (+2 frames) ---
    shift = int(round(2.0 * fs / video_fps))
    shifted = np.roll(synced_angles, -shift, axis=0)
    if shift > 0:
        shifted[-shift:] = np.nan
    valid = ~np.isnan(synced_angles).any(axis=1) & ~np.isnan(shifted).any(axis=1)
    if not valid.any():
        raise SystemExit('No valid (non-NaN) angle samples after sync/shift. Re-collect with hand in frame.')
    synced_emg = processed_emg[valid]
    synced_p = shifted[valid, 0]
    print(f'Synced: EMG {synced_emg.shape}  p-target {synced_p.shape}')

    # Velocity target (smoothed central difference) in deg/s
    dt = 1.0 / fs
    synced_s = np.gradient(synced_p, dt)

    # --- Feature extraction + 4-window stacking (32-dim) ---
    acq = EMGAcquisition(sampling_rate=int(fs), channels=synced_emg.shape[1])
    X, centers = acq.extract_stacked_features(synced_emg, window_size=320, hop=40, stack=4)
    if len(X) == 0:
        raise SystemExit('Too few samples for feature stacking.')
    p_labels = synced_p[centers]
    s_labels = synced_s[centers]
    Y = np.column_stack([p_labels, s_labels])
    print(f'Features: X={X.shape}  Y={Y.shape} (p, s)')

    windows_emg, windows_angles = X, Y
    if len(windows_emg) < 5:
        raise SystemExit('Too few windows to split. Collect more data.')

    # --- Split ---
    train_emg, train_angles, val_emg, val_angles, test_emg, test_angles = split_data(
        windows_emg, windows_angles, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True
    )
    print(f'Train: {len(train_emg)}, Val: {len(val_emg)}, Test: {len(test_emg)}')

    # --- Normalize (per-channel) ---
    train_emg_norm, norm_params = normalize_signals(train_emg, method='minmax', axis=(0, 1))
    val_emg_norm = apply_normalization(val_emg, norm_params, method='minmax')
    test_emg_norm = apply_normalization(test_emg, norm_params, method='minmax')

    # --- Save ---
    out = Path(args.output_dir)
    ensure_dir(str(out))
    np.save(out / 'train_emg.npy', train_emg_norm)
    np.save(out / 'train_angles.npy', train_angles)
    np.save(out / 'val_emg.npy', val_emg_norm)
    np.save(out / 'val_angles.npy', val_angles)
    np.save(out / 'test_emg.npy', test_emg_norm)
    np.save(out / 'test_angles.npy', test_angles)

    metadata = {
        'session_dir': str(session_dir),
        'sampling_rate': fs,
        'video_fps': video_fps,
        'window_size': args.window_size,
        'overlap': args.overlap,
        'angle_names': meta.get('angle_names', ['wrist_angle']),
        'channel_names': meta.get('channel_names', [f'Ch{i}' for i in range(n_channels)]),
        'num_channels': n_channels,
        'num_angles': int(synced_angles.shape[1]),
        'num_windows': int(len(windows_emg)),
        'norm_params': {
            'min': norm_params['min'].tolist(),
            'max': norm_params['max'].tolist(),
            'range': norm_params['range'].tolist(),
        },
    }
    save_json(metadata, str(out / 'metadata.json'))

    print('\nReprocessing complete.')
    print(f'Processed data saved to: {out}')


if __name__ == '__main__':
    main()
