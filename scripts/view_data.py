"""View past EMG/angle data (processed or raw session) without side effects."""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.utils import load_json


def _finish(fig, save_dir, filename, show):
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=150, bbox_inches='tight')
        print(f"Saved {out / filename}")
    if show:
        plt.show()
    plt.close(fig)


def summarize_processed(data_dir, train_emg, train_ang, val_emg, val_ang, test_emg, test_ang, meta):
    print("=" * 60)
    print("PROCESSED DATA SUMMARY")
    print("=" * 60)
    print(f"Directory: {data_dir}")
    for name, arr in [("train_emg", train_emg), ("train_angles", train_ang),
                      ("val_emg", val_emg), ("val_angles", val_ang),
                      ("test_emg", test_emg), ("test_angles", test_ang)]:
        print(f"  {name:14s} shape={str(arr.shape):20s} dtype={arr.dtype}")
    if meta:
        print(f"  sampling_rate : {meta.get('emg_sampling_rate', meta.get('sampling_rate', 'NA'))}")
        print(f"  video_fps     : {meta.get('video_fps', 'NA')}")
        print(f"  window_size   : {meta.get('window_size', 'NA')}")
        print(f"  overlap       : {meta.get('overlap', 'NA')}")
        print(f"  angle_names   : {meta.get('angle_names', 'NA')}")
    print()
    for name, arr in [("train_emg", train_emg), ("val_emg", val_emg), ("test_emg", test_emg)]:
        print(f"  {name}: min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
    for name, arr in [("train_angles", train_ang), ("val_angles", val_ang), ("test_angles", test_ang)]:
        if arr.ndim == 2:
            for j in range(arr.shape[1]):
                print(f"  {name}[:,{j}]: min={arr[:,j].min():.4f} max={arr[:,j].max():.4f} mean={arr[:,j].mean():.4f}")
        else:
            print(f"  {name}: min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
    print("=" * 60)


def plot_processed(train_emg, train_ang, save_dir, max_windows, show):
    # Feature-based data: (n, n_features) + (n, n_targets)
    if train_emg.ndim == 2:
        n_feat = train_emg.shape[1]
        n_tgt = train_ang.shape[1] if train_ang.ndim == 2 else 1
        ang = train_ang.reshape(-1, n_tgt)

        # p and s trajectories
        fig, axes = plt.subplots(n_tgt, 1, figsize=(10, 3 * n_tgt), sharex=True)
        if n_tgt == 1:
            axes = [axes]
        names = ['p (position)', 's (speed)']
        for j in range(n_tgt):
            axes[j].plot(ang[:, j], label=names[j] if j < len(names) else f'target {j}')
            axes[j].set_title(f'{names[j] if j < len(names) else f"target {j}"} per window (train)')
            axes[j].set_ylabel('Value')
            axes[j].grid(True, alpha=0.3)
            axes[j].legend()
        axes[-1].set_xlabel('Window index')
        plt.tight_layout()
        _finish(fig, save_dir, 'targets_trajectory.png', show)

        # distributions
        fig, axes = plt.subplots(1, n_tgt, figsize=(5 * n_tgt, 4))
        if n_tgt == 1:
            axes = [axes]
        for j in range(n_tgt):
            axes[j].hist(ang[:, j], bins=30, edgecolor='black', alpha=0.7)
            axes[j].set_title(f'{names[j] if j < len(names) else f"target {j}"} distribution')
            axes[j].set_xlabel('Value')
        plt.tight_layout()
        _finish(fig, save_dir, 'targets_hist.png', show)

        # feature-target correlation heatmap
        try:
            import seaborn as sns
            corr = np.corrcoef(train_emg.T, ang.T)[:n_feat, n_feat:]
            fig, ax = plt.subplots(figsize=(6, 8))
            sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax,
                        xticklabels=[f't{j}' for j in range(n_tgt)],
                        yticklabels=[f'f{i}' for i in range(n_feat)])
            ax.set_title('Feature-target correlation (train)')
            plt.tight_layout()
            _finish(fig, save_dir, 'feature_corr.png', show)
        except Exception as e:
            print(f'  (skipped heatmap: {e})')
        return

    # Raw-window data: (n, window, channels)
    n = min(max_windows, len(train_emg))
    n_ch = train_emg.shape[2]
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        for c in range(n_ch):
            axes[i].plot(train_emg[i, :, c], label=f'Ch{c}', alpha=0.8)
        a0 = train_ang[i]
        a0 = a0.flatten()
        axes[i].set_title(f'Window {i} | targets={a0}')
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, save_dir, 'emg_windows.png', show)

    # angle trajectory
    ang = train_ang.reshape(len(train_ang), -1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(ang.shape[1]):
        ax.plot(ang[:, j], label=f'target {j}')
    ax.set_title('Targets per window (train)')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _finish(fig, save_dir, 'targets_trajectory.png', show)


def view_processed(data_dir, save_dir, max_windows, show):
    data_dir = Path(data_dir)
    train_emg = np.load(data_dir / 'train_emg.npy')
    train_ang = np.load(data_dir / 'train_angles.npy')
    val_emg = np.load(data_dir / 'val_emg.npy')
    val_ang = np.load(data_dir / 'val_angles.npy')
    test_emg = np.load(data_dir / 'test_emg.npy')
    test_ang = np.load(data_dir / 'test_angles.npy')

    meta = None
    meta_path = data_dir / 'metadata.json'
    if meta_path.exists():
        meta = load_json(str(meta_path))

    summarize_processed(data_dir, train_emg, train_ang, val_emg, val_ang, test_emg, test_ang, meta)
    plot_processed(train_emg, train_ang, save_dir, max_windows, show)


def view_session(session_dir, save_dir, show):
    session_dir = Path(session_dir)
    print("=" * 60)
    print("RAW SESSION SUMMARY")
    print("=" * 60)
    print(f"Directory: {session_dir}")
    raw_emg = np.load(session_dir / 'raw_emg.npy') if (session_dir / 'raw_emg.npy').exists() else None
    emg_ts = np.load(session_dir / 'emg_timestamps.npy') if (session_dir / 'emg_timestamps.npy').exists() else None
    vid_ts = np.load(session_dir / 'video_timestamps.npy') if (session_dir / 'video_timestamps.npy').exists() else None
    vid_ang = np.load(session_dir / 'video_angles.npy') if (session_dir / 'video_angles.npy').exists() else None

    if raw_emg is not None:
        print(f"  raw_emg: {raw_emg.shape} {raw_emg.dtype}")
    if emg_ts is not None:
        print(f"  emg_timestamps: {emg_ts.shape} dur={emg_ts[-1]-emg_ts[0]:.2f}s")
    if vid_ts is not None:
        print(f"  video_timestamps: {vid_ts.shape}")
    if vid_ang is not None:
        print(f"  video_angles: {vid_ang.shape} min={vid_ang.min():.2f} max={vid_ang.max():.2f}")
    meta_path = session_dir / 'metadata.json'
    if meta_path.exists():
        m = load_json(str(meta_path))
        print(f"  sampling_rate: {m.get('sampling_rate', 'NA')}  video_fps: {m.get('video_fps', 'NA')}")
    print("=" * 60)

    if raw_emg is not None:
        t = emg_ts if emg_ts is not None else np.arange(len(raw_emg))
        fig, ax = plt.subplots(figsize=(12, 4))
        for c in range(raw_emg.shape[1]):
            ax.plot(t, raw_emg[:, c], label=f'Ch{c}', alpha=0.7)
        ax.set_title('Raw EMG')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _finish(fig, save_dir, 'raw_emg.png', show)

    if vid_ts is not None and vid_ang is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(vid_ts, vid_ang, label='OpenCV wrist_angle', color='tab:blue')
        ax.set_title('OpenCV angles over time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        _finish(fig, save_dir, 'video_angles.png', show)

    if raw_emg is not None and vid_ts is not None and vid_ang is not None:
        t = emg_ts if emg_ts is not None else np.arange(len(raw_emg))
        fig, ax1 = plt.subplots(figsize=(12, 5))
        for c in range(raw_emg.shape[1]):
            ax1.plot(t, raw_emg[:, c], alpha=0.4, label=f'Ch{c}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('EMG amplitude')
        ax2 = ax1.twinx()
        ax2.plot(vid_ts, vid_ang, color='red', linewidth=2, label='OpenCV angle')
        ax2.set_ylabel('Angle (deg)')
        fig.suptitle('EMG + OpenCV angle overlay')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.tight_layout()
        _finish(fig, save_dir, 'overlay.png', show)


def main():
    parser = argparse.ArgumentParser(description='View past EMG/angle data')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data_dir', type=str, default=None, help='Processed data directory')
    group.add_argument('--session_dir', type=str, default=None, help='Raw session directory')
    parser.add_argument('--max_windows', type=int, default=8, help='Max EMG windows to plot')
    parser.add_argument('--save_dir', type=str, default=None, help='Save plots to directory (no GUI needed)')
    parser.add_argument('--show', action='store_true', default=True, help='Show plots interactively')
    args = parser.parse_args()

    show = args.show and (args.save_dir is None)

    if args.session_dir:
        view_session(args.session_dir, args.save_dir, show)
    else:
        data_dir = args.data_dir or 'data/processed'
        view_processed(data_dir, args.save_dir, args.max_windows, show)


if __name__ == '__main__':
    main()
