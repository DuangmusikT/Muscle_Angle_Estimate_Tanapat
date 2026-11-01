"""Real-time oscilloscope-style viewer for raw EMG (3 differential channels)."""

import sys
import time
import argparse
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.data.realtime_emg import USB1608GXSource, SimulatedEMGSource


def main():
    parser = argparse.ArgumentParser(description='Real-time EMG oscilloscope')
    parser.add_argument('--simulate', action='store_true', help='Use simulated EMG source')
    parser.add_argument('--sampling_rate', type=float, default=1200.0)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--window_sec', type=float, default=2.0, help='Display window in seconds')
    parser.add_argument('--duration', type=float, default=0.0, help='Run duration (0 = until Ctrl+C)')
    parser.add_argument('--save', type=str, default=None, help='Save final view to PNG and exit')
    args = parser.parse_args()

    if args.simulate:
        src = SimulatedEMGSource(sampling_rate=args.sampling_rate, n_channels=args.channels)
    else:
        src_duration = args.duration if args.duration > 0 else 3600.0
        src = USB1608GXSource(sampling_rate=args.sampling_rate, n_channels=args.channels, duration=src_duration)

    src.start()

    window_n = int(args.window_sec * args.sampling_rate)
    buffers = [deque(maxlen=window_n) for _ in range(args.channels)]

    plt.ion()
    fig, axes = plt.subplots(args.channels, 1, figsize=(10, 2 * args.channels), sharex=True)
    if args.channels == 1:
        axes = [axes]
    lines = []
    for i, ax in enumerate(axes):
        (line,) = ax.plot([], [], lw=1)
        lines.append(line)
        ax.set_title('Ch%d (raw EMG)' % i)
        ax.set_ylabel('V')
        ax.grid(True, alpha=0.3)
    for ax in axes:
        ax.set_ylim(-1.5, 1.5)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()

    t0 = time.perf_counter()
    try:
        while True:
            if args.duration > 0 and (time.perf_counter() - t0) >= args.duration:
                break
            ts, data = src.read_available()
            if ts is not None and data is not None and len(ts) > 0:
                for i in range(args.channels):
                    for v in data[:, i]:
                        buffers[i].append(v)
                n = len(buffers[0])
                t_axis = (np.arange(n) / args.sampling_rate) - (n / args.sampling_rate)
                for i in range(args.channels):
                    lines[i].set_data(t_axis, np.asarray(buffers[i]))
                    axes[i].relim()
                    axes[i].autoscale_view()
            plt.pause(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print('Saved', args.save)
    plt.ioff()
    plt.close(fig)
    print('Oscilloscope stopped.')


if __name__ == '__main__':
    main()
