"""Real-time collection of synchronized sEMG + hand-angle data."""

import sys
import time
import argparse
from collections import deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

from src.data.realtime_emg import USB1608GXSource, SimulatedEMGSource
from src.data.realtime_angle import RealTimeAngleSource
from src.data.data_sync import DataSynchronizer
from src.data.angle_detection import HandAngleDetector
from src.utils import ensure_dir, save_json, split_data, normalize_signals, apply_normalization


def collect(duration, sampling_rate, video_fps, camera_index, simulate, video_file, output_dir, preview=False):
    ensure_dir(output_dir)

    if simulate:
        emg_source = SimulatedEMGSource(sampling_rate=sampling_rate, n_channels=3)
        print("Using SIMULATED EMG source")
    else:
        emg_source = USB1608GXSource(
            sampling_rate=sampling_rate, n_channels=3, duration=duration
        )
        print("Using USB-1608GX EMG source")

    emg_source.start()
    emg_start = time.perf_counter()

    if video_file:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file {video_file}")
        angle_source = None
        use_file = True
    else:
        angle_source = RealTimeAngleSource(camera_index=camera_index, fps=video_fps)
        angle_source.start()
        use_file = False

    print(f"Collecting for {duration} seconds...")
    emg_chunks = []
    video_times = []
    video_angles = []

    emg_preview = deque(maxlen=200)

    t0 = time.perf_counter()

    while (time.perf_counter() - t0) < duration:
        ts, data = emg_source.read_available()
        if ts is not None and data is not None and len(ts) > 0:
            ts_collection = ts + (emg_start - t0)
            emg_chunks.append((ts_collection, data))
            for row in data:
                emg_preview.append(row)

        if use_file:
            pass
        else:
            ts_v, angle, frame, landmarks = angle_source.read_frame()
            if angle is not None:
                video_times.append(ts_v - t0)
                video_angles.append(float(angle))

            if preview and frame is not None:
                disp = frame.copy()
                if landmarks is not None:
                    angle_source.detector._draw_landmarks(disp, landmarks)
                h, w = disp.shape[:2]
                strip_h = 80
                y0 = h - strip_h
                cv2.rectangle(disp, (0, y0), (w, h), (0, 0, 0), -1)
                if len(emg_preview) > 1:
                    arr = np.asarray(emg_preview)
                    for c in range(arr.shape[1]):
                        ch = arr[:, c]
                        mn, mx = float(ch.min()), float(ch.max())
                        rng = (mx - mn) if (mx - mn) > 1e-9 else 1.0
                        norm = (ch - mn) / rng
                        pts = [(int(x * (w - 1) / (len(ch) - 1)), int(y0 + (1 - norm[x]) * (strip_h - 4) + 2)) for x in range(len(ch))]
                        cv2.polylines(disp, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 1)
                label = "NA" if angle is None else f"{angle:.1f}"
                cv2.putText(disp, f"OpenCV angle: {label} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(disp, f"EMG samples: {sum(len(c[1]) for c in emg_chunks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow("Collection Preview", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        time.sleep(0.005)

    if preview:
        cv2.destroyAllWindows()

    emg_source.stop()
    if not use_file and angle_source is not None:
        angle_source.stop()
    if use_file:
        cap.release()

    if not emg_chunks:
        raise RuntimeError("No EMG samples collected")
    emg_timestamps = np.concatenate([c[0] for c in emg_chunks])
    raw_emg = np.concatenate([c[1] for c in emg_chunks], axis=0)
    print(f"Collected EMG: {raw_emg.shape}")

    from src.data.emg_acquisition import EMGAcquisition
    emg_acq = EMGAcquisition(
        sampling_rate=sampling_rate,
        channels=raw_emg.shape[1],
    )
    processed_emg = emg_acq.preprocess_pipeline(
        raw_emg, normalize_method="max", apply_voltage_conversion=False
    )
    print(f"Processed EMG: {processed_emg.shape}")
    if use_file:
        det = HandAngleDetector(static_image_mode=False, max_hands=1)
        cap2 = cv2.VideoCapture(video_file)
        fps = cap2.get(cv2.CAP_PROP_FPS) or video_fps
        v_times = []
        v_angles = []
        fi = 0
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            res = det.process_frame(frame)
            t = fi / fps
            if res is not None:
                v_times.append(t)
                v_angles.append(float(res["angles"]["wrist_angle"]))
            fi += 1
        cap2.release()
        det.close()
        video_times = v_times
        video_angles = v_angles
        print(f"Extracted {len(video_angles)} angles from video file")

    if not video_times:
        raise RuntimeError("No angle samples collected")

    video_times = np.asarray(video_times, dtype=float)
    video_angles = np.asarray(video_angles, dtype=float)

    # Align video timeline to collection duration if needed
    if use_file and video_times.size > 0 and video_times[-1] > 0:
        # keep as-is; interpolation handles range
        pass
    vid_speed = np.gradient(video_angles, 1.0 / video_fps)
    sync = DataSynchronizer(emg_sampling_rate=sampling_rate, video_fps=video_fps)
    synced_p0 = sync._interpolate_angles(video_times, video_angles, emg_timestamps)
    synced_s0 = sync._interpolate_angles(video_times, vid_speed, emg_timestamps)
    synced_p0 = np.asarray(synced_p0).reshape(-1)
    synced_s0 = np.asarray(synced_s0).reshape(-1)

    # --- EMD shift: pair EMG with a future vision angle (+2 frames) ---
    shift = int(round(2.0 * sampling_rate / video_fps))
    synced_p = np.roll(synced_p0, -shift)
    synced_s = np.roll(synced_s0, -shift)
    if shift > 0:
        synced_p[-shift:] = np.nan
        synced_s[-shift:] = np.nan
    valid = ~np.isnan(synced_p) & ~np.isnan(synced_s)
    if not valid.any():
        raise RuntimeError('No valid angle samples after sync/shift. Re-collect with hand in frame.')
    synced_emg = processed_emg[valid]
    synced_p = synced_p[valid]
    synced_s = synced_s[valid]
    print(f'Synchronized: EMG {synced_emg.shape}, p {synced_p.shape}, s {synced_s.shape}')

    # --- Feature extraction + 4-window stacking (32-dim) ---
    acq = EMGAcquisition(sampling_rate=int(sampling_rate), channels=synced_emg.shape[1])
    X, centers = acq.extract_stacked_features(synced_emg, window_size=320, hop=40, stack=4)
    if len(X) == 0:
        raise RuntimeError('Too few samples for feature stacking.')
    p_labels = synced_p[centers]
    s_labels = synced_s[centers]
    Y = np.column_stack([p_labels, s_labels])

    windows_emg, windows_angles = X, Y
    print(f'Features: X={X.shape}  Y={Y.shape} (p, s)')


    # --- Feature extraction + 4-window stacking (32-dim) ---
    acq = EMGAcquisition(sampling_rate=int(sampling_rate), channels=synced_emg.shape[1])
    X, centers = acq.extract_stacked_features(synced_emg, window_size=320, hop=40, stack=4)
    if len(X) == 0:
        raise RuntimeError("Too few samples for feature stacking.")
    p_labels = synced_p[centers]
    s_labels = synced_s[centers]
    Y = np.column_stack([p_labels, s_labels])

    windows_emg, windows_angles = X, Y
    print(f"Features: X={X.shape}  Y={Y.shape} (p, s)")

    session_dir = Path(output_dir) / ("session_" + str(int(time.time())))
    ensure_dir(str(session_dir))
    np.save(session_dir / "raw_emg.npy", raw_emg)
    np.save(session_dir / "emg_timestamps.npy", emg_timestamps)
    np.save(session_dir / "video_timestamps.npy", video_times)
    np.save(session_dir / "video_angles.npy", video_angles)

    window_size = 320
    overlap = 0.875
    sync = DataSynchronizer(emg_sampling_rate=sampling_rate, video_fps=video_fps)
    print(f"Created {len(windows_emg)} windows")

    train_emg, train_angles, val_emg, val_angles, test_emg, test_angles = split_data(
        windows_emg, windows_angles, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True
    )
    print(f"Train: {len(train_emg)}, Val: {len(val_emg)}, Test: {len(test_emg)}")

    train_emg_norm, norm_params = normalize_signals(train_emg, method="minmax", axis=(0, 1))
    val_emg_norm = apply_normalization(val_emg, norm_params, method="minmax")
    test_emg_norm = apply_normalization(test_emg, norm_params, method="minmax")

    processed_dir = Path("data/processed")
    ensure_dir(str(processed_dir))
    np.save(processed_dir / "train_emg.npy", train_emg_norm)
    np.save(processed_dir / "train_angles.npy", train_angles)
    np.save(processed_dir / "val_emg.npy", val_emg_norm)
    np.save(processed_dir / "val_angles.npy", val_angles)
    np.save(processed_dir / "test_emg.npy", test_emg_norm)
    np.save(processed_dir / "test_angles.npy", test_angles)

    metadata = {
        "duration": duration,
        "sampling_rate": sampling_rate,
        "video_fps": video_fps,
        "window_size": window_size,
        "overlap": overlap,
        "angle_names": ["p", "s"],
        "num_channels": int(raw_emg.shape[1]),
        "num_angles": 2,
        "num_windows": int(len(windows_emg)),
        "norm_params": {
            "min": norm_params["min"].tolist(),
            "max": norm_params["max"].tolist(),
            "range": norm_params["range"].tolist(),
        },
    }
    save_json(metadata, str(processed_dir / "metadata.json"))
    save_json(metadata, str(session_dir / "metadata.json"))

    print("Collection complete.")
    print(f"Raw session saved to: {session_dir}")
    print(f"Processed data saved to: {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect real-time sEMG + angle data")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sampling_rate", type=float, default=1200.0)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/raw_realtime")
    parser.add_argument("--preview", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    collect(
        duration=args.duration,
        sampling_rate=args.sampling_rate,
        video_fps=args.video_fps,
        camera_index=args.camera,
        simulate=args.simulate,
        video_file=args.video_file,
        output_dir=args.output_dir,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()

