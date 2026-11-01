"""Live sEMG -> hand angle inference with on-screen display."""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import torch
import yaml

from src.data.realtime_emg import USB1608GXSource, SimulatedEMGSource
from src.data.realtime_angle import RealTimeAngleSource
from src.data.realtime_preprocess import RealtimePreprocessor
from src.models.model_architectures import get_model
from src.utils import load_json


def main():
    parser = argparse.ArgumentParser(description="Live sEMG to angle inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output", type=str, default="data/live_inference")
    parser.add_argument("--sampling_rate", type=float, default=1200.0)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load metadata (norm params)
    metadata = load_json(args.metadata)
    norm_params = metadata.get("norm_params", {})

    # Load model
    model = get_model(
        config["model_name"],
        input_channels=config["input_channels"],
        sequence_length=config["sequence_length"],
        num_angles=config["num_angles"],
        hidden_dim=config.get("hidden_dim", 128),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # EMG source
    if args.simulate:
        emg_source = SimulatedEMGSource(sampling_rate=args.sampling_rate, n_channels=config["input_channels"])
    else:
        emg_source = USB1608GXSource(sampling_rate=args.sampling_rate, n_channels=config["input_channels"], duration=args.duration)
    emg_source.start()

    # Angle source (OpenCV)
    angle_source = RealTimeAngleSource(camera_index=args.camera, fps=config.get("video_fps", 30))
    angle_source.start()

    preprocessor = RealtimePreprocessor(
        sampling_rate=args.sampling_rate,
        norm_params=norm_params,
    )

    preds = []
    pred_times = []
    opencv_angles = []
    opencv_times = []

    new_samples_since_pred = 0
    t0 = time.perf_counter()

    print(f"Running live inference for {args.duration} s. Press 'q' to stop early.")
    while (time.perf_counter() - t0) < args.duration:
        ts, data = emg_source.read_available()
        if ts is not None and data is not None and len(ts) > 0:
            preprocessor.process(data)
            new_samples_since_pred += len(ts)

        # Predict every 100 new samples
        if new_samples_since_pred >= 100:
            window = preprocessor.get_window(config["sequence_length"])
            if window is not None:
                x = torch.FloatTensor(window).unsqueeze(0).to(device)  # (1, seq, ch)
                with torch.no_grad():
                    y = model(x)
                pred = float(y.cpu().numpy().flatten()[0])
                preds.append(pred)
                pred_times.append(time.perf_counter() - t0)
            new_samples_since_pred = 0

        # Webcam frame + overlay
        ts_v, angle, frame, landmarks = angle_source.read_frame()
        if frame is not None:
            if landmarks is not None:
                angle_source.detector._draw_landmarks(frame, landmarks)
            if angle is not None:
                opencv_angles.append(float(angle))
                opencv_times.append(ts_v - t0)
            # Overlay last prediction
            if preds:
                cv2.putText(frame, f"Pred: {preds[-1]:.1f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if angle is not None:
                cv2.putText(frame, f"OpenCV: {angle:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("Live Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(0.005)

    # Cleanup
    emg_source.stop()
    angle_source.stop()
    cv2.destroyAllWindows()

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", np.asarray(preds))
    np.save(out_dir / "pred_times.npy", np.asarray(pred_times))
    np.save(out_dir / "opencv_angles.npy", np.asarray(opencv_angles))
    np.save(out_dir / "opencv_times.npy", np.asarray(opencv_times))
    print(f"Saved {len(preds)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
