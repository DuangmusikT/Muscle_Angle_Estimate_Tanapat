"""Live sEMG -> hand angle inference with both OpenCV and MuJoCo visualization."""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import torch
import yaml

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError(
        "MuJoCo not installed. Install with: pip install mujoco mujoco-mjx"
    )

from src.data.realtime_emg import USB1608GXSource, SimulatedEMGSource
from src.data.realtime_angle import RealTimeAngleSource
from src.data.realtime_preprocess import RealtimePreprocessor
from src.models.model_architectures import get_model
from src.estimation.kalman_observer import KinematicKalmanFilter
from src.utils import load_json


def create_pendulum_xml():
    """Create a simple 1-joint pendulum MuJoCo XML."""
    xml = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 0 1" limited="false"/>
      <geom name="rod" type="capsule" size="0.02" fromto="0 0 0 0 0.5 0" rgba="0.8 0.2 0.2 1" mass="1"/>
      <geom name="bob" type="sphere" size="0.08" pos="0 0.5 0" rgba="0.2 0.6 0.8 1" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def main():
    parser = argparse.ArgumentParser(description="Live sEMG to angle inference with dual visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output", type=str, default="data/dual_inference")
    parser.add_argument("--sampling_rate", type=float, default=1200.0)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load metadata (norm params)
    metadata = load_json(args.metadata)
    norm_params = metadata.get("norm_params", {})

    # Load model
    model_kwargs = dict(
        input_channels=config["input_channels"],
        num_angles=config["num_angles"],
        hidden_dim=config.get("hidden_dim", 128),
    )
    if config["model_name"] == "mlp":
        model_kwargs["input_dim"] = config.get("input_dim", config["input_channels"])
    else:
        model_kwargs["sequence_length"] = config["sequence_length"]
    model = get_model(config["model_name"], **model_kwargs)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # EMG source
    if args.simulate:
        emg_source = SimulatedEMGSource(sampling_rate=args.sampling_rate, n_channels=config.get("emg_channels", 2))
    else:
        emg_source = USB1608GXSource(sampling_rate=args.sampling_rate, n_channels=config.get("emg_channels", 2), duration=args.duration)
    emg_source.start()

    # Angle source (OpenCV)
    angle_source = RealTimeAngleSource(camera_index=args.camera, fps=config.get("video_fps", 30))
    angle_source.start()

    preprocessor = RealtimePreprocessor(
        sampling_rate=args.sampling_rate,
        norm_params=norm_params,
    )

    # Create MuJoCo model
    xml = create_pendulum_xml()
    model_mj = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model_mj)

    # Initial position: pointing upward (180 degrees)
    initial_angle_deg = 180.0
    initial_angle_rad = np.deg2rad(initial_angle_deg - 90)
    data.qpos[0] = initial_angle_rad

    # Start MuJoCo viewer
    with mujoco.viewer.launch_passive(model_mj, data) as viewer:
        print(f"Running dual inference for {args.duration} s. Press 'q' in OpenCV window or close MuJoCo window to stop early.")

        preds = []
        pred_times = []
        opencv_angles = []
        opencv_times = []
        mujoco_angles = []
        mujoco_times = []

        t0 = time.perf_counter()
        last_sim_time = t0

        while (time.perf_counter() - t0) < args.duration:
            # Read EMG
            ts, emg_data = emg_source.read_available()
            if ts is not None and emg_data is not None and len(ts) > 0:
                preprocessor.process(emg_data)

            # Predict when feature vector available
            feat = preprocessor.get_feature_vector()
            if feat is not None:
                x = torch.FloatTensor(feat).unsqueeze(0).to(device)
                with torch.no_grad():
                    y = model(x)
                y = y.cpu().numpy().flatten()

                # Convert sin/cos back to angles
                sin_p, cos_p, sin_s, cos_s = y[0], y[1], y[2], y[3]
                pred_p = float(np.rad2deg(np.arctan2(sin_p, cos_p)))
                pred_s = float(np.rad2deg(np.arctan2(sin_s, cos_s)) * 100.0)

                # Ensure angle is in [0, 360)
                pred_p = pred_p % 360.0

                preds.append(pred_p)
                pred_times.append(time.perf_counter() - t0)

                # Update MuJoCo pendulum
                mujoco_angle_rad = np.deg2rad(pred_p - 90)
                data.qpos[0] = mujoco_angle_rad
                mujoco_angles.append(pred_p)
                mujoco_times.append(time.perf_counter() - t0)

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
                cv2.imshow("OpenCV Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("OpenCV window closed, stopping inference.")
                    break

            # Step MuJoCo simulation
            current_time = time.perf_counter()
            dt = current_time - last_sim_time
            last_sim_time = current_time

            mujoco.mj_step(model_mj, data)

            # Sync MuJoCo viewer
            viewer.sync()

            # Check if MuJoCo viewer closed
            if viewer.is_running() == False:
                print("MuJoCo viewer closed, stopping inference.")
                break

            time.sleep(0.005)

    # Cleanup
    emg_source.stop()
    angle_source.stop()
    cv2.destroyAllWindows()

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", np.asarray(preds))
    np.save(out_dir / "pred_times.npy", np.asarray(pred_times))
    np.save(out_dir / "opencv_angles.npy", np.asarray(opencv_angles))
    np.save(out_dir / "opencv_times.npy", np.asarray(opencv_times))
    np.save(out_dir / "mujoco_angles.npy", np.asarray(mujoco_angles))
    np.save(out_dir / "mujoco_times.npy", np.asarray(mujoco_times))
    print(f"Saved {len(preds)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
