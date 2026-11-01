"""Convert existing angle data from degrees to sin/cos representation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import json
from src.utils import save_json


def main():
    # Load existing data
    train_emg = np.load("data/processed/train_emg.npy")
    train_angles = np.load("data/processed/train_angles.npy")
    val_emg = np.load("data/processed/val_emg.npy")
    val_angles = np.load("data/processed/val_angles.npy")
    test_emg = np.load("data/processed/test_emg.npy")
    test_angles = np.load("data/processed/test_angles.npy")

    print(f"Original shapes:")
    print(f"  Train: EMG {train_emg.shape}, Angles {train_angles.shape}")
    print(f"  Val: EMG {val_emg.shape}, Angles {val_angles.shape}")
    print(f"  Test: EMG {test_emg.shape}, Angles {test_angles.shape}")

    # Convert angles to sin/cos
    def convert_to_sincos(angles):
        p = angles[:, 0]

        # Recalculate speed with proper wrap-around handling
        s = np.zeros_like(p)
        for i in range(1, len(p)):
            diff = p[i] - p[i-1]
            # Handle 360° wrap-around
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            s[i] = diff * 30.0  # Assuming 30 fps
        s[0] = s[1]  # Handle first sample

        # Clamp speed to reasonable range (±500 deg/sec)
        s = np.clip(s, -500, 500)

        # Convert to sin/cos with wrap-around handling
        p_rad = np.deg2rad(p)
        s_rad = np.deg2rad(s / 100.0)  # Scale speed to avoid extreme values

        sincos = np.column_stack([
            np.sin(p_rad), np.cos(p_rad),  # Position sin/cos
            np.sin(s_rad), np.cos(s_rad)   # Speed sin/cos
        ])
        return sincos

    train_angles_sincos = convert_to_sincos(train_angles)
    val_angles_sincos = convert_to_sincos(val_angles)
    test_angles_sincos = convert_to_sincos(test_angles)

    print(f"\nConverted shapes:")
    print(f"  Train: {train_angles_sincos.shape}")
    print(f"  Val: {val_angles_sincos.shape}")
    print(f"  Test: {test_angles_sincos.shape}")

    # Save converted data
    np.save("data/processed/train_angles.npy", train_angles_sincos)
    np.save("data/processed/val_angles.npy", val_angles_sincos)
    np.save("data/processed/test_angles.npy", test_angles_sincos)

    # Update metadata
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)

    metadata["angle_names"] = ["sin_p", "cos_p", "sin_s", "cos_s"]
    metadata["num_angles"] = 4
    metadata["representation"] = "sincos"

    save_json(metadata, "data/processed/metadata.json")

    print("\nData converted and saved successfully!")
    print("Metadata updated with representation='sincos'")


if __name__ == "__main__":
    main()
