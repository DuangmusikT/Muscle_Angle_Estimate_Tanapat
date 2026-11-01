# sEMG-based Hand Angle Estimation
**Matsuo Lab Physical AI Advanced 1 Project Submission**

## Overview

This project aims to estimate both hand joint angles (position) and movement speed from surface electromyography (sEMG) sensor signals using a supervised learning approach. The system uses OpenCV and MediaPipe-based computer vision to generate ground truth labels, which are synchronized to train a multi-task model capable of predicting kinematics solely from physical muscle activity.

### Problem Statement
Hand angle estimation from muscle activity is a challenging problem in prosthetics, human-computer interaction, and rehabilitation. This project demonstrates that applying advanced feature engineering, target shifting, and temporal window stacking to sEMG signals can effectively predict continuous hand kinematics without requiring direct visual input.

## Hardware Requirements

- **DAQ:** Digilent USB-1608GX
- **Sensors:** 4-electrode dual differential pair setup (Anterior flexors & Posterior extensors)
- **Camera:** Standard webcam for OpenCV/MediaPipe angle detection
- **Environment:** Ubuntu Linux (tested with Intel Arc GPU / external SSD setup)

## Project Structure

```text
muscle_estimate_project/
├── data/
│   ├── raw_realtime/     # Raw sEMG signals and video recordings from live sessions
│   ├── processed/        # Processed datasets (train/val/test + metadata)
│   └── models/           # Saved model checkpoints
├── src/
│   ├── data/             # Data acquisition and preprocessing
│   ├── models/           # Model architectures and training
│   ├── visualization/    # Plotting and visualization tools
│   └── utils/            # Utility functions
├── scripts/              # Executable scripts for the end-to-end workflow
├── configs/              # Configuration files (e.g., default_config.yaml)
└── README.md
```

## Setup & Installation

It is recommended to use Conda for environment management.

```bash
# Create conda environment
conda create -n muscle_estimate python=3.9 -y
conda activate muscle_estimate

# Install PyTorch (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml tqdm -y
conda install -c conda-forge opencv mediapipe -y
conda install -c conda-forge tensorboard nptdms -y

# Navigate to project directory (update path as needed)
cd /mnt/ubuntu_ext/muscle_estimate_project
```

## End-to-End AI Training Workflow

### 1. Collect Data (Real Hardware)
Plug in the USB-1608GX and verify with `lsusb`. Run the real-time collection script while keeping your hand in the webcam frame so OpenCV produces angle targets.

```bash
# Record for 180 seconds with live EMG strip and angle overlay
python scripts/collect_realtime.py --duration 180 --camera 0 --preview
```
*Outputs:* `data/raw_realtime/session_<ts>/` and `data/processed/` (train/val/test + metadata).

*(Optional)* Dry run without hardware:
```bash
python scripts/collect_realtime.py --simulate --duration 20
```

### 2. Inspect the Collected Data
Verify the integrity of your recorded signals and targets.

```bash
# View processed datasets
python scripts/view_data.py --data_dir data/processed --save_dir results/view

# Run simulated oscilloscope
python scripts/oscilloscope.py --simulate --duration 5
```
*Expected Shapes:* `data/processed/train_emg.npy` should be `(N, 32)` and `train_angles.npy` should be `(N, 2)` (representing position `p` and speed `s`).

### 3. Configure Model Parameters
Edit `configs/default_config.yaml` to set your architecture and training hyperparameters. 

```yaml
model_name: mlp            # mlp | cnn | lstm | cnn_lstm | attention
input_channels: 32         # 4 features × 2 channels × 4 stacked windows
input_dim: 32
sequence_length: 1
num_angles: 2              # Targets: [p (position), s (speed)]
hidden_dim: 128
num_epochs: 100
batch_size: 32
learning_rate: 0.001
```

### 4. Train the Model
Train the multi-task model to predict position and speed simultaneously.

```bash
python -m src.models.train --config configs/default_config.yaml --data_dir data/processed
```
*Outputs:* 
- Best checkpoint saved to `data/models/best_model.pth`
- Training logs saved to `runs/` (viewable via `tensorboard --logdir runs`)

### 5. Evaluate
Evaluate the best model checkpoint against the test dataset.

```bash
python -m src.models.evaluate --checkpoint data/models/best_model.pth --test_data data/processed
```
*Outputs:* Reports MAE, RMSE, and correlation for both **p** (position) and **s** (speed).

### 6. Live Inference (Demo)
Run the trained model in real-time to compare AI predictions against live OpenCV angles.

```bash
python scripts/run_live_inference.py --duration 180 --camera 0 \
  --checkpoint data/models/best_model.pth \
  --config configs/default_config.yaml \
  --metadata data/processed/metadata.json
```
*Outputs:* On-screen visualization; saves predictions to `data/live_inference/`.

### 6.5. MuJoCo Inference (Alternative)
Run the trained model in a simulated MuJoCo environment to compare AI predictions against synthetic targets.

```bash
python scripts/run_mujoco_inference.py --simulate --duration 60 \
  --checkpoint data/models/best_model.pth \
  --config configs/default_config.yaml \
  --metadata data/processed/metadata.json
```
*Outputs:* On-screen visualization; saves predictions to `data/mujoco_inference/`.

### 7. Data Management & Reprocessing
**Reprocess a saved session** (useful if you tweak feature engineering parameters):
```bash
python scripts/reprocess_session.py --session_dir data/raw_realtime/session_<ts>
```

**Hard Reset** (Clear all processed data, raw sessions, and models):
```bash
rm -f data/processed/*.npy data/processed/*.json
rm -rf data/raw_realtime/session_*
rm -f data/models/*.pth
```

## Methodology

### Signal Processing
- **Temporal Stacking:** Stacks 4 consecutive sliding windows to provide trajectory context to the model.
- **Feature Extraction:** Extracts 4 features per channel: RMS, MAV, WL (Waveform Length), and $\Delta$RMS (Time Derivative).
- **Input Dimension:** 4 features × 2 channels × 4 stacked windows = 32 input channels.

### Computer Vision & Synchronization
- **Hand Tracking:** MediaPipe/OpenCV extracts ground truth joint angles at 30 FPS.
- **Multi-Task Targets:** Computes both angular position and angular speed.
- **Electromechanical Delay (EMD):** Targets are forward-shifted to account for the physiological delay between muscle electrical activation and physical movement.

### Machine Learning
- **Multi-Task Regression:** By predicting both position (`p`) and speed (`s`) simultaneously, the network learns to disentangle static holding effort from rapid dynamic movements, preventing positional overshoot.

## Legacy Data Support
The project previously supported legacy datasets (TDMS/CSV format from the `sEMG_Robot_Demo` and `Senkouka` projects) sampled at 10,000 Hz. Tools for this older pipeline remain in `scripts/prepare_data.py`.

## Deadline
November 2nd, 2026 (10:00 AM JST)

## License
[To be determined]