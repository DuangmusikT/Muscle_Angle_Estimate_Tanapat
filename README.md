# sEMG-based Hand Angle Estimation

**Matsuo Lab Physical AI Advanced 1 Project Submission**

## Overview

This project aims to estimate hand joint angles from surface electromyography (sEMG) sensor signals using a supervised learning approach. The system uses OpenCV-based computer vision to generate ground truth labels, which are then used to train a model that can predict hand angles solely from sEMG signals.

### Problem Statement

Hand angle estimation from muscle activity is a challenging problem in prosthetics, human-computer interaction, and rehabilitation. This project demonstrates that sEMG signals can be effectively used to predict hand joint angles without requiring direct visual input.

### Approach

- **Input**: sEMG sensor signals (multi-channel muscle action potentials)
- **Teacher**: OpenCV-detected hand angles from video recordings
- **Objective**: Train a model to estimate hand angles from sEMG signals alone

## Project Structure

```
muscle_estimate_project/
├── data/
│   ├── raw/              # Raw sEMG signals and video recordings
│   ├── processed/        # Processed datasets ready for training
│   └── models/           # Saved model checkpoints
├── src/
│   ├── data/             # Data acquisition and preprocessing
│   ├── models/           # Model architectures and training
│   ├── visualization/    # Plotting and visualization tools
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for experimentation
├── configs/              # Configuration files
├── results/              # Experiment results and metrics
└── README.md
```

## Setup

### Installation (Conda recommended)

```bash
# Create conda environment
conda create -n muscle_estimate python=3.9 -y
conda activate muscle_estimate

# Install PyTorch (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml tqdm -y
conda install -c conda-forge opencv mediapipe -y
conda install -c conda-forge jupyter ipykernel -y
conda install -c conda-forge tensorboard -y
conda install -c conda-forge nptdms -y

# Register with Jupyter
python -m ipykernel install --user --name=muscle_estimate --display-name "Python (muscle_estimate)"
```

### Hardware Requirements

- sEMG sensor (e.g., Myo Armband, custom sEMG setup)
- Camera for OpenCV angle detection
- Computer with Python 3.8+

## Usage

### Data Collection

1. Record sEMG signals simultaneously with video
2. Use OpenCV to extract hand angles from video (ground truth)
3. Sync sEMG and angle data

### Data Preparation

The project supports multiple data formats from previous work:

**TDMS Format (from sEMG_Robot_Demo project):**
- Supports National Instruments TDMS files
- Uses notch filters (50Hz, 100Hz) and low-pass (500Hz)
- Calculates RMS and EMA features

**CSV Format (from Senkouka project):**
- Supports CSV files with metadata headers
- Uses DC blocker → Bessel → Notch filter cascade
- Sampling rate: 10,000 Hz

**Run data preparation:**
```bash
python scripts/prepare_data.py
```
This will prompt you to select the data format and provide file paths.

### Training

```bash
python src/models/train.py --config configs/default_config.yaml --data_dir data/processed
```

### Evaluation

```bash
python src/models/evaluate.py --checkpoint data/models/best_model.pth --test_data data/processed
```

## Methodology

### Signal Processing
- sEMG signal filtering and normalization
- Feature extraction (time-domain, frequency-domain)
- Windowing and segmentation

### Computer Vision
- Hand landmark detection using MediaPipe/OpenCV
- Angle calculation from joint positions
- Temporal alignment with sEMG data

### Machine Learning
- Model architecture (CNN, LSTM, or hybrid)
- Training strategy
- Evaluation metrics (MAE, RMSE, correlation)

## Results

[To be filled after experiments]

## Deadline

November 2nd, 2026

## License

[To be determined]
