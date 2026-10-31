# Project Guide for sEMG to Hand Angle Estimation

## Project Overview
This project estimates hand joint angles from surface electromyography (sEMG) sensor signals using supervised learning. OpenCV/MediaPipe provides ground truth labels from video recordings.

## Key Information

### Project Structure
- `data/`: Raw and processed data, model checkpoints
- `src/`: Source code (data processing, models, visualization, utilities)
- `notebooks/`: Jupyter notebooks for experimentation
- `configs/`: Configuration files for training
- `scripts/`: Data preparation and demo scripts
- `results/`: Experiment results and visualizations

### Data Pipeline
1. **sEMG Acquisition** (`src/data/emg_acquisition.py`):
   - Load raw sEMG signals
   - Apply notch filter (power line removal)
   - Apply bandpass filter (20-450 Hz)
   - Rectify and normalize
   - Extract time-domain features (MAV, RMS, WL, ZC, SSC)

2. **Angle Detection** (`src/data/angle_detection.py`):
   - Use MediaPipe for hand landmark detection
   - Calculate 16 joint angles (5 fingers × 3 joints + wrist)
   - Process video frame by frame

3. **Data Synchronization** (`src/data/data_sync.py`):
   - Align EMG timestamps with video timestamps
   - Interpolate angles to match EMG sampling rate
   - Create overlapping windows for training
   - Cross-correlation for lag correction

### Model Architectures (`src/models/model_architectures.py`)
- **CNN**: 1D convolutional network for spatial feature extraction
- **LSTM**: Temporal processing with recurrent layers
- **CNN-LSTM**: Hybrid model combining both approaches
- **Attention**: Transformer-based model with self-attention

### Training (`src/models/train.py`)
- Uses PyTorch
- MSE loss function
- Adam optimizer with learning rate scheduling
- TensorBoard logging
- Checkpoint saving (best model + periodic)

### Evaluation (`src/models/evaluate.py`)
- Calculates MAE, RMSE per angle
- Correlation coefficients
- Saves results to file

### Visualization (`src/visualization/plot_results.py`)
- Signal comparison plots
- Prediction vs ground truth plots
- Scatter plots with R²
- Error distribution histograms
- Metrics heatmaps
- Training history

## Usage

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
```bash
python scripts/prepare_data.py
```
Note: Adapt this script to your specific data format and pipeline.

### Training
```bash
python src/models/train.py --config configs/default_config.yaml --data_dir data/processed
```

### Evaluation
```bash
python src/models/evaluate.py --checkpoint data/models/best_model.pth --test_data data/processed
```

### Demo (synthetic data)
```bash
python scripts/demo.py
```

### Experimentation
Use `notebooks/experiment.ipynb` for interactive experimentation.

## Configuration
Edit `configs/default_config.yaml` to adjust:
- Model architecture
- Training parameters (epochs, batch size, learning rate)
- Data paths
- Window size and overlap

## Important Notes
- The data loading code in `scripts/prepare_data.py` is currently commented out and needs to be adapted to your specific data format
- Your sEMG sensor pipeline should output data in a format compatible with the EMGAcquisition class
- Video recordings should be synchronized with EMG data collection
- Adjust sampling rates in config based on your hardware

## Deadline
November 2nd, 2026
