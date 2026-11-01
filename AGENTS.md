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

**Legacy Format Support:**
The project now supports both your previous data formats:
- **TDMS format** (from sEMG_Robot_Demo project): Use `EMGAcquisitionLegacy`
- **CSV format** (from Senkouka project): Use `EMGAcquisitionCSV`

**Standard Pipeline:**
1. **sEMG Acquisition** (`src/data/emg_acquisition.py` - `EMGAcquisition`):
   - Defaults: sampling_rate=1200 Hz, channels=3, notch_freq=50 Hz, lpf_cutoff=488.652 Hz, gain=816.0, ref_voltage=0.0
   - Load raw sEMG signals:
     - `load_emg_data(filepath)` — placeholder using `np.load` (adapt to your format)
     - `load_tdms_usb1608gx(filepath)` — TDMS (Digilent USB1608GX) via `nptdms`, returns `(data_array, channel_names)`
   - Convert ADC to voltage: `(raw - ref_voltage) * 1e6 / gain` (µV)
   - Apply notch filter at 50 Hz (Q=30, `iirnotch` + `filtfilt`) for power-line removal
   - Apply 4th-order Butterworth low-pass filter at 488.652 Hz (`filtfilt`)
   - Full-wave rectification (`np.abs`)
   - Normalize per channel: methods `'max'`, `'zscore'`, or `'minmax'`
   - Extract time-domain features per sliding window (default window_size=200 samples, overlap=0.5):
     - MAV, RMS, WL, ZC, SSC (5 features × 3 channels = 15-element vector per window)
   - End-to-end via `preprocess_pipeline(raw_data, normalize_method='max', apply_voltage_conversion=True)`

**Legacy Pipeline (TDMS):**
1. **sEMG Acquisition** (`src/data/emg_acquisition_legacy.py` - `EMGAcquisitionLegacy`):
   - Load TDMS files (National Instruments format)
   - Apply notch filters at 50Hz and 100Hz
   - Apply 4th-order Butterworth low-pass at 500Hz
   - Calculate RMS features (200ms window, 20ms step)
   - Calculate EMA (exponential moving average, alpha=0.05)

**Legacy Pipeline (CSV):**
1. **sEMG Acquisition** (`src/data/emg_acquisition_legacy.py` - `EMGAcquisitionCSV`):
   - Load CSV files with header metadata
   - Apply DC blocker → Bessel filter → Notch filter cascade
   - Uses gain of 883.97 for voltage conversion
   - Sampling rate: 10,000 Hz

2. **Angle Detection** (`src/data/angle_detection.py`):
   - Use MediaPipe for hand landmark detection
   - Calculate 1 joint angle (absolute palm-to-vertical angle) — simplified single target
   - Process video frame by frame

3. **Data Synchronization** (`src/data/data_sync.py`):
   - Align EMG timestamps with video timestamps
   - Interpolate angles to match EMG sampling rate
   - Create overlapping windows for training
   - Cross-correlation for lag correction

### Real-time Collection (USB-1608GX + webcam)
- **Script**: `scripts/collect_realtime.py`
- **EMG source**: `src/data/realtime_emg.py`
  - `USB1608GXSource` (uldaq continuous scan, 1200 Hz, 8 ch) for hardware
  - `SimulatedEMGSource` for dry runs (`--simulate`)
- **Angle source**: `src/data/realtime_angle.py` (webcam + `HandAngleDetector`, single `wrist_angle`)
- **Workflow**:
  1. Collect raw EMG + angles in real time (`--duration`, default 60 s)
  2. Align via interpolation to EMG timestamps, filter invalid
  3. Window, split, normalize, save to `data/processed/`
- **Run**: `python scripts/collect_realtime.py --duration 180 --camera 0`
- **Dry run**: `python scripts/collect_realtime.py --simulate --duration 20`
- **Hardware prereq**: USB-1608GX connected (check `lsusb`); libuldaq + uldaq installed

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
