"""
Evaluation script for trained models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import argparse
import yaml

from .model_architectures import get_model
from .dataset import EMGAngleDataset, DataLoader


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, representation: str = 'angle') -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Model predictions (n_samples, n_angles)
        targets: Ground truth angles (n_samples, n_angles)
        representation: 'angle' for direct angles, 'sincos' for sin/cos representation

    Returns:
        Dictionary of metrics
    """
    # Convert sin/cos back to angles if needed
    if representation == 'sincos':
        # sin_p, cos_p, sin_s, cos_s -> p, s
        pred_p = np.rad2deg(np.arctan2(predictions[:, 0], predictions[:, 1])) % 360.0
        pred_s = np.rad2deg(np.arctan2(predictions[:, 2], predictions[:, 3])) * 100.0
        predictions = np.column_stack([pred_p, pred_s])

        targ_p = np.rad2deg(np.arctan2(targets[:, 0], targets[:, 1])) % 360.0
        targ_s = np.rad2deg(np.arctan2(targets[:, 2], targets[:, 3])) * 100.0
        targets = np.column_stack([targ_p, targ_s])

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - targets), axis=0)
    mae_mean = np.mean(mae)

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    rmse_mean = np.mean(rmse)

    # Correlation coefficient per angle
    correlations = []
    for i in range(predictions.shape[1]):
        corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)
    corr_mean = np.mean(correlations)

    metrics = {
        'mae_per_angle': mae.tolist(),
        'mae_mean': float(mae_mean),
        'rmse_per_angle': rmse.tolist(),
        'rmse_mean': float(rmse_mean),
        'correlation_per_angle': correlations,
        'correlation_mean': float(corr_mean)
    }

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on

    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for emg, angles in dataloader:
            emg = emg.to(device)
            angles = angles.to(device)

            predictions = model(emg)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(angles.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def main():
    parser = argparse.ArgumentParser(description='Evaluate sEMG to angle estimation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='results/evaluation_results.txt', help='Path to save results')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create model
    model_kwargs = dict(
        input_channels=config['input_channels'],
        num_angles=config['num_angles'],
        hidden_dim=config.get('hidden_dim', 128),
    )
    if config['model_name'] == 'mlp':
        model_kwargs['input_dim'] = config.get('input_dim', config['input_channels'])
    else:
        model_kwargs['sequence_length'] = config['sequence_length']
    model = get_model(config['model_name'], **model_kwargs)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")

    # Load test data (placeholder - adapt to your data format)
    test_emg = np.load(f"{args.test_data}/test_emg.npy")
    test_angles = np.load(f"{args.test_data}/test_angles.npy")

    # Load metadata to check representation
    import json
    metadata_path = f"{args.test_data}/../metadata.json"
    if Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        representation = metadata.get('representation', 'angle')
    else:
        representation = 'angle'

    # Create dataset and dataloader
    test_dataset = EMGAngleDataset(test_emg, test_angles)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Evaluate
    print("Evaluating model...")
    predictions, targets = evaluate_model(model, test_loader, device)

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, representation=representation)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {metrics['mae_mean']:.4f} degrees")
    print(f"Root Mean Square Error (RMSE): {metrics['rmse_mean']:.4f} degrees")
    print(f"Mean Correlation: {metrics['correlation_mean']:.4f}")
    print("\nPer-angle MAE:")
    for i, mae in enumerate(metrics['mae_per_angle']):
        print(f"  Angle {i}: {mae:.4f} degrees")
    print("\nPer-angle Correlation:")
    for i, corr in enumerate(metrics['correlation_per_angle']):
        print(f"  Angle {i}: {corr:.4f}")
    print("="*50)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Mean Absolute Error (MAE): {metrics['mae_mean']:.4f} degrees\n")
        f.write(f"Root Mean Square Error (RMSE): {metrics['rmse_mean']:.4f} degrees\n")
        f.write(f"Mean Correlation: {metrics['correlation_mean']:.4f}\n")
        f.write("\nPer-angle MAE:\n")
        for i, mae in enumerate(metrics['mae_per_angle']):
            f.write(f"  Angle {i}: {mae:.4f} degrees\n")
        f.write("\nPer-angle Correlation:\n")
        for i, corr in enumerate(metrics['correlation_per_angle']):
            f.write(f"  Angle {i}: {corr:.4f}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
