"""
Visualization tools for sEMG to angle estimation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict
import seaborn as sns


def plot_signal_comparison(
    emg_signal: np.ndarray,
    angle_signal: np.ndarray,
    timestamps: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot EMG signal and corresponding angle over time.

    Args:
        emg_signal: EMG signal (n_samples, n_channels)
        angle_signal: Angle signal (n_samples, n_angles)
        timestamps: Timestamp array
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot EMG signals
    for i in range(emg_signal.shape[1]):
        axes[0].plot(timestamps, emg_signal[:, i], alpha=0.7, label=f'Channel {i}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('sEMG Signals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot angle signals
    for i in range(angle_signal.shape[1]):
        axes[1].plot(timestamps, angle_signal[:, i], alpha=0.7, label=f'Angle {i}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angle (degrees)')
    axes[1].set_title('Hand Joint Angles')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_comparison(
    predictions: np.ndarray,
    targets: np.ndarray,
    angle_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot predicted vs actual angles.

    Args:
        predictions: Model predictions (n_samples, n_angles)
        targets: Ground truth angles (n_samples, n_angles)
        angle_names: Optional list of angle names
        save_path: Optional path to save figure
    """
    n_angles = predictions.shape[1]
    n_cols = 4
    n_rows = (n_angles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_angles > 1 else [axes]

    for i in range(n_angles):
        ax = axes[i]
        ax.plot(targets[:, i], label='Ground Truth', alpha=0.7)
        ax.plot(predictions[:, i], label='Prediction', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Angle (degrees)')
        title = f'Angle {i}' if angle_names is None else angle_names[i]
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_angles, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_scatter_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    angle_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot scatter plots of predicted vs actual angles.

    Args:
        predictions: Model predictions (n_samples, n_angles)
        targets: Ground truth angles (n_samples, n_angles)
        angle_names: Optional list of angle names
        save_path: Optional path to save figure
    """
    n_angles = predictions.shape[1]
    n_cols = 4
    n_rows = (n_angles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_angles > 1 else [axes]

    for i in range(n_angles):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10)

        # Perfect prediction line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')

        # Calculate R^2
        corr = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
        r2 = corr ** 2

        ax.set_xlabel('Ground Truth (degrees)')
        ax.set_ylabel('Prediction (degrees)')
        title = f'Angle {i} (R²={r2:.3f})' if angle_names is None else f'{angle_names[i]} (R²={r2:.3f})'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_angles, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    angle_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot error distribution for each angle.

    Args:
        predictions: Model predictions (n_samples, n_angles)
        targets: Ground truth angles (n_samples, n_angles)
        angle_names: Optional list of angle names
        save_path: Optional path to save figure
    """
    errors = predictions - targets
    n_angles = errors.shape[1]
    n_cols = 4
    n_rows = (n_angles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_angles > 1 else [axes]

    for i in range(n_angles):
        ax = axes[i]
        ax.hist(errors[:, i], bins=30, edgecolor='black', alpha=0.7)

        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])

        ax.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}')
        ax.axvline(mean_error + std_error, color='g', linestyle=':', label=f'±1 SD: {std_error:.2f}')
        ax.axvline(mean_error - std_error, color='g', linestyle=':')

        ax.set_xlabel('Error (degrees)')
        ax.set_ylabel('Frequency')
        title = f'Angle {i}' if angle_names is None else angle_names[i]
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_angles, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_heatmap(
    metrics: Dict[str, float],
    angle_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot heatmap of metrics per angle.

    Args:
        metrics: Dictionary containing 'mae_per_angle', 'rmse_per_angle', 'correlation_per_angle'
        angle_names: Optional list of angle names
        save_path: Optional path to save figure
    """
    # Create matrix of metrics
    metric_matrix = np.array([
        metrics['mae_per_angle'],
        metrics['rmse_per_angle'],
        metrics['correlation_per_angle']
    ])

    metric_labels = ['MAE (deg)', 'RMSE (deg)', 'Correlation']
    angle_labels = angle_names if angle_names else [f'Angle {i}' for i in range(len(metrics['mae_per_angle']))]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        metric_matrix,
        xticklabels=angle_labels,
        yticklabels=metric_labels,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Value'}
    )

    ax.set_title('Metrics per Joint Angle')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss over epochs.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()
