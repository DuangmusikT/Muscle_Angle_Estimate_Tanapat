"""
Dataset class for sEMG to angle estimation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class EMGAngleDataset(Dataset):
    """PyTorch Dataset for sEMG signals and corresponding angles."""

    def __init__(
        self,
        emg_data: np.ndarray,
        angle_data: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.

        Args:
            emg_data: EMG data array (n_samples, sequence_length, n_channels)
            angle_data: Angle data array (n_samples, n_angles)
            transform: Optional transform to apply
        """
        self.emg_data = torch.FloatTensor(emg_data)
        self.angle_data = torch.FloatTensor(angle_data)
        self.transform = transform

        assert len(self.emg_data) == len(self.angle_data), \
            "EMG and angle data must have same length"

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.emg_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (emg_signal, angles)
        """
        emg = self.emg_data[idx]
        angle = self.angle_data[idx]

        if self.transform:
            emg = self.transform(emg)

        return emg, angle


def create_dataloaders(
    train_emg: np.ndarray,
    train_angles: np.ndarray,
    val_emg: np.ndarray,
    val_angles: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_emg: Training EMG data
        train_angles: Training angle data
        val_emg: Validation EMG data
        val_angles: Validation angle data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = EMGAngleDataset(train_emg, train_angles)
    val_dataset = EMGAngleDataset(val_emg, val_angles)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
