"""
Training script for sEMG to angle estimation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import yaml
import argparse
from tqdm import tqdm

from .model_architectures import get_model
from .dataset import create_dataloaders


class Trainer:
    """Trainer class for sEMG to angle estimation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Tensorboard writer
        self.writer = SummaryWriter(config['log_dir'])

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for emg, angles in pbar:
            emg = emg.to(self.device)
            angles = angles.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(emg)
            loss = self.criterion(predictions, angles)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for emg, angles in self.val_loader:
                emg = emg.to(self.device)
                angles = angles.to(self.device)

                predictions = self.model(emg)
                loss = self.criterion(predictions, angles)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self, num_epochs: int):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    Path(self.config['checkpoint_dir']) / 'best_model.pth',
                    is_best=True
                )

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(
                    Path(self.config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth',
                    is_best=False
                )

        self.writer.close()
        print("Training completed!")

    def save_checkpoint(self, path: Path, is_best: bool):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, path)
        if is_best:
            print(f"Saved best model to {path} with val_loss={self.best_val_loss:.4f}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train sEMG to angle estimation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    # Load data (placeholder - adapt to your data format)
    # TODO: Implement actual data loading
    train_emg = np.load(f"{args.data_dir}/train_emg.npy")
    train_angles = np.load(f"{args.data_dir}/train_angles.npy")
    val_emg = np.load(f"{args.data_dir}/val_emg.npy")
    val_angles = np.load(f"{args.data_dir}/val_angles.npy")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_emg, train_angles,
        val_emg, val_angles,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )

    # Create model
    model = get_model(
        config['model_name'],
        input_channels=config['input_channels'],
        sequence_length=config['sequence_length'],
        num_angles=config['num_angles'],
        hidden_dim=config.get('hidden_dim', 128)
    )

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()
