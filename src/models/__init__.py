"""
Models module for sEMG to angle estimation.
"""

from .model_architectures import (
    CNNModel,
    LSTMModel,
    CNNLSTMModel,
    AttentionModel,
    get_model
)
from .dataset import EMGAngleDataset, create_dataloaders
from .train import Trainer, load_config

__all__ = [
    'CNNModel',
    'LSTMModel',
    'CNNLSTMModel',
    'AttentionModel',
    'get_model',
    'EMGAngleDataset',
    'create_dataloaders',
    'Trainer',
    'load_config'
]
