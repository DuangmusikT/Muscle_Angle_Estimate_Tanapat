"""
Model Architectures for sEMG to Angle Estimation

Implements various neural network architectures for mapping sEMG signals to hand angles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNModel(nn.Module):
    """1D CNN model for sEMG feature extraction and angle prediction."""

    def __init__(
        self,
        input_channels: int = 8,
        sequence_length: int = 200,
        num_angles: int = 16,
        hidden_dim: int = 128
    ):
        """
        Initialize CNN model.

        Args:
            input_channels: Number of EMG channels
            sequence_length: Length of input window
            num_angles: Number of output angles
            hidden_dim: Hidden dimension
        """
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate flattened size
        self.flattened_size = 128 * (sequence_length // 8)

        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_angles)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Predicted angles of shape (batch, num_angles)
        """
        x = x.transpose(1, 2)  # (batch, channels, sequence_length)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class LSTMModel(nn.Module):
    """LSTM model for temporal sEMG sequence processing."""

    def __init__(
        self,
        input_channels: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_angles: int = 16,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM model.

        Args:
            input_channels: Number of EMG channels
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_angles: Number of output angles
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_channels,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_angles)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, channels)

        Returns:
            Predicted angles of shape (batch, num_angles)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        x = h_n[-1]  # (batch, hidden_dim)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNLSTMModel(nn.Module):
    """Hybrid CNN-LSTM model combining spatial and temporal processing."""

    def __init__(
        self,
        input_channels: int = 8,
        sequence_length: int = 200,
        cnn_hidden: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_angles: int = 16,
        dropout: float = 0.3
    ):
        """
        Initialize CNN-LSTM model.

        Args:
            input_channels: Number of EMG channels
            sequence_length: Length of input window
            cnn_hidden: CNN hidden channels
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            num_angles: Number of output angles
            dropout: Dropout rate
        """
        super(CNNLSTMModel, self).__init__()

        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(input_channels, cnn_hidden, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(cnn_hidden, cnn_hidden * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_hidden * 2)
        self.pool2 = nn.MaxPool1d(2)

        # LSTM for temporal processing
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm_input_dim = cnn_hidden * 2

        self.lstm = nn.LSTM(
            self.lstm_input_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden // 2, num_angles)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, channels)

        Returns:
            Predicted angles of shape (batch, num_angles)
        """
        # Reshape for CNN: (batch, channels, sequence)
        x = x.transpose(1, 2)

        # CNN forward pass
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Reshape for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        x = h_n[-1]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class AttentionModel(nn.Module):
    """Transformer-based model with attention mechanism."""

    def __init__(
        self,
        input_channels: int = 8,
        sequence_length: int = 200,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_angles: int = 16,
        dropout: float = 0.1
    ):
        """
        Initialize Attention model.

        Args:
            input_channels: Number of EMG channels
            sequence_length: Length of input window
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_angles: Number of output angles
            dropout: Dropout rate
        """
        super(AttentionModel, self).__init__()

        self.input_projection = nn.Linear(input_channels, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim // 2, num_angles)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, channels)

        Returns:
            Predicted angles of shape (batch, num_angles)
        """
        # Project input to embedding dimension
        x = self.input_projection(x)

        # Add positional encoding (simple learnable)
        # For simplicity, we skip explicit positional encoding

        # Transformer encoder
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MLPModel(nn.Module):
    """Simple MLP regressor over stacked feature vectors."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, num_angles: int = 2, dropout: float = 0.2):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_angles),
        )

    def forward(self, x):
        # x: (batch, input_dim) or (batch, 1, input_dim)
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Get model by name.

    Args:
        model_name: Name of model architecture
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    models = {
        'mlp': MLPModel,
        'cnn': CNNModel,
        'lstm': LSTMModel,
        'cnn_lstm': CNNLSTMModel,
        'attention': AttentionModel
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    # Translate generic args into model-specific argument names
    kwargs = dict(kwargs)
    if 'hidden_dim' in kwargs:
        hd = kwargs.pop('hidden_dim')
        if model_name == 'cnn_lstm':
            kwargs.setdefault('lstm_hidden', hd)
        elif model_name == 'attention':
            kwargs.setdefault('embed_dim', hd)
        elif model_name == 'mlp':
            kwargs.setdefault('hidden_dim', hd)
        else:
            kwargs.setdefault('hidden_dim', hd)
    if model_name == 'mlp':
        kwargs.pop('input_channels', None)
        kwargs['input_dim'] = kwargs.get('input_dim', 32)

    return models[model_name](**kwargs)
