"""
Data module for sEMG and angle data processing.
"""

from .emg_acquisition import EMGAcquisition
from .angle_detection import HandAngleDetector
from .data_sync import DataSynchronizer

__all__ = ['EMGAcquisition', 'HandAngleDetector', 'DataSynchronizer']
