"""
Data module for sEMG and angle data processing.
"""

from .emg_acquisition import EMGAcquisition
from .emg_acquisition_legacy import EMGAcquisitionLegacy, EMGAcquisitionCSV
from .angle_detection import HandAngleDetector
from .data_sync import DataSynchronizer

__all__ = [
    'EMGAcquisition',
    'EMGAcquisitionLegacy',
    'EMGAcquisitionCSV',
    'HandAngleDetector',
    'DataSynchronizer'
]
