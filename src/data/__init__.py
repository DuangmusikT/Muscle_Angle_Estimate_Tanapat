"""
Data module for sEMG and angle data processing.
"""

from .emg_acquisition import EMGAcquisition
from .emg_acquisition_legacy import EMGAcquisitionLegacy, EMGAcquisitionCSV
from .angle_detection import HandAngleDetector
from .data_sync import DataSynchronizer
from .realtime_emg import RealTimeEMGSource, SimulatedEMGSource, USB1608GXSource
from .realtime_angle import RealTimeAngleSource
from .realtime_preprocess import RealtimePreprocessor

__all__ = [
    'EMGAcquisition',
    'EMGAcquisitionLegacy',
    'EMGAcquisitionCSV',
    'HandAngleDetector',
    'DataSynchronizer',
    'RealTimeEMGSource',
    'SimulatedEMGSource',
    'USB1608GXSource',
    'RealTimeAngleSource',
]
