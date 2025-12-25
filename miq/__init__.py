"""MachineIQ - Multivariate Anomaly Detection System"""
__version__ = "0.1.0"

from .core import (
    MIQDetector, DetectorConfig, DetectorState, AnomalyResult,
    Matrix, Kernel, KernelType, ChannelParameters
)

__all__ = [
    "MIQDetector", "DetectorConfig", "DetectorState", "AnomalyResult",
    "Matrix", "Kernel", "KernelType", "ChannelParameters"
]
