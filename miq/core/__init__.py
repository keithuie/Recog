"""Core detection logic modules."""
from .detector import MIQDetector, DetectorConfig, DetectorState, AnomalyResult
from .matrix import Matrix
from .kernels import Kernel, KernelType, create_kernel, TriangularKernel, ParabolicKernel
from .channel_params import ChannelParameters, ChannelParametersManager
from .binning import StaticBinner, AdaptiveBinner, FixedFrequencyBinner, CyclicBinner

__all__ = [
    "MIQDetector", "DetectorConfig", "DetectorState", "AnomalyResult",
    "Matrix", "Kernel", "KernelType", "create_kernel", "TriangularKernel", "ParabolicKernel",
    "ChannelParameters", "ChannelParametersManager",
    "StaticBinner", "AdaptiveBinner", "FixedFrequencyBinner", "CyclicBinner"
]
