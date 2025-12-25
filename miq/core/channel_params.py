import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .kernels import Kernel, KernelType, create_kernel

@dataclass
class ChannelParameters:
    name: str
    min_value: float
    max_value: float
    num_bins: int = 64
    kernel: Kernel = None
    is_cyclic: bool = False
    
    def __post_init__(self):
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if self.num_bins <= 0 or self.num_bins > 256:
            raise ValueError("num_bins must be between 1 and 256")
        self._range = self.max_value - self.min_value
    
    def bin_value(self, value: float) -> int:
        clamped = max(self.min_value, min(self.max_value, value))
        normalized = (clamped - self.min_value) / self._range
        bin_idx = int(normalized * self.num_bins)
        return min(bin_idx, self.num_bins - 1)
    
    def bin_values(self, values: np.ndarray) -> np.ndarray:
        clamped = np.clip(values, self.min_value, self.max_value)
        normalized = (clamped - self.min_value) / self._range
        indices = np.floor(normalized * self.num_bins).astype(np.int32)
        return np.minimum(indices, self.num_bins - 1).astype(np.uint8)


class ChannelParametersManager:
    def __init__(self, channel_params: List[ChannelParameters] = None):
        self._channels = channel_params or []
    
    def get_channel_names(self) -> List[str]:
        return [ch.name for ch in self._channels]
    
    def get_max_possible_score(self) -> float:
        return sum(ch.kernel.get_max_score() for ch in self._channels)
    
    def __iter__(self):
        return iter(self._channels)
    
    def __len__(self):
        return len(self._channels)
