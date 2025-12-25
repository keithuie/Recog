import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

class Binner(ABC):
    @abstractmethod
    def bin(self, value: float) -> int:
        pass
    @abstractmethod
    def bin_array(self, values: np.ndarray) -> np.ndarray:
        pass

class StaticBinner(Binner):
    def __init__(self, min_val: float, max_val: float, num_bins: int = 64):
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.range_width = max_val - min_val if max_val > min_val else 1.0

    def bin(self, value: float) -> int:
        if value <= self.min_val: return 0
        if value >= self.max_val: return self.num_bins - 1
        normalized = (value - self.min_val) / self.range_width
        return min(int(normalized * self.num_bins), self.num_bins - 1)

    def bin_array(self, values: np.ndarray) -> np.ndarray:
        clamped = np.clip(values, self.min_val, self.max_val)
        normalized = (clamped - self.min_val) / self.range_width
        indices = np.floor(normalized * self.num_bins).astype(np.int32)
        return np.minimum(indices, self.num_bins - 1).astype(np.uint8)

class AdaptiveBinner(Binner):
    def __init__(self, num_bins: int = 64, percentile_clip: float = 1.0):
        self.num_bins = num_bins
        self.percentile_clip = percentile_clip
        self.min_val = 0.0
        self.max_val = 1.0
        self._is_calibrated = False
        self.range_width = 1.0

    def calibrate(self, values: np.ndarray):
        self.min_val = float(np.percentile(values, self.percentile_clip))
        self.max_val = float(np.percentile(values, 100.0 - self.percentile_clip))
        self.range_width = self.max_val - self.min_val if self.max_val > self.min_val else 1.0
        self._is_calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def bin(self, value: float) -> int:
        if value <= self.min_val: return 0
        if value >= self.max_val: return self.num_bins - 1
        normalized = (value - self.min_val) / self.range_width
        return min(int(normalized * self.num_bins), self.num_bins - 1)

    def bin_array(self, values: np.ndarray) -> np.ndarray:
        clamped = np.clip(values, self.min_val, self.max_val)
        normalized = (clamped - self.min_val) / self.range_width
        indices = np.floor(normalized * self.num_bins).astype(np.int32)
        return np.minimum(indices, self.num_bins - 1).astype(np.uint8)

class FixedFrequencyBinner(Binner):
    def __init__(self, num_bins: int = 64):
        self.num_bins = num_bins
        self.edges = np.array([])
        self._is_calibrated = False

    def calibrate(self, values: np.ndarray):
        q_points = np.linspace(0, 100, self.num_bins + 1)
        self.edges = np.percentile(values, q_points)
        self._is_calibrated = True

    def bin(self, value: float) -> int:
        idx = np.searchsorted(self.edges, value, side='right')
        if idx == 0: return 0
        if idx >= len(self.edges): return self.num_bins - 1
        return min(max(int(idx) - 1, 0), self.num_bins - 1)

    def bin_array(self, values: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self.edges, values, side='right') - 1
        return np.clip(indices, 0, self.num_bins - 1).astype(np.uint8)

class CyclicBinner(Binner):
    def __init__(self, period: float, num_bins: int = 64):
        self.period = period
        self.num_bins = num_bins
        self.bin_width = period / num_bins

    def bin(self, value: float) -> int:
        wrapped = value % self.period
        return min(int(wrapped / self.bin_width), self.num_bins - 1)

    def bin_array(self, values: np.ndarray) -> np.ndarray:
        wrapped = values % self.period
        indices = np.floor(wrapped / self.bin_width).astype(np.int32)
        return np.minimum(indices, self.num_bins - 1).astype(np.uint8)
