import math
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Any, Dict

class KernelType(IntEnum):
    TRIANGULAR = 1
    PARABOLIC = 2

class Kernel(ABC):
    @abstractmethod
    def get_score(self, bin_difference: int) -> int:
        pass

    @abstractmethod
    def get_max_score(self) -> float:
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        pass

    @abstractmethod
    def get_type_id(self) -> int:
        pass

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


class TriangularKernel(Kernel):
    def __init__(self, num_bins: int, width: float = 0.5):
        self._num_bins = num_bins
        self._width = width
        self._strength = float(num_bins)
        self._profile: List[int] = []
        self._compute_profile()

    def _compute_profile(self) -> None:
        profile_width = self._width * self._num_bins / 2.0
        if profile_width > 0:
            gradient = (self._strength - 1) / profile_width
            limit = int(math.floor(profile_width)) + 1
            for i in range(limit):
                val = math.floor(self._strength - i * gradient + 0.5)
                self._profile.append(int(val))
        else:
            self._profile.append(int(self._strength))

    def get_score(self, bin_difference: int) -> int:
        abs_diff = abs(bin_difference)
        return self._profile[abs_diff] if abs_diff < len(self._profile) else 0

    def get_max_score(self) -> float:
        return self._strength

    @property
    def width(self) -> float:
        return self._width

    def get_type_id(self) -> int:
        return KernelType.TRIANGULAR


class ParabolicKernel(Kernel):
    def __init__(self, num_bins: int, width: float = 0.5):
        self._num_bins = num_bins
        self._width = width
        self._strength = (num_bins * num_bins) / 4.0
        self._profile: List[int] = []
        self._compute_profile()

    def _compute_profile(self) -> None:
        profile_width = self._width * self._num_bins / 2.0
        if profile_width > 0:
            curvature = (self._strength - 1) / (profile_width * profile_width)
            limit = int(math.floor(profile_width)) + 1
            for i in range(limit):
                val = math.floor(self._strength - (i * i) * curvature + 0.5)
                self._profile.append(max(0, int(val)))
        else:
            self._profile.append(int(self._strength))

    def get_score(self, bin_difference: int) -> int:
        abs_diff = abs(bin_difference)
        return self._profile[abs_diff] if abs_diff < len(self._profile) else 0

    def get_max_score(self) -> float:
        return self._strength

    @property
    def width(self) -> float:
        return self._width

    def get_type_id(self) -> int:
        return KernelType.PARABOLIC


def create_kernel(kernel_type: KernelType, num_bins: int, width: float = 0.5) -> Kernel:
    if kernel_type == KernelType.TRIANGULAR:
        return TriangularKernel(num_bins, width)
    elif kernel_type == KernelType.PARABOLIC:
        return ParabolicKernel(num_bins, width)
    raise ValueError(f"Unknown kernel type: {kernel_type}")
