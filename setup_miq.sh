#!/bin/bash
# MachineIQ Complete Setup Script for Claude Code
# Run this script to create the entire project

set -e

echo "Creating MachineIQ project structure..."

# Create directories
mkdir -p miq/core miq/preprocessing miq/utils tests gui deploy .github/workflows .do

# ============================================================================
# requirements.txt
# ============================================================================
cat > requirements.txt << 'REQEOF'
numpy>=1.24.0
scipy>=1.10.0
PyQt6>=6.4.0
pyqtgraph>=0.13.0
pytest>=7.0.0
pytest-cov>=4.0.0
pyyaml>=6.0
REQEOF

# ============================================================================
# .gitignore
# ============================================================================
cat > .gitignore << 'GITEOF'
__pycache__/
*.py[cod]
*.so
build/
dist/
*.egg-info/
.eggs/
htmlcov/
.coverage
.pytest_cache/
.env
.venv/
venv/
*.pkl
*.log
.idea/
.vscode/
.DS_Store
data/
logs/
models/
GITEOF

# ============================================================================
# miq/__init__.py
# ============================================================================
cat > miq/__init__.py << 'EOF'
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
EOF

# ============================================================================
# miq/core/__init__.py
# ============================================================================
cat > miq/core/__init__.py << 'EOF'
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
EOF

# ============================================================================
# miq/core/kernels.py
# ============================================================================
cat > miq/core/kernels.py << 'EOF'
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
EOF

# ============================================================================
# miq/core/channel_params.py
# ============================================================================
cat > miq/core/channel_params.py << 'EOF'
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
EOF

# ============================================================================
# miq/core/matrix.py
# ============================================================================
cat > miq/core/matrix.py << 'EOF'
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from .channel_params import ChannelParameters

class Matrix:
    def __init__(self, channel_params: List[ChannelParameters], initial_capacity: int = 2000):
        self.channel_params = channel_params
        self._num_channels = len(channel_params)
        self._num_trained_instances = 0
        self._capacity = initial_capacity
        self.matrix = np.zeros((self._capacity, self._num_channels), dtype=np.uint8)
        self._max_possible_raw_score = sum(p.kernel.get_max_score() for p in self.channel_params)

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def num_trained_instances(self) -> int:
        return self._num_trained_instances

    @property
    def capacity(self) -> int:
        return self._capacity

    def _exact_match(self, binned_input: np.ndarray) -> Optional[int]:
        if self._num_trained_instances == 0:
            return None
        valid_data = self.matrix[:self._num_trained_instances]
        matches = np.all(valid_data == binned_input, axis=1)
        indices = np.where(matches)[0]
        return int(indices[0]) if indices.size > 0 else None

    def train_instance(self, binned_input: np.ndarray) -> int:
        existing_idx = self._exact_match(binned_input)
        if existing_idx is not None:
            return existing_idx
        if self._num_trained_instances >= self._capacity:
            self._expand_matrix()
        new_idx = self._num_trained_instances
        self.matrix[new_idx] = binned_input
        self._num_trained_instances += 1
        return new_idx

    def _expand_matrix(self):
        new_capacity = self._capacity * 2
        new_matrix = np.zeros((new_capacity, self._num_channels), dtype=np.uint8)
        new_matrix[:self._capacity] = self.matrix
        self.matrix = new_matrix
        self._capacity = new_capacity

    def get_best_match_score(self, binned_input: np.ndarray) -> float:
        if self._num_trained_instances == 0:
            return 0.0
        total_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
        valid_matrix = self.matrix[:self._num_trained_instances]
        for c in range(self._num_channels):
            col_data = valid_matrix[:, c]
            diffs = np.abs(col_data.astype(np.int32) - int(binned_input[c]))
            kernel = self.channel_params[c].kernel
            profile = np.array(kernel._profile)
            c_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
            valid_mask = diffs < len(profile)
            c_scores[valid_mask] = profile[diffs[valid_mask]]
            total_scores += c_scores
        best_raw_score = np.max(total_scores)
        return float(best_raw_score / self._max_possible_raw_score) if self._max_possible_raw_score > 0 else 0.0

    def get_best_match_with_residuals(self, binned_input: np.ndarray) -> Tuple[float, np.ndarray, int]:
        if self._num_trained_instances == 0:
            return 0.0, np.zeros(self._num_channels), -1
        total_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
        valid_matrix = self.matrix[:self._num_trained_instances]
        for c in range(self._num_channels):
            col_data = valid_matrix[:, c]
            diffs = np.abs(col_data.astype(np.int32) - int(binned_input[c]))
            kernel = self.channel_params[c].kernel
            profile = np.array(kernel._profile)
            c_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
            valid_mask = diffs < len(profile)
            c_scores[valid_mask] = profile[diffs[valid_mask]]
            total_scores += c_scores
        best_idx = int(np.argmax(total_scores))
        best_raw_score = total_scores[best_idx]
        residuals = np.zeros(self._num_channels, dtype=np.float32)
        stored_row = self.matrix[best_idx]
        for c in range(self._num_channels):
            diff = int(binned_input[c]) - int(stored_row[c])
            residuals[c] = self.channel_params[c].kernel.get_score(diff)
        normalized_score = float(best_raw_score / self._max_possible_raw_score) if self._max_possible_raw_score > 0 else 0.0
        return normalized_score, residuals, best_idx

    def save_to_file(self, filepath: str):
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Matrix':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
EOF

# ============================================================================
# miq/core/detector.py
# ============================================================================
cat > miq/core/detector.py << 'EOF'
import numpy as np
import logging
import pickle
import os
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from .kernels import KernelType, create_kernel
from .matrix import Matrix
from .channel_params import ChannelParameters

logger = logging.getLogger(__name__)

class DetectorState(Enum):
    CREATED = auto()
    LEARNING = auto()
    MONITORING = auto()

@dataclass
class DetectorConfig:
    model_type: str = "bindiff"
    bins_per_channel: int = 64
    kernel_type: KernelType = KernelType.TRIANGULAR
    kernel_width: float = 0.5
    learning_period_seconds: Optional[int] = None

@dataclass
class AnomalyResult:
    timestamp: datetime
    match_strength: float
    anomaly_score: float
    residuals: np.ndarray
    channel_names: List[str]
    best_match_index: int
    is_anomaly: bool

    def get_top_contributors(self, n: int = 3) -> List[Tuple[str, float]]:
        count = min(len(self.residuals), len(self.channel_names))
        indexed_scores = [(i, self.residuals[i]) for i in range(count)]
        indexed_scores.sort(key=lambda x: x[1])
        return [(self.channel_names[idx], score) for idx, score in indexed_scores[:n]]


class MIQDetector:
    def __init__(self, config: DetectorConfig = None):
        self.config = config if config else DetectorConfig()
        self.state = DetectorState.CREATED
        self.matrix: Optional[Matrix] = None
        self.channel_params: List[ChannelParameters] = []
        self.anomaly_threshold = 0.3
        self.learning_start_time: Optional[datetime] = None
        self.learning_end_time: Optional[datetime] = None

    @property
    def num_trained_states(self) -> int:
        return self.matrix.num_trained_instances if self.matrix else 0

    @property
    def is_learning(self) -> bool:
        return self.state == DetectorState.LEARNING

    @property
    def is_monitoring(self) -> bool:
        return self.state == DetectorState.MONITORING

    @property
    def channel_names(self) -> List[str]:
        return [cp.name for cp in self.channel_params]

    def configure_channels(self, channel_configs: List[Dict]):
        self.channel_params = []
        for cfg in channel_configs:
            kernel = create_kernel(self.config.kernel_type, self.config.bins_per_channel, self.config.kernel_width)
            param = ChannelParameters(
                name=cfg["name"], min_value=cfg["min_value"], max_value=cfg["max_value"],
                num_bins=self.config.bins_per_channel, kernel=kernel
            )
            self.channel_params.append(param)
        self.matrix = Matrix(self.channel_params)
        logger.info(f"Configured {len(self.channel_params)} channels")

    def start_learning(self):
        if not self.matrix:
            raise RuntimeError("Channels not configured")
        self.state = DetectorState.LEARNING
        self.learning_start_time = datetime.now()

    def stop_learning(self):
        self.state = DetectorState.MONITORING
        self.learning_end_time = datetime.now()

    def _bin_feature_vector(self, feature_vector: np.ndarray) -> np.ndarray:
        binned = np.zeros(len(feature_vector), dtype=np.uint8)
        for i, val in enumerate(feature_vector):
            binned[i] = self.channel_params[i].bin_value(val)
        return binned

    def process(self, feature_vector: np.ndarray, timestamp: datetime = None) -> Optional[AnomalyResult]:
        if self.state == DetectorState.CREATED:
            raise RuntimeError("Detector not configured")
        if timestamp is None:
            timestamp = datetime.now()
        binned_input = self._bin_feature_vector(feature_vector)
        if self.state == DetectorState.LEARNING:
            self.matrix.train_instance(binned_input)
            return None
        elif self.state == DetectorState.MONITORING:
            match_strength, residuals, best_idx = self.matrix.get_best_match_with_residuals(binned_input)
            anomaly_score = 1.0 - match_strength
            return AnomalyResult(
                timestamp=timestamp, match_strength=match_strength, anomaly_score=anomaly_score,
                residuals=residuals, channel_names=self.channel_names,
                best_match_index=best_idx, is_anomaly=anomaly_score > self.anomaly_threshold
            )
        return None

    def save_detector(self, filepath: str):
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_detector(cls, filepath: str) -> 'MIQDetector':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
EOF

# ============================================================================
# miq/core/binning.py
# ============================================================================
cat > miq/core/binning.py << 'EOF'
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
EOF

# ============================================================================
# miq/preprocessing/__init__.py
# ============================================================================
cat > miq/preprocessing/__init__.py << 'EOF'
from .features import TimedomainFeatures, FrequencyFeatures, EnvelopeFeatures, BearingFrequencyCalculator
from .pipeline import PreprocessingPipeline, PipelineBuilder, create_basic_pipeline

__all__ = [
    "TimedomainFeatures", "FrequencyFeatures", "EnvelopeFeatures", "BearingFrequencyCalculator",
    "PreprocessingPipeline", "PipelineBuilder", "create_basic_pipeline"
]
EOF

# ============================================================================
# miq/preprocessing/features.py
# ============================================================================
cat > miq/preprocessing/features.py << 'EOF'
import numpy as np
import scipy.stats
import scipy.signal
from typing import Dict, List, Tuple

class TimedomainFeatures:
    @staticmethod
    def rms(signal: np.ndarray) -> float:
        return np.sqrt(np.mean(signal**2))

    @staticmethod
    def peak(signal: np.ndarray) -> float:
        return np.max(np.abs(signal))

    @staticmethod
    def crest_factor(signal: np.ndarray) -> float:
        rms_val = TimedomainFeatures.rms(signal)
        return TimedomainFeatures.peak(signal) / rms_val if rms_val > 0 else 0.0

    @staticmethod
    def kurtosis(signal: np.ndarray) -> float:
        return scipy.stats.kurtosis(signal, fisher=True)

    @classmethod
    def extract_all(cls, signal: np.ndarray) -> Dict[str, float]:
        return {
            "rms": cls.rms(signal), "peak": cls.peak(signal),
            "crest_factor": cls.crest_factor(signal), "kurtosis": cls.kurtosis(signal)
        }


class FrequencyFeatures:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def compute_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(signal)
        window = np.hanning(n)
        fft_vals = np.fft.rfft(signal * window)
        magnitudes = np.abs(fft_vals) / n * 2
        freqs = np.fft.rfftfreq(n, d=1/self.sample_rate)
        return freqs, magnitudes


class EnvelopeFeatures:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def compute_envelope(self, signal: np.ndarray) -> np.ndarray:
        analytic = scipy.signal.hilbert(signal)
        return np.abs(analytic) - np.mean(np.abs(analytic))


class BearingFrequencyCalculator:
    def __init__(self, num_balls: int, ball_diameter: float, pitch_diameter: float, contact_angle: float = 0):
        self.n = num_balls
        self.d = ball_diameter
        self.D = pitch_diameter
        self.phi = np.radians(contact_angle)

    def calculate(self, shaft_rpm: float) -> Dict[str, float]:
        freq_hz = shaft_rpm / 60.0
        ratio = self.d / self.D
        cos_phi = np.cos(self.phi)
        return {
            "bpfo": (self.n / 2.0) * freq_hz * (1.0 - ratio * cos_phi),
            "bpfi": (self.n / 2.0) * freq_hz * (1.0 + ratio * cos_phi),
            "bsf": (self.D / (2.0 * self.d)) * freq_hz * (1.0 - (ratio * cos_phi)**2),
            "ftf": 0.5 * freq_hz * (1.0 - ratio * cos_phi)
        }
EOF

# ============================================================================
# miq/preprocessing/pipeline.py
# ============================================================================
cat > miq/preprocessing/pipeline.py << 'EOF'
import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod
from .features import TimedomainFeatures

class PreprocessingLayer(ABC):
    @property
    @abstractmethod
    def output_feature_names(self) -> List[str]:
        pass
    @abstractmethod
    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        pass

class TimeDomainLayer(PreprocessingLayer):
    def __init__(self, features: List[str] = None):
        self.features = features or ["rms", "peak", "crest_factor", "kurtosis"]

    @property
    def output_feature_names(self) -> List[str]:
        return self.features

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        all_features = TimedomainFeatures.extract_all(data)
        return {k: v for k, v in all_features.items() if k in self.features}


class PreprocessingPipeline:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.layers: List[PreprocessingLayer] = []

    def add_layer(self, layer: PreprocessingLayer) -> 'PreprocessingPipeline':
        self.layers.append(layer)
        return self

    def get_feature_names(self) -> List[str]:
        names = []
        for layer in self.layers:
            names.extend(layer.output_feature_names)
        return names

    def process(self, raw_signal: np.ndarray) -> Dict[str, float]:
        results = {}
        for layer in self.layers:
            results.update(layer.process(raw_signal, self.sample_rate))
        return results

    def get_feature_vector(self, raw_signal: np.ndarray) -> np.ndarray:
        results = self.process(raw_signal)
        names = self.get_feature_names()
        return np.array([results.get(n, 0.0) for n in names], dtype=np.float32)


class PipelineBuilder:
    def __init__(self, sample_rate: float):
        self.pipeline = PreprocessingPipeline(sample_rate)

    def add_time_domain(self, features: List[str] = None) -> 'PipelineBuilder':
        self.pipeline.add_layer(TimeDomainLayer(features))
        return self

    def build(self) -> PreprocessingPipeline:
        return self.pipeline


def create_basic_pipeline(sample_rate: float) -> PreprocessingPipeline:
    return PipelineBuilder(sample_rate).add_time_domain().build()
EOF

# ============================================================================
# miq/utils/__init__.py
# ============================================================================
cat > miq/utils/__init__.py << 'EOF'
from .data_generator import TimeSeriesDataset, create_test_scenario
__all__ = ["TimeSeriesDataset", "create_test_scenario"]
EOF

# ============================================================================
# miq/utils/data_generator.py
# ============================================================================
cat > miq/utils/data_generator.py << 'EOF'
import numpy as np
from typing import List, Tuple, Dict

class TimeSeriesDataset:
    def __init__(self, num_channels: int = 1, seed: int = None):
        self.num_channels = num_channels
        if seed is not None:
            np.random.seed(seed)
        self.means = np.random.uniform(20, 80, num_channels)
        self.stds = np.random.uniform(0.5, 2.0, num_channels)

    def generate_baseline(self, num_samples: int) -> List[np.ndarray]:
        return [np.random.normal(self.means, self.stds).astype(np.float32) for _ in range(num_samples)]

    def generate_with_anomaly(self, num_samples: int, anomaly_start: int, 
                              anomaly_type: str = "shift", severity: float = 0.5) -> Tuple[List[np.ndarray], List[bool]]:
        data, labels = [], []
        affected = np.random.choice(self.num_channels, max(1, self.num_channels // 2), replace=False)
        for i in range(num_samples):
            is_anomaly = i >= anomaly_start
            labels.append(is_anomaly)
            step_means = self.means.copy()
            if is_anomaly and anomaly_type == "shift":
                step_means[affected] += severity * self.stds[affected] * 10
            data.append(np.random.normal(step_means, self.stds).astype(np.float32))
        return data, labels


def create_test_scenario(scenario_name: str, num_channels: int = 3, num_samples: int = 1000) -> Tuple[List[np.ndarray], List[bool], Dict]:
    ds = TimeSeriesDataset(num_channels)
    metadata = {"scenario": scenario_name}
    if scenario_name == "baseline_only":
        return ds.generate_baseline(num_samples), [False] * num_samples, metadata
    elif scenario_name == "sudden_fault":
        start = int(num_samples * 0.7)
        data, labels = ds.generate_with_anomaly(num_samples, start, "shift", 0.8)
        metadata["anomaly_start"] = start
        return data, labels, metadata
    return ds.generate_baseline(num_samples), [False] * num_samples, metadata
EOF

# ============================================================================
# tests/__init__.py
# ============================================================================
touch tests/__init__.py

# ============================================================================
# tests/test_integration.py
# ============================================================================
cat > tests/test_integration.py << 'EOF'
import pytest
import numpy as np
from miq.core import MIQDetector, TriangularKernel, ParabolicKernel

class TestKernels:
    def test_triangular_max_score(self):
        kernel = TriangularKernel(64, 0.5)
        assert kernel.get_max_score() == 64
        assert kernel.get_score(0) == 64
    
    def test_parabolic_max_score(self):
        kernel = ParabolicKernel(64, 0.5)
        assert kernel.get_max_score() == 1024

class TestDetector:
    def test_learning_and_monitoring(self):
        detector = MIQDetector()
        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
            {"name": "ch2", "min_value": 0, "max_value": 100}
        ])
        detector.start_learning()
        for _ in range(100):
            detector.process(np.array([50.0, 50.0]))
        detector.stop_learning()
        assert detector.num_trained_states > 0
        
        result = detector.process(np.array([50.0, 50.0]))
        assert result.match_strength > 0.9
        
        result = detector.process(np.array([95.0, 95.0]))
        assert result.anomaly_score > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# ============================================================================
# gui/__init__.py
# ============================================================================
touch gui/__init__.py

# ============================================================================
# main.py
# ============================================================================
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""MachineIQ CLI"""
import argparse
import numpy as np
from miq.core import MIQDetector
from miq.utils import create_test_scenario

def main():
    parser = argparse.ArgumentParser(description="MachineIQ Anomaly Detection")
    parser.add_argument("--mode", choices=["cli", "test"], default="cli")
    parser.add_argument("--scenario", default="sudden_fault")
    args = parser.parse_args()
    
    if args.mode == "test":
        import pytest
        pytest.main(["tests/", "-v"])
        return
    
    print("MachineIQ - Running demo...")
    data, labels, meta = create_test_scenario(args.scenario, num_channels=3, num_samples=500)
    
    detector = MIQDetector()
    detector.configure_channels([
        {"name": f"ch{i}", "min_value": 0, "max_value": 100} for i in range(3)
    ])
    
    detector.start_learning()
    for i in range(200):
        detector.process(data[i])
    detector.stop_learning()
    print(f"Learned {detector.num_trained_states} states")
    
    anomalies = 0
    for i in range(200, 500):
        result = detector.process(data[i])
        if result.is_anomaly:
            anomalies += 1
    print(f"Detected {anomalies} anomalies in monitoring phase")

if __name__ == "__main__":
    main()
EOF

# ============================================================================
# Dockerfile
# ============================================================================
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "pytest", "tests/", "-v"]
EOF

# ============================================================================
# docker-compose.yml
# ============================================================================
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  miq:
    build: .
    container_name: machineiq
    command: python -m pytest tests/ -v
EOF

# ============================================================================
# .github/workflows/ci.yml
# ============================================================================
cat > .github/workflows/ci.yml << 'EOF'
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - run: pip install -r requirements.txt
    - run: python -m pytest tests/ -v
EOF

# ============================================================================
# .do/app.yaml
# ============================================================================
cat > .do/app.yaml << 'EOF'
name: machineiq
services:
  - name: miq
    github:
      repo: keithuie/Recog
      branch: main
      deploy_on_push: true
    dockerfile_path: Dockerfile
    instance_size_slug: basic-xxs
EOF

# ============================================================================
# deploy/setup-droplet.sh
# ============================================================================
cat > deploy/setup-droplet.sh << 'EOF'
#!/bin/bash
set -e
apt-get update && apt-get install -y docker.io git
cd /opt && git clone https://github.com/keithuie/Recog.git machineiq
cd machineiq && docker build -t miq . && docker run --rm miq
EOF
chmod +x deploy/setup-droplet.sh

# ============================================================================
# README.md
# ============================================================================
cat > README.md << 'EOF'
# MachineIQ (mIQ)

Multivariate Anomaly Detection for Industrial Condition Monitoring.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python main.py --scenario sudden_fault
```

## Usage

```python
from miq.core import MIQDetector
import numpy as np

detector = MIQDetector()
detector.configure_channels([
    {"name": "vibration", "min_value": 0, "max_value": 10}
])

detector.start_learning()
for _ in range(500):
    detector.process(np.array([5.0]))
detector.stop_learning()

result = detector.process(np.array([9.0]))
print(f"Anomaly: {result.is_anomaly}, Score: {result.anomaly_score}")
```
EOF

echo ""
echo "=============================================="
echo "  MachineIQ project created successfully!"
echo "=============================================="
echo ""
echo "Files created. Now run:"
echo "  git add ."
echo "  git commit -m 'Add MachineIQ anomaly detection system'"
echo "  git push origin main"
