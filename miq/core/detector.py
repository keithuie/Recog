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
