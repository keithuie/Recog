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
