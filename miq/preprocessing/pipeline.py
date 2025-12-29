"""
MachineIQ Preprocessing Pipeline
Modular signal processing for vibration analysis
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from .features import (
    TimedomainFeatures, FrequencyFeatures, EnvelopeFeatures,
    StressWaveAnalysis, BandpassFilter, VibrationAnalyzer
)


class PreprocessingLayer(ABC):
    """Base class for preprocessing layers"""

    @property
    @abstractmethod
    def output_feature_names(self) -> List[str]:
        pass

    @abstractmethod
    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        pass


class TimeDomainLayer(PreprocessingLayer):
    """Time domain feature extraction layer"""

    DEFAULT_FEATURES = ["rms", "peak", "crest_factor", "kurtosis"]
    ALL_FEATURES = [
        "rms", "peak", "peak_to_peak", "crest_factor", "kurtosis",
        "skewness", "mean", "std", "clearance_factor", "shape_factor", "impulse_factor"
    ]

    def __init__(self, features: List[str] = None):
        self.features = features or self.DEFAULT_FEATURES

    @property
    def output_feature_names(self) -> List[str]:
        return self.features

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        all_features = TimedomainFeatures.extract_all(data)
        return {k: v for k, v in all_features.items() if k in self.features}


class FrequencyDomainLayer(PreprocessingLayer):
    """Frequency domain (FFT) feature extraction layer"""

    DEFAULT_FEATURES = ["peak_frequency", "spectral_centroid", "spectral_energy"]
    ALL_FEATURES = [
        "peak_frequency", "spectral_centroid", "spectral_spread",
        "spectral_energy", "spectral_entropy"
    ]

    def __init__(self, features: List[str] = None):
        self.features = features or self.DEFAULT_FEATURES
        self._extractor = None

    @property
    def output_feature_names(self) -> List[str]:
        return self.features

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        if self._extractor is None or self._extractor.sample_rate != sample_rate:
            self._extractor = FrequencyFeatures(sample_rate)

        all_features = self._extractor.extract_all(data)
        return {k: v for k, v in all_features.items() if k in self.features}


class BandEnergyLayer(PreprocessingLayer):
    """Extract energy in specified frequency bands"""

    def __init__(self, bands: List[Tuple[float, float]]):
        """
        bands: List of (low_freq, high_freq) tuples in Hz
        Example: [(10, 100), (100, 500), (500, 2000)]
        """
        self.bands = bands
        self._extractor = None

    @property
    def output_feature_names(self) -> List[str]:
        return [f"band_{i}_{int(low)}_{int(high)}_hz" for i, (low, high) in enumerate(self.bands)]

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        if self._extractor is None or self._extractor.sample_rate != sample_rate:
            self._extractor = FrequencyFeatures(sample_rate)

        return self._extractor.extract_band_energy(data, self.bands)


class EnvelopeLayer(PreprocessingLayer):
    """Envelope analysis layer using Hilbert transform"""

    DEFAULT_FEATURES = ["envelope_rms", "envelope_peak", "envelope_kurtosis", "envelope_crest"]

    def __init__(self, features: List[str] = None, band_filter: Optional[Tuple[float, float]] = None):
        """
        features: Which envelope features to extract
        band_filter: Optional (low, high) Hz for bandpass before envelope
        """
        self.features = features or self.DEFAULT_FEATURES
        self.band_filter = band_filter
        self._extractor = None

    @property
    def output_feature_names(self) -> List[str]:
        return self.features

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        if self._extractor is None or self._extractor.sample_rate != sample_rate:
            self._extractor = EnvelopeFeatures(sample_rate)

        # Apply optional bandpass filter
        if self.band_filter is not None:
            low, high = self.band_filter
            data = self._extractor.bandpass.apply(data, low, high)

        all_features = self._extractor.extract_envelope_features(data)
        return {k: v for k, v in all_features.items() if k in self.features}


class StressWaveLayer(PreprocessingLayer):
    """StressWave / PeakVue analysis layer"""

    DEFAULT_FEATURES = ["stresswave_rms", "stresswave_peak", "stresswave_count", "stresswave_crest"]
    ALL_FEATURES = [
        "stresswave_rms", "stresswave_peak", "stresswave_count",
        "stresswave_energy", "stresswave_crest"
    ]

    def __init__(self, features: List[str] = None, highpass_freq: float = 1000):
        """
        features: Which stress wave features to extract
        highpass_freq: High-pass filter cutoff in Hz (default 1000 Hz)
        """
        self.features = features or self.DEFAULT_FEATURES
        self.highpass_freq = highpass_freq
        self._analyzer = None

    @property
    def output_feature_names(self) -> List[str]:
        return self.features

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        # StressWave requires high sample rate
        if sample_rate < 5000:
            return {k: 0.0 for k in self.features}

        if self._analyzer is None or self._analyzer.sample_rate != sample_rate:
            self._analyzer = StressWaveAnalysis(sample_rate)

        all_features = self._analyzer.extract_stresswave_features(data)
        return {k: v for k, v in all_features.items() if k in self.features}


class BandpassFilterLayer(PreprocessingLayer):
    """Apply bandpass filter before other processing (modifies signal in-place)"""

    def __init__(self, low_freq: float, high_freq: float, order: int = 4):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self._filter = None

    @property
    def output_feature_names(self) -> List[str]:
        return []  # Filter doesn't produce features, it transforms signal

    def process(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        return {}

    def apply(self, data: np.ndarray, sample_rate: float) -> np.ndarray:
        if self._filter is None or self._filter.sample_rate != sample_rate:
            self._filter = BandpassFilter(sample_rate)
        return self._filter.apply(data, self.low_freq, self.high_freq, self.order)


class DataPipeline:
    """
    Simplified pipeline for C++ integration.
    Wraps VibrationAnalyzer with sensible defaults.
    """

    def __init__(self, sample_rate: float = 10000):
        self.sample_rate = sample_rate
        self._analyzer = VibrationAnalyzer(sample_rate)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform raw signal to feature vector"""
        if len(data) < 10:
            return np.zeros(20, dtype=np.float32)

        features = self._analyzer.analyze(data)
        return np.array(list(features.values()), dtype=np.float32)


class PreprocessingPipeline:
    """Flexible multi-layer preprocessing pipeline"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.layers: List[PreprocessingLayer] = []
        self.filters: List[BandpassFilterLayer] = []

    def add_layer(self, layer: PreprocessingLayer) -> 'PreprocessingPipeline':
        if isinstance(layer, BandpassFilterLayer):
            self.filters.append(layer)
        else:
            self.layers.append(layer)
        return self

    def get_feature_names(self) -> List[str]:
        names = []
        for layer in self.layers:
            names.extend(layer.output_feature_names)
        return names

    def process(self, raw_signal: np.ndarray) -> Dict[str, float]:
        # Apply filters first
        signal = raw_signal.copy()
        for filt in self.filters:
            signal = filt.apply(signal, self.sample_rate)

        # Extract features
        results = {}
        for layer in self.layers:
            results.update(layer.process(signal, self.sample_rate))
        return results

    def get_feature_vector(self, raw_signal: np.ndarray) -> np.ndarray:
        results = self.process(raw_signal)
        names = self.get_feature_names()
        return np.array([results.get(n, 0.0) for n in names], dtype=np.float32)


class PipelineBuilder:
    """Fluent builder for preprocessing pipelines"""

    def __init__(self, sample_rate: float):
        self.pipeline = PreprocessingPipeline(sample_rate)
        self.sample_rate = sample_rate

    def add_time_domain(self, features: List[str] = None) -> 'PipelineBuilder':
        """Add time domain feature extraction"""
        self.pipeline.add_layer(TimeDomainLayer(features))
        return self

    def add_frequency_domain(self, features: List[str] = None) -> 'PipelineBuilder':
        """Add FFT-based frequency domain features"""
        self.pipeline.add_layer(FrequencyDomainLayer(features))
        return self

    def add_band_energy(self, bands: List[Tuple[float, float]]) -> 'PipelineBuilder':
        """Add frequency band energy extraction"""
        self.pipeline.add_layer(BandEnergyLayer(bands))
        return self

    def add_envelope(self, features: List[str] = None,
                    band_filter: Optional[Tuple[float, float]] = None) -> 'PipelineBuilder':
        """Add envelope analysis (Hilbert transform)"""
        self.pipeline.add_layer(EnvelopeLayer(features, band_filter))
        return self

    def add_stresswave(self, features: List[str] = None,
                      highpass_freq: float = 1000) -> 'PipelineBuilder':
        """Add stress wave / PeakVue analysis"""
        self.pipeline.add_layer(StressWaveLayer(features, highpass_freq))
        return self

    def add_bandpass_filter(self, low_freq: float, high_freq: float,
                           order: int = 4) -> 'PipelineBuilder':
        """Add bandpass filter (applied before feature extraction)"""
        self.pipeline.add_layer(BandpassFilterLayer(low_freq, high_freq, order))
        return self

    def build(self) -> PreprocessingPipeline:
        return self.pipeline


def create_basic_pipeline(sample_rate: float) -> PreprocessingPipeline:
    """Create simple pipeline with time domain features only"""
    return PipelineBuilder(sample_rate).add_time_domain().build()


def create_vibration_pipeline(sample_rate: float) -> PreprocessingPipeline:
    """Create full vibration analysis pipeline"""
    return (PipelineBuilder(sample_rate)
            .add_time_domain()
            .add_frequency_domain()
            .add_envelope()
            .add_stresswave()
            .build())


def create_bearing_analysis_pipeline(sample_rate: float,
                                    resonance_band: Tuple[float, float] = (2000, 10000)) -> PreprocessingPipeline:
    """Create pipeline optimized for bearing fault detection"""
    return (PipelineBuilder(sample_rate)
            .add_time_domain(["rms", "peak", "kurtosis", "crest_factor"])
            .add_frequency_domain(["peak_frequency", "spectral_energy"])
            .add_envelope(band_filter=resonance_band)
            .add_stresswave()
            .build())
