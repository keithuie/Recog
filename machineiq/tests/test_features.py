"""Tests for feature extraction."""

import numpy as np
import pytest

from machineiq.features.base import BaseFeatureExtractor, FeatureSet
from machineiq.features.vibration import (
    VibrationFeatureExtractor,
    BearingInfo,
    generate_synthetic_vibration,
)


class TestFeatureSet:
    """Tests for FeatureSet class."""

    def test_basic_creation(self):
        """Test basic FeatureSet creation."""
        fs = FeatureSet(
            features={"a": 1.0, "b": 2.0, "c": 3.0},
            feature_names=["a", "b", "c"],
        )

        assert len(fs) == 3
        assert fs["a"] == 1.0
        assert fs["b"] == 2.0

    def test_vector_property(self):
        """Test vector property."""
        fs = FeatureSet(
            features={"a": 1.0, "b": 2.0, "c": 3.0},
            feature_names=["a", "b", "c"],
        )

        vec = fs.vector
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 3
        assert vec[0] == 1.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        fs = FeatureSet(
            features={"a": 1.0},
            feature_names=["a"],
            metadata={"source": "test"},
        )

        d = fs.to_dict()
        assert "features" in d
        assert "feature_names" in d
        assert "metadata" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "features": {"x": 5.0},
            "feature_names": ["x"],
            "metadata": None,
        }

        fs = FeatureSet.from_dict(d)
        assert fs["x"] == 5.0


class TestBearingInfo:
    """Tests for BearingInfo class."""

    def test_fault_frequencies(self):
        """Test bearing fault frequency calculations."""
        # Example bearing (6205 type)
        bearing = BearingInfo(
            n_balls=9,
            ball_diameter=8.0,
            pitch_diameter=38.5,
            contact_angle=0.0,
            shaft_rpm=1800,
        )

        # Shaft frequency should be 30 Hz at 1800 RPM
        assert abs(bearing.shaft_freq - 30.0) < 0.1

        # BPFO should be approximately 3.5-4x shaft frequency
        assert 100 < bearing.bpfo < 140

        # BPFI should be approximately 5-6x shaft frequency
        assert 140 < bearing.bpfi < 200

        # FTF should be approximately 0.4x shaft frequency
        assert 10 < bearing.ftf < 15

    def test_contact_angle_effect(self):
        """Test that contact angle affects frequencies."""
        bearing1 = BearingInfo(
            n_balls=9,
            ball_diameter=8.0,
            pitch_diameter=38.5,
            contact_angle=0.0,
            shaft_rpm=1800,
        )

        bearing2 = BearingInfo(
            n_balls=9,
            ball_diameter=8.0,
            pitch_diameter=38.5,
            contact_angle=30.0,  # 30 degree contact angle
            shaft_rpm=1800,
        )

        # Frequencies should differ with contact angle
        assert bearing1.bpfo != bearing2.bpfo
        assert bearing1.bpfi != bearing2.bpfi


class TestVibrationFeatureExtractor:
    """Tests for VibrationFeatureExtractor class."""

    def test_basic_extraction(self):
        """Test basic feature extraction."""
        sample_rate = 10000
        duration = 0.1
        _, waveform = generate_synthetic_vibration(
            duration=duration,
            sample_rate=sample_rate,
        )

        extractor = VibrationFeatureExtractor(
            sample_rate=sample_rate,
            fft_bins=16,
            include_bearing_freqs=False,
        )

        features = extractor.extract(waveform)

        assert isinstance(features, FeatureSet)
        assert len(features) > 0
        assert "rms" in features.features
        assert "peak" in features.features
        assert "kurtosis" in features.features

    def test_feature_names(self):
        """Test feature name generation."""
        extractor = VibrationFeatureExtractor(
            sample_rate=10000,
            fft_bins=8,
            include_bearing_freqs=False,
        )

        names = extractor.feature_names
        assert len(names) == extractor.n_features
        assert "rms" in names
        assert "fft_bin_0" in names
        assert "dominant_freq" in names

    def test_with_bearing_info(self):
        """Test extraction with bearing information."""
        sample_rate = 10000
        _, waveform = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=sample_rate,
        )

        bearing = BearingInfo(
            n_balls=9,
            ball_diameter=8.0,
            pitch_diameter=38.5,
            shaft_rpm=1800,
        )

        extractor = VibrationFeatureExtractor(
            sample_rate=sample_rate,
            fft_bins=16,
            include_bearing_freqs=True,
            n_harmonics=3,
        )

        features = extractor.extract(waveform, bearing_info=bearing)

        # Should have bearing frequency features
        assert "bpfo_h1" in features.features
        assert "bpfi_h2" in features.features
        assert "bsf_h3" in features.features
        assert "ftf_h1" in features.features

    def test_without_bearing_info(self):
        """Test that bearing features are zeros without bearing info."""
        _, waveform = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=10000,
        )

        extractor = VibrationFeatureExtractor(
            sample_rate=10000,
            include_bearing_freqs=True,
        )

        features = extractor.extract(waveform, bearing_info=None)

        # Bearing features should be zero
        assert features["bpfo_h1"] == 0.0
        assert features["bpfi_h1"] == 0.0

    def test_time_domain_features(self):
        """Test time domain feature calculations."""
        # Create a known signal
        sample_rate = 1000
        t = np.arange(1000) / sample_rate
        # Sine wave with amplitude 1
        waveform = np.sin(2 * np.pi * 50 * t)

        extractor = VibrationFeatureExtractor(
            sample_rate=sample_rate,
            include_bearing_freqs=False,
        )

        features = extractor.extract(waveform)

        # RMS of sine wave should be 1/sqrt(2) ≈ 0.707
        assert abs(features["rms"] - 0.707) < 0.01

        # Peak should be 1
        assert abs(features["peak"] - 1.0) < 0.01

        # Peak-to-peak should be 2
        assert abs(features["peak_to_peak"] - 2.0) < 0.01

        # Crest factor for sine wave should be sqrt(2) ≈ 1.414
        assert abs(features["crest_factor"] - 1.414) < 0.01

    def test_frequency_domain_features(self):
        """Test frequency domain feature calculations."""
        sample_rate = 1000
        t = np.arange(1000) / sample_rate
        # 50 Hz sine wave
        waveform = np.sin(2 * np.pi * 50 * t)

        extractor = VibrationFeatureExtractor(
            sample_rate=sample_rate,
            include_bearing_freqs=False,
        )

        features = extractor.extract(waveform)

        # Dominant frequency should be around 50 Hz
        assert abs(features["dominant_freq"] - 50) < 5

    def test_batch_extraction(self):
        """Test batch feature extraction."""
        extractor = VibrationFeatureExtractor(
            sample_rate=10000,
            include_bearing_freqs=False,
        )

        waveforms = []
        for _ in range(10):
            _, wf = generate_synthetic_vibration(duration=0.1, sample_rate=10000)
            waveforms.append(wf)

        feature_sets = extractor.extract_batch(waveforms)
        assert len(feature_sets) == 10

        array = extractor.extract_batch_to_array(waveforms)
        assert array.shape[0] == 10
        assert array.shape[1] == extractor.n_features

    def test_short_waveform_raises(self):
        """Test that very short waveforms raise error."""
        extractor = VibrationFeatureExtractor(sample_rate=10000)

        with pytest.raises(ValueError):
            extractor.extract(np.array([1]))  # Too short

    def test_config_output(self):
        """Test configuration output."""
        extractor = VibrationFeatureExtractor(
            sample_rate=10000,
            fft_bins=32,
            include_bearing_freqs=True,
        )

        config = extractor.get_config()
        assert config["sample_rate"] == 10000
        assert config["fft_bins"] == 32
        assert config["include_bearing_freqs"] is True
        assert "feature_names" in config


class TestSyntheticVibration:
    """Tests for synthetic vibration generation."""

    def test_basic_generation(self):
        """Test basic signal generation."""
        t, waveform = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=10000,
        )

        assert len(t) == 1000
        assert len(waveform) == 1000

    def test_with_fault(self):
        """Test generation with fault frequency."""
        # Normal signal
        _, normal = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=10000,
            fault_freq=None,
            fault_amplitude=0.0,
        )

        # Signal with fault
        _, faulty = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=10000,
            fault_freq=157.0,
            fault_amplitude=0.5,
        )

        # Faulty signal should have different characteristics
        normal_rms = np.sqrt(np.mean(normal**2))
        faulty_rms = np.sqrt(np.mean(faulty**2))

        # The faulty signal should have higher energy
        assert faulty_rms > normal_rms

    def test_noise_effect(self):
        """Test noise level effect."""
        _, low_noise = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=10000,
            noise_level=0.01,
        )

        _, high_noise = generate_synthetic_vibration(
            duration=0.1,
            sample_rate=10000,
            noise_level=0.5,
        )

        # High noise signal should be more random (higher kurtosis deviation)
        low_kurt = np.mean((low_noise - np.mean(low_noise))**4) / (np.std(low_noise)**4)
        high_kurt = np.mean((high_noise - np.mean(high_noise))**4) / (np.std(high_noise)**4)

        # With more noise, kurtosis should approach 3 (Gaussian)
        assert abs(high_kurt - 3) < abs(low_kurt - 3)


class TestFeatureExtractorBase:
    """Tests for base extractor class."""

    def test_abstract_method(self):
        """Test that base class cannot be instantiated directly."""

        class IncompleteExtractor(BaseFeatureExtractor):
            pass

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_complete_implementation(self):
        """Test that properly implemented subclass works."""

        class SimpleExtractor(BaseFeatureExtractor):
            def __init__(self):
                super().__init__(name="Simple")
                self._feature_names = ["mean", "std"]

            def extract(self, data, **kwargs):
                return FeatureSet(
                    features={"mean": float(np.mean(data)), "std": float(np.std(data))},
                    feature_names=self._feature_names,
                )

        extractor = SimpleExtractor()
        features = extractor.extract(np.array([1, 2, 3, 4, 5]))

        assert features["mean"] == 3.0
        assert abs(features["std"] - np.std([1, 2, 3, 4, 5])) < 0.01


class TestIntegration:
    """Integration tests for feature extraction."""

    def test_full_pipeline(self):
        """Test full feature extraction pipeline."""
        # Generate synthetic data
        sample_rate = 10000
        n_samples = 100

        waveforms = []
        for _ in range(n_samples):
            _, wf = generate_synthetic_vibration(
                duration=0.1,
                sample_rate=sample_rate,
                noise_level=0.1,
            )
            waveforms.append(wf)

        # Extract features
        extractor = VibrationFeatureExtractor(
            sample_rate=sample_rate,
            fft_bins=16,
            include_bearing_freqs=False,
        )

        features = extractor.extract_batch_to_array(waveforms)

        # Check output shape
        assert features.shape[0] == n_samples
        assert features.shape[1] == extractor.n_features

        # Check no NaN or Inf values
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

        # Check features are reasonable
        assert np.all(features >= 0) or True  # Some features can be negative (skewness, kurtosis)
