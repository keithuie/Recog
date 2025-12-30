"""
Test preprocessing pipeline and feature extraction.
"""

import pytest
import numpy as np
from miq.preprocessing.features import (
    TimedomainFeatures, FrequencyFeatures, BandpassFilter,
    EnvelopeFeatures, StressWaveAnalysis, VibrationAnalyzer
)
from miq.preprocessing.pipeline import (
    PipelineBuilder, create_basic_pipeline, create_vibration_pipeline
)


class TestTimedomainFeatures:
    """Test time domain feature calculations."""

    def test_rms_sine_wave(self):
        """RMS of sine wave should be amplitude / sqrt(2)."""
        amplitude = 2.0
        t = np.linspace(0, 1, 1000)
        signal = amplitude * np.sin(2 * np.pi * 10 * t)

        rms = TimedomainFeatures.rms(signal)
        expected = amplitude / np.sqrt(2)
        assert abs(rms - expected) < 0.05, f"RMS should be ~{expected}, got {rms}"

    def test_peak_sine_wave(self):
        """Peak of sine wave should be amplitude."""
        amplitude = 3.0
        t = np.linspace(0, 1, 1000)
        signal = amplitude * np.sin(2 * np.pi * 10 * t)

        peak = TimedomainFeatures.peak(signal)
        assert abs(peak - amplitude) < 0.01, f"Peak should be {amplitude}, got {peak}"

    def test_crest_factor_sine(self):
        """Crest factor of sine wave should be sqrt(2)."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)

        cf = TimedomainFeatures.crest_factor(signal)
        expected = np.sqrt(2)
        assert abs(cf - expected) < 0.05, f"Crest factor should be ~{expected}, got {cf}"

    def test_kurtosis_gaussian(self):
        """Kurtosis of Gaussian should be ~0 (Fisher's)."""
        np.random.seed(42)
        signal = np.random.randn(10000)

        kurt = TimedomainFeatures.kurtosis(signal)
        assert abs(kurt) < 0.2, f"Gaussian kurtosis should be ~0, got {kurt}"

    def test_kurtosis_impulsive(self):
        """Kurtosis of impulsive signal should be high."""
        signal = np.zeros(1000)
        signal[100] = 10.0  # Single spike
        signal[500] = 10.0

        kurt = TimedomainFeatures.kurtosis(signal)
        assert kurt > 10, f"Impulsive kurtosis should be high, got {kurt}"


class TestFrequencyFeatures:
    """Test frequency domain features."""

    def test_peak_frequency_single_tone(self):
        """Peak frequency should identify dominant frequency."""
        sample_rate = 1000.0
        test_freq = 50.0
        t = np.linspace(0, 1, int(sample_rate))
        signal = np.sin(2 * np.pi * test_freq * t)

        extractor = FrequencyFeatures(sample_rate)
        peak_freq = extractor.peak_frequency(signal)

        assert abs(peak_freq - test_freq) < 2.0, f"Peak freq should be {test_freq}, got {peak_freq}"

    def test_spectral_centroid_single_tone(self):
        """Spectral centroid of single tone should be near the tone frequency."""
        sample_rate = 1000.0
        test_freq = 100.0
        t = np.linspace(0, 1, int(sample_rate))
        signal = np.sin(2 * np.pi * test_freq * t)

        extractor = FrequencyFeatures(sample_rate)
        centroid = extractor.spectral_centroid(signal)

        # Centroid should be close to the tone frequency
        assert abs(centroid - test_freq) < 10.0, f"Centroid should be ~{test_freq}, got {centroid}"

    def test_band_energy(self):
        """Band energy should capture energy in specified bands."""
        sample_rate = 1000.0
        t = np.linspace(0, 1, int(sample_rate))
        # Signal with two frequencies
        signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)

        extractor = FrequencyFeatures(sample_rate)
        bands = [(10, 100), (150, 250)]
        energy = extractor.extract_band_energy(signal, bands)

        # Both bands should have significant energy
        assert energy['band_0_10_100_hz'] > 0
        assert energy['band_1_150_250_hz'] > 0


class TestBandpassFilter:
    """Test bandpass filtering."""

    def test_bandpass_removes_out_of_band(self):
        """Bandpass should attenuate frequencies outside the band."""
        sample_rate = 1000.0
        t = np.linspace(0, 1, int(sample_rate))

        # Low freq (20 Hz) + in-band (100 Hz) + high freq (400 Hz)
        low_freq = 20.0
        mid_freq = 100.0
        high_freq = 400.0
        signal = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * mid_freq * t) + np.sin(2 * np.pi * high_freq * t)

        filt = BandpassFilter(sample_rate)
        filtered = filt.apply(signal, 50, 150)  # Keep 50-150 Hz

        # Check that mid frequency dominates in output
        freq_extractor = FrequencyFeatures(sample_rate)
        peak_freq = freq_extractor.peak_frequency(filtered)

        assert abs(peak_freq - mid_freq) < 10, f"Filtered peak should be ~{mid_freq}, got {peak_freq}"


class TestEnvelopeFeatures:
    """Test envelope analysis."""

    def test_envelope_amplitude_modulation(self):
        """Envelope should capture amplitude modulation."""
        sample_rate = 10000.0
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))

        # AM signal: carrier 1000 Hz, modulation 50 Hz
        carrier = 1000.0
        modulation = 50.0
        signal = (1 + 0.5 * np.sin(2 * np.pi * modulation * t)) * np.sin(2 * np.pi * carrier * t)

        extractor = EnvelopeFeatures(sample_rate)
        envelope = extractor.compute_envelope(signal)

        # Envelope should vary (DC removed, so can be negative)
        assert np.std(envelope) > 0.1  # Envelope should have significant variation
        # Mean should be near zero after DC removal
        assert abs(np.mean(envelope)) < 0.1

    def test_envelope_spectrum_modulation_freq(self):
        """Envelope spectrum should show modulation frequency."""
        sample_rate = 10000.0
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # AM signal with 30 Hz modulation
        modulation = 30.0
        signal = (1 + 0.5 * np.sin(2 * np.pi * modulation * t)) * np.sin(2 * np.pi * 1000 * t)

        extractor = EnvelopeFeatures(sample_rate)
        freqs, magnitudes = extractor.envelope_spectrum(signal)

        # Find peak in envelope spectrum
        peak_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]

        assert abs(peak_freq - modulation) < 5, f"Envelope peak should be ~{modulation} Hz, got {peak_freq}"


class TestStressWaveAnalysis:
    """Test stress wave / PeakVue analysis."""

    def test_stresswave_detects_impacts(self):
        """Stress wave should detect impulsive events."""
        sample_rate = 10000.0
        duration = 1.0
        n_samples = int(sample_rate * duration)
        signal = np.random.randn(n_samples) * 0.1  # Background noise

        # Add periodic impacts
        impact_interval = int(sample_rate / 10)  # 10 Hz impact rate
        for i in range(0, n_samples, impact_interval):
            if i < n_samples:
                signal[i] += 5.0  # Impact

        analyzer = StressWaveAnalysis(sample_rate)
        features = analyzer.extract_stresswave_features(signal)

        # Should detect significant impacts
        assert features['stresswave_count'] > 5, f"Should detect impacts, got count={features['stresswave_count']}"
        assert features['stresswave_peak'] > 1.0, f"Peak should be significant, got {features['stresswave_peak']}"


class TestPipeline:
    """Test preprocessing pipelines."""

    def test_basic_pipeline(self):
        """Basic pipeline should extract time domain features."""
        sample_rate = 1000.0
        signal = np.random.randn(1000)

        pipeline = create_basic_pipeline(sample_rate)
        features = pipeline.process(signal)

        assert 'rms' in features
        assert 'peak' in features
        assert 'kurtosis' in features

    def test_vibration_pipeline(self):
        """Full vibration pipeline should extract all feature types."""
        sample_rate = 10000.0
        signal = np.random.randn(10000)

        pipeline = create_vibration_pipeline(sample_rate)
        features = pipeline.process(signal)

        # Should have time domain features
        assert 'rms' in features
        # Should have frequency features
        assert 'peak_frequency' in features
        # Should have envelope features
        assert 'envelope_rms' in features
        # Should have stress wave features
        assert 'stresswave_rms' in features

    def test_pipeline_builder(self):
        """Test fluent pipeline builder."""
        pipeline = (PipelineBuilder(10000)
                   .add_time_domain(['rms', 'peak'])
                   .add_frequency_domain(['peak_frequency'])
                   .build())

        names = pipeline.get_feature_names()
        assert 'rms' in names
        assert 'peak' in names
        assert 'peak_frequency' in names

    def test_pipeline_feature_vector(self):
        """Pipeline should return consistent feature vector."""
        sample_rate = 10000.0
        pipeline = create_basic_pipeline(sample_rate)

        signal1 = np.random.randn(1000)
        signal2 = np.random.randn(1000)

        vec1 = pipeline.get_feature_vector(signal1)
        vec2 = pipeline.get_feature_vector(signal2)

        # Should have same length
        assert len(vec1) == len(vec2)
        # Should match feature names
        assert len(vec1) == len(pipeline.get_feature_names())


class TestVibrationAnalyzer:
    """Test complete vibration analyzer."""

    def test_analyzer_complete(self):
        """Analyzer should extract all features."""
        sample_rate = 10000.0
        signal = np.random.randn(10000)

        analyzer = VibrationAnalyzer(sample_rate)
        features = analyzer.analyze(signal)

        # Check key features present
        assert 'rms' in features
        assert 'peak_frequency' in features
        assert 'envelope_rms' in features
        assert 'stresswave_rms' in features

    def test_analyzer_no_stresswave_low_sample_rate(self):
        """Low sample rate should skip stress wave analysis."""
        sample_rate = 1000.0  # Too low for stress wave
        signal = np.random.randn(1000)

        analyzer = VibrationAnalyzer(sample_rate)
        features = analyzer.analyze(signal, include_stresswave=True)

        # Stress wave features should not be present (sample rate too low)
        assert 'stresswave_rms' not in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
