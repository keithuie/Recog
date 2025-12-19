"""Tests for CMM detector and related components."""

import numpy as np
import pytest

from machineiq.core.binning import AdaptiveBinner, MultiChannelBinner
from machineiq.core.cmm_detector import BinaryCMM, CMMDetector
from machineiq.core.hybrid_engine import HybridEngine


class TestAdaptiveBinner:
    """Tests for AdaptiveBinner class."""

    def test_uniform_binning(self):
        """Test uniform binning method."""
        binner = AdaptiveBinner(n_bins=4, method="uniform")
        values = np.array([0, 25, 50, 75, 100])
        binner.fit(values)

        # Test transform
        pattern = binner.transform(50)
        assert len(pattern) == 4
        assert np.sum(pattern) == 1  # One-hot

    def test_quantile_binning(self):
        """Test quantile binning method."""
        binner = AdaptiveBinner(n_bins=4, method="quantile")
        # Skewed data
        values = np.array([1, 1, 1, 1, 10, 10, 100, 1000])
        binner.fit(values)

        # Should have bins that reflect the distribution
        assert binner.is_fitted

    def test_kmeans_binning(self):
        """Test k-means binning method."""
        binner = AdaptiveBinner(n_bins=3, method="kmeans")
        # Multimodal data
        values = np.concatenate([
            np.random.normal(0, 0.1, 50),
            np.random.normal(5, 0.1, 50),
            np.random.normal(10, 0.1, 50),
        ])
        binner.fit(values)

        assert binner.is_fitted

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        binner = AdaptiveBinner(n_bins=4)
        with pytest.raises(RuntimeError):
            binner.transform(50)

    def test_fit_empty_raises(self):
        """Test that fit on empty data raises error."""
        binner = AdaptiveBinner(n_bins=4)
        with pytest.raises(ValueError):
            binner.fit(np.array([]))

    def test_constant_values(self):
        """Test handling of constant values."""
        binner = AdaptiveBinner(n_bins=4)
        binner.fit(np.array([5, 5, 5, 5]))
        pattern = binner.transform(5)
        assert np.sum(pattern) == 1

    def test_batch_transform(self):
        """Test batch transformation."""
        binner = AdaptiveBinner(n_bins=8)
        binner.fit(np.arange(100))

        patterns = binner.transform_batch(np.array([10, 50, 90]))
        assert patterns.shape == (3, 8)
        assert np.all(np.sum(patterns, axis=1) == 1)

    def test_serialization(self):
        """Test state serialization and restoration."""
        binner = AdaptiveBinner(n_bins=4, method="quantile")
        binner.fit(np.arange(100))

        state = binner.get_state()
        binner2 = AdaptiveBinner()
        binner2.set_state(state)

        # Should produce same results
        p1 = binner.transform(50)
        p2 = binner2.transform(50)
        assert np.array_equal(p1, p2)


class TestMultiChannelBinner:
    """Tests for MultiChannelBinner class."""

    def test_basic_functionality(self):
        """Test basic multi-channel binning."""
        binner = MultiChannelBinner(n_channels=3, bins_per_channel=4)
        data = np.random.rand(100, 3) * 100
        binner.fit(data)

        pattern = binner.transform(np.array([50, 50, 50]))
        assert len(pattern) == 12  # 3 channels * 4 bins
        assert np.sum(pattern) == 3  # One hot per channel

    def test_pattern_width(self):
        """Test pattern width calculation."""
        binner = MultiChannelBinner(n_channels=5, bins_per_channel=8)
        assert binner.pattern_width == 40

    def test_channel_mismatch_raises(self):
        """Test that channel mismatch raises error."""
        binner = MultiChannelBinner(n_channels=3, bins_per_channel=4)
        data = np.random.rand(100, 3) * 100
        binner.fit(data)

        with pytest.raises(ValueError):
            binner.transform(np.array([1, 2]))  # Wrong number of channels


class TestBinaryCMM:
    """Tests for BinaryCMM class."""

    def test_store_and_recall(self):
        """Test basic store and recall."""
        cmm = BinaryCMM(pattern_width=8)

        # Store a pattern
        pattern = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.float32)
        cmm.store(pattern)

        # Recall same pattern should get high score
        score = cmm.recall(pattern)
        assert score > 0.9

    def test_multiple_patterns(self):
        """Test storing multiple similar patterns (simulating multi-channel)."""
        # Simulate 2 channels with 4 bins each
        cmm = BinaryCMM(pattern_width=8)

        # Store multiple similar patterns - like normal operating data
        # Channel 1: bins 0-3, Channel 2: bins 4-7
        # Normal patterns cluster around certain bins
        patterns = [
            np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),  # ch1=bin0, ch2=bin0
            np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),  # ch1=bin0, ch2=bin1
            np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32),  # ch1=bin1, ch2=bin0
            np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32),  # ch1=bin1, ch2=bin1
            np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),  # repeat for frequency
        ]

        for p in patterns:
            cmm.store(p)

        assert cmm.pattern_count == 5

        # A pattern similar to stored ones should recall well
        similar_pattern = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        score = cmm.recall(similar_pattern)
        assert score > 0.5  # Should match well since it's in training data

    def test_novel_pattern_low_score(self):
        """Test that novel patterns get lower scores."""
        cmm = BinaryCMM(pattern_width=8)

        # Store patterns with high bits on left
        for _ in range(10):
            pattern = np.zeros(8, dtype=np.float32)
            pattern[:4] = np.random.randint(0, 2, 4)
            if np.sum(pattern) > 0:
                cmm.store(pattern)

        # Query with high bits on right (novel)
        query = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        score = cmm.recall(query)

        # Should be lower than stored patterns
        stored_score = cmm.recall(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        assert score < stored_score

    def test_batch_store(self):
        """Test batch storage."""
        cmm = BinaryCMM(pattern_width=8)

        patterns = np.eye(8, dtype=np.float32)
        cmm.store_batch(patterns)

        assert cmm.pattern_count == 8

    def test_reset(self):
        """Test reset functionality."""
        cmm = BinaryCMM(pattern_width=8)
        cmm.store(np.ones(8, dtype=np.float32))
        cmm.reset()

        assert cmm.pattern_count == 0
        assert np.all(cmm.matrix == 0)


class TestCMMDetector:
    """Tests for CMMDetector class."""

    def test_basic_detection(self):
        """Test basic anomaly detection."""
        # Create simple training data
        n_channels = 4
        n_samples = 200

        # Normal data: values around 50 with small noise
        np.random.seed(42)
        normal_data = np.random.normal(50, 2, (n_samples, n_channels))

        # Create and train detector
        detector = CMMDetector(
            n_channels=n_channels,
            bins_per_channel=16,
            correlation_threshold=0.5,  # Lower threshold for this test
        )
        detector.fit(normal_data)

        assert detector.is_trained
        assert detector.pattern_count == n_samples

        # Test on normal-like data (use mean of training data)
        normal_test = np.array([50, 50, 50, 50])
        result = detector.detect(normal_test)
        # Normal data should have reasonable match
        assert result.anomaly_score < 0.8

        # Test on anomalous data (very different values - outside training range)
        anomaly_test = np.array([200, 200, 200, 200])
        result = detector.detect(anomaly_test)
        # Out-of-range values should be flagged as anomaly
        assert result.anomaly_score > 0.5

    def test_calibrate_and_learn(self):
        """Test separate calibrate and learn steps."""
        detector = CMMDetector(n_channels=3, bins_per_channel=8)

        data = np.random.rand(50, 3) * 100
        detector.calibrate(data)

        for row in data:
            detector.learn(row)

        assert detector.is_trained
        assert detector.pattern_count == 50

    def test_detect_batch(self):
        """Test batch detection."""
        detector = CMMDetector(n_channels=3, bins_per_channel=8)
        data = np.random.rand(50, 3) * 100
        detector.fit(data)

        test_data = np.random.rand(10, 3) * 100
        results = detector.detect_batch(test_data)

        assert len(results) == 10
        for r in results:
            assert 0 <= r.anomaly_score <= 1

    def test_serialization(self):
        """Test state serialization."""
        detector = CMMDetector(n_channels=3, bins_per_channel=8)
        data = np.random.rand(50, 3) * 100
        detector.fit(data)

        state = detector.get_state()

        detector2 = CMMDetector(n_channels=3, bins_per_channel=8)
        detector2.set_state(state)

        # Should produce same results
        test = np.array([50, 50, 50])
        r1 = detector.detect(test)
        r2 = detector2.detect(test)

        assert abs(r1.anomaly_score - r2.anomaly_score) < 0.01

    def test_reset(self):
        """Test detector reset."""
        detector = CMMDetector(n_channels=3, bins_per_channel=8)
        data = np.random.rand(50, 3) * 100
        detector.fit(data)

        detector.reset()

        assert not detector.is_trained
        assert detector.pattern_count == 0

    def test_memory_usage(self):
        """Test memory usage estimation."""
        detector = CMMDetector(n_channels=10, bins_per_channel=64)
        data = np.random.rand(100, 10) * 100
        detector.fit(data)

        mem = detector.get_memory_usage()
        assert "matrix_bytes" in mem
        assert "matrix_mb" in mem
        assert mem["matrix_mb"] < 50  # Should be reasonable


class TestHybridEngine:
    """Tests for HybridEngine class."""

    def test_basic_functionality(self):
        """Test basic engine functionality."""
        engine = HybridEngine(n_channels=4, bins_per_channel=16)

        data = np.random.rand(100, 4) * 100
        engine.fit(data)

        assert engine.is_trained
        result = engine.detect(data[0])
        assert 0 <= result.anomaly_score <= 1

    def test_save_and_load(self, tmp_path):
        """Test model save and load."""
        engine = HybridEngine(n_channels=3, bins_per_channel=8)
        data = np.random.rand(50, 3) * 100
        engine.fit(data)

        # Save
        path = tmp_path / "model.json"
        engine.save(str(path))

        # Load
        engine2 = HybridEngine.load(str(path))

        # Should produce same results
        test = data[0]
        r1 = engine.detect(test)
        r2 = engine2.detect(test)

        assert abs(r1.anomaly_score - r2.anomaly_score) < 0.01

    def test_detector_weights(self):
        """Test detector weight setting."""
        engine = HybridEngine(n_channels=3)
        engine.set_weights({"cmm": 1.0})

        # Should work with valid weights
        assert engine._weights["cmm"] == 1.0

        # Should raise for invalid detector
        with pytest.raises(ValueError):
            engine.set_weights({"nonexistent": 1.0})

    def test_memory_usage(self):
        """Test memory usage reporting."""
        engine = HybridEngine(n_channels=5, bins_per_channel=32)
        data = np.random.rand(100, 5) * 100
        engine.fit(data)

        mem = engine.get_memory_usage()
        assert "total_mb" in mem
        assert "detectors" in mem


class TestPerformance:
    """Performance tests."""

    def test_training_speed(self):
        """Test that training 1000+ samples takes <10 seconds."""
        import time

        n_samples = 1000
        n_channels = 20

        data = np.random.rand(n_samples, n_channels) * 100

        start = time.time()
        engine = HybridEngine(n_channels=n_channels, bins_per_channel=64)
        engine.fit(data)
        elapsed = time.time() - start

        assert elapsed < 10, f"Training took {elapsed:.2f}s, expected <10s"

    def test_detection_speed(self):
        """Test detection speed."""
        import time

        n_samples = 1000
        n_channels = 20

        data = np.random.rand(n_samples, n_channels) * 100

        engine = HybridEngine(n_channels=n_channels, bins_per_channel=64)
        engine.fit(data)

        start = time.time()
        results = engine.detect_batch(data)
        elapsed = time.time() - start

        # Should be fast
        assert elapsed < 5, f"Detection took {elapsed:.2f}s"
        assert len(results) == n_samples

    def test_memory_reasonable(self):
        """Test that memory usage is reasonable."""
        # 20 channels * 64 bins = 1280 pattern width
        # Matrix = 1280^2 * 4 bytes = ~6.5 MB
        engine = HybridEngine(n_channels=20, bins_per_channel=64)
        data = np.random.rand(1000, 20) * 100
        engine.fit(data)

        mem = engine.get_memory_usage()
        assert mem["total_mb"] < 100, f"Memory usage {mem['total_mb']:.2f} MB too high"


class TestAnomalyDetection:
    """Tests for actual anomaly detection capability."""

    def test_mean_shift_detection(self):
        """Test detection of mean shifts >3 sigma."""
        np.random.seed(42)
        n_channels = 5

        # Normal data - tight cluster
        normal_mean = 50
        normal_std = 2  # Smaller std for tighter cluster
        normal_data = np.random.normal(normal_mean, normal_std, (500, n_channels))

        # Anomalous data (mean shifted by >3 sigma)
        anomaly_mean = normal_mean + 6 * normal_std  # 6 sigma shift for clear separation
        anomaly_data = np.random.normal(anomaly_mean, normal_std, (50, n_channels))

        # Train on normal data
        detector = CMMDetector(
            n_channels=n_channels,
            bins_per_channel=32,
            correlation_threshold=0.3,  # Adjusted for realistic detection
        )
        detector.fit(normal_data)

        # Test on normal data - compute scores
        normal_results = detector.detect_batch(normal_data)
        normal_scores = [r.anomaly_score for r in normal_results]

        # Test on anomalous data
        anomaly_results = detector.detect_batch(anomaly_data)
        anomaly_scores = [r.anomaly_score for r in anomaly_results]

        # Anomaly scores should be clearly higher for anomalies
        avg_normal = np.mean(normal_scores)
        avg_anomaly = np.mean(anomaly_scores)
        assert avg_anomaly > avg_normal, f"Anomaly score ({avg_anomaly:.3f}) should be higher than normal ({avg_normal:.3f})"

        # The separation should be meaningful
        assert avg_anomaly - avg_normal > 0.05, f"Score separation too small: {avg_anomaly - avg_normal:.3f}"

    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        data = np.random.rand(100, 5) * 100

        detector1 = CMMDetector(n_channels=5, bins_per_channel=16)
        detector1.fit(data)

        detector2 = CMMDetector(n_channels=5, bins_per_channel=16)
        detector2.fit(data)

        test = np.array([50, 50, 50, 50, 50])
        r1 = detector1.detect(test)
        r2 = detector2.detect(test)

        assert r1.anomaly_score == r2.anomaly_score
