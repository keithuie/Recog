"""
Comprehensive test suite for MachineIQ multivariate anomaly detection.
Tests mathematical correctness of kernels, binning, and scoring.
"""

import pytest
import numpy as np
import math
from miq.core import MIQDetector, DetectorConfig
from miq.core.kernels import TriangularKernel, ParabolicKernel, KernelType, create_kernel
from miq.core.channel_params import ChannelParameters
from miq.core.matrix import Matrix
from miq.core.binning import StaticBinner, AdaptiveBinner, FixedFrequencyBinner, CyclicBinner


class TestKernelMath:
    """Test kernel profile computation and scoring."""

    def test_triangular_profile_values(self):
        """Verify triangular kernel profile is computed correctly."""
        num_bins = 64
        width = 0.5
        kernel = TriangularKernel(num_bins, width)

        # Max score at bin difference 0
        assert kernel.get_score(0) == num_bins  # Should be 64

        # Profile width = 0.5 * 64 / 2 = 16
        profile_width = width * num_bins / 2.0
        gradient = (num_bins - 1) / profile_width

        # Test profile values
        for i in range(len(kernel._profile)):
            expected = math.floor(num_bins - i * gradient + 0.5)
            assert kernel._profile[i] == int(expected), f"Profile mismatch at index {i}"

        # Score should be 0 beyond profile width
        assert kernel.get_score(20) == 0  # Beyond profile width of ~16

    def test_parabolic_profile_values(self):
        """Verify parabolic kernel profile is computed correctly."""
        num_bins = 64
        width = 0.5
        kernel = ParabolicKernel(num_bins, width)

        # Max score at bin difference 0
        strength = (num_bins * num_bins) / 4.0  # 64*64/4 = 1024
        assert kernel.get_max_score() == strength
        assert kernel.get_score(0) == int(strength)

        # Profile width = 0.5 * 64 / 2 = 16
        profile_width = width * num_bins / 2.0
        curvature = (strength - 1) / (profile_width * profile_width)

        # Test profile values
        for i in range(len(kernel._profile)):
            expected = math.floor(strength - (i * i) * curvature + 0.5)
            assert kernel._profile[i] == max(0, int(expected)), f"Profile mismatch at index {i}"

    def test_kernel_symmetry(self):
        """Kernels should be symmetric around bin difference 0."""
        for KernelClass in [TriangularKernel, ParabolicKernel]:
            kernel = KernelClass(64, 0.5)
            for diff in range(20):
                assert kernel.get_score(diff) == kernel.get_score(-diff), \
                    f"{KernelClass.__name__} not symmetric at diff={diff}"

    def test_kernel_monotonic_decrease(self):
        """Kernel scores should monotonically decrease as bin difference increases."""
        for KernelClass in [TriangularKernel, ParabolicKernel]:
            kernel = KernelClass(64, 0.5)
            prev_score = kernel.get_score(0)
            for diff in range(1, 20):
                score = kernel.get_score(diff)
                assert score <= prev_score, \
                    f"{KernelClass.__name__} not monotonic at diff={diff}"
                prev_score = score

    def test_different_widths(self):
        """Test kernels with different width parameters."""
        for width in [0.25, 0.5, 0.75, 1.0]:
            tri = TriangularKernel(64, width)
            para = ParabolicKernel(64, width)

            # Max score should be same regardless of width
            assert tri.get_max_score() == 64
            assert para.get_max_score() == 1024

            # Profile length should scale with width
            expected_profile_len = int(math.floor(width * 64 / 2.0)) + 1
            assert len(tri._profile) == expected_profile_len, f"Tri profile wrong length at width={width}"
            assert len(para._profile) == expected_profile_len, f"Para profile wrong length at width={width}"


class TestBinning:
    """Test binning functions for correctness."""

    def test_static_binner_boundaries(self):
        """Test static binner handles boundary conditions correctly."""
        binner = StaticBinner(0.0, 100.0, 64)

        # Test boundaries
        assert binner.bin(0.0) == 0
        assert binner.bin(100.0) == 63  # Max bin is num_bins - 1
        assert binner.bin(-10.0) == 0   # Below min clamped to 0
        assert binner.bin(110.0) == 63  # Above max clamped to max bin

        # Test midpoint
        assert binner.bin(50.0) == 32  # Should be middle bin

        # Test even distribution
        for i in range(64):
            val = (i + 0.5) * (100.0 / 64.0)
            assert binner.bin(val) == i, f"Bin mismatch for value {val}"

    def test_channel_params_binning(self):
        """Test ChannelParameters bin_value function."""
        kernel = TriangularKernel(64, 0.5)
        params = ChannelParameters("test", min_value=0.0, max_value=100.0, num_bins=64, kernel=kernel)

        # Same tests as static binner
        assert params.bin_value(0.0) == 0
        assert params.bin_value(100.0) == 63
        assert params.bin_value(-10.0) == 0
        assert params.bin_value(110.0) == 63
        assert params.bin_value(50.0) == 32

    def test_adaptive_binner_calibration(self):
        """Test adaptive binner calibration with percentiles."""
        binner = AdaptiveBinner(64, percentile_clip=5.0)

        # Generate data with outliers
        data = np.concatenate([
            np.array([0, 1000]),  # Outliers
            np.random.uniform(10, 90, 100)  # Normal data
        ])

        binner.calibrate(data)

        # Should have clipped to 5th and 95th percentile
        assert binner.is_calibrated
        assert binner.min_val > 0  # Should be above outlier
        assert binner.max_val < 1000  # Should be below outlier

    def test_cyclic_binner(self):
        """Test cyclic binner for periodic values."""
        binner = CyclicBinner(period=360.0, num_bins=36)  # 10 degrees per bin

        assert binner.bin(0.0) == 0
        assert binner.bin(90.0) == 9  # 90/10 = 9
        assert binner.bin(359.0) == 35
        assert binner.bin(360.0) == 0  # Wraps around
        assert binner.bin(370.0) == 1  # 370 % 360 = 10 -> bin 1


class TestMatrix:
    """Test Matrix state storage and matching."""

    def test_exact_match(self):
        """Test exact matching of trained states."""
        kernel = TriangularKernel(64, 0.5)
        params = [
            ChannelParameters("ch1", 0.0, 100.0, 64, kernel),
            ChannelParameters("ch2", 0.0, 100.0, 64, kernel)
        ]
        matrix = Matrix(params)

        # Train a state
        state1 = np.array([32, 32], dtype=np.uint8)
        idx = matrix.train_instance(state1)
        assert idx == 0
        assert matrix.num_trained_instances == 1

        # Same state should return same index
        idx2 = matrix.train_instance(state1)
        assert idx2 == 0
        assert matrix.num_trained_instances == 1  # No new instance

        # Different state should create new instance
        state2 = np.array([33, 32], dtype=np.uint8)
        idx3 = matrix.train_instance(state2)
        assert idx3 == 1
        assert matrix.num_trained_instances == 2

    def test_best_match_exact(self):
        """Exact match should return score of 1.0."""
        kernel = TriangularKernel(64, 0.5)
        params = [
            ChannelParameters("ch1", 0.0, 100.0, 64, kernel),
            ChannelParameters("ch2", 0.0, 100.0, 64, kernel)
        ]
        matrix = Matrix(params)

        state = np.array([32, 32], dtype=np.uint8)
        matrix.train_instance(state)

        score, residuals, best_idx = matrix.get_best_match_with_residuals(state)
        assert score == 1.0, f"Exact match should give score 1.0, got {score}"
        assert best_idx == 0

    def test_best_match_far(self):
        """Far away values should have low match score."""
        kernel = TriangularKernel(64, 0.5)
        params = [
            ChannelParameters("ch1", 0.0, 100.0, 64, kernel),
            ChannelParameters("ch2", 0.0, 100.0, 64, kernel)
        ]
        matrix = Matrix(params)

        state = np.array([10, 10], dtype=np.uint8)
        matrix.train_instance(state)

        # Query far away
        query = np.array([50, 50], dtype=np.uint8)
        score, residuals, best_idx = matrix.get_best_match_with_residuals(query)

        # 40 bin difference, which is beyond profile width (~16), so score should be 0
        assert score == 0.0, f"Far match should give score 0.0, got {score}"

    def test_max_possible_score_calculation(self):
        """Verify max possible score is sum of kernel max scores."""
        kernel = TriangularKernel(64, 0.5)
        params = [
            ChannelParameters("ch1", 0.0, 100.0, 64, kernel),
            ChannelParameters("ch2", 0.0, 100.0, 64, kernel),
            ChannelParameters("ch3", 0.0, 100.0, 64, kernel),
        ]
        matrix = Matrix(params)

        # 3 channels * 64 max score per channel = 192
        expected_max = 3 * 64
        assert matrix._max_possible_raw_score == expected_max


class TestDetector:
    """Test full detector pipeline."""

    def test_detector_exact_learned_state(self):
        """Detector should recognize exact learned states."""
        config = DetectorConfig(bins_per_channel=64, kernel_type=KernelType.TRIANGULAR)
        detector = MIQDetector(config)

        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
            {"name": "ch2", "min_value": 0, "max_value": 100}
        ])

        # Learn a specific state
        detector.start_learning()
        detector.process(np.array([50.0, 50.0]))
        detector.stop_learning()

        # Query same state
        result = detector.process(np.array([50.0, 50.0]))
        assert result.match_strength == 1.0, f"Exact learned state should match 1.0, got {result.match_strength}"
        assert result.anomaly_score == 0.0
        assert not result.is_anomaly

    def test_detector_anomaly_detection(self):
        """Detector should detect anomalies (values far from learned states)."""
        config = DetectorConfig(bins_per_channel=64, kernel_type=KernelType.TRIANGULAR)
        detector = MIQDetector(config)

        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
            {"name": "ch2", "min_value": 0, "max_value": 100}
        ])

        # Learn states around 50
        detector.start_learning()
        for _ in range(10):
            detector.process(np.array([50.0, 50.0]))
        detector.stop_learning()

        # Query very different state
        result = detector.process(np.array([10.0, 10.0]))

        # Should be anomaly (low match, high anomaly score)
        assert result.match_strength < 0.5
        assert result.anomaly_score > 0.5
        assert result.is_anomaly

    def test_detector_multiple_learned_states(self):
        """Detector should find best match among multiple learned states."""
        config = DetectorConfig(bins_per_channel=64, kernel_type=KernelType.TRIANGULAR, kernel_width=0.5)
        detector = MIQDetector(config)

        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
        ])

        # Learn two distinct states
        detector.start_learning()
        detector.process(np.array([25.0]))  # Bin ~16
        detector.process(np.array([75.0]))  # Bin ~48
        detector.stop_learning()

        assert detector.num_trained_states == 2

        # Query near first state
        result1 = detector.process(np.array([25.0]))
        assert result1.match_strength == 1.0
        assert result1.best_match_index == 0

        # Query near second state
        result2 = detector.process(np.array([75.0]))
        assert result2.match_strength == 1.0
        assert result2.best_match_index == 1

    def test_detector_parabolic_kernel(self):
        """Test detector with parabolic kernel."""
        config = DetectorConfig(bins_per_channel=64, kernel_type=KernelType.PARABOLIC)
        detector = MIQDetector(config)

        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
        ])

        detector.start_learning()
        detector.process(np.array([50.0]))
        detector.stop_learning()

        # Exact match
        result = detector.process(np.array([50.0]))
        assert result.match_strength == 1.0

    def test_residuals_identify_anomalous_channels(self):
        """Residuals should identify which channels are anomalous."""
        config = DetectorConfig(bins_per_channel=64, kernel_type=KernelType.TRIANGULAR)
        detector = MIQDetector(config)

        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
            {"name": "ch2", "min_value": 0, "max_value": 100},
            {"name": "ch3", "min_value": 0, "max_value": 100}
        ])

        # Learn a state
        detector.start_learning()
        detector.process(np.array([50.0, 50.0, 50.0]))
        detector.stop_learning()

        # Query with only channel 2 different
        result = detector.process(np.array([50.0, 10.0, 50.0]))

        # Channel 1 and 3 should have high residuals (good match)
        # Channel 2 should have low residual (poor match)
        top_contributors = result.get_top_contributors(3)

        # The anomalous channel should be at the top (lowest score/residual)
        assert top_contributors[0][0] == "ch2", f"ch2 should be top contributor, got {top_contributors}"


class TestEndToEnd:
    """End-to-end tests simulating real usage."""

    def test_typical_usage_scenario(self):
        """Test typical industrial monitoring scenario."""
        detector = MIQDetector()

        detector.configure_channels([
            {"name": "temperature", "min_value": 20, "max_value": 80},
            {"name": "pressure", "min_value": 0, "max_value": 100},
            {"name": "vibration", "min_value": 0, "max_value": 10}
        ])

        # Learning phase - normal operation
        detector.start_learning()
        for _ in range(100):
            temp = 45 + np.random.randn() * 2  # ~45°C ± 2
            pressure = 50 + np.random.randn() * 5  # ~50 psi ± 5
            vibration = 2 + np.random.randn() * 0.5  # ~2 mm/s ± 0.5
            detector.process(np.array([temp, pressure, vibration]))
        detector.stop_learning()

        print(f"Trained states: {detector.num_trained_states}")

        # Monitoring phase - normal data should have high match
        normal_scores = []
        for _ in range(20):
            temp = 45 + np.random.randn() * 2
            pressure = 50 + np.random.randn() * 5
            vibration = 2 + np.random.randn() * 0.5
            result = detector.process(np.array([temp, pressure, vibration]))
            normal_scores.append(result.match_strength)

        avg_normal = np.mean(normal_scores)
        assert avg_normal > 0.7, f"Normal data should have high match, got avg={avg_normal}"

        # Anomaly - sudden temperature spike
        result = detector.process(np.array([75.0, 50.0, 2.0]))
        assert result.is_anomaly, "Temperature spike should be detected as anomaly"

        # Check that temperature is identified as contributor
        contributors = result.get_top_contributors(1)
        assert contributors[0][0] == "temperature", "Temperature should be top contributor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
