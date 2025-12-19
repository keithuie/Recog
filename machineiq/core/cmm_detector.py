"""
Binary Correlation Matrix Memory (CMM) Detector.

A neuromorphic-inspired approach to multivariate anomaly detection
based on the AURA system's associative memory concepts.
"""

from typing import Literal, Optional
import numpy as np

from machineiq.core.base import BaseDetector, DetectionResult
from machineiq.core.binning import MultiChannelBinner


class BinaryCMM:
    """Binary Correlation Matrix Memory.

    Stores correlations between binary pattern elements as a matrix.
    Instead of storing individual patterns, we accumulate the outer
    products, which provides efficient storage and fast matching.

    The correlation matrix M[i,j] stores how many times bits i and j
    were both active (1) together across all stored patterns.

    Attributes:
        width: Total width of binary patterns.
        matrix: The correlation matrix.
        pattern_count: Number of patterns stored.
    """

    def __init__(self, pattern_width: int):
        """Initialize the CMM.

        Args:
            pattern_width: Width of binary patterns to store.
        """
        self.width = pattern_width
        self.matrix = np.zeros((pattern_width, pattern_width), dtype=np.float32)
        self.pattern_count = 0
        self._diagonal_sum = 0.0  # For normalization

    def store(self, binary_pattern: np.ndarray) -> None:
        """Store a binary pattern in the correlation matrix.

        Uses outer product to update correlations:
        M += pattern @ pattern.T

        Args:
            binary_pattern: 1D binary pattern array.
        """
        binary_pattern = np.asarray(binary_pattern, dtype=np.float32)
        if len(binary_pattern) != self.width:
            raise ValueError(f"Pattern width mismatch: expected {self.width}, got {len(binary_pattern)}")

        # Outer product adds correlations for this pattern
        self.matrix += np.outer(binary_pattern, binary_pattern)
        self.pattern_count += 1
        self._diagonal_sum += np.sum(binary_pattern)

    def store_batch(self, patterns: np.ndarray) -> None:
        """Store multiple patterns efficiently.

        Args:
            patterns: 2D array of shape (n_patterns, pattern_width).
        """
        patterns = np.asarray(patterns, dtype=np.float32)
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)

        if patterns.shape[1] != self.width:
            raise ValueError(f"Pattern width mismatch: expected {self.width}, got {patterns.shape[1]}")

        # Batch outer product: M += sum(p @ p.T for p in patterns)
        # Equivalent to patterns.T @ patterns
        self.matrix += patterns.T @ patterns
        self.pattern_count += patterns.shape[0]
        self._diagonal_sum += np.sum(patterns)

    def recall(self, query_pattern: np.ndarray) -> float:
        """Compute how well a query pattern matches stored patterns.

        The recall score indicates how similar the query is to the
        patterns in memory. Higher scores = better match = more normal.

        Uses a normalized similarity measure that accounts for the
        statistical distribution of stored patterns.

        Args:
            query_pattern: Binary query pattern.

        Returns:
            Correlation score between 0 and 1.
            1.0 = perfect match to stored patterns.
            0.0 = no match at all.
        """
        if self.pattern_count == 0:
            return 0.0

        query_pattern = np.asarray(query_pattern, dtype=np.float32)

        # Get diagonal of matrix - these are the counts for each individual bit
        diag = np.diag(self.matrix)

        # Compute activation: how much the query aligns with stored correlations
        activation = self.matrix @ query_pattern

        query_active = np.sum(query_pattern)
        if query_active == 0:
            return 0.0

        # For each active query bit, compute what fraction of stored patterns
        # also had that bit active
        query_indices = np.where(query_pattern > 0)[0]

        if len(query_indices) == 0:
            return 0.0

        # Calculate per-bit match scores
        bit_scores = []
        for idx in query_indices:
            # How many patterns had this bit active?
            bit_count = diag[idx]
            if bit_count > 0:
                # What fraction of those patterns also matched our other active bits?
                # Use the activation normalized by the bit's frequency
                bit_scores.append(activation[idx] / (bit_count * query_active))
            else:
                # This bit was never seen in training - strong anomaly signal
                bit_scores.append(0.0)

        # Average score across all active bits
        score = np.mean(bit_scores) if bit_scores else 0.0
        return float(np.clip(score, 0.0, 1.0))

    def recall_detailed(self, query_pattern: np.ndarray) -> dict:
        """Detailed recall with per-channel information.

        Args:
            query_pattern: Binary query pattern.

        Returns:
            Dictionary with detailed match information.
        """
        query_pattern = np.asarray(query_pattern, dtype=np.float32)
        activation = self.matrix @ query_pattern

        return {
            "score": self.recall(query_pattern),
            "activation": activation.copy(),
            "query_active_bits": int(np.sum(query_pattern > 0)),
            "max_activation": float(np.max(activation)),
            "mean_activation": float(np.mean(activation)),
        }

    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "width": self.width,
            "matrix": self.matrix.tolist(),
            "pattern_count": self.pattern_count,
            "diagonal_sum": self._diagonal_sum,
        }

    def set_state(self, state: dict) -> None:
        """Restore from serialized state."""
        self.width = state["width"]
        self.matrix = np.array(state["matrix"], dtype=np.float32)
        self.pattern_count = state["pattern_count"]
        self._diagonal_sum = state.get("diagonal_sum", 0.0)

    def reset(self) -> None:
        """Clear all stored patterns."""
        self.matrix = np.zeros((self.width, self.width), dtype=np.float32)
        self.pattern_count = 0
        self._diagonal_sum = 0.0


class CMMDetector(BaseDetector):
    """Correlation Matrix Memory Detector for anomaly detection.

    This detector learns normal patterns from feature vectors using
    binary encoding and correlation storage. For detection, it measures
    how well new patterns match the learned normal patterns.

    How it works:
    1. Convert continuous feature values to binary patterns using adaptive binning
    2. Store binary patterns in a correlation matrix (outer product accumulation)
    3. For detection: convert new data to binary, match against stored patterns
    4. Anomaly = poor match to stored patterns (low correlation score)

    Attributes:
        n_channels: Number of input features/channels.
        bins_per_channel: Number of bins for discretizing each channel.
        correlation_threshold: Threshold for anomaly detection.
        binning_method: Method for adaptive binning.
    """

    def __init__(
        self,
        n_channels: int,
        bins_per_channel: int = 64,
        correlation_threshold: float = 0.85,
        binning_method: Literal["uniform", "quantile", "kmeans"] = "quantile",
        name: str = "CMM",
    ):
        """Initialize the CMM detector.

        Args:
            n_channels: Number of input feature channels.
            bins_per_channel: Number of bins per channel for discretization.
            correlation_threshold: Match threshold (0-1). Lower score = anomaly.
            binning_method: Binning strategy - 'uniform', 'quantile', or 'kmeans'.
            name: Human-readable name for this detector.
        """
        super().__init__(name=name)
        self.n_channels = n_channels
        self.bins_per_channel = bins_per_channel
        self.correlation_threshold = correlation_threshold
        self.binning_method = binning_method

        # Create binner and CMM
        self._binner = MultiChannelBinner(
            n_channels=n_channels,
            bins_per_channel=bins_per_channel,
            method=binning_method,
        )
        self._cmm = BinaryCMM(pattern_width=n_channels * bins_per_channel)

        # Track statistics for confidence calculation
        self._scores: list = []
        self._score_mean: float = 0.0
        self._score_std: float = 1.0

    def _require_fitted_binner(self) -> None:
        """Raise error if binner not fitted."""
        if not self._binner.is_fitted:
            raise RuntimeError("Detector must be calibrated with calibrate() before learning")

    def calibrate(self, feature_vectors: np.ndarray) -> "CMMDetector":
        """Calibrate the binner on sample data.

        This should be called before learn() to set up bin boundaries.
        Can be called with the same data used for learning.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).

        Returns:
            self for method chaining.
        """
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        self._binner.fit(feature_vectors)
        return self

    def learn(self, feature_vector: np.ndarray) -> None:
        """Learn a single normal pattern.

        Args:
            feature_vector: 1D array of feature values.
        """
        self._require_fitted_binner()
        feature_vector = np.asarray(feature_vector).flatten()

        if len(feature_vector) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} features, got {len(feature_vector)}")

        # Convert to binary and store
        binary_pattern = self._binner.transform(feature_vector)
        self._cmm.store(binary_pattern)
        self._pattern_count += 1
        self._is_trained = True

    def learn_batch(self, feature_vectors: np.ndarray) -> None:
        """Learn from a batch of normal patterns.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).
        """
        self._require_fitted_binner()
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        if feature_vectors.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {feature_vectors.shape[1]}")

        # Convert all to binary patterns
        binary_patterns = self._binner.transform_batch(feature_vectors)

        # Store all patterns
        self._cmm.store_batch(binary_patterns)
        self._pattern_count += feature_vectors.shape[0]
        self._is_trained = True

    def fit(self, feature_vectors: np.ndarray) -> "CMMDetector":
        """Convenience method to calibrate and learn in one step.

        Args:
            feature_vectors: 2D array of normal patterns.

        Returns:
            self for method chaining.
        """
        self.calibrate(feature_vectors)
        self.learn_batch(feature_vectors)
        self._update_statistics(feature_vectors)
        return self

    def _update_statistics(self, feature_vectors: np.ndarray) -> None:
        """Update score statistics for confidence calculation."""
        scores = []
        for fv in feature_vectors:
            result = self.detect(fv)
            scores.append(1.0 - result.anomaly_score)  # Correlation score

        self._scores = scores
        if len(scores) > 1:
            self._score_mean = float(np.mean(scores))
            self._score_std = float(np.std(scores))
            if self._score_std < 0.01:
                self._score_std = 0.01  # Minimum std to avoid division issues

    def detect(self, feature_vector: np.ndarray) -> DetectionResult:
        """Detect if a feature vector is anomalous.

        Args:
            feature_vector: 1D array of feature values.

        Returns:
            DetectionResult with anomaly score and classification.
        """
        if not self._is_trained:
            raise RuntimeError("Detector must be trained before detection")

        feature_vector = np.asarray(feature_vector).flatten()
        if len(feature_vector) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} features, got {len(feature_vector)}")

        # Convert to binary and recall
        binary_pattern = self._binner.transform(feature_vector)
        correlation_score = self._cmm.recall(binary_pattern)

        # Anomaly score is inverse of correlation
        # High correlation = normal, Low correlation = anomaly
        anomaly_score = 1.0 - correlation_score

        # Determine if it's an anomaly based on threshold
        is_anomaly = correlation_score < self.correlation_threshold

        # Calculate confidence based on how far from threshold
        if self._score_std > 0:
            z_score = abs(correlation_score - self.correlation_threshold) / self._score_std
            confidence = float(np.clip(1.0 - np.exp(-z_score), 0.5, 1.0))
        else:
            confidence = 0.8 if is_anomaly else 0.9

        return DetectionResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            details={
                "correlation_score": correlation_score,
                "threshold": self.correlation_threshold,
                "pattern_count": self._pattern_count,
            },
        )

    def detect_batch(self, feature_vectors: np.ndarray) -> list:
        """Detect anomalies in a batch of feature vectors.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).

        Returns:
            List of DetectionResult objects.
        """
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        results = []
        for fv in feature_vectors:
            results.append(self.detect(fv))

        return results

    def reset(self) -> None:
        """Reset detector to untrained state."""
        self._binner = MultiChannelBinner(
            n_channels=self.n_channels,
            bins_per_channel=self.bins_per_channel,
            method=self.binning_method,
        )
        self._cmm = BinaryCMM(pattern_width=self.n_channels * self.bins_per_channel)
        self._is_trained = False
        self._pattern_count = 0
        self._scores = []
        self._score_mean = 0.0
        self._score_std = 1.0

    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "name": self.name,
            "n_channels": self.n_channels,
            "bins_per_channel": self.bins_per_channel,
            "correlation_threshold": self.correlation_threshold,
            "binning_method": self.binning_method,
            "binner": self._binner.get_state(),
            "cmm": self._cmm.get_state(),
            "is_trained": self._is_trained,
            "pattern_count": self._pattern_count,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
        }

    def set_state(self, state: dict) -> None:
        """Restore from serialized state."""
        self.name = state["name"]
        self.n_channels = state["n_channels"]
        self.bins_per_channel = state["bins_per_channel"]
        self.correlation_threshold = state["correlation_threshold"]
        self.binning_method = state["binning_method"]

        self._binner = MultiChannelBinner(
            n_channels=self.n_channels,
            bins_per_channel=self.bins_per_channel,
            method=self.binning_method,
        )
        self._binner.set_state(state["binner"])

        self._cmm = BinaryCMM(pattern_width=self.n_channels * self.bins_per_channel)
        self._cmm.set_state(state["cmm"])

        self._is_trained = state["is_trained"]
        self._pattern_count = state["pattern_count"]
        self._score_mean = state.get("score_mean", 0.0)
        self._score_std = state.get("score_std", 1.0)

    def get_memory_usage(self) -> dict:
        """Get estimated memory usage in bytes."""
        # Matrix is (n_channels * bins_per_channel)^2 * 4 bytes (float32)
        matrix_size = (self.n_channels * self.bins_per_channel) ** 2 * 4

        return {
            "matrix_bytes": matrix_size,
            "matrix_mb": matrix_size / (1024 * 1024),
            "pattern_count": self._pattern_count,
        }
