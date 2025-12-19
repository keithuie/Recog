"""
Hybrid ML Engine for anomaly detection.

Combines multiple detection algorithms and provides a unified interface.
Phase 1 includes only CMM detector; future phases will add autoencoder and LSTM.
"""

from typing import Dict, List, Literal, Optional
import json
import numpy as np
from pathlib import Path

from machineiq.core.base import BaseDetector, DetectionResult
from machineiq.core.cmm_detector import CMMDetector


class HybridEngine:
    """Hybrid anomaly detection engine combining multiple detectors.

    The hybrid engine manages multiple detection algorithms and combines
    their outputs for more robust anomaly detection. Currently supports:
    - CMM (Correlation Matrix Memory): Fast binary pattern matching

    Future phases will add:
    - Autoencoder: Deep learning for complex pattern reconstruction
    - LSTM: Temporal pattern analysis for sequence-based anomalies

    Attributes:
        n_channels: Number of input feature channels.
        detectors: Dictionary of active detector instances.
        weights: Weights for combining detector outputs.
    """

    def __init__(
        self,
        n_channels: int,
        bins_per_channel: int = 64,
        correlation_threshold: float = 0.85,
        binning_method: Literal["uniform", "quantile", "kmeans"] = "quantile",
        enable_cmm: bool = True,
        enable_autoencoder: bool = False,  # Phase 2
        enable_lstm: bool = False,  # Phase 2
    ):
        """Initialize the hybrid engine.

        Args:
            n_channels: Number of input feature channels.
            bins_per_channel: Bins for CMM discretization.
            correlation_threshold: CMM anomaly threshold.
            binning_method: CMM binning method.
            enable_cmm: Whether to use CMM detector.
            enable_autoencoder: Placeholder for Phase 2.
            enable_lstm: Placeholder for Phase 2.
        """
        self.n_channels = n_channels
        self.bins_per_channel = bins_per_channel
        self.correlation_threshold = correlation_threshold
        self.binning_method = binning_method

        self._detectors: Dict[str, BaseDetector] = {}
        self._weights: Dict[str, float] = {}
        self._is_trained = False
        self._pattern_count = 0

        # Initialize enabled detectors
        if enable_cmm:
            self._detectors["cmm"] = CMMDetector(
                n_channels=n_channels,
                bins_per_channel=bins_per_channel,
                correlation_threshold=correlation_threshold,
                binning_method=binning_method,
                name="CMM",
            )
            self._weights["cmm"] = 1.0

        # Placeholder for future detectors
        if enable_autoencoder:
            raise NotImplementedError("Autoencoder detector coming in Phase 2")
        if enable_lstm:
            raise NotImplementedError("LSTM detector coming in Phase 2")

    @property
    def is_trained(self) -> bool:
        """Check if the engine has been trained."""
        return self._is_trained

    @property
    def pattern_count(self) -> int:
        """Number of patterns learned."""
        return self._pattern_count

    @property
    def detectors(self) -> Dict[str, BaseDetector]:
        """Get active detectors."""
        return self._detectors

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for detector combination.

        Args:
            weights: Dictionary mapping detector names to weights.
        """
        for name in weights:
            if name not in self._detectors:
                raise ValueError(f"Unknown detector: {name}")

        # Normalize weights
        total = sum(weights.values())
        self._weights = {k: v / total for k, v in weights.items()}

    def calibrate(self, feature_vectors: np.ndarray) -> "HybridEngine":
        """Calibrate all detectors on sample data.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).

        Returns:
            self for method chaining.
        """
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        for detector in self._detectors.values():
            if hasattr(detector, "calibrate"):
                detector.calibrate(feature_vectors)

        return self

    def learn(self, feature_vector: np.ndarray) -> None:
        """Learn a single normal pattern.

        Args:
            feature_vector: 1D array of feature values.
        """
        for detector in self._detectors.values():
            detector.learn(feature_vector)

        self._pattern_count += 1
        self._is_trained = True

    def learn_batch(self, feature_vectors: np.ndarray) -> None:
        """Learn from a batch of normal patterns.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).
        """
        for detector in self._detectors.values():
            detector.learn_batch(feature_vectors)

        self._pattern_count += len(feature_vectors)
        self._is_trained = True

    def fit(self, feature_vectors: np.ndarray) -> "HybridEngine":
        """Calibrate and learn in one step.

        Args:
            feature_vectors: 2D array of normal patterns.

        Returns:
            self for method chaining.
        """
        self.calibrate(feature_vectors)
        self.learn_batch(feature_vectors)
        return self

    def detect(self, feature_vector: np.ndarray) -> DetectionResult:
        """Detect anomalies in a single feature vector.

        Combines results from all active detectors using weighted averaging.

        Args:
            feature_vector: 1D array of feature values.

        Returns:
            Combined DetectionResult.
        """
        if not self._is_trained:
            raise RuntimeError("Engine must be trained before detection")

        # Get results from all detectors
        detector_results: Dict[str, DetectionResult] = {}
        for name, detector in self._detectors.items():
            detector_results[name] = detector.detect(feature_vector)

        # Combine scores using weights
        combined_score = sum(
            self._weights[name] * result.anomaly_score
            for name, result in detector_results.items()
        )

        # Combined confidence (weighted average)
        combined_confidence = sum(
            self._weights[name] * result.confidence
            for name, result in detector_results.items()
        )

        # Determine anomaly by voting or threshold
        # For now, use weighted average score
        is_anomaly = combined_score > (1.0 - self.correlation_threshold)

        return DetectionResult(
            anomaly_score=combined_score,
            is_anomaly=is_anomaly,
            confidence=combined_confidence,
            details={
                "detector_results": {
                    name: result.to_dict()
                    for name, result in detector_results.items()
                },
                "weights": self._weights.copy(),
            },
        )

    def detect_batch(self, feature_vectors: np.ndarray) -> List[DetectionResult]:
        """Detect anomalies in a batch of feature vectors.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).

        Returns:
            List of DetectionResult objects.
        """
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        return [self.detect(fv) for fv in feature_vectors]

    def reset(self) -> None:
        """Reset all detectors to untrained state."""
        for detector in self._detectors.values():
            detector.reset()

        self._is_trained = False
        self._pattern_count = 0

    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "n_channels": self.n_channels,
            "bins_per_channel": self.bins_per_channel,
            "correlation_threshold": self.correlation_threshold,
            "binning_method": self.binning_method,
            "weights": self._weights,
            "detectors": {
                name: detector.get_state()
                for name, detector in self._detectors.items()
            },
            "is_trained": self._is_trained,
            "pattern_count": self._pattern_count,
        }

    def set_state(self, state: dict) -> None:
        """Restore from serialized state."""
        self.n_channels = state["n_channels"]
        self.bins_per_channel = state["bins_per_channel"]
        self.correlation_threshold = state["correlation_threshold"]
        self.binning_method = state["binning_method"]
        self._weights = state["weights"]
        self._is_trained = state["is_trained"]
        self._pattern_count = state["pattern_count"]

        # Restore detectors
        self._detectors = {}
        for name, dstate in state["detectors"].items():
            if name == "cmm":
                detector = CMMDetector(
                    n_channels=self.n_channels,
                    bins_per_channel=self.bins_per_channel,
                )
                detector.set_state(dstate)
                self._detectors[name] = detector

    def save(self, path: str) -> None:
        """Save engine state to file.

        Args:
            path: Path to save file (JSON format).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = self.get_state()

        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return super().default(obj)

        with open(path, "w") as f:
            json.dump(state, f, indent=2, cls=NumpyEncoder)

    @classmethod
    def load(cls, path: str) -> "HybridEngine":
        """Load engine state from file.

        Args:
            path: Path to saved state file.

        Returns:
            Restored HybridEngine instance.
        """
        with open(path, "r") as f:
            state = json.load(f)

        engine = cls(
            n_channels=state["n_channels"],
            bins_per_channel=state["bins_per_channel"],
            correlation_threshold=state["correlation_threshold"],
            binning_method=state["binning_method"],
        )
        engine.set_state(state)
        return engine

    def get_memory_usage(self) -> dict:
        """Get estimated memory usage from all detectors."""
        usage = {"total_mb": 0.0, "detectors": {}}

        for name, detector in self._detectors.items():
            if hasattr(detector, "get_memory_usage"):
                d_usage = detector.get_memory_usage()
                usage["detectors"][name] = d_usage
                usage["total_mb"] += d_usage.get("matrix_mb", 0)

        return usage

    def __repr__(self) -> str:
        detector_str = ", ".join(self._detectors.keys())
        return (
            f"HybridEngine(n_channels={self.n_channels}, "
            f"detectors=[{detector_str}], "
            f"trained={self._is_trained}, "
            f"patterns={self._pattern_count})"
        )
