"""Base detector interface for all ML-based anomaly detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """Result from an anomaly detection operation.

    Attributes:
        anomaly_score: Float between 0 and 1, where 1 = definite anomaly.
        is_anomaly: Boolean indicating whether the score exceeds threshold.
        confidence: Confidence level in the detection (0-1).
        matched_pattern: ID of the closest matching normal pattern, if any.
        details: Additional detector-specific information.
    """
    anomaly_score: float
    is_anomaly: bool
    confidence: float = 1.0
    matched_pattern: Optional[str] = None
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "matched_pattern": self.matched_pattern,
            "details": self.details or {},
        }


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors.

    All detectors in MachineIQ implement this interface to ensure
    consistent behavior and easy swapping of detection algorithms.
    """

    def __init__(self, name: str = "BaseDetector"):
        """Initialize the detector.

        Args:
            name: Human-readable name for this detector instance.
        """
        self.name = name
        self._is_trained = False
        self._pattern_count = 0

    @property
    def is_trained(self) -> bool:
        """Check if the detector has been trained with normal patterns."""
        return self._is_trained

    @property
    def pattern_count(self) -> int:
        """Return the number of patterns learned."""
        return self._pattern_count

    @abstractmethod
    def learn(self, feature_vector: np.ndarray) -> None:
        """Learn a normal pattern from a feature vector.

        This method should be called multiple times with different
        examples of normal operating conditions.

        Args:
            feature_vector: 1D numpy array of extracted features.
        """
        pass

    @abstractmethod
    def learn_batch(self, feature_vectors: np.ndarray) -> None:
        """Learn from a batch of normal patterns.

        Args:
            feature_vectors: 2D numpy array of shape (n_samples, n_features).
        """
        pass

    @abstractmethod
    def detect(self, feature_vector: np.ndarray) -> DetectionResult:
        """Detect anomalies in a single feature vector.

        Args:
            feature_vector: 1D numpy array of extracted features.

        Returns:
            DetectionResult with anomaly score and related information.
        """
        pass

    @abstractmethod
    def detect_batch(self, feature_vectors: np.ndarray) -> list:
        """Detect anomalies in a batch of feature vectors.

        Args:
            feature_vectors: 2D numpy array of shape (n_samples, n_features).

        Returns:
            List of DetectionResult objects.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the detector to its initial untrained state."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get the current state of the detector for serialization.

        Returns:
            Dictionary containing all state needed to restore the detector.
        """
        pass

    @abstractmethod
    def set_state(self, state: dict) -> None:
        """Restore the detector from a saved state.

        Args:
            state: Dictionary returned by get_state().
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained}, patterns={self.pattern_count})"
