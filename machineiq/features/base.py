"""Base classes for feature extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FeatureSet:
    """Container for extracted features.

    Attributes:
        features: Dictionary of feature name to value.
        feature_names: Ordered list of feature names.
        vector: Feature values as numpy array (in order of feature_names).
        metadata: Optional metadata about the extraction.
    """
    features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    metadata: Optional[Dict] = None

    @property
    def vector(self) -> np.ndarray:
        """Get features as a numpy array in consistent order."""
        if not self.feature_names:
            self.feature_names = sorted(self.features.keys())
        return np.array([self.features[name] for name in self.feature_names])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, key: str) -> float:
        return self.features[key]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "features": self.features.copy(),
            "feature_names": self.feature_names.copy(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSet":
        """Create from dictionary representation."""
        return cls(
            features=data["features"],
            feature_names=data["feature_names"],
            metadata=data.get("metadata"),
        )


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    All feature extractors should inherit from this class and implement
    the extract method.
    """

    def __init__(self, name: str = "BaseExtractor"):
        """Initialize the feature extractor.

        Args:
            name: Human-readable name for this extractor.
        """
        self.name = name
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        """List of feature names produced by this extractor."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Number of features produced."""
        return len(self._feature_names)

    @abstractmethod
    def extract(self, data: np.ndarray, **kwargs) -> FeatureSet:
        """Extract features from input data.

        Args:
            data: Input data (format depends on extractor type).
            **kwargs: Additional extraction parameters.

        Returns:
            FeatureSet containing extracted features.
        """
        pass

    def extract_batch(self, data_batch: List[np.ndarray], **kwargs) -> List[FeatureSet]:
        """Extract features from multiple data samples.

        Args:
            data_batch: List of input data arrays.
            **kwargs: Additional extraction parameters.

        Returns:
            List of FeatureSet objects.
        """
        return [self.extract(data, **kwargs) for data in data_batch]

    def extract_to_array(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Extract features and return as numpy array.

        Args:
            data: Input data.
            **kwargs: Additional extraction parameters.

        Returns:
            1D numpy array of feature values.
        """
        return self.extract(data, **kwargs).vector

    def extract_batch_to_array(
        self, data_batch: List[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Extract features from batch and return as 2D array.

        Args:
            data_batch: List of input data arrays.
            **kwargs: Additional extraction parameters.

        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        feature_sets = self.extract_batch(data_batch, **kwargs)
        return np.array([fs.vector for fs in feature_sets])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', n_features={self.n_features})"
