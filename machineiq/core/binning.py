"""Adaptive binning for converting continuous values to binary patterns."""

from typing import Literal, Optional
import numpy as np


class AdaptiveBinner:
    """Adaptive binning for converting continuous values to binary patterns.

    Supports multiple binning strategies:
    - 'uniform': Equal-width bins based on min/max values
    - 'quantile': Equal-frequency bins (better for skewed data)
    - 'kmeans': Cluster-based bins for multimodal distributions

    Example:
        >>> binner = AdaptiveBinner(n_bins=8, method='quantile')
        >>> binner.fit(np.array([1, 2, 3, 10, 11, 12, 50, 100]))
        >>> pattern = binner.transform(5.0)
        >>> print(pattern)  # One-hot binary pattern
        [0. 0. 1. 0. 0. 0. 0. 0.]
    """

    def __init__(
        self,
        n_bins: int = 64,
        method: Literal["uniform", "quantile", "kmeans"] = "quantile",
    ):
        """Initialize the adaptive binner.

        Args:
            n_bins: Number of bins for discretization.
            method: Binning method - 'uniform', 'quantile', or 'kmeans'.
        """
        self.n_bins = n_bins
        self.method = method
        self._bin_edges: Optional[np.ndarray] = None
        self._is_fitted = False
        self._min_val: Optional[float] = None
        self._max_val: Optional[float] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the binner has been fitted to data."""
        return self._is_fitted

    def fit(self, values: np.ndarray) -> "AdaptiveBinner":
        """Learn bin boundaries from training data.

        Args:
            values: 1D array of continuous values to learn bins from.

        Returns:
            self for method chaining.
        """
        values = np.asarray(values).flatten()
        if len(values) == 0:
            raise ValueError("Cannot fit on empty data")

        self._min_val = float(np.min(values))
        self._max_val = float(np.max(values))

        # Handle edge case where all values are the same
        if self._min_val == self._max_val:
            self._bin_edges = np.array([self._min_val - 0.5, self._max_val + 0.5])
            self._is_fitted = True
            return self

        if self.method == "uniform":
            self._fit_uniform(values)
        elif self.method == "quantile":
            self._fit_quantile(values)
        elif self.method == "kmeans":
            self._fit_kmeans(values)
        else:
            raise ValueError(f"Unknown binning method: {self.method}")

        self._is_fitted = True
        return self

    def _fit_uniform(self, values: np.ndarray) -> None:
        """Fit uniform (equal-width) bins."""
        self._bin_edges = np.linspace(self._min_val, self._max_val, self.n_bins + 1)

    def _fit_quantile(self, values: np.ndarray) -> None:
        """Fit quantile (equal-frequency) bins."""
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        self._bin_edges = np.percentile(values, percentiles)
        # Ensure unique bin edges
        self._bin_edges = np.unique(self._bin_edges)
        # If we have fewer unique edges than bins, fall back to uniform
        if len(self._bin_edges) < 3:
            self._fit_uniform(values)

    def _fit_kmeans(self, values: np.ndarray) -> None:
        """Fit cluster-based bins using k-means."""
        # Simple 1D k-means implementation
        values_sorted = np.sort(values)
        n_samples = len(values_sorted)

        if n_samples < self.n_bins:
            # Not enough samples, fall back to uniform
            self._fit_uniform(values)
            return

        # Initialize centroids by taking evenly spaced samples
        indices = np.linspace(0, n_samples - 1, self.n_bins, dtype=int)
        centroids = values_sorted[indices].copy()

        # Run k-means for a few iterations
        for _ in range(10):
            # Assign each value to nearest centroid
            distances = np.abs(values.reshape(-1, 1) - centroids.reshape(1, -1))
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros(self.n_bins)
            for i in range(self.n_bins):
                mask = assignments == i
                if np.any(mask):
                    new_centroids[i] = np.mean(values[mask])
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Create bin edges midway between sorted centroids
        centroids_sorted = np.sort(centroids)
        edges = [self._min_val]
        for i in range(len(centroids_sorted) - 1):
            edges.append((centroids_sorted[i] + centroids_sorted[i + 1]) / 2)
        edges.append(self._max_val)
        self._bin_edges = np.array(edges)

    def transform(self, value: float) -> np.ndarray:
        """Convert a continuous value to a one-hot binary pattern.

        Args:
            value: Continuous value to transform.

        Returns:
            One-hot binary pattern as numpy array.

        Raises:
            RuntimeError: If the binner has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Binner must be fitted before transform")

        # Find which bin the value falls into
        bin_idx = np.searchsorted(self._bin_edges, value, side="right") - 1
        # Clamp to valid range
        bin_idx = max(0, min(len(self._bin_edges) - 2, bin_idx))

        # Create one-hot pattern
        actual_n_bins = len(self._bin_edges) - 1
        pattern = np.zeros(self.n_bins)
        if bin_idx < self.n_bins:
            pattern[bin_idx] = 1.0
        else:
            pattern[-1] = 1.0

        return pattern

    def transform_batch(self, values: np.ndarray) -> np.ndarray:
        """Transform a batch of values to binary patterns.

        Args:
            values: 1D array of continuous values.

        Returns:
            2D array of shape (n_values, n_bins) with one-hot patterns.
        """
        values = np.asarray(values).flatten()
        patterns = np.zeros((len(values), self.n_bins))
        for i, v in enumerate(values):
            patterns[i] = self.transform(v)
        return patterns

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit the binner and transform values in one step.

        Args:
            values: 1D array of continuous values.

        Returns:
            2D array of binary patterns.
        """
        self.fit(values)
        return self.transform_batch(values)

    def get_state(self) -> dict:
        """Get the current state for serialization."""
        return {
            "n_bins": self.n_bins,
            "method": self.method,
            "bin_edges": self._bin_edges.tolist() if self._bin_edges is not None else None,
            "min_val": self._min_val,
            "max_val": self._max_val,
            "is_fitted": self._is_fitted,
        }

    def set_state(self, state: dict) -> None:
        """Restore state from serialization."""
        self.n_bins = state["n_bins"]
        self.method = state["method"]
        self._bin_edges = np.array(state["bin_edges"]) if state["bin_edges"] else None
        self._min_val = state["min_val"]
        self._max_val = state["max_val"]
        self._is_fitted = state["is_fitted"]


class MultiChannelBinner:
    """Handles binning for multiple feature channels.

    Creates one AdaptiveBinner per feature channel, allowing each
    channel to have its own bin boundaries.
    """

    def __init__(
        self,
        n_channels: int,
        bins_per_channel: int = 64,
        method: Literal["uniform", "quantile", "kmeans"] = "quantile",
    ):
        """Initialize the multi-channel binner.

        Args:
            n_channels: Number of feature channels.
            bins_per_channel: Number of bins per channel.
            method: Binning method for all channels.
        """
        self.n_channels = n_channels
        self.bins_per_channel = bins_per_channel
        self.method = method
        self._binners = [
            AdaptiveBinner(n_bins=bins_per_channel, method=method)
            for _ in range(n_channels)
        ]
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if all channel binners have been fitted."""
        return self._is_fitted

    @property
    def pattern_width(self) -> int:
        """Total width of the concatenated binary pattern."""
        return self.n_channels * self.bins_per_channel

    def fit(self, feature_vectors: np.ndarray) -> "MultiChannelBinner":
        """Fit binners for all channels.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).

        Returns:
            self for method chaining.
        """
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        if feature_vectors.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {feature_vectors.shape[1]}"
            )

        for i in range(self.n_channels):
            self._binners[i].fit(feature_vectors[:, i])

        self._is_fitted = True
        return self

    def transform(self, feature_vector: np.ndarray) -> np.ndarray:
        """Convert a feature vector to a concatenated binary pattern.

        Args:
            feature_vector: 1D array of shape (n_channels,).

        Returns:
            Concatenated one-hot binary pattern.
        """
        if not self._is_fitted:
            raise RuntimeError("MultiChannelBinner must be fitted before transform")

        feature_vector = np.asarray(feature_vector).flatten()
        if len(feature_vector) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} features, got {len(feature_vector)}"
            )

        patterns = []
        for i in range(self.n_channels):
            patterns.append(self._binners[i].transform(feature_vector[i]))

        return np.concatenate(patterns)

    def transform_batch(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Transform a batch of feature vectors.

        Args:
            feature_vectors: 2D array of shape (n_samples, n_channels).

        Returns:
            2D array of shape (n_samples, n_channels * bins_per_channel).
        """
        feature_vectors = np.asarray(feature_vectors)
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        patterns = np.zeros((feature_vectors.shape[0], self.pattern_width))
        for i in range(feature_vectors.shape[0]):
            patterns[i] = self.transform(feature_vectors[i])

        return patterns

    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            "n_channels": self.n_channels,
            "bins_per_channel": self.bins_per_channel,
            "method": self.method,
            "binners": [b.get_state() for b in self._binners],
            "is_fitted": self._is_fitted,
        }

    def set_state(self, state: dict) -> None:
        """Restore from serialized state."""
        self.n_channels = state["n_channels"]
        self.bins_per_channel = state["bins_per_channel"]
        self.method = state["method"]
        self._binners = []
        for bstate in state["binners"]:
            binner = AdaptiveBinner()
            binner.set_state(bstate)
            self._binners.append(binner)
        self._is_fitted = state["is_fitted"]
