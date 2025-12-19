"""Weighted kernels for pattern matching in CMM detector."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseKernel(ABC):
    """Abstract base class for weighting kernels.

    Kernels are used to weight the contribution of pattern matches
    based on distance or similarity metrics.
    """

    @abstractmethod
    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """Apply the kernel to a distance value.

        Args:
            distance: Distance or dissimilarity value(s).

        Returns:
            Weight value(s) between 0 and 1.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get kernel parameters."""
        pass


class GaussianKernel(BaseKernel):
    """Gaussian (RBF) kernel for distance weighting.

    Weight = exp(-distance^2 / (2 * sigma^2))

    Good for smooth, continuous similarity measures.
    """

    def __init__(self, sigma: float = 1.0):
        """Initialize Gaussian kernel.

        Args:
            sigma: Bandwidth parameter controlling spread.
        """
        self.sigma = sigma

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """Apply Gaussian kernel."""
        distance = np.asarray(distance)
        return np.exp(-distance**2 / (2 * self.sigma**2))

    def get_params(self) -> dict:
        return {"type": "gaussian", "sigma": self.sigma}


class WeightedKernel(BaseKernel):
    """Weighted kernel with configurable decay.

    Weight = 1 / (1 + alpha * distance^beta)

    Parameters:
        alpha: Controls how quickly weight decreases with distance.
        beta: Controls the shape of the decay curve.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        """Initialize weighted kernel.

        Args:
            alpha: Decay rate parameter.
            beta: Decay shape parameter.
        """
        self.alpha = alpha
        self.beta = beta

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """Apply weighted kernel."""
        distance = np.asarray(distance)
        return 1.0 / (1.0 + self.alpha * np.power(distance, self.beta))

    def get_params(self) -> dict:
        return {"type": "weighted", "alpha": self.alpha, "beta": self.beta}


class LinearKernel(BaseKernel):
    """Linear decay kernel.

    Weight = max(0, 1 - distance / max_distance)

    Simple linear interpolation from 1 at distance=0 to 0 at max_distance.
    """

    def __init__(self, max_distance: float = 1.0):
        """Initialize linear kernel.

        Args:
            max_distance: Distance at which weight becomes zero.
        """
        self.max_distance = max_distance

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """Apply linear kernel."""
        distance = np.asarray(distance)
        return np.maximum(0.0, 1.0 - distance / self.max_distance)

    def get_params(self) -> dict:
        return {"type": "linear", "max_distance": self.max_distance}


class ExponentialKernel(BaseKernel):
    """Exponential decay kernel.

    Weight = exp(-lambda * distance)

    Faster decay than Gaussian for larger distances.
    """

    def __init__(self, decay_rate: float = 1.0):
        """Initialize exponential kernel.

        Args:
            decay_rate: Lambda parameter controlling decay speed.
        """
        self.decay_rate = decay_rate

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """Apply exponential kernel."""
        distance = np.asarray(distance)
        return np.exp(-self.decay_rate * distance)

    def get_params(self) -> dict:
        return {"type": "exponential", "decay_rate": self.decay_rate}


class ThresholdKernel(BaseKernel):
    """Hard threshold kernel.

    Weight = 1 if distance < threshold else 0

    Binary decision boundary for pattern matching.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize threshold kernel.

        Args:
            threshold: Distance threshold for acceptance.
        """
        self.threshold = threshold

    def __call__(self, distance: np.ndarray) -> np.ndarray:
        """Apply threshold kernel."""
        distance = np.asarray(distance)
        return (distance < self.threshold).astype(np.float64)

    def get_params(self) -> dict:
        return {"type": "threshold", "threshold": self.threshold}


class ChannelWeightedKernel:
    """Applies different weights to different feature channels.

    Useful when some features are more important than others
    for anomaly detection.
    """

    def __init__(
        self,
        channel_weights: np.ndarray,
        bins_per_channel: int,
        base_kernel: Optional[BaseKernel] = None,
    ):
        """Initialize channel-weighted kernel.

        Args:
            channel_weights: Weight for each channel (higher = more important).
            bins_per_channel: Number of bins per channel.
            base_kernel: Optional base kernel to apply after weighting.
        """
        self.channel_weights = np.asarray(channel_weights)
        self.bins_per_channel = bins_per_channel
        self.base_kernel = base_kernel or GaussianKernel(sigma=1.0)

        # Expand channel weights to pattern width
        self._expanded_weights = np.repeat(channel_weights, bins_per_channel)
        self._expanded_weights = self._expanded_weights / np.sum(self._expanded_weights)

    def weight_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Apply channel weights to a binary pattern.

        Args:
            pattern: Binary pattern of shape (n_channels * bins_per_channel,).

        Returns:
            Weighted pattern.
        """
        return pattern * self._expanded_weights

    def weighted_similarity(
        self, pattern1: np.ndarray, pattern2: np.ndarray
    ) -> float:
        """Compute weighted similarity between two patterns.

        Args:
            pattern1: First binary pattern.
            pattern2: Second binary pattern.

        Returns:
            Weighted similarity score.
        """
        weighted1 = self.weight_pattern(pattern1)
        weighted2 = self.weight_pattern(pattern2)

        # Compute weighted dot product similarity
        similarity = np.sum(weighted1 * weighted2)
        normalization = np.sqrt(np.sum(weighted1**2) * np.sum(weighted2**2))

        if normalization > 0:
            return similarity / normalization
        return 0.0

    def get_params(self) -> dict:
        return {
            "type": "channel_weighted",
            "channel_weights": self.channel_weights.tolist(),
            "bins_per_channel": self.bins_per_channel,
            "base_kernel": self.base_kernel.get_params(),
        }


def create_kernel(kernel_type: str, **params) -> BaseKernel:
    """Factory function to create kernels by name.

    Args:
        kernel_type: One of 'gaussian', 'weighted', 'linear', 'exponential', 'threshold'.
        **params: Kernel-specific parameters.

    Returns:
        Instantiated kernel object.
    """
    kernels = {
        "gaussian": GaussianKernel,
        "weighted": WeightedKernel,
        "linear": LinearKernel,
        "exponential": ExponentialKernel,
        "threshold": ThresholdKernel,
    }

    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernels[kernel_type](**params)
