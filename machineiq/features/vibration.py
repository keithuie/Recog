"""Vibration signal feature extraction for machinery condition monitoring."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from machineiq.features.base import BaseFeatureExtractor, FeatureSet


@dataclass
class BearingInfo:
    """Bearing geometry for fault frequency calculation.

    Attributes:
        n_balls: Number of rolling elements.
        ball_diameter: Diameter of rolling elements.
        pitch_diameter: Pitch diameter of bearing.
        contact_angle: Contact angle in degrees.
        shaft_rpm: Shaft rotational speed in RPM.
    """
    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle: float = 0.0
    shaft_rpm: float = 1800.0

    @property
    def shaft_freq(self) -> float:
        """Shaft frequency in Hz."""
        return self.shaft_rpm / 60.0

    @property
    def bpfo(self) -> float:
        """Ball Pass Frequency Outer race."""
        cos_angle = np.cos(np.radians(self.contact_angle))
        return (self.n_balls / 2) * self.shaft_freq * (1 - (self.ball_diameter / self.pitch_diameter) * cos_angle)

    @property
    def bpfi(self) -> float:
        """Ball Pass Frequency Inner race."""
        cos_angle = np.cos(np.radians(self.contact_angle))
        return (self.n_balls / 2) * self.shaft_freq * (1 + (self.ball_diameter / self.pitch_diameter) * cos_angle)

    @property
    def bsf(self) -> float:
        """Ball Spin Frequency."""
        cos_angle = np.cos(np.radians(self.contact_angle))
        return (self.pitch_diameter / (2 * self.ball_diameter)) * self.shaft_freq * (1 - ((self.ball_diameter / self.pitch_diameter) * cos_angle) ** 2)

    @property
    def ftf(self) -> float:
        """Fundamental Train Frequency (cage)."""
        cos_angle = np.cos(np.radians(self.contact_angle))
        return 0.5 * self.shaft_freq * (1 - (self.ball_diameter / self.pitch_diameter) * cos_angle)


class VibrationFeatureExtractor(BaseFeatureExtractor):
    """Extract features from vibration waveforms.

    Extracts time-domain, frequency-domain, and optionally bearing-specific
    features from raw vibration waveforms.

    Time Domain Features:
        - RMS: Root Mean Square amplitude
        - Peak: Maximum absolute amplitude
        - Peak-to-Peak: Difference between max and min
        - Crest Factor: Peak / RMS ratio
        - Kurtosis: Fourth moment (peakedness)
        - Skewness: Third moment (asymmetry)
        - Form Factor: RMS / Mean absolute value

    Frequency Domain Features:
        - FFT magnitude bins (configurable resolution)
        - Dominant frequency
        - Spectral centroid
        - Spectral spread

    Bearing Features (optional):
        - Amplitudes at BPFO, BPFI, BSF, FTF and their harmonics
    """

    def __init__(
        self,
        sample_rate: float,
        fft_bins: int = 32,
        max_freq: Optional[float] = None,
        include_bearing_freqs: bool = True,
        n_harmonics: int = 3,
        name: str = "VibrationExtractor",
    ):
        """Initialize the vibration feature extractor.

        Args:
            sample_rate: Sampling rate of input waveforms in Hz.
            fft_bins: Number of frequency bins for FFT features.
            max_freq: Maximum frequency to analyze (default: Nyquist).
            include_bearing_freqs: Whether to extract bearing fault frequencies.
            n_harmonics: Number of harmonics to extract for bearing frequencies.
            name: Human-readable name.
        """
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.fft_bins = fft_bins
        self.max_freq = max_freq or (sample_rate / 2)
        self.include_bearing_freqs = include_bearing_freqs
        self.n_harmonics = n_harmonics

        # Build feature name list
        self._build_feature_names()

    def _build_feature_names(self) -> None:
        """Build the list of feature names."""
        self._feature_names = []

        # Time domain features
        time_features = [
            "rms", "peak", "peak_to_peak", "crest_factor",
            "kurtosis", "skewness", "form_factor", "impulse_factor"
        ]
        self._feature_names.extend(time_features)

        # Frequency domain features
        for i in range(self.fft_bins):
            self._feature_names.append(f"fft_bin_{i}")

        freq_features = ["dominant_freq", "spectral_centroid", "spectral_spread"]
        self._feature_names.extend(freq_features)

        # Bearing frequency features (if enabled)
        if self.include_bearing_freqs:
            for fault_type in ["bpfo", "bpfi", "bsf", "ftf"]:
                for h in range(1, self.n_harmonics + 1):
                    self._feature_names.append(f"{fault_type}_h{h}")

    def extract(
        self,
        waveform: np.ndarray,
        bearing_info: Optional[BearingInfo] = None,
        **kwargs
    ) -> FeatureSet:
        """Extract features from a vibration waveform.

        Args:
            waveform: 1D array of vibration amplitude samples.
            bearing_info: Optional bearing geometry for fault frequency extraction.
            **kwargs: Additional parameters (ignored).

        Returns:
            FeatureSet containing all extracted features.
        """
        waveform = np.asarray(waveform).flatten()

        if len(waveform) < 2:
            raise ValueError("Waveform must have at least 2 samples")

        features: Dict[str, float] = {}

        # Extract time domain features
        time_features = self._extract_time_domain(waveform)
        features.update(time_features)

        # Extract frequency domain features
        freq_features = self._extract_frequency_domain(waveform)
        features.update(freq_features)

        # Extract bearing frequency features if bearing info provided
        if self.include_bearing_freqs:
            if bearing_info is not None:
                bearing_features = self._extract_bearing_frequencies(waveform, bearing_info)
            else:
                # Fill with zeros if no bearing info
                bearing_features = {}
                for fault_type in ["bpfo", "bpfi", "bsf", "ftf"]:
                    for h in range(1, self.n_harmonics + 1):
                        bearing_features[f"{fault_type}_h{h}"] = 0.0
            features.update(bearing_features)

        return FeatureSet(
            features=features,
            feature_names=self._feature_names.copy(),
            metadata={
                "sample_rate": self.sample_rate,
                "waveform_length": len(waveform),
                "has_bearing_info": bearing_info is not None,
            },
        )

    def _extract_time_domain(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract time domain features."""
        features = {}

        # Basic statistics
        mean_abs = np.mean(np.abs(waveform))
        rms = np.sqrt(np.mean(waveform ** 2))
        peak = np.max(np.abs(waveform))

        features["rms"] = float(rms)
        features["peak"] = float(peak)
        features["peak_to_peak"] = float(np.max(waveform) - np.min(waveform))

        # Crest factor
        features["crest_factor"] = float(peak / rms) if rms > 0 else 0.0

        # Higher order statistics
        n = len(waveform)
        mean = np.mean(waveform)
        std = np.std(waveform)

        if std > 0:
            # Kurtosis (using unbiased estimator)
            m4 = np.mean((waveform - mean) ** 4)
            features["kurtosis"] = float(m4 / (std ** 4) - 3)  # Excess kurtosis

            # Skewness
            m3 = np.mean((waveform - mean) ** 3)
            features["skewness"] = float(m3 / (std ** 3))
        else:
            features["kurtosis"] = 0.0
            features["skewness"] = 0.0

        # Form factor
        features["form_factor"] = float(rms / mean_abs) if mean_abs > 0 else 0.0

        # Impulse factor
        features["impulse_factor"] = float(peak / mean_abs) if mean_abs > 0 else 0.0

        return features

    def _extract_frequency_domain(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features."""
        features = {}

        # Compute FFT
        n = len(waveform)
        fft_result = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        magnitude = np.abs(fft_result)

        # Limit to max frequency
        freq_mask = freqs <= self.max_freq
        freqs = freqs[freq_mask]
        magnitude = magnitude[freq_mask]

        # Bin the spectrum into fft_bins frequency bands
        bin_magnitudes = self._bin_spectrum(magnitude, freqs, self.fft_bins)
        for i, mag in enumerate(bin_magnitudes):
            features[f"fft_bin_{i}"] = float(mag)

        # Dominant frequency
        if len(magnitude) > 0:
            dominant_idx = np.argmax(magnitude)
            features["dominant_freq"] = float(freqs[dominant_idx])
        else:
            features["dominant_freq"] = 0.0

        # Spectral centroid
        total_magnitude = np.sum(magnitude)
        if total_magnitude > 0:
            features["spectral_centroid"] = float(np.sum(freqs * magnitude) / total_magnitude)
        else:
            features["spectral_centroid"] = 0.0

        # Spectral spread (standard deviation of spectrum)
        if total_magnitude > 0:
            centroid = features["spectral_centroid"]
            variance = np.sum(((freqs - centroid) ** 2) * magnitude) / total_magnitude
            features["spectral_spread"] = float(np.sqrt(variance))
        else:
            features["spectral_spread"] = 0.0

        return features

    def _bin_spectrum(
        self, magnitude: np.ndarray, freqs: np.ndarray, n_bins: int
    ) -> np.ndarray:
        """Bin the spectrum into n_bins frequency bands.

        Uses logarithmic spacing for better low-frequency resolution.
        """
        if len(magnitude) == 0:
            return np.zeros(n_bins)

        # Create bin edges (log-spaced for better resolution at low frequencies)
        min_freq = max(freqs[0], 1.0)  # Avoid log(0)
        max_freq = max(freqs[-1], min_freq + 1.0)

        bin_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins + 1)
        bin_magnitudes = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
            if np.any(mask):
                # Use max magnitude in bin (peak detection)
                bin_magnitudes[i] = np.max(magnitude[mask])

        return bin_magnitudes

    def _extract_bearing_frequencies(
        self, waveform: np.ndarray, bearing_info: BearingInfo
    ) -> Dict[str, float]:
        """Extract amplitude at bearing fault frequencies."""
        features = {}

        # Compute FFT
        n = len(waveform)
        fft_result = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        magnitude = np.abs(fft_result)

        # Get bearing fault frequencies
        fault_freqs = {
            "bpfo": bearing_info.bpfo,
            "bpfi": bearing_info.bpfi,
            "bsf": bearing_info.bsf,
            "ftf": bearing_info.ftf,
        }

        # Extract amplitude at each frequency and its harmonics
        freq_resolution = self.sample_rate / n

        for fault_name, base_freq in fault_freqs.items():
            for h in range(1, self.n_harmonics + 1):
                target_freq = base_freq * h

                # Find closest frequency bin
                if target_freq < self.max_freq:
                    idx = int(round(target_freq / freq_resolution))
                    if 0 <= idx < len(magnitude):
                        # Use a small window around the target frequency
                        window_size = max(1, int(5 / freq_resolution))  # ~5 Hz window
                        start = max(0, idx - window_size // 2)
                        end = min(len(magnitude), idx + window_size // 2 + 1)
                        features[f"{fault_name}_h{h}"] = float(np.max(magnitude[start:end]))
                    else:
                        features[f"{fault_name}_h{h}"] = 0.0
                else:
                    features[f"{fault_name}_h{h}"] = 0.0

        return features

    def get_config(self) -> dict:
        """Get extractor configuration."""
        return {
            "sample_rate": self.sample_rate,
            "fft_bins": self.fft_bins,
            "max_freq": self.max_freq,
            "include_bearing_freqs": self.include_bearing_freqs,
            "n_harmonics": self.n_harmonics,
            "n_features": self.n_features,
            "feature_names": self._feature_names.copy(),
        }


def generate_synthetic_vibration(
    duration: float,
    sample_rate: float = 10000.0,
    base_freq: float = 60.0,
    noise_level: float = 0.1,
    fault_freq: Optional[float] = None,
    fault_amplitude: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic vibration signal for testing.

    Args:
        duration: Signal duration in seconds.
        sample_rate: Sampling rate in Hz.
        base_freq: Base rotational frequency in Hz.
        noise_level: Noise amplitude relative to signal.
        fault_freq: Optional fault frequency to add.
        fault_amplitude: Amplitude of fault frequency component.

    Returns:
        Tuple of (time array, waveform array).
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Base signal (fundamental + harmonic)
    signal = np.sin(2 * np.pi * base_freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * base_freq * t)  # 2nd harmonic
    signal += 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)  # 3rd harmonic

    # Add fault frequency if specified
    if fault_freq is not None and fault_amplitude > 0:
        signal += fault_amplitude * np.sin(2 * np.pi * fault_freq * t)
        # Also add some modulation at fault frequency
        signal *= (1 + 0.1 * fault_amplitude * np.sin(2 * np.pi * fault_freq * t))

    # Add noise
    signal += noise_level * np.random.randn(n_samples)

    return t, signal
