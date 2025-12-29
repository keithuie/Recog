"""
MachineIQ Preprocessing Features Module
Advanced signal processing for vibration analysis and anomaly detection
"""
import numpy as np
import scipy.stats
import scipy.signal
from typing import Dict, List, Tuple, Optional


class TimedomainFeatures:
    """Time domain feature extraction for vibration signals"""

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        """Root Mean Square - overall signal energy"""
        return np.sqrt(np.mean(signal**2))

    @staticmethod
    def peak(signal: np.ndarray) -> float:
        """Peak amplitude - maximum absolute value"""
        return np.max(np.abs(signal))

    @staticmethod
    def peak_to_peak(signal: np.ndarray) -> float:
        """Peak-to-peak amplitude"""
        return np.max(signal) - np.min(signal)

    @staticmethod
    def crest_factor(signal: np.ndarray) -> float:
        """Crest factor - peak/RMS ratio, indicates impulsiveness"""
        rms_val = TimedomainFeatures.rms(signal)
        return TimedomainFeatures.peak(signal) / rms_val if rms_val > 0 else 0.0

    @staticmethod
    def kurtosis(signal: np.ndarray) -> float:
        """Kurtosis - peakedness indicator, sensitive to impacts"""
        return scipy.stats.kurtosis(signal, fisher=True)

    @staticmethod
    def skewness(signal: np.ndarray) -> float:
        """Skewness - asymmetry indicator"""
        return scipy.stats.skew(signal)

    @staticmethod
    def mean(signal: np.ndarray) -> float:
        """Mean value - DC offset"""
        return np.mean(signal)

    @staticmethod
    def std(signal: np.ndarray) -> float:
        """Standard deviation - spread measure"""
        return np.std(signal)

    @staticmethod
    def variance(signal: np.ndarray) -> float:
        """Variance"""
        return np.var(signal)

    @staticmethod
    def clearance_factor(signal: np.ndarray) -> float:
        """Clearance factor - sensitive to early fault detection"""
        peak = TimedomainFeatures.peak(signal)
        mean_sqrt = np.mean(np.sqrt(np.abs(signal)))**2
        return peak / mean_sqrt if mean_sqrt > 0 else 0.0

    @staticmethod
    def shape_factor(signal: np.ndarray) -> float:
        """Shape factor - RMS/mean absolute value"""
        mean_abs = np.mean(np.abs(signal))
        return TimedomainFeatures.rms(signal) / mean_abs if mean_abs > 0 else 0.0

    @staticmethod
    def impulse_factor(signal: np.ndarray) -> float:
        """Impulse factor - peak/mean absolute value"""
        mean_abs = np.mean(np.abs(signal))
        return TimedomainFeatures.peak(signal) / mean_abs if mean_abs > 0 else 0.0

    @classmethod
    def extract_all(cls, signal: np.ndarray) -> Dict[str, float]:
        """Extract all time domain features"""
        return {
            "rms": cls.rms(signal),
            "peak": cls.peak(signal),
            "peak_to_peak": cls.peak_to_peak(signal),
            "crest_factor": cls.crest_factor(signal),
            "kurtosis": cls.kurtosis(signal),
            "skewness": cls.skewness(signal),
            "mean": cls.mean(signal),
            "std": cls.std(signal),
            "clearance_factor": cls.clearance_factor(signal),
            "shape_factor": cls.shape_factor(signal),
            "impulse_factor": cls.impulse_factor(signal)
        }


class FrequencyFeatures:
    """Frequency domain feature extraction using FFT"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def compute_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT magnitude spectrum"""
        n = len(signal)
        window = np.hanning(n)
        fft_vals = np.fft.rfft(signal * window)
        magnitudes = np.abs(fft_vals) / n * 2
        freqs = np.fft.rfftfreq(n, d=1/self.sample_rate)
        return freqs, magnitudes

    def compute_power_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density using Welch's method"""
        freqs, psd = scipy.signal.welch(
            signal,
            fs=self.sample_rate,
            nperseg=min(256, len(signal)),
            noverlap=None
        )
        return freqs, psd

    def extract_band_energy(self, signal: np.ndarray, bands: List[Tuple[float, float]]) -> Dict[str, float]:
        """Extract energy in specified frequency bands"""
        freqs, magnitudes = self.compute_spectrum(signal)
        results = {}

        for i, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(magnitudes[mask]**2)
            results[f"band_{i}_{int(low)}_{int(high)}_hz"] = band_energy

        return results

    def peak_frequency(self, signal: np.ndarray) -> float:
        """Find dominant frequency"""
        freqs, magnitudes = self.compute_spectrum(signal)
        return freqs[np.argmax(magnitudes)]

    def spectral_centroid(self, signal: np.ndarray) -> float:
        """Spectral centroid - center of mass of spectrum"""
        freqs, magnitudes = self.compute_spectrum(signal)
        total = np.sum(magnitudes)
        return np.sum(freqs * magnitudes) / total if total > 0 else 0.0

    def spectral_spread(self, signal: np.ndarray) -> float:
        """Spectral spread - bandwidth around centroid"""
        freqs, magnitudes = self.compute_spectrum(signal)
        centroid = self.spectral_centroid(signal)
        total = np.sum(magnitudes)
        return np.sqrt(np.sum(((freqs - centroid)**2) * magnitudes) / total) if total > 0 else 0.0

    def extract_all(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract all frequency domain features"""
        freqs, magnitudes = self.compute_spectrum(signal)
        mag_sum = np.sum(magnitudes)
        return {
            "peak_frequency": self.peak_frequency(signal),
            "spectral_centroid": self.spectral_centroid(signal),
            "spectral_spread": self.spectral_spread(signal),
            "spectral_energy": np.sum(magnitudes**2),
            "spectral_entropy": scipy.stats.entropy(magnitudes / mag_sum) if mag_sum > 0 else 0.0
        }


class BandpassFilter:
    """Bandpass filtering for isolating frequency bands"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def apply(self, signal: np.ndarray, low_freq: float, high_freq: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth bandpass filter"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))

        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return scipy.signal.filtfilt(b, a, signal)

    def highpass(self, signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth highpass filter"""
        nyquist = self.sample_rate / 2
        normalized = max(0.001, min(cutoff / nyquist, 0.999))
        b, a = scipy.signal.butter(order, normalized, btype='high')
        return scipy.signal.filtfilt(b, a, signal)

    def lowpass(self, signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth lowpass filter"""
        nyquist = self.sample_rate / 2
        normalized = max(0.001, min(cutoff / nyquist, 0.999))
        b, a = scipy.signal.butter(order, normalized, btype='low')
        return scipy.signal.filtfilt(b, a, signal)


class EnvelopeFeatures:
    """Envelope analysis using Hilbert transform - for bearing/gear fault detection"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.bandpass = BandpassFilter(sample_rate)

    def compute_envelope(self, signal: np.ndarray) -> np.ndarray:
        """Compute envelope using Hilbert transform"""
        analytic = scipy.signal.hilbert(signal)
        envelope = np.abs(analytic)
        return envelope - np.mean(envelope)  # Remove DC

    def envelope_spectrum(self, signal: np.ndarray, band_low: Optional[float] = None,
                         band_high: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute envelope spectrum - key technique for bearing fault detection
        Optionally bandpass filter before envelope extraction
        """
        # Optional bandpass filtering to isolate resonance band
        if band_low is not None and band_high is not None:
            signal = self.bandpass.apply(signal, band_low, band_high)

        # Compute envelope
        envelope = self.compute_envelope(signal)

        # FFT of envelope
        n = len(envelope)
        window = np.hanning(n)
        fft_vals = np.fft.rfft(envelope * window)
        magnitudes = np.abs(fft_vals) / n * 2
        freqs = np.fft.rfftfreq(n, d=1/self.sample_rate)

        return freqs, magnitudes

    def extract_envelope_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract features from envelope signal"""
        envelope = self.compute_envelope(signal)
        env_rms = np.sqrt(np.mean(envelope**2))
        return {
            "envelope_rms": env_rms,
            "envelope_peak": np.max(np.abs(envelope)),
            "envelope_kurtosis": scipy.stats.kurtosis(envelope, fisher=True),
            "envelope_crest": np.max(np.abs(envelope)) / env_rms if env_rms > 0 else 0.0
        }


class StressWaveAnalysis:
    """
    StressWave / PeakVue analysis - emulates Emerson's PeakVue technique
    Detects high-frequency stress waves from metal-to-metal impacts
    """

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.bandpass = BandpassFilter(sample_rate)

    def compute_stresswave(self, signal: np.ndarray, highpass_freq: float = 1000,
                          peak_threshold_percentile: float = 90) -> np.ndarray:
        """
        Compute stress wave (PeakVue-like) signal:
        1. High-pass filter to isolate high-frequency impacts
        2. Rectify the signal
        3. Peak detection - capture only peak values
        4. Reconstruct sparse peak signal
        """
        # High-pass filter to get stress wave components
        nyquist = self.sample_rate / 2
        if highpass_freq < nyquist * 0.9:  # Ensure valid frequency
            filtered = self.bandpass.highpass(signal, highpass_freq)
        else:
            filtered = signal

        # Rectify (absolute value)
        rectified = np.abs(filtered)

        # Find peaks above threshold
        threshold = np.percentile(rectified, peak_threshold_percentile)
        peaks, properties = scipy.signal.find_peaks(rectified, height=threshold, distance=5)

        # Create sparse peak signal
        peak_signal = np.zeros_like(signal)
        if len(peaks) > 0:
            peak_signal[peaks] = rectified[peaks]

        return peak_signal

    def stresswave_spectrum(self, signal: np.ndarray, highpass_freq: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spectrum of stress wave signal"""
        sw = self.compute_stresswave(signal, highpass_freq)

        n = len(sw)
        window = np.hanning(n)
        fft_vals = np.fft.rfft(sw * window)
        magnitudes = np.abs(fft_vals) / n * 2
        freqs = np.fft.rfftfreq(n, d=1/self.sample_rate)

        return freqs, magnitudes

    def extract_stresswave_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract stress wave analysis features"""
        sw = self.compute_stresswave(signal)

        # Count significant peaks (impacts)
        nonzero = sw[sw > 0]
        threshold = np.percentile(nonzero, 50) if len(nonzero) > 0 else 0
        peak_count = np.sum(sw > threshold)
        sw_rms = np.sqrt(np.mean(sw**2))

        return {
            "stresswave_rms": sw_rms,
            "stresswave_peak": np.max(sw),
            "stresswave_count": float(peak_count),
            "stresswave_energy": np.sum(sw**2),
            "stresswave_crest": np.max(sw) / sw_rms if sw_rms > 0 else 0.0
        }


class BearingFrequencyCalculator:
    """Calculate theoretical bearing defect frequencies"""

    def __init__(self, num_balls: int, ball_diameter: float, pitch_diameter: float, contact_angle: float = 0):
        self.n = num_balls
        self.d = ball_diameter
        self.D = pitch_diameter
        self.phi = np.radians(contact_angle)

    def calculate(self, shaft_rpm: float) -> Dict[str, float]:
        """
        Calculate bearing defect frequencies:
        - BPFO: Ball Pass Frequency Outer race
        - BPFI: Ball Pass Frequency Inner race
        - BSF: Ball Spin Frequency
        - FTF: Fundamental Train Frequency (cage)
        """
        freq_hz = shaft_rpm / 60.0
        ratio = self.d / self.D
        cos_phi = np.cos(self.phi)

        return {
            "bpfo": (self.n / 2.0) * freq_hz * (1.0 - ratio * cos_phi),
            "bpfi": (self.n / 2.0) * freq_hz * (1.0 + ratio * cos_phi),
            "bsf": (self.D / (2.0 * self.d)) * freq_hz * (1.0 - (ratio * cos_phi)**2),
            "ftf": 0.5 * freq_hz * (1.0 - ratio * cos_phi),
            "shaft_freq": freq_hz,
            "2x_shaft": 2.0 * freq_hz
        }


class VibrationAnalyzer:
    """Complete vibration analysis combining all techniques"""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.time_features = TimedomainFeatures()
        self.freq_features = FrequencyFeatures(sample_rate)
        self.envelope = EnvelopeFeatures(sample_rate)
        self.stresswave = StressWaveAnalysis(sample_rate)
        self.bandpass = BandpassFilter(sample_rate)

    def analyze(self, signal: np.ndarray, include_stresswave: bool = True,
               include_envelope: bool = True) -> Dict[str, float]:
        """Run complete vibration analysis"""
        results = {}

        # Time domain
        results.update(self.time_features.extract_all(signal))

        # Frequency domain
        results.update(self.freq_features.extract_all(signal))

        # Envelope analysis
        if include_envelope:
            results.update(self.envelope.extract_envelope_features(signal))

        # Stress wave / PeakVue
        if include_stresswave and self.sample_rate >= 5000:  # Only for high sample rates
            results.update(self.stresswave.extract_stresswave_features(signal))

        return results
