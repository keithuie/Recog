import numpy as np
import scipy.stats
import scipy.signal
from typing import Dict, List, Tuple

class TimedomainFeatures:
    @staticmethod
    def rms(signal: np.ndarray) -> float:
        return np.sqrt(np.mean(signal**2))

    @staticmethod
    def peak(signal: np.ndarray) -> float:
        return np.max(np.abs(signal))

    @staticmethod
    def crest_factor(signal: np.ndarray) -> float:
        rms_val = TimedomainFeatures.rms(signal)
        return TimedomainFeatures.peak(signal) / rms_val if rms_val > 0 else 0.0

    @staticmethod
    def kurtosis(signal: np.ndarray) -> float:
        return scipy.stats.kurtosis(signal, fisher=True)

    @classmethod
    def extract_all(cls, signal: np.ndarray) -> Dict[str, float]:
        return {
            "rms": cls.rms(signal), "peak": cls.peak(signal),
            "crest_factor": cls.crest_factor(signal), "kurtosis": cls.kurtosis(signal)
        }


class FrequencyFeatures:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def compute_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(signal)
        window = np.hanning(n)
        fft_vals = np.fft.rfft(signal * window)
        magnitudes = np.abs(fft_vals) / n * 2
        freqs = np.fft.rfftfreq(n, d=1/self.sample_rate)
        return freqs, magnitudes


class EnvelopeFeatures:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def compute_envelope(self, signal: np.ndarray) -> np.ndarray:
        analytic = scipy.signal.hilbert(signal)
        return np.abs(analytic) - np.mean(np.abs(analytic))


class BearingFrequencyCalculator:
    def __init__(self, num_balls: int, ball_diameter: float, pitch_diameter: float, contact_angle: float = 0):
        self.n = num_balls
        self.d = ball_diameter
        self.D = pitch_diameter
        self.phi = np.radians(contact_angle)

    def calculate(self, shaft_rpm: float) -> Dict[str, float]:
        freq_hz = shaft_rpm / 60.0
        ratio = self.d / self.D
        cos_phi = np.cos(self.phi)
        return {
            "bpfo": (self.n / 2.0) * freq_hz * (1.0 - ratio * cos_phi),
            "bpfi": (self.n / 2.0) * freq_hz * (1.0 + ratio * cos_phi),
            "bsf": (self.D / (2.0 * self.d)) * freq_hz * (1.0 - (ratio * cos_phi)**2),
            "ftf": 0.5 * freq_hz * (1.0 - ratio * cos_phi)
        }
