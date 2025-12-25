import pytest
import numpy as np
from miq.core import MIQDetector, TriangularKernel, ParabolicKernel

class TestKernels:
    def test_triangular_max_score(self):
        kernel = TriangularKernel(64, 0.5)
        assert kernel.get_max_score() == 64
        assert kernel.get_score(0) == 64
    
    def test_parabolic_max_score(self):
        kernel = ParabolicKernel(64, 0.5)
        assert kernel.get_max_score() == 1024

class TestDetector:
    def test_learning_and_monitoring(self):
        detector = MIQDetector()
        detector.configure_channels([
            {"name": "ch1", "min_value": 0, "max_value": 100},
            {"name": "ch2", "min_value": 0, "max_value": 100}
        ])
        detector.start_learning()
        for _ in range(100):
            detector.process(np.array([50.0, 50.0]))
        detector.stop_learning()
        assert detector.num_trained_states > 0
        
        result = detector.process(np.array([50.0, 50.0]))
        assert result.match_strength > 0.9
        
        result = detector.process(np.array([95.0, 95.0]))
        assert result.anomaly_score > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
