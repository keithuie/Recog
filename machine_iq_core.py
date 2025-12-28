import sys
import numpy as np
from miq.core.detector import AnomalyDetector
from miq.preprocessing.pipeline import DataPipeline

# Initialize global instances to avoid reloading models every call
pipeline = DataPipeline()
detector = AnomalyDetector()

def analyze_pattern(data_array):
    """
    Called from C++:
    Input: numpy array (from C++ vector)
    Output: dict { "confidence": float, "detected": bool }
    """
    try:
        # 1. Preprocess
        processed_data = pipeline.transform(data_array)
        
        # 2. Detect
        result = detector.predict(processed_data)
        
        return {
            "confidence": float(result.confidence),
            "detected": bool(result.is_anomaly),
            "pattern": str(result.pattern_name)
        }
    except Exception as e:
        print(f"[Python-Adapter] Error: {e}")
        return {
            "confidence": 0.0,
            "detected": False,
            "pattern": "Error"
        }
