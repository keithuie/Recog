#!/usr/bin/env python3
"""MachineIQ CLI"""
import argparse
import numpy as np
from miq.core import MIQDetector
from miq.utils import create_test_scenario

def main():
    parser = argparse.ArgumentParser(description="MachineIQ Anomaly Detection")
    parser.add_argument("--mode", choices=["cli", "test"], default="cli")
    parser.add_argument("--scenario", default="sudden_fault")
    args = parser.parse_args()
    
    if args.mode == "test":
        import pytest
        pytest.main(["tests/", "-v"])
        return
    
    print("MachineIQ - Running demo...")
    data, labels, meta = create_test_scenario(args.scenario, num_channels=3, num_samples=500)
    
    detector = MIQDetector()
    detector.configure_channels([
        {"name": f"ch{i}", "min_value": 0, "max_value": 100} for i in range(3)
    ])
    
    detector.start_learning()
    for i in range(200):
        detector.process(data[i])
    detector.stop_learning()
    print(f"Learned {detector.num_trained_states} states")
    
    anomalies = 0
    for i in range(200, 500):
        result = detector.process(data[i])
        if result.is_anomaly:
            anomalies += 1
    print(f"Detected {anomalies} anomalies in monitoring phase")

if __name__ == "__main__":
    main()
