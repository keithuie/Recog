import numpy as np
from typing import List, Tuple, Dict

class TimeSeriesDataset:
    def __init__(self, num_channels: int = 1, seed: int = None):
        self.num_channels = num_channels
        if seed is not None:
            np.random.seed(seed)
        self.means = np.random.uniform(20, 80, num_channels)
        self.stds = np.random.uniform(0.5, 2.0, num_channels)

    def generate_baseline(self, num_samples: int) -> List[np.ndarray]:
        return [np.random.normal(self.means, self.stds).astype(np.float32) for _ in range(num_samples)]

    def generate_with_anomaly(self, num_samples: int, anomaly_start: int, 
                              anomaly_type: str = "shift", severity: float = 0.5) -> Tuple[List[np.ndarray], List[bool]]:
        data, labels = [], []
        affected = np.random.choice(self.num_channels, max(1, self.num_channels // 2), replace=False)
        for i in range(num_samples):
            is_anomaly = i >= anomaly_start
            labels.append(is_anomaly)
            step_means = self.means.copy()
            if is_anomaly and anomaly_type == "shift":
                step_means[affected] += severity * self.stds[affected] * 10
            data.append(np.random.normal(step_means, self.stds).astype(np.float32))
        return data, labels


def create_test_scenario(scenario_name: str, num_channels: int = 3, num_samples: int = 1000) -> Tuple[List[np.ndarray], List[bool], Dict]:
    ds = TimeSeriesDataset(num_channels)
    metadata = {"scenario": scenario_name}
    if scenario_name == "baseline_only":
        return ds.generate_baseline(num_samples), [False] * num_samples, metadata
    elif scenario_name == "sudden_fault":
        start = int(num_samples * 0.7)
        data, labels = ds.generate_with_anomaly(num_samples, start, "shift", 0.8)
        metadata["anomaly_start"] = start
        return data, labels, metadata
    return ds.generate_baseline(num_samples), [False] * num_samples, metadata
