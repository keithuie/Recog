# MachineIQ (mIQ)

Multivariate Anomaly Detection for Industrial Condition Monitoring.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python main.py --scenario sudden_fault
```

## Usage

```python
from miq.core import MIQDetector
import numpy as np

detector = MIQDetector()
detector.configure_channels([
    {"name": "vibration", "min_value": 0, "max_value": 10}
])

detector.start_learning()
for _ in range(500):
    detector.process(np.array([5.0]))
detector.stop_learning()

result = detector.process(np.array([9.0]))
print(f"Anomaly: {result.is_anomaly}, Score: {result.anomaly_score}")
```
