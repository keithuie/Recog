#!/usr/bin/env python3
"""
MachineIQ CLI Demo - Simple command-line tool to test the ML pipeline.

This demonstrates the ML core working correctly without any GUI complexity.
Run with: python cli_demo.py

Usage:
    python cli_demo.py                    # Run with defaults
    python cli_demo.py --engine 5         # Use engine 5
    python cli_demo.py --train-cycles 50  # Train on 50 cycles
    python cli_demo.py --speed 10         # 10x playback speed
"""

import sys
import time
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_status(label: str, value: str, color: str = None):
    """Print a status line with optional color."""
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    if color and color in colors:
        print(f"  {label}: {colors[color]}{value}{colors['reset']}")
    else:
        print(f"  {label}: {value}")

def print_bar(value: float, width: int = 40, threshold: float = 0.3):
    """Print a visual progress bar."""
    filled = int(value * width)
    bar = '█' * filled + '░' * (width - filled)

    # Color based on value vs threshold
    if value < threshold:
        color = '\033[91m'  # Red - anomaly
    elif value < threshold + 0.2:
        color = '\033[93m'  # Yellow - warning
    else:
        color = '\033[92m'  # Green - normal

    reset = '\033[0m'
    print(f"  [{color}{bar}{reset}] {value*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='MachineIQ CLI Demo')
    parser.add_argument('--engine', type=int, default=1, help='Engine number (1-100)')
    parser.add_argument('--train-cycles', type=int, default=30, help='Number of cycles to train on')
    parser.add_argument('--monitor-cycles', type=int, default=0, help='Cycles to monitor (0=all remaining)')
    parser.add_argument('--speed', type=float, default=5, help='Playback speed multiplier')
    parser.add_argument('--threshold', type=float, default=0.3, help='Anomaly threshold (0-1)')
    parser.add_argument('--bins', type=int, default=64, help='Bins per channel')
    args = parser.parse_args()

    print_header("MachineIQ CLI Demo")
    print("  Testing ML pipeline without GUI complexity")
    print()

    # ========================================
    # Step 1: Load ML Core
    # ========================================
    print_header("Step 1: Loading ML Core")

    try:
        from miq.core.detector import MIQDetector, DetectorConfig
        from miq.core.kernels import KernelType
        print_status("ML Core", "Loaded successfully", "green")
    except ImportError as e:
        print_status("ML Core", f"Failed to load: {e}", "red")
        return 1

    # ========================================
    # Step 2: Load NASA CMAPSS Data
    # ========================================
    print_header("Step 2: Loading NASA CMAPSS Data")

    try:
        from miq.samples.nasa_cmapss import NASACMAPSSLoader, DEFAULT_SENSORS, SENSOR_DICTIONARY

        loader = NASACMAPSSLoader()
        print_status("Loader", "Created", "green")

        if not loader.is_loaded():
            print("  Downloading dataset (first time only)...")
            loader.download_and_load()

        print_status("Dataset", "Loaded", "green")

        # Get engine info
        engines = loader.get_engine_list()
        engine = next((e for e in engines if e['unit_nr'] == args.engine), None)

        if not engine:
            print_status("Engine", f"Engine {args.engine} not found", "red")
            return 1

        total_cycles = engine['total_cycles']
        print_status("Engine", f"#{args.engine} ({total_cycles} cycles to failure)", "green")

        # Get engine data
        sensors = DEFAULT_SENSORS
        engine_data = loader.get_engine_data(args.engine, sensors)

        print_status("Sensors", f"{len(sensors)} channels", "green")
        for s in sensors:
            info = SENSOR_DICTIONARY.get(s, {})
            name = info.get('name', s)
            unit = info.get('unit', '')
            print(f"    - {s}: {name} ({unit})")

    except Exception as e:
        print_status("Data Load", f"Failed: {e}", "red")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================
    # Step 3: Configure Detector
    # ========================================
    print_header("Step 3: Configuring Detector")

    # Get sensor ranges from statistics
    stats = loader.get_statistics()
    sensor_ranges = stats.get('sensor_ranges', {})

    config = DetectorConfig(
        bins_per_channel=args.bins,
        kernel_type=KernelType.TRIANGULAR,
        kernel_width=0.5
    )

    detector = MIQDetector(config)

    # Configure channels
    channel_configs = []
    for s in sensors:
        if s in sensor_ranges:
            r = sensor_ranges[s]
            channel_configs.append({
                "name": s,
                "min_value": r['min'],
                "max_value": r['max']
            })
        else:
            channel_configs.append({
                "name": s,
                "min_value": 0.0,
                "max_value": 1000.0
            })

    detector.configure_channels(channel_configs)
    detector.anomaly_threshold = args.threshold

    print_status("Detector", "Configured", "green")
    print_status("Bins/Channel", str(args.bins), "blue")
    print_status("Kernel", "Triangular", "blue")
    print_status("Threshold", f"{args.threshold*100:.0f}%", "blue")

    # ========================================
    # Step 4: Training Phase
    # ========================================
    print_header(f"Step 4: Training ({args.train_cycles} cycles)")

    import numpy as np

    detector.start_learning()
    print_status("State", "LEARNING", "yellow")

    train_start = time.time()

    for i in range(min(args.train_cycles, len(engine_data))):
        record = engine_data[i]

        # Build feature vector
        feature_vector = np.array([float(record[s]) for s in sensors], dtype=np.float32)

        # Process (training)
        detector.process(feature_vector)

        # Progress
        progress = (i + 1) / args.train_cycles
        cycle = int(record['time_cycles'])
        rul = int(record['rul'])

        # Print progress every 10%
        if (i + 1) % max(1, args.train_cycles // 10) == 0 or i == 0:
            print(f"  Training: {progress*100:.0f}% (cycle {cycle}, RUL {rul})")

        time.sleep(0.01 / args.speed)  # Small delay for visual

    detector.stop_learning()
    train_time = time.time() - train_start

    print_status("State", "MONITORING", "green")
    print_status("Trained States", str(detector.num_trained_states), "green")
    print_status("Training Time", f"{train_time:.2f}s", "blue")

    # ========================================
    # Step 5: Monitoring Phase
    # ========================================
    monitor_start = args.train_cycles
    monitor_end = len(engine_data) if args.monitor_cycles == 0 else min(monitor_start + args.monitor_cycles, len(engine_data))

    print_header(f"Step 5: Monitoring (cycles {monitor_start+1} to {monitor_end})")
    print(f"  Press Ctrl+C to stop early")
    print()

    anomaly_count = 0
    results_history = []

    try:
        for i in range(monitor_start, monitor_end):
            record = engine_data[i]

            # Build feature vector
            feature_vector = np.array([float(record[s]) for s in sensors], dtype=np.float32)

            # Process (monitoring)
            result = detector.process(feature_vector)

            cycle = int(record['time_cycles'])
            rul = int(record['rul'])

            if result:
                confidence = result.match_strength
                is_anomaly = result.is_anomaly

                if is_anomaly:
                    anomaly_count += 1

                results_history.append({
                    'cycle': cycle,
                    'rul': rul,
                    'confidence': confidence,
                    'is_anomaly': is_anomaly
                })

                # Print status
                status = "ANOMALY" if is_anomaly else "NORMAL"
                color = "red" if is_anomaly else "green"

                # Clear line and print
                print(f"\r  Cycle {cycle:3d} | RUL {rul:3d} | ", end='')
                print_bar(confidence, width=30, threshold=args.threshold)

                # Show anomaly alert
                if is_anomaly:
                    print(f"\033[91m  ⚠ ANOMALY DETECTED at cycle {cycle} (RUL: {rul})\033[0m")
                    top = result.get_top_contributors(3)
                    for ch, score in top:
                        print(f"      Top contributor: {ch} (score: {score:.2f})")

            # Delay for playback speed
            time.sleep(1.0 / args.speed)

    except KeyboardInterrupt:
        print("\n\n  Monitoring stopped by user")

    # ========================================
    # Step 6: Summary
    # ========================================
    print_header("Summary")

    total_monitored = len(results_history)

    print_status("Engine", f"#{args.engine}", "blue")
    print_status("Total Cycles", str(total_cycles), "blue")
    print_status("Trained On", f"{args.train_cycles} cycles", "blue")
    print_status("Monitored", f"{total_monitored} cycles", "blue")
    print_status("Trained States", str(detector.num_trained_states), "green")
    print_status("Anomalies", str(anomaly_count), "red" if anomaly_count > 0 else "green")

    if results_history:
        confidences = [r['confidence'] for r in results_history]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        print()
        print_status("Avg Confidence", f"{avg_conf*100:.1f}%", "blue")
        print_status("Min Confidence", f"{min_conf*100:.1f}%", "yellow" if min_conf < args.threshold else "blue")

        # Find first anomaly
        anomalies = [r for r in results_history if r['is_anomaly']]
        if anomalies:
            first = anomalies[0]
            print_status("First Anomaly", f"Cycle {first['cycle']} (RUL: {first['rul']})", "red")

    print()
    print("  ML Pipeline: WORKING")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
