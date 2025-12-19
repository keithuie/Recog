#!/usr/bin/env python3
"""
MachineIQ Command Line Interface.

Usage:
    machineiq learn <data_file> --machine-id <id>
    machineiq detect <data_file> --machine-id <id>
    machineiq status --machine-id <id>
    machineiq test
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import numpy as np
import yaml

from machineiq.core.hybrid_engine import HybridEngine
from machineiq.features.vibration import VibrationFeatureExtractor, generate_synthetic_vibration
from machineiq.memory.memory_bank import MemoryBank, MachineRecord, PatternRecord


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Try default locations
        for path in ["./config/default.yaml", "./machineiq/config/default.yaml"]:
            if Path(path).exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Return defaults
    return {
        "cmm": {
            "bins_per_channel": 64,
            "binning_method": "quantile",
            "correlation_threshold": 0.85,
        },
        "features": {
            "vibration": {
                "sample_rate": 10000,
                "fft_bins": 32,
                "include_bearing_freqs": False,
            }
        },
        "memory": {"db_path": "./data/memory.db"},
    }


def load_data(file_path: str) -> np.ndarray:
    """Load data from CSV or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise click.ClickException(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        # Assume numeric columns are features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].values

    elif suffix == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            if isinstance(data[0], list):
                return np.array(data)
            elif isinstance(data[0], dict):
                # List of dicts - extract values
                return np.array([list(d.values()) for d in data])
        elif isinstance(data, dict):
            if "data" in data:
                return np.array(data["data"])
            elif "features" in data:
                return np.array(data["features"])

        raise click.ClickException("Unsupported JSON format")

    elif suffix == ".npy":
        return np.load(file_path)

    else:
        raise click.ClickException(f"Unsupported file format: {suffix}")


@click.group()
@click.option("--config", "-c", type=str, help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """MachineIQ - ML-powered anomaly detection for industrial machinery."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.argument("data_file", type=str)
@click.option("--machine-id", "-m", required=True, help="Machine identifier")
@click.option("--machine-name", "-n", default=None, help="Machine name")
@click.pass_context
def learn(ctx, data_file, machine_id, machine_name):
    """Learn normal patterns from data file."""
    config = ctx.obj["config"]
    click.echo(f"Loading data from {data_file}...")

    try:
        data = load_data(data_file)
    except Exception as e:
        raise click.ClickException(f"Failed to load data: {e}")

    click.echo(f"Loaded {len(data)} samples with {data.shape[1]} features")

    # Initialize memory bank
    db_path = config["memory"]["db_path"]
    memory = MemoryBank(db_path)

    # Register or update machine
    machine_name = machine_name or machine_id
    machine = MachineRecord(
        id=machine_id,
        name=machine_name,
        config={"n_features": data.shape[1]},
    )
    memory.register_machine(machine)
    click.echo(f"Registered machine: {machine_id}")

    # Initialize and train engine
    cmm_config = config["cmm"]
    engine = HybridEngine(
        n_channels=data.shape[1],
        bins_per_channel=cmm_config["bins_per_channel"],
        correlation_threshold=cmm_config["correlation_threshold"],
        binning_method=cmm_config["binning_method"],
    )

    click.echo("Training CMM detector...")
    start_time = time.time()
    engine.fit(data)
    elapsed = time.time() - start_time

    click.echo(f"Training completed in {elapsed:.2f} seconds")
    click.echo(f"Learned {engine.pattern_count} patterns")

    # Save engine state
    model_path = Path(db_path).parent / f"{machine_id}_model.json"
    engine.save(str(model_path))
    click.echo(f"Model saved to {model_path}")

    # Store patterns in memory bank
    click.echo("Storing patterns in memory bank...")
    patterns = []
    for fv in data:
        patterns.append(PatternRecord(
            machine_id=machine_id,
            timestamp=datetime.now(),
            feature_vector=fv,
            is_anomaly=False,
            label="normal",
        ))
    memory.store_patterns(patterns)

    stats = memory.get_statistics(machine_id)
    click.echo(f"Total patterns stored: {stats['total_patterns']}")

    # Show memory usage
    mem_usage = engine.get_memory_usage()
    click.echo(f"Memory usage: {mem_usage['total_mb']:.2f} MB")


@cli.command()
@click.argument("data_file", type=str)
@click.option("--machine-id", "-m", required=True, help="Machine identifier")
@click.option("--output", "-o", default=None, help="Output file for results")
@click.pass_context
def detect(ctx, data_file, machine_id, output):
    """Run anomaly detection on data file."""
    config = ctx.obj["config"]
    click.echo(f"Loading data from {data_file}...")

    try:
        data = load_data(data_file)
    except Exception as e:
        raise click.ClickException(f"Failed to load data: {e}")

    click.echo(f"Loaded {len(data)} samples")

    # Load model
    db_path = config["memory"]["db_path"]
    model_path = Path(db_path).parent / f"{machine_id}_model.json"

    if not model_path.exists():
        raise click.ClickException(f"Model not found: {model_path}. Run 'learn' first.")

    click.echo(f"Loading model from {model_path}...")
    engine = HybridEngine.load(str(model_path))

    # Run detection
    click.echo("Running anomaly detection...")
    start_time = time.time()
    results = engine.detect_batch(data)
    elapsed = time.time() - start_time

    click.echo(f"Detection completed in {elapsed:.2f} seconds")

    # Analyze results
    anomalies = [r for r in results if r.is_anomaly]
    scores = [r.anomaly_score for r in results]

    click.echo(f"\nResults:")
    click.echo(f"  Total samples: {len(results)}")
    click.echo(f"  Anomalies detected: {len(anomalies)} ({100*len(anomalies)/len(results):.1f}%)")
    click.echo(f"  Anomaly score - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")

    # Store results in memory bank
    memory = MemoryBank(db_path)
    patterns = []
    for i, (fv, result) in enumerate(zip(data, results)):
        patterns.append(PatternRecord(
            machine_id=machine_id,
            timestamp=datetime.now(),
            feature_vector=fv,
            is_anomaly=result.is_anomaly,
            anomaly_score=result.anomaly_score,
            label="anomaly" if result.is_anomaly else "normal",
        ))
    memory.store_patterns(patterns)

    # Output results
    if output:
        output_data = []
        for i, result in enumerate(results):
            output_data.append({
                "index": i,
                "anomaly_score": result.anomaly_score,
                "is_anomaly": result.is_anomaly,
                "confidence": result.confidence,
            })

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.option("--machine-id", "-m", required=True, help="Machine identifier")
@click.pass_context
def status(ctx, machine_id):
    """Show status for a machine."""
    config = ctx.obj["config"]
    db_path = config["memory"]["db_path"]

    memory = MemoryBank(db_path)
    machine = memory.get_machine(machine_id)

    if not machine:
        raise click.ClickException(f"Machine not found: {machine_id}")

    click.echo(f"\nMachine: {machine.name} (ID: {machine.id})")
    click.echo(f"Created: {machine.created_at}")

    stats = memory.get_statistics(machine_id)

    click.echo(f"\nPattern Statistics:")
    click.echo(f"  Total patterns: {stats['total_patterns']}")
    click.echo(f"  Normal patterns: {stats['normal_patterns']}")
    click.echo(f"  Anomaly patterns: {stats['anomaly_patterns']}")
    click.echo(f"  Anomaly rate: {100*stats['anomaly_rate']:.1f}%")

    if stats['total_patterns'] > 0:
        click.echo(f"\nAnomaly Scores:")
        click.echo(f"  Average: {stats['score_avg']:.4f}")
        click.echo(f"  Min: {stats['score_min']:.4f}")
        click.echo(f"  Max: {stats['score_max']:.4f}")

        click.echo(f"\nTime Range:")
        click.echo(f"  First pattern: {stats['first_pattern']}")
        click.echo(f"  Last pattern: {stats['last_pattern']}")

        click.echo(f"\nLabels:")
        for label, count in stats['label_counts'].items():
            click.echo(f"  {label}: {count}")

    # Check for model
    model_path = Path(db_path).parent / f"{machine_id}_model.json"
    if model_path.exists():
        engine = HybridEngine.load(str(model_path))
        click.echo(f"\nModel Status:")
        click.echo(f"  Trained: {engine.is_trained}")
        click.echo(f"  Patterns learned: {engine.pattern_count}")
        mem = engine.get_memory_usage()
        click.echo(f"  Memory usage: {mem['total_mb']:.2f} MB")


@cli.command()
@click.option("--samples", "-n", default=1000, help="Number of test samples")
@click.option("--anomaly-rate", "-a", default=0.05, help="Fraction of anomalies")
@click.pass_context
def test(ctx, samples, anomaly_rate):
    """Run with synthetic data to verify everything works."""
    click.echo("MachineIQ Self-Test")
    click.echo("=" * 40)

    # Generate synthetic vibration data
    click.echo("\n1. Generating synthetic vibration data...")
    sample_rate = 10000
    duration = 0.1  # 100ms per sample

    normal_waveforms = []
    anomaly_waveforms = []

    n_normal = int(samples * (1 - anomaly_rate))
    n_anomaly = samples - n_normal

    for _ in range(n_normal):
        _, waveform = generate_synthetic_vibration(
            duration=duration,
            sample_rate=sample_rate,
            base_freq=60.0,
            noise_level=0.1,
        )
        normal_waveforms.append(waveform)

    for _ in range(n_anomaly):
        _, waveform = generate_synthetic_vibration(
            duration=duration,
            sample_rate=sample_rate,
            base_freq=60.0,
            noise_level=0.1,
            fault_freq=157.0,  # Simulated bearing defect
            fault_amplitude=0.5,
        )
        anomaly_waveforms.append(waveform)

    click.echo(f"   Generated {n_normal} normal + {n_anomaly} anomaly waveforms")

    # Extract features
    click.echo("\n2. Extracting vibration features...")
    extractor = VibrationFeatureExtractor(
        sample_rate=sample_rate,
        fft_bins=32,
        include_bearing_freqs=False,
    )

    normal_features = []
    for wf in normal_waveforms:
        features = extractor.extract(wf)
        normal_features.append(features.vector)
    normal_features = np.array(normal_features)

    anomaly_features = []
    for wf in anomaly_waveforms:
        features = extractor.extract(wf)
        anomaly_features.append(features.vector)
    anomaly_features = np.array(anomaly_features)

    click.echo(f"   Feature vector size: {normal_features.shape[1]}")
    click.echo(f"   Feature names: {extractor.feature_names[:5]}...")

    # Train CMM detector
    click.echo("\n3. Training CMM detector...")
    start_time = time.time()

    engine = HybridEngine(
        n_channels=normal_features.shape[1],
        bins_per_channel=64,
        correlation_threshold=0.3,  # Lower threshold for high-dimensional features
        binning_method="quantile",
    )
    engine.fit(normal_features)

    train_time = time.time() - start_time
    click.echo(f"   Training time: {train_time:.3f} seconds")
    click.echo(f"   Patterns learned: {engine.pattern_count}")

    mem = engine.get_memory_usage()
    click.echo(f"   Memory usage: {mem['total_mb']:.2f} MB")

    # Test detection on normal data
    click.echo("\n4. Testing detection on normal data...")
    normal_results = engine.detect_batch(normal_features)
    normal_false_positives = sum(1 for r in normal_results if r.is_anomaly)
    normal_scores = [r.anomaly_score for r in normal_results]

    click.echo(f"   False positives: {normal_false_positives}/{n_normal} ({100*normal_false_positives/n_normal:.1f}%)")
    click.echo(f"   Avg anomaly score: {np.mean(normal_scores):.4f}")

    # Test detection on anomaly data
    click.echo("\n5. Testing detection on anomaly data...")
    anomaly_results = engine.detect_batch(anomaly_features)
    anomaly_true_positives = sum(1 for r in anomaly_results if r.is_anomaly)
    anomaly_scores = [r.anomaly_score for r in anomaly_results]

    click.echo(f"   True positives: {anomaly_true_positives}/{n_anomaly} ({100*anomaly_true_positives/n_anomaly:.1f}%)")
    click.echo(f"   Avg anomaly score: {np.mean(anomaly_scores):.4f}")

    # Test memory bank
    click.echo("\n6. Testing memory bank...")
    memory = MemoryBank(":memory:")

    machine = MachineRecord(
        id="test-machine",
        name="Test Machine",
        config={"n_features": normal_features.shape[1]},
    )
    memory.register_machine(machine)

    patterns = []
    for fv in normal_features[:100]:
        patterns.append(PatternRecord(
            machine_id="test-machine",
            timestamp=datetime.now(),
            feature_vector=fv,
            is_anomaly=False,
            label="normal",
        ))
    memory.store_patterns(patterns)

    stats = memory.get_statistics("test-machine")
    click.echo(f"   Stored {stats['total_patterns']} patterns")

    # Summary
    click.echo("\n" + "=" * 40)
    click.echo("SELF-TEST SUMMARY")
    click.echo("=" * 40)

    fp_rate = normal_false_positives / n_normal
    tp_rate = anomaly_true_positives / n_anomaly

    click.echo(f"False Positive Rate: {100*fp_rate:.1f}%")
    click.echo(f"True Positive Rate: {100*tp_rate:.1f}% (Detection Rate)")

    # Success criteria
    success = True
    if train_time > 10:
        click.echo("[FAIL] Training too slow (>10s for 1000 samples)")
        success = False
    else:
        click.echo("[PASS] Training time acceptable")

    if fp_rate > 0.1:
        click.echo("[FAIL] False positive rate too high (>10%)")
        success = False
    else:
        click.echo("[PASS] False positive rate acceptable")

    if tp_rate < 0.5:
        click.echo("[FAIL] Detection rate too low (<50%)")
        success = False
    else:
        click.echo("[PASS] Detection rate acceptable")

    if mem['total_mb'] > 500:
        click.echo("[FAIL] Memory usage too high (>500MB)")
        success = False
    else:
        click.echo("[PASS] Memory usage acceptable")

    click.echo("")
    if success:
        click.echo("All tests PASSED!")
        sys.exit(0)
    else:
        click.echo("Some tests FAILED!")
        sys.exit(1)


@cli.command()
@click.pass_context
def list_machines(ctx):
    """List all registered machines."""
    config = ctx.obj["config"]
    db_path = config["memory"]["db_path"]

    if not Path(db_path).exists():
        click.echo("No database found. Run 'learn' first.")
        return

    memory = MemoryBank(db_path)
    machines = memory.list_machines()

    if not machines:
        click.echo("No machines registered.")
        return

    click.echo(f"\nRegistered Machines ({len(machines)}):")
    click.echo("-" * 60)

    for m in machines:
        stats = memory.get_statistics(m.id)
        click.echo(f"  {m.id}: {m.name}")
        click.echo(f"    Patterns: {stats['total_patterns']} (anomalies: {stats['anomaly_patterns']})")
        click.echo("")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
