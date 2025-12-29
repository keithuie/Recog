"""
MachineIQ ML Core Service
Provides REST API for high-fidelity anomaly detection

DATA INGESTION ARCHITECTURE:
===========================
HIGH-FIDELITY DATA (time series, waveforms):
  - REST API POST /api/analyze (batch processing)
  - Direct pybind11 calls from C++ host (zero-copy, preferred)
  - Direct SQL queries with binary BLOB extraction
  - gRPC streaming (for real-time high-throughput)

  DO NOT USE MQTT for time series data - it adds latency,
  serialization overhead, and is not designed for high-throughput.

CONTROL PLANE (MQTT is appropriate):
  - Start/stop training commands
  - Configuration updates
  - Status notifications
  - Anomaly alerts
"""
import os
import json
import threading
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from influxdb import InfluxDBClient
import paho.mqtt.client as mqtt

from miq.core.detector import MIQDetector, DetectorConfig
from miq.core.kernels import KernelType
from miq.preprocessing.pipeline import PipelineBuilder

app = Flask(__name__)

# Configuration from environment
INFLUXDB_HOST = os.environ.get('INFLUXDB_HOST', 'localhost')
INFLUXDB_PORT = int(os.environ.get('INFLUXDB_PORT', 8086))
INFLUXDB_DB = os.environ.get('INFLUXDB_DB', 'machineiq_data')
MQTT_HOST = os.environ.get('MQTT_HOST', 'localhost')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))

# Global detector instances (per group)
detectors = {}
pipelines = {}
influx_client = None
mqtt_client = None


def get_influx_client():
    """Get or create InfluxDB client"""
    global influx_client
    if influx_client is None:
        influx_client = InfluxDBClient(
            host=INFLUXDB_HOST,
            port=INFLUXDB_PORT,
            database=INFLUXDB_DB
        )
    return influx_client


def setup_mqtt():
    """
    Setup MQTT client for CONTROL PLANE ONLY.

    IMPORTANT: MQTT is NOT used for time series data ingestion.
    Data should flow via:
      - REST API POST /api/analyze (recommended for web clients)
      - Direct pybind11 from C++ host (zero-copy, highest performance)
      - SQL/database polling
    """
    global mqtt_client

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        # Subscribe to CONTROL topics only - NOT data topics
        client.subscribe("machineiq/control/#")
        # NOTE: We do NOT subscribe to machineiq/data/#
        # Time series data must use REST API or direct calls for high fidelity

    def on_message(client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            if topic.startswith("machineiq/control/"):
                handle_control_message(topic, payload)
            # Data messages are NOT handled via MQTT - use REST API instead
        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"[MQTT] Connected to {MQTT_HOST}:{MQTT_PORT}")
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")


def handle_control_message(topic, payload):
    """Handle control plane messages"""
    action = topic.split("/")[-1]

    if action == "start_training":
        group_name = payload.get("group")
        duration = payload.get("duration", 60)
        start_training(group_name, duration)
    elif action == "stop_training":
        group_name = payload.get("group")
        stop_training(group_name)
    elif action == "configure":
        configure_detector(payload)


# NOTE: handle_data_message removed - MQTT is not suitable for high-fidelity time series
# Use REST API POST /api/analyze or direct pybind11 calls from C++ host instead


def publish_result(group_name, result):
    """Publish analysis result via MQTT"""
    if mqtt_client:
        payload = {
            "group": group_name,
            "timestamp": result.timestamp.isoformat(),
            "confidence": float(1.0 - result.anomaly_score) * 100,
            "detected": result.is_anomaly,
            "pattern": "Anomaly" if result.is_anomaly else "Normal",
            "match_strength": float(result.match_strength),
            "top_contributors": result.get_top_contributors(3)
        }
        mqtt_client.publish(
            f"machineiq/analysis/result/{group_name}",
            json.dumps(payload)
        )


def store_result(group_name, result):
    """Store result in InfluxDB"""
    try:
        client = get_influx_client()
        point = {
            "measurement": "anomaly_detection",
            "tags": {"group": group_name},
            "time": result.timestamp.isoformat(),
            "fields": {
                "match_strength": float(result.match_strength),
                "anomaly_score": float(result.anomaly_score),
                "is_anomaly": int(result.is_anomaly)
            }
        }
        client.write_points([point])
    except Exception as e:
        print(f"[InfluxDB] Write error: {e}")


def start_training(group_name, duration):
    """Start training mode for a group"""
    if group_name in detectors:
        detectors[group_name].start_learning()
        print(f"[ML] Started training for group: {group_name}, duration: {duration}s")

        # Schedule stop after duration
        timer = threading.Timer(duration, lambda: stop_training(group_name))
        timer.start()


def stop_training(group_name):
    """Stop training mode for a group"""
    if group_name in detectors:
        detectors[group_name].stop_learning()
        print(f"[ML] Stopped training for group: {group_name}")

        # Publish status update
        if mqtt_client:
            mqtt_client.publish(
                f"machineiq/status/{group_name}",
                json.dumps({"status": "monitoring", "trained_states": detectors[group_name].num_trained_states})
            )


def configure_detector(config):
    """Configure a detector for a channel group"""
    group_name = config.get("group")
    channels = config.get("channels", [])
    sample_rate = config.get("sample_rate", 1000)

    # Create detector config
    det_config = DetectorConfig(
        bins_per_channel=config.get("bins", 64),
        kernel_type=KernelType[config.get("kernel", "TRIANGULAR").upper()],
        kernel_width=config.get("kernel_width", 0.5)
    )

    # Create detector
    detector = MIQDetector(det_config)
    detector.anomaly_threshold = config.get("threshold", 0.3)

    # Configure channels
    channel_configs = [
        {"name": ch["name"], "min_value": ch.get("min", 0), "max_value": ch.get("max", 100)}
        for ch in channels
    ]
    detector.configure_channels(channel_configs)

    # Create preprocessing pipeline
    pipeline = PipelineBuilder(sample_rate).add_time_domain().build()

    # Store
    detectors[group_name] = detector
    pipelines[group_name] = pipeline

    print(f"[ML] Configured detector for group: {group_name} with {len(channels)} channels")


# REST API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "detectors": len(detectors)})


@app.route('/api/configure', methods=['POST'])
def api_configure():
    """Configure a detector via REST API"""
    config = request.json
    configure_detector(config)
    return jsonify({"status": "configured", "group": config.get("group")})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Analyze high-fidelity time series data via REST API.

    This is the PRIMARY endpoint for data ingestion from web clients.
    For highest performance, use direct pybind11 calls from C++ host.

    Expected JSON payload:
    {
        "group": "group_name",
        "values": [1.23, 4.56, ...],  // Raw waveform samples
        "sample_rate": 10000          // Optional, for preprocessing selection
    }
    """
    data = request.json
    group_name = data.get("group")
    values = np.array(data.get("values", []))

    if group_name not in detectors:
        return jsonify({"error": "Group not configured"}), 400

    detector = detectors[group_name]
    pipeline = pipelines.get(group_name)

    if pipeline and len(values) > 0:
        features = pipeline.get_feature_vector(values)
        result = detector.process(features)

        if result:
            return jsonify({
                "confidence": float(1.0 - result.anomaly_score) * 100,
                "detected": result.is_anomaly,
                "pattern": "Anomaly" if result.is_anomaly else "Normal",
                "match_strength": float(result.match_strength)
            })

    return jsonify({"status": "processing"})


@app.route('/api/training/start', methods=['POST'])
def api_start_training():
    """Start training via REST API"""
    data = request.json
    group_name = data.get("group")
    duration = data.get("duration", 60)

    start_training(group_name, duration)
    return jsonify({"status": "training_started", "group": group_name, "duration": duration})


@app.route('/api/training/stop', methods=['POST'])
def api_stop_training():
    """Stop training via REST API"""
    data = request.json
    group_name = data.get("group")

    stop_training(group_name)
    return jsonify({"status": "training_stopped", "group": group_name})


@app.route('/api/status/<group_name>', methods=['GET'])
def api_status(group_name):
    """Get detector status"""
    if group_name not in detectors:
        return jsonify({"error": "Group not found"}), 404

    detector = detectors[group_name]
    return jsonify({
        "group": group_name,
        "state": detector.state.name,
        "trained_states": detector.num_trained_states,
        "threshold": detector.anomaly_threshold
    })


if __name__ == '__main__':
    print("[ML Service] Starting MachineIQ ML Core...")
    setup_mqtt()
    app.run(host='0.0.0.0', port=5000, debug=False)
