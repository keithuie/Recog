"""
MachineIQ ML Core Service
Provides REST API for high-fidelity anomaly detection

DATA INGESTION ARCHITECTURE:
============================

1. HIGH-FIDELITY DATA (vibration, ultrasound, waveforms):
   -------------------------------------------------------
   - REST API POST /api/analyze (batch processing)
   - Direct pybind11 calls from C++ host (zero-copy, preferred)
   - Direct SQL queries with binary BLOB extraction
   - gRPC streaming (for real-time high-throughput)

   NEVER USE MQTT for waveforms - it adds latency, serialization
   overhead, and causes sample loss at high frequencies.

2. LOW-FREQUENCY IoT SENSORS (MQTT is acceptable):
   ------------------------------------------------
   Topic: machineiq/iot/{group_name}

   ONLY for slow-changing scalar values:
   - Temperature sensors (LoRaWAN, Zigbee, etc.)
   - Humidity, pressure, level sensors
   - Digital I/O states (on/off, open/closed)
   - Current/voltage spot readings (NOT waveforms)
   - GPS coordinates, battery levels

   STRICT LIMITS (enforced):
   - Max 10 values per message (scalar readings, not arrays)
   - Max 10 Hz effective sample rate
   - NO waveforms, NO vibration, NO ultrasound, NO audio

3. CONTROL PLANE (MQTT):
   ----------------------
   - machineiq/control/* - Start/stop training, configuration
   - machineiq/status/* - Status notifications
   - machineiq/analysis/result/* - Anomaly alerts
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

# =============================================================================
# IoT SENSOR DATA PATHWAY (Low-frequency MQTT)
# =============================================================================
# Separate storage for IoT groups - these are NOT waveform analyzers
# They process slow scalar readings from LoRaWAN/Zigbee/IoT sensors

IOT_MAX_VALUES_PER_MESSAGE = 10  # Reject messages with more values (likely waveforms)
IOT_MAX_SAMPLE_RATE_HZ = 10     # Maximum acceptable sample rate for IoT path
IOT_BUFFER_SIZE = 60            # Accumulate N readings before analysis

iot_detectors = {}      # Separate from high-fidelity detectors
iot_buffers = {}        # Accumulate scalar readings: {group: {channel: [values]}}
iot_last_update = {}    # Track timing to detect abuse: {group: timestamp}


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
    Setup MQTT client for:
    1. Control plane (start/stop, config)
    2. IoT sensor data (LOW-FREQUENCY ONLY - temperature, humidity, I/O)

    HIGH-FIDELITY DATA (vibration, ultrasound, waveforms) must use REST API
    or direct pybind11 calls - NEVER MQTT.
    """
    global mqtt_client

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        # Subscribe to CONTROL topics
        client.subscribe("machineiq/control/#")

        # Subscribe to IoT sensor topics (LOW-FREQUENCY ONLY)
        # Topic format: machineiq/iot/{group_name}
        client.subscribe("machineiq/iot/#")
        print("[MQTT] Subscribed to machineiq/iot/# for LOW-FREQUENCY IoT sensors only")

        # NOTE: We do NOT subscribe to machineiq/data/#
        # High-fidelity waveform data must use REST API or direct calls

    def on_message(client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            if topic.startswith("machineiq/control/"):
                handle_control_message(topic, payload)
            elif topic.startswith("machineiq/iot/"):
                # IoT sensor pathway - strictly for low-frequency scalar data
                handle_iot_sensor_message(topic, payload)
            # machineiq/data/* is NOT handled - use REST API for waveforms
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


# =============================================================================
# IoT SENSOR HANDLER (Low-frequency MQTT pathway)
# =============================================================================

def handle_iot_sensor_message(topic, payload):
    """
    Handle LOW-FREQUENCY IoT sensor data from MQTT.

    STRICTLY FOR:
    - Temperature, humidity, pressure sensors
    - Digital I/O states
    - LoRaWAN/Zigbee/NB-IoT devices
    - Slow-changing scalar values (< 10 Hz)

    NEVER FOR:
    - Vibration waveforms
    - Ultrasound data
    - Audio signals
    - Any high-frequency time series

    Expected payload format:
    {
        "device_id": "sensor_001",
        "readings": {
            "temperature": 23.5,
            "humidity": 65.2,
            "pressure": 1013.25
        },
        "timestamp": "2024-01-01T12:00:00Z"  # Optional
    }
    """
    group_name = topic.split("/")[-1]  # machineiq/iot/{group_name}

    # Extract readings
    readings = payload.get("readings", {})
    device_id = payload.get("device_id", "unknown")

    # =========================================================================
    # STRICT VALIDATION - Reject anything that looks like waveform data
    # =========================================================================

    # Check 1: Reject if too many values (likely a waveform array)
    if isinstance(readings, list):
        if len(readings) > IOT_MAX_VALUES_PER_MESSAGE:
            print(f"[IoT] REJECTED: {group_name} sent {len(readings)} values - "
                  f"exceeds limit of {IOT_MAX_VALUES_PER_MESSAGE}. Use REST API for waveforms.")
            return
        # Convert list to dict
        readings = {f"ch_{i}": v for i, v in enumerate(readings)}

    if isinstance(readings, dict):
        total_values = sum(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in readings.values()
        )
        if total_values > IOT_MAX_VALUES_PER_MESSAGE:
            print(f"[IoT] REJECTED: {group_name} sent {total_values} total values - "
                  f"exceeds limit of {IOT_MAX_VALUES_PER_MESSAGE}. Use REST API for waveforms.")
            return

    # Check 2: Reject any value that is an array (waveforms are arrays)
    for key, value in readings.items():
        if isinstance(value, (list, np.ndarray)):
            if len(value) > 1:
                print(f"[IoT] REJECTED: {group_name}.{key} contains array of {len(value)} samples. "
                      f"IoT path is for scalar values only. Use REST API for waveforms.")
                return

    # Check 3: Rate limiting - reject if updates too fast (> 10 Hz)
    now = datetime.now()
    if group_name in iot_last_update:
        elapsed = (now - iot_last_update[group_name]).total_seconds()
        if elapsed < (1.0 / IOT_MAX_SAMPLE_RATE_HZ):
            # Too fast - this might be waveform data trying to sneak through
            print(f"[IoT] RATE LIMITED: {group_name} updating faster than {IOT_MAX_SAMPLE_RATE_HZ} Hz. "
                  f"Use REST API for high-frequency data.")
            return
    iot_last_update[group_name] = now

    # =========================================================================
    # Process validated IoT sensor readings
    # =========================================================================

    # Initialize buffer for this group if needed
    if group_name not in iot_buffers:
        iot_buffers[group_name] = {}

    # Add readings to buffer
    for channel, value in readings.items():
        if channel not in iot_buffers[group_name]:
            iot_buffers[group_name][channel] = []

        # Handle scalar or single-element array
        scalar_value = value[0] if isinstance(value, (list, np.ndarray)) else value
        iot_buffers[group_name][channel].append(float(scalar_value))

        # Trim buffer to max size
        if len(iot_buffers[group_name][channel]) > IOT_BUFFER_SIZE:
            iot_buffers[group_name][channel] = iot_buffers[group_name][channel][-IOT_BUFFER_SIZE:]

    # Store in InfluxDB
    store_iot_reading(group_name, device_id, readings)

    # Check if we have enough data to analyze
    if group_name in iot_detectors:
        min_samples = min(len(buf) for buf in iot_buffers[group_name].values())
        if min_samples >= IOT_BUFFER_SIZE:
            analyze_iot_buffer(group_name)


def store_iot_reading(group_name, device_id, readings):
    """Store IoT sensor reading in InfluxDB"""
    try:
        client = get_influx_client()
        fields = {}
        for key, value in readings.items():
            scalar = value[0] if isinstance(value, (list, np.ndarray)) else value
            fields[key] = float(scalar)

        point = {
            "measurement": "iot_sensors",
            "tags": {"group": group_name, "device": device_id},
            "time": datetime.now().isoformat(),
            "fields": fields
        }
        client.write_points([point])
    except Exception as e:
        print(f"[InfluxDB] IoT write error: {e}")


def analyze_iot_buffer(group_name):
    """Analyze accumulated IoT sensor buffer"""
    if group_name not in iot_detectors or group_name not in iot_buffers:
        return

    detector = iot_detectors[group_name]
    buffer = iot_buffers[group_name]

    # Build feature vector from buffer (use latest values)
    channel_names = sorted(buffer.keys())
    feature_vector = np.array([buffer[ch][-1] for ch in channel_names], dtype=np.float32)

    # Process with detector
    result = detector.process(feature_vector)

    if result:
        publish_result(group_name, result)
        store_result(group_name, result)


def configure_iot_detector(config):
    """
    Configure a detector for IoT sensor group.

    This creates a simple detector for scalar IoT values.
    NO preprocessing pipeline - just direct value comparison.
    """
    group_name = config.get("group")
    channels = config.get("channels", [])

    # Create detector config - simpler settings for IoT
    det_config = DetectorConfig(
        bins_per_channel=config.get("bins", 32),  # Fewer bins for slow data
        kernel_type=KernelType.TRIANGULAR,
        kernel_width=config.get("kernel_width", 0.7)  # Wider kernel for noisy IoT
    )

    detector = MIQDetector(det_config)
    detector.anomaly_threshold = config.get("threshold", 0.4)

    # Configure channels
    channel_configs = [
        {"name": ch["name"], "min_value": ch.get("min", 0), "max_value": ch.get("max", 100)}
        for ch in channels
    ]
    detector.configure_channels(channel_configs)

    # Store in IoT-specific storage
    iot_detectors[group_name] = detector
    iot_buffers[group_name] = {}

    print(f"[IoT] Configured detector for group: {group_name} with {len(channels)} channels")


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
