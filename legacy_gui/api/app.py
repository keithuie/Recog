"""
MachineIQ Flask API Server
Provides REST endpoints for training and monitoring with the MIQ detector
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from miq.core.detector import MIQDetector, DetectorConfig
from miq.core.kernels import KernelType
from miq.utils.data_generator import TimeSeriesDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

# Global detector instance
detector = None
training_data = None
channel_ranges = None
monitoring_results = []


def generate_multivariate_data(n_samples: int, n_channels: int, anomaly_ratio: float = 0.0) -> np.ndarray:
    """Generate multivariate data compatible with the API interface."""
    dataset = TimeSeriesDataset(num_channels=n_channels)

    if anomaly_ratio > 0:
        anomaly_start = int(n_samples * (1 - anomaly_ratio))
        data_list, _ = dataset.generate_with_anomaly(n_samples, anomaly_start, "shift", 0.8)
    else:
        data_list = dataset.generate_baseline(n_samples)

    return np.array(data_list, dtype=np.float32)


@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current detector status"""
    try:
        return jsonify({
            'detector_initialized': detector is not None,
            'is_trained': detector.state.name == 'MONITORING' if detector else False,
            'num_states': detector.num_trained_states if detector else 0,
            'num_channels': len(detector.channel_params) if detector else 0,
            'monitoring_results_count': len(monitoring_results)
        })
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Generate synthetic test data"""
    global training_data, channel_ranges

    params = request.json or {}
    n_samples = params.get('n_samples', 1000)
    n_channels = params.get('n_channels', 6)
    anomaly_ratio = params.get('anomaly_ratio', 0.0)

    # Validate inputs
    if not isinstance(n_samples, int) or n_samples < 10 or n_samples > 100000:
        return jsonify({'success': False, 'error': 'n_samples must be between 10 and 100000'}), 400
    if not isinstance(n_channels, int) or n_channels < 1 or n_channels > 100:
        return jsonify({'success': False, 'error': 'n_channels must be between 1 and 100'}), 400

    try:
        training_data = generate_multivariate_data(
            n_samples=n_samples,
            n_channels=n_channels,
            anomaly_ratio=anomaly_ratio
        )

        # Store channel ranges for later use
        channel_ranges = []
        for i in range(n_channels):
            col = training_data[:, i]
            channel_ranges.append({
                'min': float(col.min()) - 1.0,  # Add margin
                'max': float(col.max()) + 1.0
            })

        logger.info(f"Generated {n_samples} samples with {n_channels} channels")

        return jsonify({
            'success': True,
            'n_samples': n_samples,
            'n_channels': n_channels,
            'data_shape': list(training_data.shape)
        })
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_detector():
    """Train the MIQ detector"""
    global detector, training_data, channel_ranges

    if training_data is None:
        return jsonify({'success': False, 'error': 'No training data available. Generate data first.'}), 400

    params = request.json or {}
    kernel_type = params.get('kernel', 'triangular')
    n_bins = params.get('n_bins', 64)

    # Validate inputs
    if kernel_type not in ['triangular', 'parabolic']:
        return jsonify({'success': False, 'error': 'kernel must be "triangular" or "parabolic"'}), 400
    if not isinstance(n_bins, int) or n_bins < 5 or n_bins > 256:
        return jsonify({'success': False, 'error': 'n_bins must be between 5 and 256'}), 400

    try:
        # Create detector config
        config = DetectorConfig(
            bins_per_channel=n_bins,
            kernel_type=KernelType.TRIANGULAR if kernel_type == 'triangular' else KernelType.PARABOLIC
        )

        # Initialize detector
        detector = MIQDetector(config=config)

        # Configure all channels at once
        n_channels = training_data.shape[1]
        channel_configs = []
        for i in range(n_channels):
            channel_configs.append({
                'name': f'Channel_{i}',
                'min_value': channel_ranges[i]['min'],
                'max_value': channel_ranges[i]['max']
            })

        detector.configure_channels(channel_configs)

        # Train
        detector.start_learning()
        for sample in training_data:
            detector.process(sample)
        detector.stop_learning()

        logger.info(f"Training complete: {detector.num_trained_states} states, {kernel_type} kernel, {n_bins} bins")

        return jsonify({
            'success': True,
            'n_states': detector.num_trained_states,
            'n_channels': len(detector.channel_params),
            'kernel': kernel_type,
            'n_bins': n_bins
        })
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/monitor', methods=['POST'])
def monitor_state():
    """Monitor a single state or batch of states"""
    global detector, monitoring_results

    if detector is None:
        return jsonify({'success': False, 'error': 'Detector not initialized'}), 400
    if detector.state.name != 'MONITORING':
        return jsonify({'success': False, 'error': 'Detector not trained. Train the detector first.'}), 400

    data = request.json
    if not data or 'state' not in data:
        return jsonify({'success': False, 'error': 'No state data provided'}), 400

    state = np.array(data.get('state', []), dtype=np.float32)

    if len(state) == 0:
        return jsonify({'success': False, 'error': 'Empty state data'}), 400

    try:
        # Handle single state or batch
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Validate channel count
        if state.shape[1] != len(detector.channel_params):
            return jsonify({
                'success': False,
                'error': f'Expected {len(detector.channel_params)} channels, got {state.shape[1]}'
            }), 400

        results = []
        for s in state:
            result = detector.process(s)
            result_dict = {
                'score': float(result.match_strength),
                'is_anomaly': result.is_anomaly,
                'anomaly_score': float(result.anomaly_score),
                'state': s.tolist()
            }
            results.append(result_dict)
            monitoring_results.append(result_dict)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/monitor-stream', methods=['POST'])
def monitor_stream():
    """Monitor a stream of data (for demo/testing)"""
    global detector, monitoring_results

    if detector is None:
        return jsonify({'success': False, 'error': 'Detector not initialized'}), 400
    if detector.state.name != 'MONITORING':
        return jsonify({'success': False, 'error': 'Detector not trained. Train the detector first.'}), 400

    params = request.json or {}
    n_samples = params.get('n_samples', 100)
    anomaly_ratio = params.get('anomaly_ratio', 0.1)

    # Validate inputs
    if not isinstance(n_samples, int) or n_samples < 1 or n_samples > 10000:
        return jsonify({'success': False, 'error': 'n_samples must be between 1 and 10000'}), 400
    if not isinstance(anomaly_ratio, (int, float)) or anomaly_ratio < 0 or anomaly_ratio > 1:
        return jsonify({'success': False, 'error': 'anomaly_ratio must be between 0 and 1'}), 400

    try:
        # Generate test stream with same channel count
        n_channels = len(detector.channel_params)
        test_data = generate_multivariate_data(
            n_samples=n_samples,
            n_channels=n_channels,
            anomaly_ratio=anomaly_ratio
        )

        # Monitor all states
        results = []
        for state in test_data:
            result = detector.process(state)
            result_dict = {
                'score': float(result.match_strength),
                'is_anomaly': result.is_anomaly,
                'anomaly_score': float(result.anomaly_score),
                'state': state.tolist()
            }
            results.append(result_dict)

        # Keep only recent results (limit memory usage)
        monitoring_results.extend(results)
        if len(monitoring_results) > 1000:
            monitoring_results = monitoring_results[-1000:]

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        logger.info(f"Monitored {n_samples} samples, detected {anomaly_count} anomalies")

        return jsonify({
            'success': True,
            'n_samples': n_samples,
            'results': results,
            'anomaly_count': anomaly_count
        })
    except Exception as e:
        logger.error(f"Stream monitoring failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get monitoring results history"""
    try:
        limit = request.args.get('limit', 100, type=int)
        limit = max(1, min(limit, 1000))  # Clamp to reasonable range

        return jsonify({
            'success': True,
            'results': monitoring_results[-limit:] if monitoring_results else [],
            'total_count': len(monitoring_results)
        })
    except Exception as e:
        logger.error(f"Results fetch failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the detector and results"""
    global detector, training_data, channel_ranges, monitoring_results

    detector = None
    training_data = None
    channel_ranges = None
    monitoring_results = []

    logger.info("System reset complete")

    return jsonify({'success': True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
