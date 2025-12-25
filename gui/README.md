# MachineIQ Web Dashboard

Browser-based GUI for testing the MachineIQ anomaly detection system.

## Quick Start

### Install GUI Dependencies
```bash
pip install -r gui/requirements.txt
```

### Start the Server
```bash
cd gui/api
python app.py
```

Server runs at: http://localhost:5000

## Usage

1. **Generate Training Data** - Create synthetic sensor data
2. **Train Detector** - Learn normal operating patterns
3. **Monitor Stream** - Test anomaly detection with new data

## Features

- Real-time anomaly score visualization
- Configurable kernels (Triangular/Parabolic) and binning
- Batch monitoring with adjustable anomaly ratios
- Results history tracking

## Architecture

- **Backend**: Flask REST API (`gui/api/app.py`)
- **Frontend**: Single-page HTML dashboard (`gui/static/index.html`)
- **Integration**: Uses existing `miq.core` modules (no modifications)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get system status |
| `/api/generate-data` | POST | Generate training data |
| `/api/train` | POST | Train the detector |
| `/api/monitor` | POST | Monitor single/batch states |
| `/api/monitor-stream` | POST | Monitor generated test stream |
| `/api/results` | GET | Get monitoring history |
| `/api/reset` | POST | Reset the system |

## Configuration

Environment variables:
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: true)

## Production Deployment

For production use:
1. Set `FLASK_DEBUG=false`
2. Configure proper CORS origins
3. Add authentication as needed
4. Use a production WSGI server (gunicorn, uWSGI)
