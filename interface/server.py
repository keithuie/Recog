"""
MachineIQ Web Interface Server
Custom dashboard replacing Grafana for IP ownership
"""
import os
import sys
import json
import asyncio
import aiohttp
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from io import StringIO
from enum import Enum

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Configuration
INFLUXDB_HOST = os.environ.get('INFLUXDB_HOST', 'localhost')
INFLUXDB_PORT = int(os.environ.get('INFLUXDB_PORT', 8086))
MQTT_HOST = os.environ.get('MQTT_HOST', 'localhost')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))
ML_CORE_HOST = os.environ.get('ML_CORE_HOST', 'localhost')
ML_CORE_PORT = int(os.environ.get('ML_CORE_PORT', 5000))

# Initialize FastAPI app
app = FastAPI(
    title="MachineIQ Dashboard",
    description="Real-time anomaly detection monitoring system",
    version="1.0.0"
)

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Create directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Group State Enum - 4 states as specified
class GroupState(str, Enum):
    IDLE = "idle"                    # No data streaming
    STREAMING = "streaming"          # Data streaming, no trained model
    TRAINING = "training"            # Data streaming, training in progress
    STANDBY = "standby"              # Trained model exists, not monitoring
    MONITORING = "monitoring"        # Trained model, actively monitoring
    ERROR = "error"                  # Error state


# In-memory state (in production, use Redis or database)
channel_groups: Dict[str, dict] = {}
data_sources: Dict[str, dict] = {}
active_websockets: List[WebSocket] = []
data_polling_task: Optional[asyncio.Task] = None
ml_service_process: Optional[subprocess.Popen] = None

# Data cache for streaming
latest_data: Dict[str, dict] = {}  # {group_name: {channel_values, confidence, timestamp}}

# ML Service status
ml_service_status: dict = {
    "running": False,
    "configured_groups": [],
    "error": None
}

# Model configuration (global defaults)
current_model_config: dict = {
    "kernel_type": "triangular",
    "num_bins": 64,
    "kernel_width": 0.5,
    "threshold": 0.3,
    "training_duration": 60,
    "preprocessing": "basic",
    "auto_scale": True,
    "outlier_rejection": True,
    "smoothing_window": 5
}


# Pydantic models for API
class DataSourceConfig(BaseModel):
    name: str
    source_type: str  # api, mqtt, sql, csv
    url: Optional[str] = None
    poll_interval: int = 5
    mqtt_topic: Optional[str] = None
    sql_connection: Optional[str] = None
    sql_query: Optional[str] = None


class ChannelGroupConfig(BaseModel):
    name: str
    channels: List[str]
    color: str = "#007AFF"
    sample_rate: float = 1000
    preprocessing: str = "basic"  # basic, vibration, bearing
    source: Optional[str] = None  # Link to data source name


class ModelConfig(BaseModel):
    kernel_type: str = "triangular"
    num_bins: int = 64
    kernel_width: float = 0.5
    threshold: float = 0.3
    training_duration: int = 60
    preprocessing: str = "basic"
    auto_scale: bool = True
    outlier_rejection: bool = True
    smoothing_window: int = 5


class TrainingRequest(BaseModel):
    group_name: str
    duration: int = 60


# Page Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "MachineIQ Dashboard",
        "groups": list(channel_groups.values()),
        "sources": list(data_sources.values())
    })


@app.get("/setup", response_class=HTMLResponse)
async def setup_wizard(request: Request):
    """Render the setup wizard."""
    return templates.TemplateResponse("setup.html", {
        "request": request,
        "title": "Setup Wizard"
    })


@app.get("/dataflow", response_class=HTMLResponse)
async def dataflow_page(request: Request):
    """Render the dataflow/data sources page."""
    return templates.TemplateResponse("dataflow.html", {
        "request": request,
        "title": "Dataflow",
        "sources": list(data_sources.values())
    })


@app.get("/model-config", response_class=HTMLResponse)
async def model_config_page(request: Request):
    """Render the model configuration page."""
    return templates.TemplateResponse("model_config.html", {
        "request": request,
        "title": "Model Configuration"
    })


# API Routes
@app.post("/api/sources")
async def add_data_source(config: DataSourceConfig):
    """Add a new data source."""
    data_sources[config.name] = {
        "name": config.name,
        "type": config.source_type,
        "url": config.url,
        "poll_interval": config.poll_interval,
        "mqtt_topic": config.mqtt_topic,
        "sql_connection": config.sql_connection,
        "sql_query": config.sql_query,
        "status": "connected",
        "created_at": datetime.now().isoformat()
    }
    await broadcast_update("source_added", data_sources[config.name])
    return {"status": "success", "source": data_sources[config.name]}


@app.get("/api/sources")
async def list_data_sources():
    """List all data sources."""
    return list(data_sources.values())


@app.delete("/api/sources/{name}")
async def remove_data_source(name: str):
    """Remove a data source."""
    if name in data_sources:
        del data_sources[name]
        await broadcast_update("source_removed", {"name": name})
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Source not found")


@app.post("/api/groups")
async def add_channel_group(config: ChannelGroupConfig):
    """Add a new channel group."""
    channel_groups[config.name] = {
        "name": config.name,
        "channels": config.channels,
        "color": config.color,
        "sample_rate": config.sample_rate,
        "preprocessing": config.preprocessing,
        "source": config.source,  # Link to data source
        "state": GroupState.IDLE,
        # Training metadata
        "trained_states": 0,
        "training_start_time": None,
        "training_end_time": None,
        "training_duration": None,
        "last_trained": None,
        # Streaming metadata
        "streaming_start_time": None,
        "stream_duration_total": 0,
        "samples_processed": 0,
        # Error tracking
        "last_error": None,
        "created_at": datetime.now().isoformat()
    }
    await broadcast_update("group_added", channel_groups[config.name])
    return {"status": "success", "group": channel_groups[config.name]}


@app.get("/api/groups")
async def list_channel_groups():
    """List all channel groups."""
    return list(channel_groups.values())


@app.delete("/api/groups/{name}")
async def remove_channel_group(name: str):
    """Remove a channel group."""
    if name in channel_groups:
        del channel_groups[name]
        await broadcast_update("group_removed", {"name": name})
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Group not found")


# Model Configuration API
@app.get("/api/model-config")
async def get_model_config():
    """Get current model configuration."""
    return current_model_config


@app.post("/api/model-config")
async def save_model_config(config: ModelConfig):
    """Save model configuration."""
    global current_model_config
    current_model_config = {
        "kernel_type": config.kernel_type,
        "num_bins": config.num_bins,
        "kernel_width": config.kernel_width,
        "threshold": config.threshold,
        "training_duration": config.training_duration,
        "preprocessing": config.preprocessing,
        "auto_scale": config.auto_scale,
        "outlier_rejection": config.outlier_rejection,
        "smoothing_window": config.smoothing_window
    }
    await broadcast_update("config_updated", current_model_config)
    return {"status": "success", "config": current_model_config}


# =============================================================================
# Group State Management API - 4-State System
# =============================================================================

@app.post("/api/groups/{group_name}/start-stream")
async def start_streaming(group_name: str):
    """Start data streaming for a group (State: IDLE -> STREAMING)."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    # Can start streaming from IDLE or ERROR state
    if group["state"] not in [GroupState.IDLE, GroupState.ERROR]:
        raise HTTPException(status_code=400, detail=f"Cannot start streaming from state: {group['state']}")

    # Update state
    group["state"] = GroupState.STREAMING
    group["streaming_start_time"] = datetime.now().isoformat()
    group["last_error"] = None

    # Ensure ML service is running
    await ensure_ml_service_running()

    await broadcast_update("stream_started", {
        "group": group_name,
        "state": group["state"],
        "streaming_start_time": group["streaming_start_time"]
    })

    print(f"[Stream] Started for {group_name}")
    return {"status": "success", "state": group["state"], "group": group}


@app.post("/api/groups/{group_name}/stop-stream")
async def stop_streaming(group_name: str):
    """Stop data streaming for a group (Any State -> IDLE)."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    # Calculate total stream time
    if group["streaming_start_time"]:
        start = datetime.fromisoformat(group["streaming_start_time"])
        duration = (datetime.now() - start).total_seconds()
        group["stream_duration_total"] += duration

    # Reset to IDLE
    group["state"] = GroupState.IDLE
    group["streaming_start_time"] = None

    await broadcast_update("stream_stopped", {
        "group": group_name,
        "state": group["state"]
    })

    print(f"[Stream] Stopped for {group_name}")
    return {"status": "success", "state": group["state"]}


@app.post("/api/groups/{group_name}/start-training")
async def start_training(group_name: str, req: TrainingRequest):
    """Start training for a group (STREAMING -> TRAINING)."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    # Must be streaming to train
    if group["state"] != GroupState.STREAMING:
        raise HTTPException(status_code=400, detail=f"Must be streaming to train. Current state: {group['state']}")

    # Update state
    group["state"] = GroupState.TRAINING
    group["training_start_time"] = datetime.now().isoformat()
    group["training_duration"] = req.duration

    # Try to call ML service to start training
    ml_training_started = False
    try:
        async with aiohttp.ClientSession() as session:
            # First configure the detector if not done
            channel_configs = []
            for ch in group.get("channels", []):
                ch_info = CHANNEL_INFO.get(ch, {})
                channel_configs.append({
                    "name": ch,
                    "min_value": 0,  # Will be auto-scaled
                    "max_value": 1000  # Will be auto-scaled
                })

            # Configure ML service
            config_payload = {
                "group": group_name,
                "channels": channel_configs,
                "sample_rate": group.get("sample_rate", 1000),
                "bins": current_model_config.get("num_bins", 64),
                "kernel": current_model_config.get("kernel_type", "triangular"),
                "kernel_width": current_model_config.get("kernel_width", 0.5),
                "threshold": current_model_config.get("threshold", 0.3)
            }

            async with session.post(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/api/configure",
                json=config_payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    print(f"[Training] Configured ML service for {group_name}")

            # Start training
            async with session.post(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/api/training/start",
                json={"group": group_name, "duration": req.duration},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    ml_training_started = True
                    print(f"[Training] Started ML training for {group_name}")
    except Exception as e:
        print(f"[Training] ML service not available: {e}")
        group["last_error"] = f"ML service unavailable: {str(e)}"

    await broadcast_update("training_started", {
        "group": group_name,
        "state": group["state"],
        "duration": req.duration,
        "ml_service_connected": ml_training_started,
        "training_start_time": group["training_start_time"]
    })

    # Schedule training completion (if duration > 0)
    if req.duration > 0:
        asyncio.create_task(complete_training_after(group_name, req.duration))

    return {
        "status": "training_started",
        "state": group["state"],
        "duration": req.duration,
        "ml_service_connected": ml_training_started
    }


async def complete_training_after(group_name: str, duration: int):
    """Complete training after specified duration."""
    await asyncio.sleep(duration)
    if group_name in channel_groups and channel_groups[group_name]["state"] == GroupState.TRAINING:
        await finish_training(group_name)


async def finish_training(group_name: str):
    """Finish training and transition to STANDBY."""
    if group_name not in channel_groups:
        return

    group = channel_groups[group_name]

    # Calculate training duration
    if group["training_start_time"]:
        start = datetime.fromisoformat(group["training_start_time"])
        actual_duration = (datetime.now() - start).total_seconds()
    else:
        actual_duration = group.get("training_duration", 0)

    # Try to stop ML training and get trained states count
    trained_states = 0
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/api/training/stop",
                json={"group": group_name},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    trained_states = result.get("trained_states", 1)
                    print(f"[Training] ML service stopped training for {group_name}, {trained_states} states learned")
    except Exception as e:
        print(f"[Training] Could not stop ML training: {e}")
        trained_states = 1  # Assume at least 1 state was learned

    # Update group metadata
    group["state"] = GroupState.STANDBY
    group["training_end_time"] = datetime.now().isoformat()
    group["last_trained"] = datetime.now().isoformat()
    group["training_duration"] = actual_duration
    group["trained_states"] = group.get("trained_states", 0) + trained_states

    await broadcast_update("training_complete", {
        "group": group_name,
        "state": group["state"],
        "trained_states": group["trained_states"],
        "training_duration": actual_duration
    })

    print(f"[Training] Completed for {group_name}: {trained_states} states, {actual_duration:.1f}s")


@app.post("/api/groups/{group_name}/stop-training")
async def stop_training(group_name: str):
    """Manually stop training (TRAINING -> STANDBY)."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    if group["state"] != GroupState.TRAINING:
        raise HTTPException(status_code=400, detail=f"Not currently training. State: {group['state']}")

    await finish_training(group_name)

    return {"status": "success", "state": group["state"], "trained_states": group["trained_states"]}


@app.post("/api/groups/{group_name}/start-monitoring")
async def start_monitoring(group_name: str):
    """Start monitoring (STANDBY -> MONITORING)."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    # Must have trained states to monitor
    if group["trained_states"] == 0:
        raise HTTPException(status_code=400, detail="No trained states. Train the model first.")

    # Must be in STANDBY to start monitoring
    if group["state"] != GroupState.STANDBY:
        raise HTTPException(status_code=400, detail=f"Must be in standby to monitor. Current state: {group['state']}")

    group["state"] = GroupState.MONITORING

    await broadcast_update("monitoring_started", {
        "group": group_name,
        "state": group["state"],
        "trained_states": group["trained_states"]
    })

    print(f"[Monitoring] Started for {group_name}")
    return {"status": "success", "state": group["state"]}


@app.post("/api/groups/{group_name}/stop-monitoring")
async def stop_monitoring(group_name: str):
    """Stop monitoring (MONITORING -> STANDBY)."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    if group["state"] != GroupState.MONITORING:
        raise HTTPException(status_code=400, detail=f"Not currently monitoring. State: {group['state']}")

    group["state"] = GroupState.STANDBY

    await broadcast_update("monitoring_stopped", {
        "group": group_name,
        "state": group["state"]
    })

    print(f"[Monitoring] Stopped for {group_name}")
    return {"status": "success", "state": group["state"]}


@app.get("/api/groups/{group_name}/state")
async def get_group_state(group_name: str):
    """Get current state and metadata for a group."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    group = channel_groups[group_name]

    # Calculate current stream duration if streaming
    current_stream_duration = group["stream_duration_total"]
    if group["streaming_start_time"] and group["state"] in [GroupState.STREAMING, GroupState.TRAINING, GroupState.MONITORING]:
        start = datetime.fromisoformat(group["streaming_start_time"])
        current_stream_duration += (datetime.now() - start).total_seconds()

    return {
        "group": group_name,
        "state": group["state"],
        "trained_states": group["trained_states"],
        "last_trained": group["last_trained"],
        "training_duration": group["training_duration"],
        "stream_duration_total": current_stream_duration,
        "samples_processed": group["samples_processed"],
        "last_error": group["last_error"]
    }


# Training duration presets
TRAINING_PRESETS = [
    {"label": "Quick (30s)", "duration": 30},
    {"label": "Standard (60s)", "duration": 60},
    {"label": "Extended (2min)", "duration": 120},
    {"label": "Thorough (5min)", "duration": 300},
    {"label": "Manual", "duration": 0}  # 0 means manual stop
]

@app.get("/api/training/presets")
async def get_training_presets():
    """Get available training duration presets."""
    return {"presets": TRAINING_PRESETS}


# =============================================================================
# ML Service Management
# =============================================================================

async def ensure_ml_service_running():
    """Ensure the ML service is running, start if necessary."""
    global ml_service_status

    # Check if already running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/health",
                timeout=aiohttp.ClientTimeout(total=2)
            ) as response:
                if response.status == 200:
                    ml_service_status["running"] = True
                    ml_service_status["error"] = None
                    return True
    except Exception:
        pass

    # Not running - try to start it
    try:
        print("[ML Service] Attempting to start ML service...")
        ml_service_path = os.path.join(os.path.dirname(BASE_DIR), "ml_service.py")

        if os.path.exists(ml_service_path):
            # Start ML service in background
            global ml_service_process
            ml_service_process = subprocess.Popen(
                [sys.executable, ml_service_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(BASE_DIR)
            )

            # Wait a moment for it to start
            await asyncio.sleep(2)

            # Verify it started
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/health",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as response:
                        if response.status == 200:
                            ml_service_status["running"] = True
                            ml_service_status["error"] = None
                            print("[ML Service] Started successfully")
                            return True
            except Exception:
                pass

        ml_service_status["running"] = False
        ml_service_status["error"] = "Failed to start ML service"
        print("[ML Service] Failed to start")
        return False

    except Exception as e:
        ml_service_status["running"] = False
        ml_service_status["error"] = str(e)
        print(f"[ML Service] Error starting: {e}")
        return False


@app.get("/api/ml-service/status")
async def get_ml_service_status():
    """Get ML service status."""
    # Check if ML service is responsive
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/health",
                timeout=aiohttp.ClientTimeout(total=2)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    ml_service_status["running"] = True
                    ml_service_status["error"] = None
                    ml_service_status["configured_groups"] = list(data.get("detectors", {}).keys()) if isinstance(data.get("detectors"), dict) else []
                    return {**ml_service_status, "health": data}
    except Exception as e:
        ml_service_status["running"] = False
        ml_service_status["error"] = str(e)

    return ml_service_status


@app.get("/api/csv-template")
async def download_csv_template():
    """Download CSV template for data upload."""
    csv_content = """timestamp,channel_1,channel_2,channel_3,channel_4
2024-01-01T00:00:00Z,0.523,1.234,-0.876,2.345
2024-01-01T00:00:01Z,0.612,1.156,-0.923,2.412
2024-01-01T00:00:02Z,0.498,1.298,-0.845,2.289
2024-01-01T00:00:03Z,0.567,1.201,-0.901,2.378
2024-01-01T00:00:04Z,0.534,1.245,-0.867,2.356

# CSV Template Instructions:
# 1. First column must be 'timestamp' in ISO 8601 format
# 2. Subsequent columns are your data channels
# 3. Channel names become the column headers
# 4. Values should be numeric (float or integer)
# 5. Missing values can be left empty or marked as NaN
# 6. Sample rate is determined by timestamp intervals
#
# Example use cases:
# - Vibration data: acceleration values in g or mm/s
# - Temperature: values in Celsius or Fahrenheit
# - Pressure: values in PSI or bar
# - Current/Voltage: electrical measurements
"""
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=machineiq_template.csv"}
    )


@app.get("/api/preprocessing-info")
async def get_preprocessing_info():
    """Get information about available preprocessing algorithms."""
    return {
        "algorithms": [
            {
                "id": "basic",
                "name": "Basic Time Domain",
                "description": "RMS, Peak, Crest Factor, Kurtosis - suitable for general monitoring",
                "features": ["rms", "peak", "crest_factor", "kurtosis"],
                "sample_rate_min": 100,
                "tip": "Good starting point for any signal type"
            },
            {
                "id": "vibration",
                "name": "Full Vibration Analysis",
                "description": "Time + Frequency + Envelope + StressWave - comprehensive vibration diagnostics",
                "features": ["rms", "peak", "kurtosis", "peak_frequency", "spectral_energy",
                           "envelope_rms", "envelope_kurtosis", "stresswave_rms", "stresswave_count"],
                "sample_rate_min": 5000,
                "tip": "Best for rotating machinery with sample rate >= 5kHz"
            },
            {
                "id": "bearing",
                "name": "Bearing Fault Detection",
                "description": "Optimized for detecting bearing defects using envelope and stress wave analysis",
                "features": ["rms", "peak", "kurtosis", "crest_factor", "peak_frequency",
                           "envelope_rms", "envelope_kurtosis", "stresswave_rms", "stresswave_peak"],
                "sample_rate_min": 10000,
                "tip": "Requires high sample rate (10kHz+) for accurate bearing frequency detection"
            },
            {
                "id": "frequency",
                "name": "Frequency Analysis",
                "description": "FFT-based spectral analysis for identifying frequency components",
                "features": ["peak_frequency", "spectral_centroid", "spectral_spread",
                           "spectral_energy", "spectral_entropy"],
                "sample_rate_min": 1000,
                "tip": "Use when you need to identify specific frequency content"
            }
        ],
        "filters": [
            {
                "id": "bandpass",
                "name": "Bandpass Filter",
                "description": "Isolate specific frequency range",
                "parameters": ["low_freq", "high_freq"],
                "tip": "Useful for removing noise outside frequency range of interest"
            },
            {
                "id": "highpass",
                "name": "Highpass Filter",
                "description": "Remove low-frequency drift and DC offset",
                "parameters": ["cutoff_freq"],
                "tip": "Common cutoff: 10 Hz for vibration, 0.1 Hz for slow processes"
            }
        ]
    }


@app.get("/api/model-config-info")
async def get_model_config_info():
    """Get information about model configuration parameters."""
    return {
        "parameters": [
            {
                "id": "kernel_type",
                "name": "Kernel Type",
                "options": ["triangular", "parabolic"],
                "default": "triangular",
                "description": "Shape of the kernel used for pattern matching",
                "tip": "Triangular: Sharp response, good for detecting sudden changes. "
                      "Parabolic: Smoother response, better for gradual variations."
            },
            {
                "id": "num_bins",
                "name": "Number of Bins",
                "options": [32, 64, 128, 256],
                "default": 64,
                "description": "Resolution of the histogram for each channel",
                "tip": "More bins = higher resolution but needs more training data. "
                      "64 bins is a good balance for most applications."
            },
            {
                "id": "kernel_width",
                "name": "Kernel Width",
                "min": 0.1,
                "max": 2.0,
                "default": 0.5,
                "description": "Width of the kernel relative to bin width",
                "tip": "Lower values (0.3-0.5): More sensitive to exact matches. "
                      "Higher values (0.7-1.0): More tolerant of variation."
            },
            {
                "id": "threshold",
                "name": "Anomaly Threshold",
                "min": 0.1,
                "max": 0.9,
                "default": 0.3,
                "description": "Match strength below which triggers anomaly alert",
                "tip": "Lower threshold = more sensitive (more alerts). "
                      "Start with 0.3 and adjust based on false positive rate."
            }
        ],
        "training": {
            "duration_options": [30, 60, 120, 300, 600],
            "default_duration": 60,
            "description": "Duration in seconds for learning normal patterns",
            "tip": "Training should capture full operating cycle. "
                  "For machinery: at least one complete rotation at slowest speed."
        }
    }


# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe":
                # Handle subscription requests
                pass
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass  # Normal disconnect
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        # Always remove from active list
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def broadcast_update(event_type: str, data: dict):
    """Broadcast update to all connected WebSocket clients."""
    message = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now().isoformat()})
    disconnected = []

    for ws in active_websockets:
        try:
            await ws.send_text(message)
        except Exception as e:
            print(f"[Broadcast] Failed to send to client: {e}")
            disconnected.append(ws)

    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)


# =============================================================================
# Data Streaming & API Integration
# =============================================================================

async def poll_api_source(source: dict) -> Optional[dict]:
    """Poll data from an API source."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(source['url'], timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
    except Exception as e:
        print(f"[API Poll] Error fetching from {source.get('name', 'unknown')}: {e}")
    return None


async def analyze_with_ml_service(group_name: str, values: list) -> Optional[dict]:
    """Send data to ML service for analysis."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"group": group_name, "values": values}
            async with session.post(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/api/analyze",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        # ML service may not be running, fall back to simulation
        pass
    return None


async def data_polling_loop():
    """Background task to poll data from sources and stream to dashboard.

    NO SIMULATION - Only real data from sources and real ML analysis.
    """
    while True:
        try:
            # Poll each data source (API or sample)
            for source_name, source in data_sources.items():
                data = None

                # Handle API sources
                if source.get('type') == 'api' and source.get('url'):
                    data = await poll_api_source(source)

                # Handle sample data sources
                elif source.get('type') == 'sample':
                    data = await poll_sample_source(source_name, source)

                if data:
                    # Process data for each group using this source
                    for group_name, group in channel_groups.items():
                        if group.get('source') == source_name:
                            # Only process if group is in active state
                            group_state = group.get('state', GroupState.IDLE)
                            if group_state == GroupState.IDLE:
                                continue  # Not streaming, skip

                            channel_values = {}
                            values_list = []

                            for channel in group.get('channels', []):
                                # Try to find channel value in response
                                value = data.get(channel, data.get('values', {}).get(channel))
                                if value is not None:
                                    channel_values[channel] = float(value)
                                    values_list.append(float(value))

                            if channel_values:
                                # Update samples processed count
                                group["samples_processed"] = group.get("samples_processed", 0) + 1

                                # Get RUL if available (for turbofan data)
                                rul = data.get('RUL')

                                # Determine confidence based on state - NO SIMULATION
                                confidence = None  # Default: no confidence (not monitoring)
                                ml_connected = False

                                if group_state == GroupState.MONITORING:
                                    # Only get ML analysis when actively monitoring
                                    ml_result = await analyze_with_ml_service(group_name, values_list)
                                    if ml_result:
                                        confidence = ml_result.get('confidence')
                                        ml_connected = True
                                    else:
                                        # ML service unavailable during monitoring - this is an error
                                        group["last_error"] = "ML service unavailable during monitoring"

                                elif group_state == GroupState.TRAINING:
                                    # During training, send data to ML service for learning
                                    ml_result = await analyze_with_ml_service(group_name, values_list)
                                    ml_connected = ml_result is not None
                                    confidence = None  # No confidence during training

                                # Broadcast real data to dashboard
                                await broadcast_update("data_update", {
                                    "group": group_name,
                                    "state": group_state,
                                    "confidence": confidence,  # None if not monitoring
                                    "channel_values": channel_values,
                                    "timestamp": datetime.now().isoformat(),
                                    "rul": rul,
                                    "ml_connected": ml_connected,
                                    "samples_processed": group["samples_processed"]
                                })

            # Poll interval (auto-managed based on number of sources)
            poll_interval = max(0.5, min(2.0, 1.0 / max(1, len(data_sources))))
            await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Polling] Error in data polling loop: {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    global data_polling_task
    data_polling_task = asyncio.create_task(data_polling_loop())
    print("[Server] Data polling task started (NO SIMULATION MODE)")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on shutdown."""
    global data_polling_task, ml_service_process
    if data_polling_task:
        data_polling_task.cancel()
    if ml_service_process:
        ml_service_process.terminate()


# API Test & Channel Detection
@app.post("/api/test-connection")
async def test_api_connection(request: Request):
    """Test API connection and detect available channels."""
    body = await request.json()
    url = body.get('url')

    if not url:
        return JSONResponse({"success": False, "error": "URL is required"}, status_code=400)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()

                    # Detect channels from response
                    channels = []
                    if isinstance(data, dict):
                        # Look for numeric values that could be channels
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                channels.append(key)
                            elif isinstance(value, dict):
                                # Nested structure - look for numeric values
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, (int, float)):
                                        channels.append(f"{key}.{subkey}" if key else subkey)

                        # Also check for 'values', 'channels', 'data' keys
                        for container_key in ['values', 'channels', 'data', 'readings']:
                            container = data.get(container_key)
                            if isinstance(container, dict):
                                for key, value in container.items():
                                    if isinstance(value, (int, float)) and key not in channels:
                                        channels.append(key)
                            elif isinstance(container, list):
                                # List of channel names or values
                                for i, item in enumerate(container):
                                    if isinstance(item, str):
                                        if item not in channels:
                                            channels.append(item)
                                    elif isinstance(item, dict) and 'name' in item:
                                        if item['name'] not in channels:
                                            channels.append(item['name'])

                    return {
                        "success": True,
                        "channels": channels,
                        "sample_data": data
                    }
                else:
                    return JSONResponse({
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }, status_code=400)

    except asyncio.TimeoutError:
        return JSONResponse({
            "success": False,
            "error": "Connection timeout"
        }, status_code=408)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "influxdb": INFLUXDB_HOST,
            "mqtt": MQTT_HOST,
            "ml_core": ML_CORE_HOST
        },
        "groups": len(channel_groups),
        "sources": len(data_sources),
        "websockets": len(active_websockets)
    }


# =============================================================================
# Sample Data Support
# =============================================================================

# Available sample datasets with detailed channel information
CHANNEL_INFO = {
    "cycle": {"name": "Cycle", "unit": "count", "description": "Operating cycle number"},
    "T2": {"name": "Fan Inlet Temp", "unit": "°R", "description": "Total temperature at fan inlet"},
    "T24": {"name": "LPC Outlet Temp", "unit": "°R", "description": "Total temperature at LPC outlet"},
    "T30": {"name": "HPC Outlet Temp", "unit": "°R", "description": "Total temperature at HPC outlet"},
    "T50": {"name": "LPT Outlet Temp", "unit": "°R", "description": "Total temperature at LPT outlet"},
    "P2": {"name": "Fan Inlet Pressure", "unit": "psia", "description": "Pressure at fan inlet"},
    "P15": {"name": "Bypass Duct Pressure", "unit": "psia", "description": "Total pressure in bypass-duct"},
    "P30": {"name": "HPC Outlet Pressure", "unit": "psia", "description": "Total pressure at HPC outlet"},
    "Nf": {"name": "Fan Speed", "unit": "rpm", "description": "Physical fan speed"},
    "Nc": {"name": "Core Speed", "unit": "rpm", "description": "Physical core speed"},
    "epr": {"name": "Engine Pressure Ratio", "unit": "ratio", "description": "Engine pressure ratio (P50/P2)"},
    "Ps30": {"name": "HPC Static Pressure", "unit": "psia", "description": "Static pressure at HPC outlet"},
    "phi": {"name": "Fuel Flow Ratio", "unit": "pps/psi", "description": "Ratio of fuel flow to Ps30"},
    "NRf": {"name": "Corrected Fan Speed", "unit": "rpm", "description": "Corrected fan speed"},
    "NRc": {"name": "Corrected Core Speed", "unit": "rpm", "description": "Corrected core speed"},
    "BPR": {"name": "Bypass Ratio", "unit": "ratio", "description": "Bypass ratio"},
    "farB": {"name": "Fuel-Air Ratio", "unit": "ratio", "description": "Burner fuel-air ratio"},
    "htBleed": {"name": "Bleed Enthalpy", "unit": "BTU/lb", "description": "Bleed enthalpy"},
    "Nf_dmd": {"name": "Demanded Fan Speed", "unit": "rpm", "description": "Demanded fan speed"},
    "W31": {"name": "HPT Coolant Bleed", "unit": "lbm/s", "description": "HPT coolant bleed flow"},
    "W32": {"name": "LPT Coolant Bleed", "unit": "lbm/s", "description": "LPT coolant bleed flow"},
    "RUL": {"name": "Remaining Life", "unit": "cycles", "description": "Remaining Useful Life until failure"}
}

SAMPLE_DATASETS = {
    "NASA Turbofan - Normal Degradation": {
        "file": "turbofan_normal.json",
        "description": "Gradual HPC degradation over ~190 cycles. Good for training baseline patterns.",
        "channels": ["cycle", "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
                    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
                    "W31", "W32", "RUL"],
        "poll_interval": 2,
        "total_cycles": 191,
        "fault_mode": "HPC Degradation"
    },
    "NASA Turbofan - Rapid Failure": {
        "file": "turbofan_rapid.json",
        "description": "Accelerated failure in ~80 cycles. Tests anomaly detection responsiveness.",
        "channels": ["cycle", "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
                    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
                    "W31", "W32", "RUL"],
        "poll_interval": 2,
        "total_cycles": 82,
        "fault_mode": "HPC Degradation (Accelerated)"
    },
    "NASA Turbofan - Extended Life": {
        "file": "turbofan_extended.json",
        "description": "Long-running engine with ~300 cycles. Extended monitoring scenario.",
        "channels": ["cycle", "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
                    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
                    "W31", "W32", "RUL"],
        "poll_interval": 2,
        "total_cycles": 305,
        "fault_mode": "HPC Degradation (Slow)"
    },
    "NASA Turbofan - Multi-Fault": {
        "file": "turbofan_multifault.json",
        "description": "Combined HPC and fan degradation. Complex failure pattern.",
        "channels": ["cycle", "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
                    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
                    "W31", "W32", "RUL"],
        "poll_interval": 2,
        "total_cycles": 145,
        "fault_mode": "HPC + Fan Degradation"
    }
}

# API endpoint for channel metadata
@app.get("/api/channel-info")
async def get_channel_info():
    """Get detailed information about all available channels."""
    return CHANNEL_INFO

# Track sample data state for streaming
sample_data_state: Dict[str, dict] = {}


@app.get("/api/sample-datasets")
async def list_sample_datasets():
    """List available sample datasets."""
    datasets = []
    for name, info in SAMPLE_DATASETS.items():
        datasets.append({
            "name": name,
            "description": info["description"],
            "channels": info["channels"],
            "poll_interval": info["poll_interval"]
        })
    return {"datasets": datasets}


@app.post("/api/sample-data/test")
async def test_sample_data(request: Request):
    """Test sample data connection and return channels."""
    body = await request.json()
    dataset_name = body.get('dataset')

    if not dataset_name or dataset_name not in SAMPLE_DATASETS:
        return JSONResponse({"success": False, "error": "Unknown dataset"}, status_code=400)

    dataset_info = SAMPLE_DATASETS[dataset_name]
    data_path = os.path.join(BASE_DIR, "data", dataset_info["file"])

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Extract channels from the data
        channels = []
        if data.get('type') == 'FeatureCollection':
            features = data.get('features', [])
            if features:
                props = features[0].get('properties', {})
                for key, value in props.items():
                    if isinstance(value, (int, float)) and key not in ('time',):
                        channels.append(key)

        return {
            "success": True,
            "channels": channels,
            "num_samples": len(data.get('features', [])),
            "description": dataset_info["description"]
        }
    except FileNotFoundError:
        return JSONResponse({"success": False, "error": "Sample data file not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/sources/sample")
async def add_sample_data_source(request: Request):
    """Add a sample data source."""
    body = await request.json()
    dataset_name = body.get('dataset')
    source_name = body.get('name', dataset_name)

    if not dataset_name or dataset_name not in SAMPLE_DATASETS:
        return JSONResponse({"success": False, "error": "Unknown dataset"}, status_code=400)

    dataset_info = SAMPLE_DATASETS[dataset_name]

    # Add to data sources
    data_sources[source_name] = {
        "name": source_name,
        "type": "sample",
        "dataset": dataset_name,
        "poll_interval": dataset_info["poll_interval"],
        "status": "connected",
        "created_at": datetime.now().isoformat()
    }

    # Initialize sample data state for streaming
    data_path = os.path.join(BASE_DIR, "data", dataset_info["file"])
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        sample_data_state[source_name] = {
            "data": data,
            "current_index": 0,
            "features": data.get('features', [])
        }
    except Exception as e:
        print(f"[Sample Data] Error loading {dataset_name}: {e}")

    await broadcast_update("source_added", data_sources[source_name])
    return {"status": "success", "source": data_sources[source_name]}


async def poll_sample_source(source_name: str, source: dict) -> Optional[dict]:
    """Get next data point from sample data source."""
    if source_name not in sample_data_state:
        return None

    state = sample_data_state[source_name]
    features = state.get('features', [])
    if not features:
        return None

    # Get current feature and advance index (cycling through)
    current_index = state['current_index']
    feature = features[current_index % len(features)]
    state['current_index'] = (current_index + 1) % len(features)

    # Extract properties
    props = feature.get('properties', {})
    return props


def start():
    """Launch the application server."""
    print("Starting MachineIQ Web Interface...")
    print(f"Dashboard: http://127.0.0.1:8000")
    print(f"Setup Wizard: http://127.0.0.1:8000/setup")
    uvicorn.run("interface.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
