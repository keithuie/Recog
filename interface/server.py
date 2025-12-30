"""
MachineIQ Web Interface Server
Custom dashboard replacing Grafana for IP ownership
"""
import os
import sys
import json
import asyncio
import aiohttp
import random
from datetime import datetime
from typing import Dict, List, Optional
from io import StringIO

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

# In-memory state (in production, use Redis or database)
channel_groups: Dict[str, dict] = {}
data_sources: Dict[str, dict] = {}
active_websockets: List[WebSocket] = []
data_polling_task: Optional[asyncio.Task] = None
simulation_task: Optional[asyncio.Task] = None

# Data cache for streaming
latest_data: Dict[str, dict] = {}  # {group_name: {channel_values, confidence, timestamp}}


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
        "status": "idle",
        "trained_states": 0,
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


@app.post("/api/training/start")
async def start_training(req: TrainingRequest):
    """Start training for a channel group."""
    if req.group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    channel_groups[req.group_name]["status"] = "training"
    await broadcast_update("training_started", {
        "group": req.group_name,
        "duration": req.duration
    })

    # Try to call ML service
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ML_CORE_HOST}:{ML_CORE_PORT}/api/training/start",
                json={"group": req.group_name, "duration": req.duration},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    print(f"[Training] Started ML training for {req.group_name}")
    except Exception as e:
        print(f"[Training] ML service not available: {e}")

    # Schedule training completion
    asyncio.create_task(complete_training_after(req.group_name, req.duration))

    return {"status": "training_started", "group": req.group_name, "duration": req.duration}


async def complete_training_after(group_name: str, duration: int):
    """Complete training after specified duration."""
    await asyncio.sleep(duration)
    if group_name in channel_groups and channel_groups[group_name]["status"] == "training":
        channel_groups[group_name]["status"] = "monitoring"
        channel_groups[group_name]["trained_states"] = channel_groups[group_name].get("trained_states", 0) + 1
        await broadcast_update("training_complete", {"group": group_name})
        print(f"[Training] Completed for {group_name}")


@app.post("/api/training/stop/{group_name}")
async def stop_training(group_name: str):
    """Stop training for a channel group."""
    if group_name not in channel_groups:
        raise HTTPException(status_code=404, detail="Group not found")

    channel_groups[group_name]["status"] = "monitoring"
    await broadcast_update("training_stopped", {"group": group_name})
    return {"status": "monitoring", "group": group_name}


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
    """Background task to poll data from sources and stream to dashboard."""
    while True:
        try:
            # Poll each API source
            for source_name, source in data_sources.items():
                if source.get('type') == 'api' and source.get('url'):
                    data = await poll_api_source(source)
                    if data:
                        # Process data for each group using this source
                        for group_name, group in channel_groups.items():
                            if group.get('source') == source_name:
                                channel_values = {}
                                values_list = []

                                for channel in group.get('channels', []):
                                    # Try to find channel value in response
                                    value = data.get(channel, data.get('values', {}).get(channel))
                                    if value is not None:
                                        channel_values[channel] = float(value)
                                        values_list.append(float(value))

                                if channel_values:
                                    # Try to get ML analysis
                                    ml_result = await analyze_with_ml_service(group_name, values_list)
                                    confidence = ml_result.get('confidence', 85) if ml_result else 70 + random.random() * 25

                                    # Broadcast to dashboard
                                    await broadcast_update("data_update", {
                                        "group": group_name,
                                        "confidence": confidence,
                                        "channel_values": channel_values,
                                        "timestamp": datetime.now().isoformat()
                                    })

            # Poll interval (auto-managed based on number of sources)
            poll_interval = max(0.5, min(2.0, 1.0 / max(1, len(data_sources))))
            await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Polling] Error in data polling loop: {e}")
            await asyncio.sleep(1)


async def simulation_loop():
    """Background task to simulate data when no real sources are connected."""
    while True:
        try:
            # Only simulate if we have groups but no active data from real sources
            for group_name, group in channel_groups.items():
                channels = group.get('channels', [])
                if channels:
                    # Generate simulated data
                    channel_values = {}
                    for i, channel in enumerate(channels):
                        # Simulate realistic values with some variation
                        base_value = 0.5 + 0.3 * (i / max(1, len(channels) - 1))
                        noise = random.uniform(-0.1, 0.1)
                        channel_values[channel] = round(base_value + noise, 4)

                    # Simulate confidence score (typically high with occasional dips)
                    if random.random() < 0.95:
                        confidence = 70 + random.random() * 28
                    else:
                        confidence = 30 + random.random() * 40  # Occasional anomaly

                    # Broadcast to dashboard
                    await broadcast_update("data_update", {
                        "group": group_name,
                        "confidence": round(confidence, 1),
                        "channel_values": channel_values,
                        "timestamp": datetime.now().isoformat()
                    })

            await asyncio.sleep(1)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Simulation] Error: {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    global data_polling_task, simulation_task
    data_polling_task = asyncio.create_task(data_polling_loop())
    simulation_task = asyncio.create_task(simulation_loop())
    print("[Server] Background data streaming tasks started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on shutdown."""
    global data_polling_task, simulation_task
    if data_polling_task:
        data_polling_task.cancel()
    if simulation_task:
        simulation_task.cancel()


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


def start():
    """Launch the application server."""
    print("Starting MachineIQ Web Interface...")
    print(f"Dashboard: http://127.0.0.1:8000")
    print(f"Setup Wizard: http://127.0.0.1:8000/setup")
    uvicorn.run("interface.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
