"""
Data Player for Time Series Playback and Real-Time Streaming

Supports:
- CSV file import with playback controls
- API connections for real-time data
- Play, pause, speed control
- Timestamp-based navigation
"""

import csv
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from queue import Queue

import numpy as np


class PlaybackState(Enum):
    """Playback state enumeration"""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


@dataclass
class DataPoint:
    """Single data point with timestamp and channel values"""
    timestamp: datetime
    values: Dict[str, float]

    @property
    def unix_timestamp(self) -> float:
        return self.timestamp.timestamp()


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    source_type: str  # 'csv', 'api', 'mqtt', 'opcua'
    name: str
    config: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for data sources"""

    def __init__(self, name: str):
        self.name = name
        self._channels: List[str] = []
        self._on_data_callback: Optional[Callable[[DataPoint], None]] = None

    @property
    def channels(self) -> List[str]:
        return self._channels.copy()

    def set_on_data_callback(self, callback: Callable[[DataPoint], None]):
        """Set callback for when new data arrives"""
        self._on_data_callback = callback

    def _emit_data(self, point: DataPoint):
        """Emit data point to callback"""
        if self._on_data_callback:
            self._on_data_callback(point)

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the data source"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the data source"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected"""
        pass


class CSVDataSource(DataSource):
    """
    CSV file data source with playback capabilities.

    Supports:
    - Loading CSV files with timestamp column
    - Play/pause/stop controls
    - Speed adjustment
    - Seeking to specific timestamps
    """

    def __init__(self, name: str, filepath: str = None):
        super().__init__(name)
        self.filepath = filepath
        self._data: List[DataPoint] = []
        self._current_index: int = 0
        self._state = PlaybackState.STOPPED
        self._playback_speed: float = 1.0
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._timestamp_column: str = 'timestamp'
        self._connected: bool = False

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def total_points(self) -> int:
        return len(self._data)

    @property
    def progress(self) -> float:
        """Get playback progress as 0-1"""
        if not self._data:
            return 0.0
        return self._current_index / len(self._data)

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Get current playback timestamp"""
        if self._data and 0 <= self._current_index < len(self._data):
            return self._data[self._current_index].timestamp
        return None

    @property
    def start_timestamp(self) -> Optional[datetime]:
        """Get first timestamp in data"""
        return self._data[0].timestamp if self._data else None

    @property
    def end_timestamp(self) -> Optional[datetime]:
        """Get last timestamp in data"""
        return self._data[-1].timestamp if self._data else None

    def set_timestamp_column(self, column_name: str):
        """Set the name of the timestamp column in CSV"""
        self._timestamp_column = column_name

    def connect(self) -> bool:
        """Load the CSV file"""
        if not self.filepath or not Path(self.filepath).exists():
            return False

        try:
            self._load_csv()
            self._connected = True
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def disconnect(self):
        """Stop playback and clear data"""
        self.stop()
        self._data.clear()
        self._channels.clear()
        self._current_index = 0
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _load_csv(self):
        """Load and parse the CSV file"""
        self._data.clear()

        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)

            # Get channel names (all columns except timestamp)
            self._channels = [
                col for col in reader.fieldnames
                if col.lower() != self._timestamp_column.lower()
            ]

            for row in reader:
                # Parse timestamp
                ts_str = row.get(self._timestamp_column) or row.get('time') or row.get('Time')
                if ts_str:
                    try:
                        # Try various timestamp formats
                        for fmt in [
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%d %H:%M:%S.%f',
                            '%Y/%m/%d %H:%M:%S',
                            '%d/%m/%Y %H:%M:%S',
                            '%m/%d/%Y %H:%M:%S',
                        ]:
                            try:
                                timestamp = datetime.strptime(ts_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            # Try Unix timestamp
                            timestamp = datetime.fromtimestamp(float(ts_str))
                    except (ValueError, TypeError):
                        continue

                    # Parse channel values
                    values = {}
                    for channel in self._channels:
                        try:
                            values[channel] = float(row[channel])
                        except (ValueError, KeyError, TypeError):
                            values[channel] = 0.0

                    self._data.append(DataPoint(timestamp=timestamp, values=values))

        # Sort by timestamp
        self._data.sort(key=lambda p: p.timestamp)

    def play(self):
        """Start or resume playback"""
        if not self._data:
            return

        if self._state == PlaybackState.PLAYING:
            return

        self._state = PlaybackState.PLAYING
        self._stop_event.clear()

        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def pause(self):
        """Pause playback"""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED
            self._stop_event.set()
            if self._playback_thread:
                self._playback_thread.join(timeout=1.0)

    def stop(self):
        """Stop playback and reset to beginning"""
        self._state = PlaybackState.STOPPED
        self._stop_event.set()
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
        self._current_index = 0

    def seek(self, index: int):
        """Seek to specific index"""
        self._current_index = max(0, min(index, len(self._data) - 1))

    def seek_timestamp(self, timestamp: datetime):
        """Seek to specific timestamp"""
        for i, point in enumerate(self._data):
            if point.timestamp >= timestamp:
                self._current_index = i
                return
        self._current_index = len(self._data) - 1

    def set_playback_speed(self, speed: float):
        """Set playback speed (1.0 = real-time, 2.0 = 2x speed)"""
        self._playback_speed = max(0.1, min(100.0, speed))

    def _playback_loop(self):
        """Background thread for playback"""
        while not self._stop_event.is_set() and self._current_index < len(self._data):
            point = self._data[self._current_index]
            self._emit_data(point)

            self._current_index += 1

            if self._current_index < len(self._data):
                # Calculate delay based on timestamp difference
                current_ts = point.timestamp
                next_ts = self._data[self._current_index].timestamp
                delay = (next_ts - current_ts).total_seconds() / self._playback_speed
                delay = max(0.001, min(delay, 10.0))  # Clamp delay

                self._stop_event.wait(delay)

        if self._current_index >= len(self._data):
            self._state = PlaybackState.STOPPED

    def get_data_range(self) -> Dict[str, tuple]:
        """Get min/max range for each channel"""
        if not self._data:
            return {}

        ranges = {}
        for channel in self._channels:
            values = [p.values.get(channel, 0) for p in self._data]
            ranges[channel] = (min(values), max(values))

        return ranges

    def get_all_data(self) -> List[DataPoint]:
        """Get all loaded data points"""
        return self._data.copy()


class APIDataSource(DataSource):
    """
    REST API data source for real-time streaming.

    Polls an API endpoint at regular intervals.
    Supports flat JSON, GeoJSON, and nested structures.
    """

    def __init__(self, name: str, url: str = None, poll_interval: float = 5.0, use_wall_clock: bool = False):
        super().__init__(name)
        self.url = url
        self.poll_interval = poll_interval
        self.use_wall_clock = use_wall_clock
        self.headers: Dict[str, str] = {'User-Agent': 'MachineIQ/1.0'}
        self.auth_token: Optional[str] = None
        self._connected = False
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._data_path: Optional[str] = None  # JSON path to data array
        self._channel_map: Dict[str, str] = {}  # channel -> json path
        self._on_status_callback: Optional[Callable[[str], None]] = None

    def set_auth(self, token: str):
        """Set authentication token"""
        self.auth_token = token
        self.headers['Authorization'] = f'Bearer {token}'

    def set_header(self, key: str, value: str):
        """Set custom header"""
        self.headers[key] = value

    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates"""
        self._on_status_callback = callback

    def _emit_status(self, status: str):
        """Emit status to callback"""
        if self._on_status_callback:
            self._on_status_callback(status)

    def _fetch_data(self) -> Optional[Any]:
        """Fetch data from API"""
        import urllib.request
        req = urllib.request.Request(self.url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode())

    def _extract_channels_from_response(self, data: Any) -> List[str]:
        """Extract channel names from API response"""
        channels = []

        # Handle GeoJSON (USGS earthquake, etc.)
        if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
            features = data.get('features', [])
            if features and isinstance(features[0], dict):
                props = features[0].get('properties', {})
                # Extract numeric properties as channels
                for key, value in props.items():
                    if isinstance(value, (int, float)) and key not in ('time', 'updated', 'tz'):
                        channels.append(key)
                self._data_path = 'features'
                self._channel_map = {ch: f'properties.{ch}' for ch in channels}
            return channels

        # Handle array of objects
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (int, float)) and key not in ('timestamp', 'time'):
                        channels.append(key)
            return channels

        # Handle flat dict
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)) and key not in ('timestamp', 'time'):
                    channels.append(key)
            return channels

        return channels

    def _parse_data_point(self, data: Any) -> Optional[DataPoint]:
        """Parse a data point from API response"""
        timestamp = datetime.now()
        values = {}

        # Handle GeoJSON
        if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
            features = data.get('features', [])
            if not features:
                return None

            # Get most recent feature (first in list for USGS)
            feature = features[0]
            props = feature.get('properties', {})

            # Try to get timestamp
            if not self.use_wall_clock and 'time' in props:
                try:
                    # USGS uses milliseconds since epoch
                    ts_val = props['time']
                    if isinstance(ts_val, (int, float)) and ts_val > 1e10:
                        timestamp = datetime.fromtimestamp(ts_val / 1000)
                    elif isinstance(ts_val, (int, float)):
                        timestamp = datetime.fromtimestamp(ts_val)
                except (ValueError, OSError):
                    pass

            # Extract channel values
            for channel in self._channels:
                path = self._channel_map.get(channel, channel)
                try:
                    if '.' in path:
                        parts = path.split('.')
                        val = props
                        for part in parts[1:]:  # Skip 'properties'
                            val = val.get(part, 0)
                    else:
                        val = props.get(channel, 0)
                    values[channel] = float(val) if val is not None else 0.0
                except (ValueError, TypeError, AttributeError):
                    values[channel] = 0.0

            return DataPoint(timestamp=timestamp, values=values)

        # Handle flat dict or array
        if isinstance(data, dict):
            if not self.use_wall_clock and 'timestamp' in data:
                try:
                    timestamp = datetime.fromisoformat(str(data['timestamp']))
                except (ValueError, TypeError):
                    pass

            for channel in self._channels:
                try:
                    values[channel] = float(data.get(channel, 0))
                except (ValueError, TypeError):
                    values[channel] = 0.0

            return DataPoint(timestamp=timestamp, values=values)

        return None

    def test_connection(self) -> tuple:
        """
        Test connection and discover channels.

        Returns:
            (success: bool, channels: List[str], message: str)
        """
        if not self.url:
            return False, [], "No URL configured"

        try:
            data = self._fetch_data()
            channels = self._extract_channels_from_response(data)
            self._channels = channels

            if channels:
                return True, channels, f"Found {len(channels)} channels"
            else:
                return True, [], "Connected but no numeric channels found"

        except Exception as e:
            return False, [], f"Connection failed: {str(e)}"

    def connect(self) -> bool:
        """Start polling the API"""
        if not self.url:
            return False

        try:
            self._emit_status("Connecting...")

            # Test connection and discover channels
            success, channels, message = self.test_connection()
            if not success:
                self._emit_status(f"Failed: {message}")
                return False

            self._channels = channels
            self._connected = True
            self._stop_event.clear()

            self._emit_status(f"Connected: {len(channels)} channels")

            self._polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._polling_thread.start()
            return True

        except Exception as e:
            self._emit_status(f"Error: {e}")
            print(f"API connection error: {e}")
            return False

    def disconnect(self):
        """Stop polling"""
        self._stop_event.set()
        if self._polling_thread:
            self._polling_thread.join(timeout=2.0)
        self._connected = False
        self._emit_status("Disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def _poll_loop(self):
        """Background polling loop"""
        poll_count = 0

        while not self._stop_event.is_set():
            try:
                data = self._fetch_data()
                point = self._parse_data_point(data)

                if point:
                    self._emit_data(point)
                    poll_count += 1
                    self._emit_status(f"Monitoring: {poll_count} updates")

            except Exception as e:
                self._emit_status(f"Poll error: {str(e)[:50]}")
                print(f"API poll error: {e}")

            self._stop_event.wait(self.poll_interval)


class DataPlayer:
    """
    High-level data player managing multiple sources.

    Coordinates:
    - Multiple data sources (CSV, API, etc.)
    - Unified playback control
    - Data buffering and history
    """

    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
        self._active_source: Optional[str] = None
        self._data_buffer: List[DataPoint] = []
        self._buffer_size: int = 10000
        self._on_data_callbacks: List[Callable[[DataPoint], None]] = []

    @property
    def sources(self) -> List[str]:
        """Get list of source names"""
        return list(self._sources.keys())

    @property
    def active_source(self) -> Optional[DataSource]:
        """Get the active data source"""
        if self._active_source:
            return self._sources.get(self._active_source)
        return None

    @property
    def channels(self) -> List[str]:
        """Get channels from active source"""
        source = self.active_source
        return source.channels if source else []

    @property
    def buffer(self) -> List[DataPoint]:
        """Get buffered data points"""
        return self._data_buffer.copy()

    def add_source(self, source: DataSource):
        """Add a data source"""
        source.set_on_data_callback(self._on_source_data)
        self._sources[source.name] = source

    def remove_source(self, name: str):
        """Remove a data source"""
        if name in self._sources:
            self._sources[name].disconnect()
            del self._sources[name]
            if self._active_source == name:
                self._active_source = None

    def set_active_source(self, name: str) -> bool:
        """Set the active data source"""
        if name not in self._sources:
            return False

        # Disconnect current source
        if self._active_source and self._active_source != name:
            self._sources[self._active_source].disconnect()

        self._active_source = name
        return True

    def connect(self) -> bool:
        """Connect the active source"""
        source = self.active_source
        if source:
            return source.connect()
        return False

    def disconnect(self):
        """Disconnect the active source"""
        source = self.active_source
        if source:
            source.disconnect()

    def add_data_callback(self, callback: Callable[[DataPoint], None]):
        """Add callback for data events"""
        self._on_data_callbacks.append(callback)

    def remove_data_callback(self, callback: Callable[[DataPoint], None]):
        """Remove data callback"""
        if callback in self._on_data_callbacks:
            self._on_data_callbacks.remove(callback)

    def _on_source_data(self, point: DataPoint):
        """Handle data from source"""
        # Add to buffer
        self._data_buffer.append(point)

        # Trim buffer if needed
        if len(self._data_buffer) > self._buffer_size:
            self._data_buffer = self._data_buffer[-self._buffer_size:]

        # Notify callbacks
        for callback in self._on_data_callbacks:
            try:
                callback(point)
            except Exception as e:
                print(f"Data callback error: {e}")

    def clear_buffer(self):
        """Clear the data buffer"""
        self._data_buffer.clear()

    def get_channel_data(self, channel: str, limit: int = None) -> tuple:
        """
        Get timestamps and values for a specific channel.

        Returns:
            (timestamps, values) as numpy arrays
        """
        data = self._data_buffer if limit is None else self._data_buffer[-limit:]

        timestamps = np.array([p.unix_timestamp for p in data])
        values = np.array([p.values.get(channel, 0.0) for p in data])

        return timestamps, values
