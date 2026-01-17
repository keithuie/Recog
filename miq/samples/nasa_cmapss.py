"""
NASA CMAPSS Turbofan Engine Degradation Dataset Loader

This module downloads and manages the NASA Commercial Modular Aero-Propulsion
System Simulation (CMAPSS) dataset for turbofan engine degradation simulation.

The dataset contains run-to-failure trajectories of simulated turbofan engines
with 21 sensor measurements over time until failure.

Reference: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
Data source: https://github.com/biswajitsahoo1111/rul_codes_open
"""

import os
import sqlite3
import urllib.request
import io
from typing import Dict, List, Optional, Tuple, Generator
from datetime import datetime, timedelta
import json

# NASA CMAPSS Data URL (reliable GitHub mirror)
NASA_DATA_URL = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt"

# Column definitions for the raw data (no headers in file)
INDEX_COLS = ['unit_nr', 'time_cycles']
SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
SENSOR_COLS = [f's_{i}' for i in range(1, 22)]  # 21 sensors
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensor mapping - what each sensor actually measures
SENSOR_DICTIONARY = {
    's_1': {'name': 'Fan Inlet Temp (T2)', 'unit': 'R', 'type': 'temperature'},
    's_2': {'name': 'LPC Outlet Temp (T24)', 'unit': 'R', 'type': 'temperature'},
    's_3': {'name': 'HPC Outlet Temp (T30)', 'unit': 'R', 'type': 'temperature'},
    's_4': {'name': 'LPT Outlet Temp (T50)', 'unit': 'R', 'type': 'temperature'},
    's_5': {'name': 'Fan Inlet Pressure (P2)', 'unit': 'psia', 'type': 'pressure'},
    's_6': {'name': 'Bypass Duct Pressure (P15)', 'unit': 'psia', 'type': 'pressure'},
    's_7': {'name': 'HPC Outlet Pressure (P30)', 'unit': 'psia', 'type': 'pressure'},
    's_8': {'name': 'Physical Fan Speed (Nf)', 'unit': 'rpm', 'type': 'speed'},
    's_9': {'name': 'Physical Core Speed (Nc)', 'unit': 'rpm', 'type': 'speed'},
    's_10': {'name': 'Engine Pressure Ratio (epr)', 'unit': '', 'type': 'ratio'},
    's_11': {'name': 'Static Pressure (Ps30)', 'unit': 'psia', 'type': 'pressure'},
    's_12': {'name': 'Fuel Flow Ratio (phi)', 'unit': 'pps/psi', 'type': 'flow'},
    's_13': {'name': 'Corrected Fan Speed (NRf)', 'unit': 'rpm', 'type': 'speed'},
    's_14': {'name': 'Corrected Core Speed (NRc)', 'unit': 'rpm', 'type': 'speed'},
    's_15': {'name': 'Bypass Ratio (BPR)', 'unit': '', 'type': 'ratio'},
    's_16': {'name': 'Burner Fuel-Air Ratio (farB)', 'unit': '', 'type': 'ratio'},
    's_17': {'name': 'Bleed Enthalpy (htBleed)', 'unit': '', 'type': 'energy'},
    's_18': {'name': 'Demanded Fan Speed (Nf_dmd)', 'unit': 'rpm', 'type': 'speed'},
    's_19': {'name': 'Demanded Corrected Fan Speed (PCNfR_dmd)', 'unit': 'rpm', 'type': 'speed'},
    's_20': {'name': 'HPT Coolant Bleed (W31)', 'unit': 'lbm/s', 'type': 'flow'},
    's_21': {'name': 'LPT Coolant Bleed (W32)', 'unit': 'lbm/s', 'type': 'flow'},
}

# Sensors with most diagnostic value (good variance, not constant)
KEY_SENSORS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# Default recommended sensors for anomaly detection
DEFAULT_SENSORS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12']


class NASACMAPSSLoader:
    """
    Loader for NASA CMAPSS Turbofan Engine Degradation Dataset.

    Downloads the dataset, stores it in SQLite for efficient querying,
    and provides streaming playback for real-time simulation.
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize the loader.

        Args:
            data_dir: Directory to store downloaded data. Defaults to ~/.machineiq/data
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser('~'), '.machineiq', 'data')

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = os.path.join(data_dir, 'nasa_cmapss.db')
        self._initialized = False

    def is_loaded(self) -> bool:
        """Check if the dataset has been downloaded and loaded."""
        if not os.path.exists(self.db_path):
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM sensor_data")
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except:
            return False

    def download_and_load(self, progress_callback=None) -> Dict:
        """
        Download the NASA CMAPSS dataset and load into SQLite.

        Args:
            progress_callback: Optional callback(percent, message) for progress updates

        Returns:
            Dictionary with loading statistics
        """
        if progress_callback:
            progress_callback(0, "Downloading NASA CMAPSS dataset...")

        # Download the data
        try:
            with urllib.request.urlopen(NASA_DATA_URL, timeout=30) as response:
                raw_data = response.read().decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to download NASA dataset: {e}")

        if progress_callback:
            progress_callback(20, "Parsing data...")

        # Parse the data
        lines = raw_data.strip().split('\n')
        records = []

        for i, line in enumerate(lines):
            values = line.split()
            if len(values) >= len(ALL_COLS):
                record = {col: float(values[j]) for j, col in enumerate(ALL_COLS)}
                records.append(record)

            if progress_callback and i % 1000 == 0:
                progress_callback(20 + int(30 * i / len(lines)), f"Parsing row {i}/{len(lines)}...")

        if progress_callback:
            progress_callback(50, "Creating database...")

        # Create SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS sensor_data")
        cursor.execute("DROP TABLE IF EXISTS engine_info")
        cursor.execute("DROP TABLE IF EXISTS metadata")

        # Create sensor data table
        cols_def = ', '.join([f"{col} REAL" for col in ALL_COLS])
        cursor.execute(f"""
            CREATE TABLE sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols_def},
                rul INTEGER
            )
        """)

        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX idx_unit ON sensor_data(unit_nr)")
        cursor.execute("CREATE INDEX idx_time ON sensor_data(time_cycles)")
        cursor.execute("CREATE INDEX idx_unit_time ON sensor_data(unit_nr, time_cycles)")

        if progress_callback:
            progress_callback(60, "Inserting data...")

        # Calculate RUL (Remaining Useful Life) for each record
        # Group by unit to find max cycle (failure point)
        engine_max_cycles = {}
        for record in records:
            unit = int(record['unit_nr'])
            cycle = int(record['time_cycles'])
            engine_max_cycles[unit] = max(engine_max_cycles.get(unit, 0), cycle)

        # Insert data with RUL
        cols = ALL_COLS + ['rul']
        placeholders = ', '.join(['?' for _ in cols])

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            rows = []
            for record in batch:
                unit = int(record['unit_nr'])
                cycle = int(record['time_cycles'])
                rul = engine_max_cycles[unit] - cycle
                row = [record[col] for col in ALL_COLS] + [rul]
                rows.append(row)

            cursor.executemany(f"INSERT INTO sensor_data ({', '.join(cols)}) VALUES ({placeholders})", rows)

            if progress_callback:
                pct = 60 + int(30 * (i + len(batch)) / len(records))
                progress_callback(pct, f"Inserted {i + len(batch)}/{len(records)} rows...")

        # Create engine info table
        cursor.execute("""
            CREATE TABLE engine_info (
                unit_nr INTEGER PRIMARY KEY,
                total_cycles INTEGER,
                failure_point INTEGER
            )
        """)

        for unit, max_cycle in engine_max_cycles.items():
            cursor.execute(
                "INSERT INTO engine_info (unit_nr, total_cycles, failure_point) VALUES (?, ?, ?)",
                (unit, max_cycle, max_cycle)
            )

        # Create metadata table
        cursor.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        metadata = {
            'source': NASA_DATA_URL,
            'downloaded_at': datetime.now().isoformat(),
            'total_records': len(records),
            'total_engines': len(engine_max_cycles),
            'sensor_dictionary': json.dumps(SENSOR_DICTIONARY),
            'key_sensors': json.dumps(KEY_SENSORS),
            'default_sensors': json.dumps(DEFAULT_SENSORS)
        }

        for key, value in metadata.items():
            cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (key, str(value)))

        conn.commit()
        conn.close()

        if progress_callback:
            progress_callback(100, "Complete!")

        self._initialized = True

        return {
            'total_records': len(records),
            'total_engines': len(engine_max_cycles),
            'engines': list(engine_max_cycles.keys()),
            'sensor_count': len(SENSOR_COLS),
            'db_path': self.db_path
        }

    def get_engine_list(self) -> List[Dict]:
        """Get list of all engines with their info."""
        if not self.is_loaded():
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT unit_nr, total_cycles, failure_point
            FROM engine_info
            ORDER BY unit_nr
        """)

        engines = []
        for row in cursor:
            engines.append({
                'unit_nr': int(row[0]),
                'total_cycles': int(row[1]),
                'failure_point': int(row[2])
            })

        conn.close()
        return engines

    def get_engine_data(self, unit_nr: int, sensors: List[str] = None) -> List[Dict]:
        """
        Get all data for a specific engine.

        Args:
            unit_nr: Engine unit number
            sensors: List of sensor columns to include (default: all)

        Returns:
            List of dictionaries with sensor readings per cycle
        """
        if not self.is_loaded():
            return []

        if sensors is None:
            sensors = SENSOR_COLS

        # Ensure sensors are valid
        sensors = [s for s in sensors if s in SENSOR_COLS]

        conn = sqlite3.connect(self.db_path)
        cols = ['time_cycles', 'rul'] + sensors
        cursor = conn.execute(f"""
            SELECT {', '.join(cols)}
            FROM sensor_data
            WHERE unit_nr = ?
            ORDER BY time_cycles
        """, (unit_nr,))

        data = []
        for row in cursor:
            record = {col: row[i] for i, col in enumerate(cols)}
            data.append(record)

        conn.close()
        return data

    def stream_engine_data(
        self,
        unit_nr: int,
        sensors: List[str] = None,
        speed_multiplier: float = 1.0,
        cycle_interval_ms: int = 1000
    ) -> Generator[Dict, None, None]:
        """
        Generator that yields engine data as a stream.

        This simulates real-time sensor data by yielding one cycle at a time
        with configurable timing.

        Args:
            unit_nr: Engine unit number
            sensors: List of sensor columns to stream
            speed_multiplier: Speed up factor (2.0 = twice as fast)
            cycle_interval_ms: Base interval between cycles in milliseconds

        Yields:
            Dictionary with sensor values, cycle number, and RUL
        """
        data = self.get_engine_data(unit_nr, sensors)

        for i, record in enumerate(data):
            yield {
                'cycle': int(record['time_cycles']),
                'rul': int(record['rul']),
                'values': {k: v for k, v in record.items() if k in (sensors or SENSOR_COLS)},
                'progress': (i + 1) / len(data),
                'is_final': i == len(data) - 1
            }

    def get_sensor_info(self) -> Dict:
        """Get sensor dictionary with descriptions."""
        return {
            'sensors': SENSOR_DICTIONARY,
            'key_sensors': KEY_SENSORS,
            'default_sensors': DEFAULT_SENSORS,
            'all_sensors': SENSOR_COLS
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.is_loaded():
            return {}

        conn = sqlite3.connect(self.db_path)

        # Get basic counts
        cursor = conn.execute("SELECT COUNT(*) FROM sensor_data")
        total_records = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM engine_info")
        total_engines = cursor.fetchone()[0]

        # Get cycle range
        cursor = conn.execute("SELECT MIN(total_cycles), MAX(total_cycles), AVG(total_cycles) FROM engine_info")
        min_cycles, max_cycles, avg_cycles = cursor.fetchone()

        # Get sensor value ranges for key sensors
        sensor_ranges = {}
        for sensor in KEY_SENSORS:
            cursor = conn.execute(f"SELECT MIN({sensor}), MAX({sensor}), AVG({sensor}) FROM sensor_data")
            min_val, max_val, avg_val = cursor.fetchone()
            sensor_ranges[sensor] = {
                'min': round(min_val, 4) if min_val else None,
                'max': round(max_val, 4) if max_val else None,
                'avg': round(avg_val, 4) if avg_val else None,
                'name': SENSOR_DICTIONARY[sensor]['name']
            }

        conn.close()

        return {
            'total_records': total_records,
            'total_engines': total_engines,
            'cycles': {
                'min': min_cycles,
                'max': max_cycles,
                'avg': round(avg_cycles, 1) if avg_cycles else None
            },
            'sensor_ranges': sensor_ranges
        }


def get_nasa_sample_config() -> Dict:
    """
    Get pre-configured setup for NASA sample data.

    Returns a configuration that can be used to quickly set up
    the NASA CMAPSS dataset in MachineIQ.
    """
    return {
        'source': {
            'name': 'NASA Turbofan Engine',
            'source_type': 'sample',
            'sample_type': 'nasa_cmapss',
            'description': 'NASA CMAPSS Turbofan Engine Degradation Simulation'
        },
        'group': {
            'name': 'Engine Sensors',
            'channels': DEFAULT_SENSORS,
            'color': '#5794f2',
            'sample_rate': 1.0,  # 1 Hz (1 cycle per second simulation)
            'preprocessing': 'basic'
        },
        'channel_info': {
            sensor: {
                'name': info['name'],
                'unit': info['unit'],
                'type': info['type'],
                'preprocessing': 'basic'  # Per-channel preprocessing
            }
            for sensor, info in SENSOR_DICTIONARY.items()
            if sensor in DEFAULT_SENSORS
        },
        'recommended_preprocessing': {
            # Temperature sensors - basic is fine
            's_2': 'basic',
            's_3': 'basic',
            's_4': 'basic',
            # Pressure sensors - basic
            's_7': 'basic',
            's_11': 'basic',
            # Speed sensors - may benefit from frequency analysis
            's_8': 'frequency',
            's_9': 'frequency',
            # Flow ratio - basic
            's_12': 'basic'
        },
        'engines': {
            'recommended': [1, 5, 10],  # Good variety of failure timelines
            'description': 'Engine 1: 192 cycles, Engine 5: 128 cycles, Engine 10: 156 cycles'
        }
    }


# Convenience function for quick testing
def test_loader():
    """Test the NASA CMAPSS loader."""
    loader = NASACMAPSSLoader()

    if not loader.is_loaded():
        print("Downloading NASA CMAPSS dataset...")
        result = loader.download_and_load(
            progress_callback=lambda p, m: print(f"[{p:3d}%] {m}")
        )
        print(f"\nLoaded {result['total_records']} records from {result['total_engines']} engines")
    else:
        print("Dataset already loaded.")

    stats = loader.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Total engines: {stats['total_engines']}")
    print(f"  Cycle range: {stats['cycles']['min']} - {stats['cycles']['max']}")

    print(f"\nSensor Ranges (key sensors):")
    for sensor, info in list(stats['sensor_ranges'].items())[:5]:
        print(f"  {sensor} ({info['name']}): {info['min']:.2f} - {info['max']:.2f}")

    # Test streaming
    print(f"\nStreaming first 5 cycles from Engine 1:")
    for i, data in enumerate(loader.stream_engine_data(1, DEFAULT_SENSORS[:3])):
        if i >= 5:
            break
        print(f"  Cycle {data['cycle']}, RUL: {data['rul']}, Values: {data['values']}")


if __name__ == "__main__":
    test_loader()
