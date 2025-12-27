
import sys
import os
import time
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.desktop.core.data_player import APIDataSource

def test_api_source():
    # Use the same URL as the app
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=10"
    source = APIDataSource("Test API", url, poll_interval=1.0, use_wall_clock=True)
    
    print(f"Connecting to {url}...")
    if not source.connect():
        print("Failed to connect")
        return

    print("Connected. Listening for data...")
    
    last_ts = None
    count = 0
    
    def on_data(point):
        nonlocal last_ts, count
        count += 1
        print(f"Update {count}: Time={point.timestamp} Unix={point.unix_timestamp}")
        
        if last_ts and point.unix_timestamp == last_ts:
            print("  WARNING: Timestamp is identical to previous point!")
        
        last_ts = point.unix_timestamp

    source.set_on_data_callback(on_data)
    
    # Run for 5 seconds
    time.sleep(5)
    source.disconnect()
    print("Test finished")

if __name__ == "__main__":
    test_api_source()
