"""
Anomaly marker overlay for channel plotter

Displays vertical lines at anomaly timestamps with severity-based colors
"""

import pyqtgraph as pg
from PyQt6 import QtCore
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AnomalyMarker:
    """Represents a single anomaly marker"""
    timestamp: float
    score: float  # 0-1, lower = more anomalous
    channels: List[str]  # Which channels contributed
    severity: str = 'warning'  # 'info', 'warning', 'critical'
    metadata: Optional[Dict] = None

    @property
    def color(self) -> str:
        """Get marker color based on severity"""
        colors = {
            'info': (255, 255, 0),  # Yellow
            'warning': (255, 165, 0),  # Orange
            'critical': (255, 0, 0)  # Red
        }
        return colors.get(self.severity, (255, 255, 0))

    @property
    def severity_label(self) -> str:
        """Get human-readable severity label"""
        return {
            'info': '⚠️ Info',
            'warning': '⚠️ Warning',
            'critical': '🚨 Critical'
        }.get(self.severity, '⚠️')

    @classmethod
    def from_detection(cls, timestamp: float, score: float,
                       channels: List[str], threshold: float = 0.3):
        """
        Create marker from detection result

        Args:
            timestamp: Unix timestamp
            score: Match score (0-1)
            channels: Contributing channel names
            threshold: Detection threshold
        """
        # Determine severity based on how far below threshold
        if score < threshold * 0.5:
            severity = 'critical'
        elif score < threshold * 0.75:
            severity = 'warning'
        else:
            severity = 'info'

        return cls(
            timestamp=timestamp,
            score=score,
            channels=channels,
            severity=severity
        )


class AnomalyMarkerOverlay:
    """Manages anomaly markers on a PyQtGraph plot"""

    def __init__(self, plot_widget: pg.PlotWidget):
        self.plot_widget = plot_widget
        self.markers: List[AnomalyMarker] = []
        self.marker_lines: List[pg.InfiniteLine] = []
        self.marker_clicked_callback = None

    def add_marker(self, marker: AnomalyMarker):
        """
        Add an anomaly marker to the plot

        Args:
            marker: AnomalyMarker instance
        """
        self.markers.append(marker)

        # Create vertical line
        line = pg.InfiniteLine(
            pos=marker.timestamp,
            angle=90,
            pen=pg.mkPen(marker.color, width=2, style=QtCore.Qt.PenStyle.DashLine),
            movable=False
        )

        # Add label
        dt = datetime.fromtimestamp(marker.timestamp)
        label_text = f"{marker.severity_label}\n{dt.strftime('%H:%M:%S')}\nScore: {marker.score:.3f}"
        text_item = pg.TextItem(
            text=label_text,
            color=marker.color,
            anchor=(0.5, 1.0),
            fill=(0, 0, 0, 128)  # Semi-transparent black background
        )
        text_item.setPos(marker.timestamp, 0)

        # Make line clickable
        line.setZValue(100)  # Ensure markers are on top

        # Store reference to marker in line's userData
        line.marker = marker

        self.plot_widget.addItem(line)
        self.plot_widget.addItem(text_item)
        self.marker_lines.append(line)

    def add_markers_from_detections(self, timestamps: np.ndarray,
                                    scores: np.ndarray,
                                    contributing_channels: List[List[str]],
                                    threshold: float = 0.3):
        """
        Add multiple markers from detection results

        Args:
            timestamps: Array of Unix timestamps
            scores: Array of match scores
            contributing_channels: List of channel lists for each detection
            threshold: Detection threshold
        """
        for i, (ts, score) in enumerate(zip(timestamps, scores)):
            if score < threshold:  # Only add if anomaly detected
                channels = contributing_channels[i] if i < len(contributing_channels) else []
                marker = AnomalyMarker.from_detection(ts, score, channels, threshold)
                self.add_marker(marker)

    def clear_markers(self):
        """Remove all markers from the plot"""
        for line in self.marker_lines:
            self.plot_widget.removeItem(line)
        self.marker_lines.clear()
        self.markers.clear()

    def get_markers_in_range(self, start_time: float, end_time: float) -> List[AnomalyMarker]:
        """Get all markers within a time range"""
        return [m for m in self.markers if start_time <= m.timestamp <= end_time]

    def get_marker_at_time(self, timestamp: float, tolerance: float = 10.0) -> Optional[AnomalyMarker]:
        """
        Get marker closest to timestamp within tolerance

        Args:
            timestamp: Target timestamp
            tolerance: Maximum time difference in seconds
        """
        closest_marker = None
        min_distance = float('inf')

        for marker in self.markers:
            distance = abs(marker.timestamp - timestamp)
            if distance < min_distance and distance <= tolerance:
                min_distance = distance
                closest_marker = marker

        return closest_marker

    def set_marker_clicked_callback(self, callback):
        """
        Set callback function for when marker is clicked

        Callback signature: callback(marker: AnomalyMarker)
        """
        self.marker_clicked_callback = callback

    def get_anomaly_density(self, window_size: float = 3600) -> np.ndarray:
        """
        Calculate anomaly density over time

        Args:
            window_size: Time window in seconds for density calculation

        Returns:
            Array of (timestamp, density) tuples
        """
        if not self.markers:
            return np.array([])

        timestamps = np.array([m.timestamp for m in self.markers])
        min_time, max_time = timestamps.min(), timestamps.max()

        # Create time bins
        num_bins = int((max_time - min_time) / window_size) + 1
        time_bins = np.linspace(min_time, max_time, num_bins)

        # Count anomalies in each bin
        densities = []
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            count = sum(1 for m in self.markers
                       if bin_start <= m.timestamp < bin_end)
            densities.append((bin_start, count))

        return np.array(densities)


# Demo function
def demo_anomaly_markers():
    """Demonstrate anomaly markers with channel plotter"""
    import sys
    from PyQt6 import QtWidgets
    from datetime import timedelta

    # Import channel plotter
    from channel_plotter import ChannelPlotter

    app = QtWidgets.QApplication(sys.argv)

    # Create plotter with data
    plotter = ChannelPlotter(title="Anomaly Detection Demo")
    plotter.add_channel("Vibration", color='r')
    plotter.add_channel("Temperature", color='b')

    # Generate data with anomalies
    num_points = 500
    base_time = datetime.now()
    timestamps = np.array([(base_time + timedelta(seconds=i)).timestamp()
                          for i in range(num_points)])

    # Normal vibration with spikes
    vibration = 3.0 + 0.3 * np.sin(np.linspace(0, 10*np.pi, num_points))
    vibration[100:105] += 4.0  # Anomaly 1
    vibration[250:255] += 6.0  # Anomaly 2
    vibration[400:410] += 3.5  # Anomaly 3

    # Temperature
    temperature = 65 + 2 * np.sin(np.linspace(0, 5*np.pi, num_points))

    plotter.update_data("Vibration", timestamps, vibration)
    plotter.update_data("Temperature", timestamps, temperature)

    # Add anomaly markers
    overlay = AnomalyMarkerOverlay(plotter.plot_widget)

    # Add markers at anomaly locations
    anomaly_times = [
        (timestamps[102], 0.15, 'critical'),
        (timestamps[252], 0.10, 'critical'),
        (timestamps[405], 0.25, 'warning')
    ]

    for ts, score, severity in anomaly_times:
        marker = AnomalyMarker(
            timestamp=ts,
            score=score,
            channels=["Vibration"],
            severity=severity
        )
        overlay.add_marker(marker)

    # Set up click callback
    def on_marker_info():
        markers = overlay.markers
        info_text = f"Total Anomalies: {len(markers)}\n\n"
        for i, m in enumerate(markers, 1):
            dt = datetime.fromtimestamp(m.timestamp)
            info_text += f"{i}. {m.severity_label} at {dt.strftime('%H:%M:%S')}\n"
            info_text += f"   Score: {m.score:.3f}\n"
            info_text += f"   Channels: {', '.join(m.channels)}\n\n"

        QtWidgets.QMessageBox.information(plotter, "Anomaly Summary", info_text)

    # Add info button
    info_btn = QtWidgets.QPushButton("📊 Anomaly Summary")
    info_btn.clicked.connect(on_marker_info)
    plotter.layout().addWidget(info_btn)

    plotter.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    demo_anomaly_markers()
