#!/usr/bin/env python3
"""
MachineIQ Desktop Application - Complete Integration Demo

This demonstrates the full visualization stack:
1. Multi-channel trend plotting
2. Anomaly marker overlays
3. Click-to-drill-down functionality
4. FFT/Waveform/Envelope/NeuraPeak analysis

Run this to see the complete system in action!
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from PyQt6 import QtWidgets, QtCore

# Import our visualization components
from visualization.channel_plotter import ChannelPlotter
from visualization.anomaly_markers import AnomalyMarker, AnomalyMarkerOverlay
from visualization.drill_down_viewer import DrillDownViewer


class MachineIQDesktopApp(QtWidgets.QMainWindow):
    """
    Main desktop application window

    Integrates:
    - Channel trending
    - Anomaly detection
    - Drill-down analysis
    """

    def __init__(self):
        super().__init__()

        # Storage for raw waveform data (for drill-down)
        self.waveform_database = {}

        self._init_ui()
        self._generate_demo_data()

    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('MachineIQ - Industrial Condition Monitoring')
        self.setGeometry(100, 100, 1400, 800)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout()

        # Header
        header = QtWidgets.QLabel('<h1>🏭 MachineIQ Desktop</h1>')
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Status bar
        self.status_label = QtWidgets.QLabel('Status: Initializing...')
        layout.addWidget(self.status_label)

        # Channel plotter
        self.plotter = ChannelPlotter(title="Motor DE Bearing - Real-Time Monitoring")
        layout.addWidget(self.plotter)

        # Connect plot click to drill-down
        self.plotter.plot_clicked.connect(self._on_plot_clicked)

        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()

        refresh_btn = QtWidgets.QPushButton('🔄 Refresh Data')
        refresh_btn.clicked.connect(self._generate_demo_data)
        button_layout.addWidget(refresh_btn)

        clear_btn = QtWidgets.QPushButton('🗑️ Clear Anomalies')
        clear_btn.clicked.connect(self._clear_anomalies)
        button_layout.addWidget(clear_btn)

        button_layout.addStretch()

        help_btn = QtWidgets.QPushButton('❓ Help')
        help_btn.clicked.connect(self._show_help)
        button_layout.addWidget(help_btn)

        layout.addLayout(button_layout)

        central_widget.setLayout(layout)

        # Initialize anomaly overlay
        self.anomaly_overlay = None

        # Drill-down viewer (created on demand)
        self.drill_down_viewer = None

    def _generate_demo_data(self):
        """Generate demo data with synthetic bearing faults"""
        self.status_label.setText('Status: Generating data...')
        QtWidgets.QApplication.processEvents()

        # Clear existing channels
        for channel_name in list(self.plotter.channels.keys()):
            del self.plotter.channels[channel_name]
            del self.plotter.plot_curves[channel_name]
        self.plotter.plot_widget.clear()

        # Add channels
        self.plotter.add_channel("DE Radial Accel", color='r', line_width=2)
        self.plotter.add_channel("DE Axial Accel", color='b', line_width=2)
        self.plotter.add_channel("DE Temperature", color='g', line_width=2)
        self.plotter.add_channel("DE Ultrasound", color='y', line_width=2)

        # Generate time series
        num_points = 1000
        base_time = datetime.now() - timedelta(hours=1)
        timestamps = np.array([(base_time + timedelta(seconds=i*3.6)).timestamp()
                              for i in range(num_points)])

        # Generate vibration data with anomalies
        t = np.linspace(0, num_points, num_points)

        # Normal vibration + fault development
        radial = 3.0 + 0.5 * np.sin(t * 0.1)
        radial += 0.2 * np.random.randn(num_points)

        # Inject bearing fault progression
        fault_locations = [400, 500, 600, 700, 750]
        for loc in fault_locations:
            if loc < num_points:
                radial[loc-5:loc+5] += 4.0 * (1 + (loc - 400) / 350)  # Increasing severity

        # Correlated axial vibration
        axial = 2.5 + 0.3 * np.cos(t * 0.08)
        axial += 0.15 * np.random.randn(num_points)
        for loc in fault_locations:
            if loc < num_points:
                axial[loc-3:loc+3] += 2.5 * (1 + (loc - 400) / 350)

        # Temperature rises with fault
        temperature = 65 + 2 * np.sin(t * 0.05)
        for loc in fault_locations:
            if loc < num_points:
                temperature[loc:] += 0.5  # Cumulative heating

        # Ultrasound increases with fault
        ultrasound = 35 + 3 * np.sin(t * 0.12)
        ultrasound += 0.5 * np.random.randn(num_points)
        for loc in fault_locations:
            if loc < num_points:
                ultrasound[loc-2:loc+2] += 10 * (1 + (loc - 400) / 350)

        # Update plotter
        self.plotter.update_data("DE Radial Accel", timestamps, radial)
        self.plotter.update_data("DE Axial Accel", timestamps, axial)
        self.plotter.update_data("DE Temperature", timestamps, temperature)
        self.plotter.update_data("DE Ultrasound", timestamps, ultrasound)

        # Store high-frequency waveforms for drill-down
        # (In real app, these would come from actual data acquisition)
        sample_rate = 25600
        self.waveform_database = {}
        for loc in fault_locations:
            ts = timestamps[loc]
            waveform = self._generate_bearing_fault_waveform(sample_rate)
            self.waveform_database[ts] = {
                'waveform': waveform,
                'sample_rate': sample_rate,
                'channel': 'DE Radial Accel'
            }

        # Add anomaly markers
        if self.anomaly_overlay:
            self.anomaly_overlay.clear_markers()
        self.anomaly_overlay = AnomalyMarkerOverlay(self.plotter.plot_widget)

        for i, loc in enumerate(fault_locations):
            severity_map = ['info', 'info', 'warning', 'warning', 'critical']
            score = 0.35 - (i * 0.05)  # Decreasing score = worsening

            marker = AnomalyMarker(
                timestamp=timestamps[loc],
                score=score,
                channels=["DE Radial Accel", "DE Axial Accel"],
                severity=severity_map[i]
            )
            self.anomaly_overlay.add_marker(marker)

        self.status_label.setText(f'Status: Monitoring {len(self.plotter.channels)} channels | {len(fault_locations)} anomalies detected')

    def _generate_bearing_fault_waveform(self, sample_rate: float, duration: float = 1.0) -> np.ndarray:
        """Generate realistic bearing fault waveform for drill-down"""
        num_samples = int(sample_rate * duration)
        time = np.linspace(0, duration, num_samples)

        # Bearing parameters
        shaft_speed = 30  # Hz (1800 RPM)
        bpfo_freq = 120  # Outer race defect frequency

        # Synthesize signal
        waveform = (
            1.5 * np.sin(2 * np.pi * shaft_speed * time) +
            0.8 * np.sin(2 * np.pi * shaft_speed * 2 * time) +
            4.0 * np.sin(2 * np.pi * bpfo_freq * time) +
            1.5 * np.sin(2 * np.pi * bpfo_freq * 2 * time) +
            0.7 * np.random.randn(num_samples)
        )

        # Add impulses
        for i in range(0, num_samples, int(sample_rate / bpfo_freq)):
            if i < num_samples - 20:
                impulse = 8.0 * np.exp(-np.arange(20) / 5)
                waveform[i:i+20] += impulse

        return waveform

    def _on_plot_clicked(self, timestamp: float, channel_name: str):
        """Handle plot click - open drill-down viewer"""
        # Find closest anomaly
        marker = self.anomaly_overlay.get_marker_at_time(timestamp, tolerance=100)

        if marker:
            dt = datetime.fromtimestamp(marker.timestamp)

            # Check if we have waveform data
            if marker.timestamp in self.waveform_database:
                # Open drill-down viewer
                if self.drill_down_viewer is None or not self.drill_down_viewer.isVisible():
                    self.drill_down_viewer = DrillDownViewer()

                data = self.waveform_database[marker.timestamp]

                # Bearing frequencies (would come from configuration in real app)
                bearing_freqs = {
                    'BPFO': 120,
                    'BPFI': 180,
                    'BSF': 45,
                    'FTF': 12
                }

                self.drill_down_viewer.load_anomaly_data(
                    channel_name=data['channel'],
                    timestamp=marker.timestamp,
                    waveform=data['waveform'],
                    sample_rate=data['sample_rate'],
                    bearing_freqs=bearing_freqs
                )

                self.drill_down_viewer.show()
                self.drill_down_viewer.raise_()
                self.drill_down_viewer.activateWindow()
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Anomaly Details",
                    f"Anomaly: {marker.severity_label}\n"
                    f"Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Score: {marker.score:.3f}\n"
                    f"Channels: {', '.join(marker.channels)}\n\n"
                    f"Raw waveform data not available for this anomaly."
                )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "No Anomaly",
                f"No anomaly detected near this time.\n"
                f"Clicked: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}"
            )

    def _clear_anomalies(self):
        """Clear all anomaly markers"""
        if self.anomaly_overlay:
            self.anomaly_overlay.clear_markers()
            self.status_label.setText('Status: Anomalies cleared')

    def _show_help(self):
        """Show help dialog"""
        help_text = """
<h2>MachineIQ Desktop - Quick Guide</h2>

<h3>Features:</h3>
<ul>
<li><b>Multi-Channel Trends:</b> View multiple sensor channels simultaneously</li>
<li><b>Anomaly Detection:</b> Vertical markers show detected anomalies</li>
<li><b>Interactive Zoom/Pan:</b> Use mouse to zoom and pan the plot</li>
<li><b>Drill-Down Analysis:</b> Click near anomalies to see detailed analysis</li>
</ul>

<h3>How to Use:</h3>
<ol>
<li><b>Enable/Disable Channels:</b> Use checkboxes below plot</li>
<li><b>Change Time Scale:</b> Use dropdown in control panel</li>
<li><b>View Anomaly Details:</b> Click on or near vertical anomaly markers</li>
<li><b>Drill-Down:</b> Opens 4-panel view showing:
   <ul>
   <li>Time Waveform</li>
   <li>FFT Spectrum</li>
   <li>Envelope Spectrum (demodulated)</li>
   <li>NeuraPeak™ Spectrum (stress wave)</li>
   </ul>
</li>
<li><b>Export:</b> Use export buttons to save plots and data</li>
</ol>

<h3>Anomaly Severity:</h3>
<ul>
<li>🟡 <b>Info:</b> Minor deviation from normal</li>
<li>🟠 <b>Warning:</b> Moderate anomaly - investigate soon</li>
<li>🔴 <b>Critical:</b> Severe anomaly - immediate action required</li>
</ul>

<h3>Keyboard Shortcuts:</h3>
<ul>
<li><b>Ctrl+Q:</b> Quit application</li>
<li><b>Ctrl+R:</b> Refresh data</li>
</ul>
        """

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("MachineIQ Help")
        msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.exec()


def main():
    """Main entry point"""
    app = QtWidgets.QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create main window
    window = MachineIQDesktopApp()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
