"""
Multi-channel trend plotting with PyQtGraph

Features:
- Overlay multiple channels on single plot
- Individual channel enable/disable
- Color assignment per channel
- Configurable time window
- Zoom, pan, export functionality
"""

import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ChannelConfig:
    """Configuration for a single channel"""
    name: str
    color: str  # PyQtGraph color string ('r', 'g', 'b', '#FF0000', etc.)
    enabled: bool = True
    line_width: float = 2.0
    style: str = 'solid'  # 'solid', 'dashed', 'dotted'


class ChannelPlotter(QtWidgets.QWidget):
    """
    Multi-channel time-series plotter with interactive controls

    Example usage:
        plotter = ChannelPlotter()
        plotter.add_channel("Vibration RMS", color='r')
        plotter.add_channel("Temperature", color='b')
        plotter.update_data("Vibration RMS", timestamps, values)
        plotter.show()
    """

    # Signal emitted when user clicks on the plot (for drill-down)
    plot_clicked = QtCore.pyqtSignal(float, str)  # (timestamp, channel_name)

    def __init__(self, parent=None, title="MachineIQ Channel Trends"):
        super().__init__(parent)

        self.title = title
        self.channels: Dict[str, ChannelConfig] = {}
        self.plot_curves: Dict[str, pg.PlotDataItem] = {}
        self.data_buffers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface"""
        layout = QtWidgets.QVBoxLayout()

        # Title
        title_label = QtWidgets.QLabel(f"<h2>{self.title}</h2>")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)

        # Main plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)

        # Connect click events
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        layout.addWidget(self.plot_widget)

        # Channel control panel (checkboxes for enable/disable)
        self.channel_controls_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.channel_controls_layout)

        self.setLayout(layout)
        self.setWindowTitle('MachineIQ Channel Plotter')
        self.resize(1200, 700)

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """Create the control panel with buttons and options"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        # Time scale selector
        layout.addWidget(QtWidgets.QLabel('Time Scale:'))
        self.time_scale_combo = QtWidgets.QComboBox()
        self.time_scale_combo.addItems([
            '1 minute', '5 minutes', '15 minutes',
            '1 hour', '6 hours', '24 hours', 'All'
        ])
        self.time_scale_combo.setCurrentText('1 hour')
        self.time_scale_combo.currentTextChanged.connect(self._on_time_scale_changed)
        layout.addWidget(self.time_scale_combo)

        layout.addStretch()

        # Export button
        self.export_btn = QtWidgets.QPushButton('📊 Export PNG')
        self.export_btn.clicked.connect(self._export_png)
        layout.addWidget(self.export_btn)

        # Reset zoom button
        self.reset_btn = QtWidgets.QPushButton('🔍 Reset Zoom')
        self.reset_btn.clicked.connect(self.plot_widget.autoRange)
        layout.addWidget(self.reset_btn)

        panel.setLayout(layout)
        return panel

    def add_channel(self, name: str, color: str = 'w',
                    line_width: float = 2.0, enabled: bool = True):
        """
        Add a new channel to the plotter

        Args:
            name: Channel name (must be unique)
            color: PyQtGraph color specifier
            line_width: Width of the plot line
            enabled: Whether channel is initially visible
        """
        if name in self.channels:
            raise ValueError(f"Channel '{name}' already exists")

        config = ChannelConfig(
            name=name,
            color=color,
            enabled=enabled,
            line_width=line_width
        )

        self.channels[name] = config

        # Create plot curve
        pen = pg.mkPen(color=color, width=line_width)
        curve = self.plot_widget.plot(
            [], [],
            pen=pen,
            name=name
        )
        curve.setVisible(enabled)

        self.plot_curves[name] = curve
        self.data_buffers[name] = (np.array([]), np.array([]))

        # Add checkbox control
        self._add_channel_control(name, config)

    def _add_channel_control(self, name: str, config: ChannelConfig):
        """Add checkbox control for channel enable/disable"""
        checkbox = QtWidgets.QCheckBox(name)
        checkbox.setChecked(config.enabled)
        checkbox.setStyleSheet(f"QCheckBox {{ color: {config.color}; font-weight: bold; }}")
        checkbox.stateChanged.connect(
            lambda state: self.set_channel_enabled(name, state == QtCore.Qt.CheckState.Checked.value)
        )
        self.channel_controls_layout.addWidget(checkbox)

    def update_data(self, channel_name: str,
                   timestamps: np.ndarray, values: np.ndarray):
        """
        Update data for a specific channel

        Args:
            channel_name: Name of channel to update
            timestamps: Unix timestamps or datetime objects
            values: Sensor values
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' does not exist")

        # Convert datetime to Unix timestamps if needed
        if len(timestamps) > 0 and isinstance(timestamps[0], datetime):
            timestamps = np.array([t.timestamp() for t in timestamps])

        # Store in buffer
        self.data_buffers[channel_name] = (timestamps, values)

        # Update plot
        if self.channels[channel_name].enabled:
            self.plot_curves[channel_name].setData(timestamps, values)

    def set_channel_enabled(self, channel_name: str, enabled: bool):
        """Enable or disable a channel"""
        if channel_name in self.channels:
            self.channels[channel_name].enabled = enabled
            self.plot_curves[channel_name].setVisible(enabled)

            # Update data if enabling
            if enabled:
                timestamps, values = self.data_buffers[channel_name]
                self.plot_curves[channel_name].setData(timestamps, values)

    def _on_time_scale_changed(self, scale_text: str):
        """Handle time scale changes"""
        # Get all timestamps across all channels
        all_times = []
        for timestamps, _ in self.data_buffers.values():
            if len(timestamps) > 0:
                all_times.extend(timestamps)

        if not all_times:
            return

        max_time = max(all_times)

        # Calculate time window
        scale_map = {
            '1 minute': 60,
            '5 minutes': 300,
            '15 minutes': 900,
            '1 hour': 3600,
            '6 hours': 21600,
            '24 hours': 86400,
            'All': None
        }

        window = scale_map.get(scale_text)

        if window is None:
            self.plot_widget.autoRange()
        else:
            min_time = max_time - window
            self.plot_widget.setXRange(min_time, max_time, padding=0.02)

    def _export_png(self):
        """Export current plot as PNG"""
        from pyqtgraph.exporters import ImageExporter

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export PNG', '', 'PNG Files (*.png)'
        )

        if filename:
            if not filename.endswith('.png'):
                filename += '.png'
            exporter = ImageExporter(self.plot_widget.plotItem)
            exporter.parameters()['width'] = 1920  # High res
            exporter.export(filename)
            QtWidgets.QMessageBox.information(
                self,
                'Export Success',
                f'Plot exported to:\n{filename}'
            )

    def _on_plot_clicked(self, event):
        """Handle plot click events"""
        # Get mouse position in data coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        timestamp = mouse_point.x()

        # Find closest enabled channel
        min_distance = float('inf')
        closest_channel = None

        for channel_name, config in self.channels.items():
            if not config.enabled:
                continue

            timestamps, values = self.data_buffers[channel_name]
            if len(timestamps) == 0:
                continue

            # Find closest timestamp
            idx = np.abs(timestamps - timestamp).argmin()
            distance = abs(timestamps[idx] - timestamp)

            if distance < min_distance:
                min_distance = distance
                closest_channel = channel_name

        if closest_channel:
            self.plot_clicked.emit(timestamp, closest_channel)


def demo_plotter():
    """Demonstrate the channel plotter with simulated data"""
    import sys

    app = QtWidgets.QApplication(sys.argv)

    plotter = ChannelPlotter(title="Demo: Motor Bearing Monitoring")

    # Add some channels with different colors
    plotter.add_channel("DE Radial Accel", color='r', line_width=2)
    plotter.add_channel("DE Axial Accel", color='b', line_width=2)
    plotter.add_channel("Temperature", color='g', line_width=2)
    plotter.add_channel("Pressure", color='y', line_width=2)

    # Generate simulated data
    num_points = 1000
    base_time = datetime.now()
    timestamps = [base_time + timedelta(seconds=i) for i in range(num_points)]
    timestamps_unix = np.array([t.timestamp() for t in timestamps])

    # Simulated vibration with anomalies
    vib_radial = 3.0 + 0.5 * np.sin(np.linspace(0, 20*np.pi, num_points))
    vib_radial[500:520] += 5.0  # Simulate anomaly spike
    vib_radial += 0.2 * np.random.randn(num_points)  # Add noise

    # Simulated axial vibration
    vib_axial = 2.5 + 0.3 * np.cos(np.linspace(0, 15*np.pi, num_points))
    vib_axial[500:520] += 3.0  # Correlated anomaly
    vib_axial += 0.15 * np.random.randn(num_points)

    # Simulated temperature drift
    temp_data = 65 + 5 * np.sin(np.linspace(0, 4*np.pi, num_points))
    temp_data[500:550] += 2.0  # Slight temperature rise during anomaly

    # Simulated pressure
    press_data = 100 + 10 * np.cos(np.linspace(0, 10*np.pi, num_points))
    press_data += 0.5 * np.random.randn(num_points)

    # Update data
    plotter.update_data("DE Radial Accel", timestamps_unix, vib_radial)
    plotter.update_data("DE Axial Accel", timestamps_unix, vib_axial)
    plotter.update_data("Temperature", timestamps_unix, temp_data)
    plotter.update_data("Pressure", timestamps_unix, press_data)

    # Connect click handler
    def on_plot_click(timestamp, channel):
        dt = datetime.fromtimestamp(timestamp)
        print(f"Clicked: {channel} at {dt.strftime('%H:%M:%S')}")
        QtWidgets.QMessageBox.information(
            plotter,
            "Plot Clicked",
            f"Channel: {channel}\nTime: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    plotter.plot_clicked.connect(on_plot_click)

    plotter.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    demo_plotter()
