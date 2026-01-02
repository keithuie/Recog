"""
Dataflow Page

Unified data source management combining CSV import, connections, and sample data.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
    QFormLayout, QHeaderView, QStackedWidget, QScrollArea,
    QLineEdit, QTextEdit, QCheckBox, QDialog, QDialogButtonBox,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..styles import COLORS, get_auto_contrast_color


# Sample APIs available for quick setup
SAMPLE_APIS = {
    'USGS Earthquakes': {
        'url': 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=10',
        'description': 'Real-time earthquake data from USGS',
        'interval': 30
    },
    'OpenWeather (requires API key)': {
        'url': 'https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_API_KEY',
        'description': 'Weather data - requires free API key',
        'interval': 60
    },
}

# Sample datasets available (local data, no network required)
SAMPLE_DATASETS = {
    'NASA Space Weather': {
        'description': 'Solar flux, plasma density, magnetic field, and geomagnetic indices',
        'channels': ['solar_flux', 'proton_density', 'plasma_speed', 'plasma_temp',
                    'mag_field_bt', 'mag_field_bz', 'kp_index', 'dst_index'],
        'interval': 5
    }
}


class DataSourceCard(QFrame):
    """Card displaying a data source"""

    connect_clicked = pyqtSignal(str)  # source_name
    disconnect_clicked = pyqtSignal(str)  # source_name
    delete_clicked = pyqtSignal(str)  # source_name

    def __init__(self, name: str, source_type: str, status: str = "Disconnected",
                 details: str = "", parent=None):
        super().__init__(parent)
        self.name = name
        self.source_type = source_type
        self._status = status
        self._details = details

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        self.setFixedHeight(90)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        name_label = QLabel(self.name)
        name_label.setStyleSheet(f"font-weight: 600; font-size: 14px; color: {COLORS['text_primary']};")
        info_layout.addWidget(name_label)

        type_label = QLabel(self.source_type)
        type_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        info_layout.addWidget(type_label)

        if self._details:
            details_label = QLabel(self._details)
            details_label.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 11px;")
            info_layout.addWidget(details_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Status
        self.status_label = QLabel(self._status)
        self._update_status_style()
        layout.addWidget(self.status_label)

        # Action button
        self.action_btn = QPushButton("Connect")
        self.action_btn.setFixedWidth(90)
        self.action_btn.clicked.connect(self._on_action_click)
        layout.addWidget(self.action_btn)

        # Delete button
        delete_btn = QPushButton("Remove")
        delete_btn.setFixedWidth(70)
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['danger']};
                border: 1px solid {COLORS['danger']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']};
                color: white;
            }}
        """)
        delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.name))
        layout.addWidget(delete_btn)

    def _on_action_click(self):
        if self._status in ["Connected", "Monitoring"]:
            self.disconnect_clicked.emit(self.name)
        else:
            self.connect_clicked.emit(self.name)

    def set_status(self, status: str):
        self._status = status
        self.status_label.setText(status)
        self._update_status_style()

        if status in ["Connected", "Monitoring"]:
            self.action_btn.setText("Disconnect")
            self.action_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['warning']};
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: #E68600;
                }}
            """)
        else:
            self.action_btn.setText("Connect")
            self.action_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['success']};
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: #2DB84D;
                }}
            """)

    def _update_status_style(self):
        if self._status in ["Connected", "Monitoring"]:
            color = COLORS['success']
        elif self._status == "Connecting...":
            color = COLORS['warning']
        elif self._status == "Error":
            color = COLORS['danger']
        else:
            color = COLORS['text_tertiary']

        self.status_label.setStyleSheet(f"""
            color: {color};
            font-weight: 500;
            padding: 4px 12px;
            background-color: {color}20;
            border-radius: 4px;
        """)


class SampleDataDialog(QDialog):
    """Dialog for generating sample vibration data"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Sample Data")
        self.setMinimumWidth(400)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        layout.addWidget(QLabel("Generate sample vibration monitoring data for testing."))

        form = QFormLayout()
        form.setSpacing(12)

        # Duration
        duration_row = QHBoxLayout()
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 1440)  # Up to 24 hours
        self.duration_spin.setValue(5)
        self.duration_spin.setFixedWidth(80)
        duration_row.addWidget(self.duration_spin)

        self.duration_unit = QComboBox()
        self.duration_unit.addItems(["minutes", "hours"])
        self.duration_unit.setFixedWidth(100)
        duration_row.addWidget(self.duration_unit)
        duration_row.addStretch()
        form.addRow("Duration:", duration_row)

        # Number of channels
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 20)
        self.channels_spin.setValue(4)
        self.channels_spin.setFixedWidth(80)
        form.addRow("Channels:", self.channels_spin)

        # Sample rate
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(0.1, 100.0)
        self.sample_rate_spin.setValue(10.0)
        self.sample_rate_spin.setSuffix(" Hz")
        self.sample_rate_spin.setFixedWidth(100)
        form.addRow("Sample Rate:", self.sample_rate_spin)

        # Include anomaly
        self.anomaly_check = QCheckBox("Include simulated anomaly")
        self.anomaly_check.setChecked(True)
        form.addRow("", self.anomaly_check)

        # Anomaly position
        anomaly_row = QHBoxLayout()
        self.anomaly_spin = QSpinBox()
        self.anomaly_spin.setRange(10, 90)
        self.anomaly_spin.setValue(70)
        self.anomaly_spin.setSuffix("%")
        self.anomaly_spin.setFixedWidth(80)
        anomaly_row.addWidget(self.anomaly_spin)
        anomaly_row.addWidget(QLabel("through data"))
        anomaly_row.addStretch()
        form.addRow("Anomaly at:", anomaly_row)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict:
        """Get the configuration for sample data generation"""
        duration = self.duration_spin.value()
        if self.duration_unit.currentText() == "hours":
            duration *= 60  # Convert to minutes

        return {
            'duration_minutes': duration,
            'num_channels': self.channels_spin.value(),
            'sample_rate_hz': self.sample_rate_spin.value(),
            'include_anomaly': self.anomaly_check.isChecked(),
            'anomaly_position': self.anomaly_spin.value() / 100.0,
        }


class AddSourceDialog(QDialog):
    """Dialog for adding a new data source"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Data Source")
        self.setMinimumWidth(500)
        self._config = {}
        self._discovered_channels = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Source type selection
        type_group = QGroupBox("Source Type")
        type_layout = QVBoxLayout(type_group)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["CSV File", "REST API", "MQTT Broker", "OPC-UA Server"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)

        layout.addWidget(type_group)

        # Config stack
        self.config_stack = QStackedWidget()

        # CSV config
        csv_widget = QWidget()
        csv_layout = QFormLayout(csv_widget)
        csv_layout.setSpacing(12)

        file_row = QHBoxLayout()
        self.csv_path = QLineEdit()
        self.csv_path.setPlaceholderText("Select a CSV file...")
        self.csv_path.setReadOnly(True)
        file_row.addWidget(self.csv_path)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_csv)
        file_row.addWidget(browse_btn)
        csv_layout.addRow("File:", file_row)

        self.timestamp_combo = QComboBox()
        self.timestamp_combo.addItems(["timestamp", "time", "Time", "datetime"])
        self.timestamp_combo.setEditable(True)
        csv_layout.addRow("Timestamp Column:", self.timestamp_combo)

        self.config_stack.addWidget(csv_widget)

        # REST API config
        api_widget = QWidget()
        api_layout = QFormLayout(api_widget)
        api_layout.setSpacing(12)

        # Sample API selector
        sample_row = QHBoxLayout()
        self.sample_combo = QComboBox()
        self.sample_combo.addItem("-- Select sample API --")
        for name in SAMPLE_APIS.keys():
            self.sample_combo.addItem(name)
        self.sample_combo.currentTextChanged.connect(self._on_sample_selected)
        sample_row.addWidget(self.sample_combo)
        api_layout.addRow("Sample APIs:", sample_row)

        self.api_url = QLineEdit()
        self.api_url.setPlaceholderText("https://api.example.com/data")
        api_layout.addRow("API URL:", self.api_url)

        self.api_interval = QSpinBox()
        self.api_interval.setRange(1, 3600)
        self.api_interval.setValue(5)
        self.api_interval.setSuffix(" seconds")
        api_layout.addRow("Poll Interval:", self.api_interval)

        # Test connection button
        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_api_connection)
        test_row.addWidget(self.test_btn)

        self.test_result = QLabel("")
        test_row.addWidget(self.test_result)
        test_row.addStretch()
        api_layout.addRow("", test_row)

        # Discovered channels
        self.channels_label = QLabel("Channels: (test connection to discover)")
        self.channels_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        api_layout.addRow("", self.channels_label)

        self.config_stack.addWidget(api_widget)

        # MQTT config
        mqtt_widget = QWidget()
        mqtt_layout = QFormLayout(mqtt_widget)
        mqtt_layout.setSpacing(12)

        self.mqtt_host = QLineEdit()
        self.mqtt_host.setPlaceholderText("broker.example.com")
        mqtt_layout.addRow("Broker Host:", self.mqtt_host)

        self.mqtt_port = QSpinBox()
        self.mqtt_port.setRange(1, 65535)
        self.mqtt_port.setValue(1883)
        mqtt_layout.addRow("Port:", self.mqtt_port)

        self.mqtt_topic = QLineEdit()
        self.mqtt_topic.setPlaceholderText("sensors/#")
        mqtt_layout.addRow("Topic:", self.mqtt_topic)

        self.config_stack.addWidget(mqtt_widget)

        # OPC-UA config
        opcua_widget = QWidget()
        opcua_layout = QFormLayout(opcua_widget)
        opcua_layout.setSpacing(12)

        self.opcua_url = QLineEdit()
        self.opcua_url.setPlaceholderText("opc.tcp://localhost:4840")
        opcua_layout.addRow("Server URL:", self.opcua_url)

        self.config_stack.addWidget(opcua_widget)

        layout.addWidget(self.config_stack)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_type_changed(self, type_text: str):
        type_indices = {
            "CSV File": 0,
            "REST API": 1,
            "MQTT Broker": 2,
            "OPC-UA Server": 3,
        }
        self.config_stack.setCurrentIndex(type_indices.get(type_text, 0))

    def _on_sample_selected(self, name: str):
        if name in SAMPLE_APIS:
            api_info = SAMPLE_APIS[name]
            self.api_url.setText(api_info['url'])
            self.api_interval.setValue(api_info['interval'])
            self.test_result.setText("")
            self._discovered_channels = []
            self.channels_label.setText("Channels: (test connection to discover)")

    def _browse_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if filepath:
            self.csv_path.setText(filepath)

    def _test_api_connection(self):
        """Test API connection and discover channels"""
        url = self.api_url.text().strip()
        if not url:
            self.test_result.setText("Enter a URL first")
            self.test_result.setStyleSheet(f"color: {COLORS['danger']};")
            return

        self.test_result.setText("Testing...")
        self.test_result.setStyleSheet(f"color: {COLORS['text_secondary']};")

        # Import here to avoid circular imports
        try:
            from ..core import APIDataSource

            source = APIDataSource("test", url)
            success, channels, message = source.test_connection()

            if success:
                self._discovered_channels = channels
                self.test_result.setText("Connected!")
                self.test_result.setStyleSheet(f"color: {COLORS['success']};")
                if channels:
                    self.channels_label.setText(f"Channels: {', '.join(channels[:5])}{'...' if len(channels) > 5 else ''}")
                else:
                    self.channels_label.setText("Channels: None found (may need data mapping)")
            else:
                self.test_result.setText("Failed")
                self.test_result.setStyleSheet(f"color: {COLORS['danger']};")
                self.channels_label.setText(f"Error: {message}")

        except Exception as e:
            self.test_result.setText("Error")
            self.test_result.setStyleSheet(f"color: {COLORS['danger']};")
            self.channels_label.setText(f"Error: {str(e)}")

    def _validate_and_accept(self):
        source_type = self.type_combo.currentText()

        if source_type == "CSV File":
            if not self.csv_path.text():
                QMessageBox.warning(self, "Missing File", "Please select a CSV file.")
                return
            self._config = {
                'type': 'csv',
                'path': self.csv_path.text(),
                'timestamp_column': self.timestamp_combo.currentText(),
            }
        elif source_type == "REST API":
            if not self.api_url.text():
                QMessageBox.warning(self, "Missing URL", "Please enter an API URL.")
                return
            self._config = {
                'type': 'api',
                'url': self.api_url.text(),
                'interval': self.api_interval.value(),
                'channels': self._discovered_channels,
            }
        elif source_type == "MQTT Broker":
            if not self.mqtt_host.text():
                QMessageBox.warning(self, "Missing Host", "Please enter a broker host.")
                return
            self._config = {
                'type': 'mqtt',
                'host': self.mqtt_host.text(),
                'port': self.mqtt_port.value(),
                'topic': self.mqtt_topic.text(),
            }
        elif source_type == "OPC-UA Server":
            if not self.opcua_url.text():
                QMessageBox.warning(self, "Missing URL", "Please enter a server URL.")
                return
            self._config = {
                'type': 'opcua',
                'url': self.opcua_url.text(),
            }

        self.accept()

    def get_config(self) -> dict:
        return self._config


class DataflowPage(QWidget):
    """
    Unified data source management page.

    Features:
    - Active data sources display
    - CSV upload
    - Sample data generation
    - CSV template download
    - Connection wizard
    """

    # Signals
    source_added = pyqtSignal(dict)  # source config
    source_removed = pyqtSignal(str)  # source name
    source_connected = pyqtSignal(str)  # source name
    source_disconnected = pyqtSignal(str)  # source name
    file_loaded = pyqtSignal(str)  # filepath
    sample_data_generated = pyqtSignal(str)  # filepath

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources = {}
        self._total_points = 0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 16, 0)
        scroll_layout.setSpacing(20)

        # Active Data Sources Section
        sources_group = QGroupBox("Active Data Sources")
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.setSpacing(12)

        # Header with Add button
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Manage your data connections"))
        header_row.addStretch()

        self.add_source_btn = QPushButton("+ Add Source")
        self.add_source_btn.clicked.connect(self._show_add_source_dialog)
        header_row.addWidget(self.add_source_btn)

        sources_layout.addLayout(header_row)

        # Sources container
        self.sources_container = QVBoxLayout()
        self.sources_container.setSpacing(8)

        # Placeholder
        self.sources_placeholder = QLabel("No data sources configured. Click '+ Add Source' to add one.")
        self.sources_placeholder.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            padding: 30px;
            background-color: {COLORS['surface']};
            border: 1px dashed {COLORS['border']};
            border-radius: 8px;
        """)
        self.sources_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sources_container.addWidget(self.sources_placeholder)

        sources_layout.addLayout(self.sources_container)
        scroll_layout.addWidget(sources_group)

        # Quick Actions Section
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setSpacing(12)

        # Upload CSV button
        upload_btn = QPushButton("Upload CSV")
        upload_btn.setMinimumHeight(50)
        upload_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface']};
                color: {get_auto_contrast_color(COLORS['surface'])};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {COLORS['primary']};
                background-color: {COLORS['primary']}10;
            }}
        """)
        upload_btn.clicked.connect(self._upload_csv)
        actions_layout.addWidget(upload_btn)

        # Generate Sample Data button
        sample_btn = QPushButton("Generate Sample Data")
        sample_btn.setMinimumHeight(50)
        sample_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface']};
                color: {get_auto_contrast_color(COLORS['surface'])};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {COLORS['primary']};
                background-color: {COLORS['primary']}10;
            }}
        """)
        sample_btn.clicked.connect(self._show_sample_dialog)
        actions_layout.addWidget(sample_btn)

        # Download Template button
        template_btn = QPushButton("Download CSV Template")
        template_btn.setMinimumHeight(50)
        template_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface']};
                color: {get_auto_contrast_color(COLORS['surface'])};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {COLORS['primary']};
                background-color: {COLORS['primary']}10;
            }}
        """)
        template_btn.clicked.connect(self._download_template)
        actions_layout.addWidget(template_btn)

        scroll_layout.addWidget(actions_group)

        # Data Preview Section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Info labels
        info_row = QHBoxLayout()

        self.rows_label = QLabel("Rows: 0")
        self.rows_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        info_row.addWidget(self.rows_label)

        self.cols_label = QLabel("Channels: 0")
        self.cols_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        info_row.addWidget(self.cols_label)

        self.time_range_label = QLabel("Time Range: N/A")
        self.time_range_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        info_row.addWidget(self.time_range_label)

        info_row.addStretch()
        preview_layout.addLayout(info_row)

        # Table
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['surface']};
                alternate-background-color: {COLORS['surface_secondary']};
                gridline-color: {COLORS['border_light']};
            }}
        """)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.setMaximumHeight(250)
        preview_layout.addWidget(self.preview_table)

        scroll_layout.addWidget(preview_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def _show_add_source_dialog(self):
        """Show the add source dialog"""
        dialog = AddSourceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._add_source(config)

    def _add_source(self, config: dict):
        """Add a new data source"""
        source_type = config.get('type', '')

        if source_type == 'csv':
            name = Path(config['path']).name
            source_type_display = "CSV File"
            details = config['path']
        elif source_type == 'api':
            name = "REST API"
            source_type_display = "REST API"
            details = config['url'][:50] + "..." if len(config.get('url', '')) > 50 else config.get('url', '')
        elif source_type == 'mqtt':
            name = f"MQTT: {config['host']}"
            source_type_display = "MQTT Broker"
            details = f"{config['host']}:{config['port']}"
        elif source_type == 'opcua':
            name = "OPC-UA Server"
            source_type_display = "OPC-UA"
            details = config['url']
        else:
            return

        # Hide placeholder
        self.sources_placeholder.hide()

        # Create card
        card = DataSourceCard(name, source_type_display, "Disconnected", details)
        card.connect_clicked.connect(lambda n: self.source_connected.emit(n))
        card.disconnect_clicked.connect(lambda n: self.source_disconnected.emit(n))
        card.delete_clicked.connect(self._remove_source)

        self._sources[name] = {'card': card, 'config': config}
        self.sources_container.insertWidget(self.sources_container.count() - 1, card)

        # Emit signal
        config['name'] = name
        self.source_added.emit(config)

        # If CSV, load immediately
        if source_type == 'csv':
            self.file_loaded.emit(config['path'])

    def _remove_source(self, name: str):
        """Remove a data source"""
        if name in self._sources:
            self._sources[name]['card'].deleteLater()
            del self._sources[name]

            if not self._sources:
                self.sources_placeholder.show()

            self.source_removed.emit(name)

    def _upload_csv(self):
        """Open file browser and upload CSV"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filepath:
            self._add_source({
                'type': 'csv',
                'path': filepath,
                'timestamp_column': 'timestamp',
            })

    def _show_sample_dialog(self):
        """Show sample data generation dialog"""
        dialog = SampleDataDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._generate_sample_data(config)

    def _generate_sample_data(self, config: dict):
        """Generate sample vibration data"""
        import numpy as np
        import pandas as pd

        duration_minutes = config['duration_minutes']
        num_channels = config['num_channels']
        sample_rate = config['sample_rate_hz']
        include_anomaly = config['include_anomaly']
        anomaly_position = config['anomaly_position']

        # Calculate number of samples
        num_samples = int(duration_minutes * 60 * sample_rate)

        # Generate timestamps
        start_time = datetime.now()
        timestamps = pd.date_range(
            start=start_time,
            periods=num_samples,
            freq=f'{1000/sample_rate:.3f}ms'
        )

        # Generate channel data
        t = np.arange(num_samples) / sample_rate
        data = {'timestamp': timestamps}

        for i in range(num_channels):
            channel_name = f'sensor_{i+1}'

            # Base signal: combination of frequencies (motor harmonics)
            base_freq = 10 + i * 2  # Different base frequency per channel
            signal = (
                np.sin(2 * np.pi * base_freq * t) * 0.5 +  # Fundamental
                np.sin(2 * np.pi * base_freq * 2 * t) * 0.3 +  # 2nd harmonic
                np.sin(2 * np.pi * base_freq * 3 * t) * 0.1 +  # 3rd harmonic
                np.random.randn(num_samples) * 0.05  # Noise
            )

            # Add slow drift (temperature effect)
            signal += np.sin(2 * np.pi * t / (duration_minutes * 60)) * 0.2

            # Add anomaly if requested
            if include_anomaly:
                anomaly_start = int(num_samples * anomaly_position)
                anomaly_duration = int(num_samples * 0.1)  # 10% of data

                if anomaly_start + anomaly_duration <= num_samples:
                    # Bearing defect frequency pattern
                    anomaly_t = np.arange(anomaly_duration) / sample_rate
                    defect_freq = 45.3  # Typical ball pass frequency
                    anomaly_signal = (
                        np.sin(2 * np.pi * defect_freq * anomaly_t) * 0.8 +
                        np.sin(2 * np.pi * defect_freq * 2 * anomaly_t) * 0.4
                    )

                    # Apply exponential growth
                    growth = np.linspace(0, 1, anomaly_duration) ** 2
                    signal[anomaly_start:anomaly_start + anomaly_duration] += anomaly_signal * growth

            data[channel_name] = signal

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f'machineiq_sample_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(temp_path, index=False)

        # Create unique name for this sample
        sample_name = f"Sample Data ({datetime.now().strftime('%H:%M')})"

        # Hide placeholder
        self.sources_placeholder.hide()

        # Create card directly (don't use _add_source which emits file_loaded)
        card = DataSourceCard(sample_name, "Sample Data", "Connected", f"{num_channels} channels, {duration_minutes} min")
        card.connect_clicked.connect(lambda n: self.source_connected.emit(n))
        card.disconnect_clicked.connect(lambda n: self.source_disconnected.emit(n))
        card.delete_clicked.connect(self._remove_source)

        config = {
            'type': 'sample',
            'path': temp_path,
            'timestamp_column': 'timestamp',
            'name': sample_name,
        }

        self._sources[sample_name] = {'card': card, 'config': config}
        self.sources_container.insertWidget(self.sources_container.count() - 1, card)

        QMessageBox.information(
            self,
            "Sample Data Generated",
            f"Generated {num_samples:,} samples across {num_channels} channels.\n\n"
            f"Duration: {duration_minutes} minutes\n"
            f"Sample rate: {sample_rate} Hz\n"
            f"{'Includes simulated anomaly at ' + str(int(anomaly_position * 100)) + '%' if include_anomaly else 'No anomaly included'}\n\n"
            f"Click 'Connect' on the card to start playback."
        )

        # Emit signal with config for the app to pick up
        config['name'] = sample_name
        self.source_added.emit(config)

    def _download_template(self):
        """Download CSV template file"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV Template",
            "machineiq_template.csv",
            "CSV Files (*.csv)"
        )

        if filepath:
            # Create template content
            template_content = """timestamp,sensor_1,sensor_2,sensor_3,sensor_4
# MachineIQ CSV Template
# -----------------------
# Timestamp format: ISO 8601 (YYYY-MM-DDTHH:MM:SS) or Unix timestamp
# Channel values: Numeric values (float or integer)
# Add as many sensor columns as needed
#
2024-01-01T00:00:00,1.234,2.345,3.456,4.567
2024-01-01T00:00:01,1.235,2.346,3.457,4.568
2024-01-01T00:00:02,1.236,2.347,3.458,4.569
2024-01-01T00:00:03,1.237,2.348,3.459,4.570
2024-01-01T00:00:04,1.238,2.349,3.460,4.571
"""
            with open(filepath, 'w') as f:
                f.write(template_content)

            QMessageBox.information(
                self,
                "Template Saved",
                f"CSV template saved to:\n{filepath}\n\n"
                "Edit this file with your data and upload it to MachineIQ."
            )

    def set_source_status(self, name: str, status: str):
        """Update the status of a data source"""
        if name in self._sources:
            self._sources[name]['card'].set_status(status)

    def set_file_info(self, rows: int, columns: list, start_time: str = None, end_time: str = None):
        """Update file info display"""
        self._total_points = rows
        self.rows_label.setText(f"Rows: {rows:,}")
        self.cols_label.setText(f"Channels: {len(columns)}")

        if start_time and end_time:
            self.time_range_label.setText(f"Time Range: {start_time} to {end_time}")

        # Update table columns
        self.preview_table.setColumnCount(len(columns))
        self.preview_table.setHorizontalHeaderLabels(columns)

    def set_preview_data(self, rows: list):
        """Set preview table data"""
        self.preview_table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)

    def get_source_config(self, name: str) -> dict:
        """Get configuration for a source"""
        if name in self._sources:
            return self._sources[name].get('config', {})
        return {}

    def add_source_card(self, name: str, source_type: str, details: str, status: str = "Disconnected"):
        """Add a source card programmatically (e.g., from wizard or other sources)"""
        # Check if source already exists
        if name in self._sources:
            # Update existing source status
            self._sources[name]['card'].set_status(status)
            return

        # Hide placeholder
        self.sources_placeholder.hide()

        # Create card
        card = DataSourceCard(name, source_type, status, details)
        card.connect_clicked.connect(lambda n: self.source_connected.emit(n))
        card.disconnect_clicked.connect(lambda n: self.source_disconnected.emit(n))
        card.delete_clicked.connect(self._remove_source)

        # Determine config based on type
        config = {'name': name}
        if source_type == "CSV File":
            config['type'] = 'csv'
            config['path'] = details
        elif source_type == "Sample Data" or source_type == "NASA Sample Data":
            config['type'] = 'sample'
            config['sample_dataset'] = details
            config['path'] = details
        elif source_type == "REST API":
            config['type'] = 'api'
            config['url'] = details

        self._sources[name] = {'card': card, 'config': config}
        self.sources_container.insertWidget(self.sources_container.count() - 1, card)

    def get_all_sources(self) -> list:
        """Get list of all source names"""
        return list(self._sources.keys())

    def has_source(self, name: str) -> bool:
        """Check if a source exists"""
        return name in self._sources
