"""
Setup Wizard

Step-by-step guided configuration for new users.
Walks through data connection, model setup, and alarm configuration.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QComboBox, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QStackedWidget,
    QCheckBox, QListWidget, QListWidgetItem, QFileDialog,
    QProgressBar, QTextEdit, QWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from ..styles import COLORS, get_auto_contrast_color


class WizardStep(QFrame):
    """Base class for wizard steps"""

    completed = pyqtSignal(bool)  # Emitted when step validation changes

    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        self.step_title = title
        self.step_description = description
        self._setup_base_ui()

    def _setup_base_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background']};
            }}
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(24)

        # Title
        title_label = QLabel(self.step_title)
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        self.main_layout.addWidget(title_label)

        # Description
        desc_label = QLabel(self.step_description)
        desc_label.setStyleSheet(f"""
            font-size: 14px;
            color: {COLORS['text_secondary']};
            line-height: 1.5;
        """)
        desc_label.setWordWrap(True)
        self.main_layout.addWidget(desc_label)

        # Content area (to be filled by subclasses)
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(0, 16, 0, 0)
        self.main_layout.addWidget(self.content_frame)

        self.main_layout.addStretch()

    def is_valid(self) -> bool:
        """Check if step is complete and valid"""
        return True

    def get_data(self) -> dict:
        """Get configuration data from this step"""
        return {}


class DataSourceStep(WizardStep):
    """Step 1: Configure data source"""

    # Sample APIs for testing
    SAMPLE_APIS = {
        "-- Select sample API --": "",
        "USGS Earthquake Data (GeoJSON)": "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=10",
    }

    def __init__(self, parent=None):
        super().__init__(
            "Connect Your Data",
            "Choose how you want to feed data into MachineIQ. You can upload CSV files for testing or connect to a live data source for real-time monitoring.",
            parent
        )
        self._discovered_channels = []
        self._setup_content()

    def _setup_content(self):
        # Source type selection
        type_group = QGroupBox("Data Source Type")
        type_layout = QVBoxLayout(type_group)
        type_layout.setSpacing(12)

        self.source_type = QComboBox()
        self.source_type.addItems([
            "CSV File Upload",
            "REST API",
            "MQTT Broker",
            "OPC-UA Server"
        ])
        self.source_type.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.source_type)

        self.content_layout.addWidget(type_group)

        # CSV configuration
        self.csv_group = QGroupBox("CSV Configuration")
        csv_layout = QFormLayout(self.csv_group)

        file_row = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a CSV file...")
        self.file_path.textChanged.connect(lambda: self.completed.emit(self.is_valid()))
        file_row.addWidget(self.file_path)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)

        csv_layout.addRow("File:", file_row)

        self.timestamp_col = QComboBox()
        self.timestamp_col.setEnabled(False)
        csv_layout.addRow("Timestamp Column:", self.timestamp_col)

        self.content_layout.addWidget(self.csv_group)

        # API configuration
        self.api_group = QGroupBox("API Configuration")
        api_layout = QVBoxLayout(self.api_group)

        # Sample APIs dropdown
        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Try a sample API:"))
        self.sample_api_combo = QComboBox()
        self.sample_api_combo.addItems(list(self.SAMPLE_APIS.keys()))
        self.sample_api_combo.currentTextChanged.connect(self._on_sample_selected)
        sample_row.addWidget(self.sample_api_combo)
        api_layout.addLayout(sample_row)

        # URL input
        url_form = QFormLayout()
        self.api_url = QLineEdit()
        self.api_url.setPlaceholderText("https://api.example.com/data")
        self.api_url.textChanged.connect(lambda: self.completed.emit(self.is_valid()))
        url_form.addRow("Endpoint URL:", self.api_url)

        self.api_interval = QSpinBox()
        self.api_interval.setRange(1, 3600)
        self.api_interval.setValue(5)
        self.api_interval.setSuffix(" seconds")
        url_form.addRow("Poll Interval:", self.api_interval)
        api_layout.addLayout(url_form)

        # Test connection button
        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_api_connection)
        test_row.addWidget(self.test_btn)

        self.test_status = QLabel("")
        self.test_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        test_row.addWidget(self.test_status)
        test_row.addStretch()
        api_layout.addLayout(test_row)

        # Discovered channels
        self.channels_label = QLabel("")
        self.channels_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            padding: 8px;
            background-color: {COLORS['surface_secondary']};
            border-radius: 4px;
        """)
        self.channels_label.setWordWrap(True)
        self.channels_label.hide()
        api_layout.addWidget(self.channels_label)

        # Help text
        api_help = QLabel(
            "REST APIs should return JSON with numeric values. "
            "Supports flat JSON objects, arrays, and GeoJSON formats."
        )
        api_help.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 11px;")
        api_help.setWordWrap(True)
        api_layout.addWidget(api_help)

        self.api_group.hide()
        self.content_layout.addWidget(self.api_group)

        # MQTT configuration
        self.mqtt_group = QGroupBox("MQTT Configuration")
        mqtt_layout = QFormLayout(self.mqtt_group)

        self.mqtt_broker = QLineEdit()
        self.mqtt_broker.setPlaceholderText("broker.example.com")
        self.mqtt_broker.textChanged.connect(lambda: self.completed.emit(self.is_valid()))
        mqtt_layout.addRow("Broker:", self.mqtt_broker)

        self.mqtt_port = QSpinBox()
        self.mqtt_port.setRange(1, 65535)
        self.mqtt_port.setValue(1883)
        mqtt_layout.addRow("Port:", self.mqtt_port)

        self.mqtt_topic = QLineEdit()
        self.mqtt_topic.setPlaceholderText("sensors/+/data")
        mqtt_layout.addRow("Topic:", self.mqtt_topic)

        self.mqtt_group.hide()
        self.content_layout.addWidget(self.mqtt_group)

        # OPC-UA configuration
        self.opcua_group = QGroupBox("OPC-UA Configuration")
        opcua_layout = QFormLayout(self.opcua_group)

        self.opcua_endpoint = QLineEdit()
        self.opcua_endpoint.setPlaceholderText("opc.tcp://localhost:4840")
        self.opcua_endpoint.textChanged.connect(lambda: self.completed.emit(self.is_valid()))
        opcua_layout.addRow("Endpoint:", self.opcua_endpoint)

        self.opcua_group.hide()
        self.content_layout.addWidget(self.opcua_group)

    def _on_type_changed(self, type_text: str):
        self.csv_group.setVisible("CSV" in type_text)
        self.api_group.setVisible("REST" in type_text)
        self.mqtt_group.setVisible("MQTT" in type_text)
        self.opcua_group.setVisible("OPC" in type_text)
        self.completed.emit(self.is_valid())

    def _on_sample_selected(self, sample_name: str):
        """Handle sample API selection"""
        url = self.SAMPLE_APIS.get(sample_name, "")
        if url:
            self.api_url.setText(url)

    def _test_api_connection(self):
        """Test API connection and discover channels"""
        url = self.api_url.text().strip()
        if not url:
            self.test_status.setText("Enter a URL first")
            self.test_status.setStyleSheet(f"color: {COLORS['warning']};")
            return

        self.test_status.setText("Testing...")
        self.test_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.test_btn.setEnabled(False)

        # Force UI update
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from ..core import APIDataSource
            source = APIDataSource("test", url)
            success, channels, message = source.test_connection()

            if success and channels:
                self._discovered_channels = channels
                self.test_status.setText(f"Connected - {len(channels)} channels found")
                self.test_status.setStyleSheet(f"color: {COLORS['success']};")
                self.channels_label.setText(f"Channels: {', '.join(channels[:10])}" +
                    (f" (+{len(channels)-10} more)" if len(channels) > 10 else ""))
                self.channels_label.show()
            elif success:
                self.test_status.setText("Connected but no numeric data found")
                self.test_status.setStyleSheet(f"color: {COLORS['warning']};")
                self.channels_label.hide()
            else:
                self.test_status.setText(f"Failed: {message[:50]}")
                self.test_status.setStyleSheet(f"color: {COLORS['danger']};")
                self.channels_label.hide()

        except Exception as e:
            self.test_status.setText(f"Error: {str(e)[:50]}")
            self.test_status.setStyleSheet(f"color: {COLORS['danger']};")
            self.channels_label.hide()

        self.test_btn.setEnabled(True)
        self.completed.emit(self.is_valid())

    def get_discovered_channels(self) -> list:
        """Return channels discovered from API test"""
        return self._discovered_channels

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self._load_csv_columns(file_path)

    def _load_csv_columns(self, file_path: str):
        try:
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                self.timestamp_col.clear()
                self.timestamp_col.addItems(headers)
                self.timestamp_col.setEnabled(True)
        except Exception:
            pass

    def is_valid(self) -> bool:
        source_type = self.source_type.currentText()
        if "CSV" in source_type:
            return bool(self.file_path.text().strip())
        elif "REST" in source_type:
            return bool(self.api_url.text().strip())
        elif "MQTT" in source_type:
            return bool(self.mqtt_broker.text().strip())
        elif "OPC" in source_type:
            return bool(self.opcua_endpoint.text().strip())
        return False

    def get_data(self) -> dict:
        source_type = self.source_type.currentText()
        data = {'source_type': source_type}

        if "CSV" in source_type:
            data['file_path'] = self.file_path.text()
            data['timestamp_column'] = self.timestamp_col.currentText()
        elif "REST" in source_type:
            data['url'] = self.api_url.text()
            data['interval'] = self.api_interval.value()
        elif "MQTT" in source_type:
            data['broker'] = self.mqtt_broker.text()
            data['port'] = self.mqtt_port.value()
            data['topic'] = self.mqtt_topic.text()
        elif "OPC" in source_type:
            data['endpoint'] = self.opcua_endpoint.text()

        return data


class ModelConfigStep(WizardStep):
    """Step 2: Configure ML model"""

    def __init__(self, parent=None):
        super().__init__(
            "Configure Detection Model",
            "Set up the anomaly detection parameters. The defaults work well for most industrial applications. Adjust if needed for your specific use case.",
            parent
        )
        self._setup_content()

    def _setup_content(self):
        # Kernel settings
        kernel_group = QGroupBox("Detection Kernel")
        kernel_layout = QFormLayout(kernel_group)
        kernel_layout.setSpacing(12)

        self.kernel_type = QComboBox()
        self.kernel_type.addItems(["Triangular", "Parabolic"])
        kernel_layout.addRow("Kernel Type:", self.kernel_type)

        self.num_bins = QSpinBox()
        self.num_bins.setRange(10, 500)
        self.num_bins.setMinimumWidth(120)
        self.num_bins.setValue(100)
        kernel_layout.addRow("Number of Bins:", self.num_bins)

        self.kernel_width = QDoubleSpinBox()
        self.kernel_width.setRange(0.01, 1.0)
        self.kernel_width.setMinimumWidth(120)
        self.kernel_width.setValue(0.1)
        self.kernel_width.setSingleStep(0.01)
        kernel_layout.addRow("Kernel Width:", self.kernel_width)

        self.content_layout.addWidget(kernel_group)

        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout(detection_group)
        detection_layout.setSpacing(12)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(1, 100)
        self.threshold.setMinimumWidth(120)
        self.threshold.setValue(30)
        self.threshold.setSuffix(" %")
        detection_layout.addRow("Alert Threshold:", self.threshold)

        self.window_size = QSpinBox()
        self.window_size.setRange(1, 1000)
        self.window_size.setMinimumWidth(120)
        self.window_size.setValue(50)
        self.window_size.setSuffix(" samples")
        detection_layout.addRow("Window Size:", self.window_size)

        self.content_layout.addWidget(detection_group)

        # Help text
        help_label = QLabel(
            "Tip: Lower thresholds are more sensitive but may produce more false alarms. "
            "Start with the default of 30% and adjust based on your results."
        )
        help_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 12px;
            padding: 12px;
            background-color: {COLORS['surface_secondary']};
            border-radius: 6px;
        """)
        help_label.setWordWrap(True)
        self.content_layout.addWidget(help_label)

    def get_data(self) -> dict:
        return {
            'kernel_type': self.kernel_type.currentText().lower(),
            'num_bins': self.num_bins.value(),
            'kernel_width': self.kernel_width.value(),
            'threshold': self.threshold.value() / 100,
            'window_size': self.window_size.value()
        }


class ChannelGroupStep(WizardStep):
    """Step 3: Configure channel groups"""

    def __init__(self, parent=None):
        super().__init__(
            "Group Your Channels",
            "Organize channels into groups for multivariate analysis. Channels in the same group will be analyzed together to detect correlated anomalies.",
            parent
        )
        self._groups = []
        self._setup_content()

    def _setup_content(self):
        # Available channels (will be populated from data source)
        channels_group = QGroupBox("Available Channels")
        channels_layout = QVBoxLayout(channels_group)

        self.channels_list = QListWidget()
        self.channels_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.channels_list.setMinimumHeight(300)
        channels_layout.addWidget(self.channels_list)

        self.content_layout.addWidget(channels_group)

        # Group creation
        create_group = QGroupBox("Create Channel Group")
        create_layout = QVBoxLayout(create_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Group Name:"))
        self.group_name = QLineEdit()
        self.group_name.setPlaceholderText("e.g., Motor Bearings")
        name_row.addWidget(self.group_name)
        create_layout.addLayout(name_row)

        create_btn = QPushButton("Create Group from Selected")
        create_btn.clicked.connect(self._create_group)
        create_layout.addWidget(create_btn)

        self.content_layout.addWidget(create_group)

        # Created groups
        groups_group = QGroupBox("Channel Groups")
        groups_layout = QVBoxLayout(groups_group)

        self.groups_list = QListWidget()
        self.groups_list = QListWidget()
        self.groups_list.setMinimumHeight(150)
        groups_layout.addWidget(self.groups_list)

        remove_btn = QPushButton("Remove Selected Group")
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
        """)
        remove_btn.clicked.connect(self._remove_group)
        groups_layout.addWidget(remove_btn)

        self.content_layout.addWidget(groups_group)

    def set_channels(self, channels: list):
        """Set available channels from data source"""
        self.channels_list.clear()
        for channel in channels:
            item = QListWidgetItem(channel)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.channels_list.addItem(item)

    def _create_group(self):
        name = self.group_name.text().strip()
        if not name:
            return

        selected = [item.text() for item in self.channels_list.selectedItems()]
        if not selected:
            return

        self._groups.append({
            'name': name,
            'channels': selected
        })

        self.groups_list.addItem(f"{name}: {', '.join(selected)}")
        self.group_name.clear()
        self.channels_list.clearSelection()
        self.completed.emit(self.is_valid())

    def _remove_group(self):
        current = self.groups_list.currentRow()
        if current >= 0:
            self.groups_list.takeItem(current)
            del self._groups[current]
            self.completed.emit(self.is_valid())

    def is_valid(self) -> bool:
        return len(self._groups) > 0

    def get_data(self) -> dict:
        return {'groups': self._groups}


class AlarmConfigStep(WizardStep):
    """Step 4: Configure alarms and notifications"""

    def __init__(self, parent=None):
        super().__init__(
            "Set Up Alerts",
            "Configure when to trigger alerts and who should be notified. You can add multiple email recipients for alarm notifications.",
            parent
        )
        self._setup_content()

    def _setup_content(self):
        # Alarm conditions
        conditions_group = QGroupBox("Alert Conditions")
        conditions_layout = QVBoxLayout(conditions_group)

        self.alert_critical = QCheckBox("Critical: Match score below 15%")
        self.alert_critical.setChecked(True)
        conditions_layout.addWidget(self.alert_critical)

        self.alert_warning = QCheckBox("Warning: Match score below 30%")
        self.alert_warning.setChecked(True)
        conditions_layout.addWidget(self.alert_warning)

        self.alert_info = QCheckBox("Info: Match score below 50%")
        conditions_layout.addWidget(self.alert_info)

        self.content_layout.addWidget(conditions_group)

        # Email configuration
        email_group = QGroupBox("Email Notifications")
        email_layout = QFormLayout(email_group)
        email_layout.setSpacing(12)

        self.email_enabled = QCheckBox("Enable email notifications")
        email_layout.addRow("", self.email_enabled)

        self.smtp_server = QLineEdit()
        self.smtp_server.setPlaceholderText("smtp.gmail.com")
        email_layout.addRow("SMTP Server:", self.smtp_server)

        self.smtp_port = QSpinBox()
        self.smtp_port.setRange(1, 65535)
        self.smtp_port.setValue(587)
        email_layout.addRow("Port:", self.smtp_port)

        self.smtp_user = QLineEdit()
        self.smtp_user.setPlaceholderText("your@email.com")
        email_layout.addRow("Username:", self.smtp_user)

        self.smtp_pass = QLineEdit()
        self.smtp_pass.setEchoMode(QLineEdit.EchoMode.Password)
        email_layout.addRow("Password:", self.smtp_pass)

        self.content_layout.addWidget(email_group)

        email_layout.setContentsMargins(12, 20, 12, 12)
        email_layout.setVerticalSpacing(16)

        # Recipients
        recipients_group = QGroupBox("Notification Recipients")
        recipients_layout = QVBoxLayout(recipients_group)

        self.recipients_list = QListWidget()
        self.recipients_list = QListWidget()
        self.recipients_list.setMinimumHeight(120)
        recipients_layout.addWidget(self.recipients_list)

        add_row = QHBoxLayout()
        self.new_email = QLineEdit()
        self.new_email.setPlaceholderText("email@example.com")
        add_row.addWidget(self.new_email)

        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(60)
        add_btn.clicked.connect(self._add_recipient)
        add_row.addWidget(add_btn)

        recipients_layout.addLayout(add_row)
        self.content_layout.addWidget(recipients_group)

    def _add_recipient(self):
        email = self.new_email.text().strip()
        if email and '@' in email:
            self.recipients_list.addItem(email)
            self.new_email.clear()

    def get_data(self) -> dict:
        recipients = []
        for i in range(self.recipients_list.count()):
            recipients.append(self.recipients_list.item(i).text())

        conditions = []
        if self.alert_critical.isChecked():
            conditions.append({'type': 'MATCH_BELOW', 'threshold': 0.15, 'severity': 'CRITICAL'})
        if self.alert_warning.isChecked():
            conditions.append({'type': 'MATCH_BELOW', 'threshold': 0.30, 'severity': 'WARNING'})
        if self.alert_info.isChecked():
            conditions.append({'type': 'MATCH_BELOW', 'threshold': 0.50, 'severity': 'INFO'})

        return {
            'conditions': conditions,
            'email': {
                'enabled': self.email_enabled.isChecked(),
                'smtp_server': self.smtp_server.text(),
                'smtp_port': self.smtp_port.value(),
                'username': self.smtp_user.text(),
                'password': self.smtp_pass.text(),
                'recipients': recipients
            }
        }


class ReviewStep(WizardStep):
    """Step 5: Review and confirm"""

    def __init__(self, parent=None):
        super().__init__(
            "Review Configuration",
            "Review your settings before starting monitoring. You can go back to make changes if needed.",
            parent
        )
        self._setup_content()

    def _setup_content(self):
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 16px;
                color: {COLORS['text_primary']};
                font-family: monospace;
            }}
        """)
        self.content_layout.addWidget(self.summary_text)

    def set_summary(self, config: dict):
        """Set the summary text from configuration"""
        lines = []
        lines.append("=== Data Source ===")
        if 'data_source' in config:
            ds = config['data_source']
            lines.append(f"Type: {ds.get('source_type', 'Unknown')}")
            if 'file_path' in ds:
                lines.append(f"File: {ds['file_path']}")
            if 'url' in ds:
                lines.append(f"URL: {ds['url']}")

        lines.append("\n=== Detection Model ===")
        if 'model' in config:
            model = config['model']
            lines.append(f"Kernel: {model.get('kernel_type', 'triangular').title()}")
            lines.append(f"Bins: {model.get('num_bins', 100)}")
            lines.append(f"Threshold: {model.get('threshold', 0.3) * 100:.0f}%")

        lines.append("\n=== Channel Groups ===")
        if 'channels' in config:
            groups = config['channels'].get('groups', [])
            for g in groups:
                lines.append(f"- {g['name']}: {', '.join(g['channels'])}")
            if not groups:
                lines.append("No groups configured")

        lines.append("\n=== Alerts ===")
        if 'alarms' in config:
            alarms = config['alarms']
            conditions = alarms.get('conditions', [])
            for c in conditions:
                lines.append(f"- {c['severity']}: Below {c['threshold']*100:.0f}%")

            email = alarms.get('email', {})
            if email.get('enabled'):
                lines.append(f"\nEmail notifications: Enabled")
                lines.append(f"Recipients: {', '.join(email.get('recipients', []))}")
            else:
                lines.append(f"\nEmail notifications: Disabled")

        self.summary_text.setText('\n'.join(lines))


class SetupWizard(QDialog):
    """
    Multi-step setup wizard for new users.

    Guides through:
    1. Data source configuration
    2. Model parameters
    3. Channel grouping
    4. Alarm setup
    5. Review and confirm
    """

    wizard_completed = pyqtSignal(dict)  # Emitted with full configuration

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MachineIQ Setup Wizard")
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self._config = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
            }}
            QGroupBox {{
                font-weight: 600;
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding: 16px;
                padding-top: 24px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                padding: 6px 8px;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                background-color: {COLORS['surface']};
                color: {get_auto_contrast_color(COLORS['surface'])};
                min-height: 20px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QPushButton {{
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 600;
                background-color: {COLORS['primary']};
                color: {COLORS['text_inverse']};
                border: none;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton.secondary {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Progress indicator
        progress_frame = QFrame()
        progress_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        progress_layout = QHBoxLayout(progress_frame)
        progress_layout.setContentsMargins(40, 20, 40, 20)

        self.step_labels = []
        step_names = ["Data Source", "Model", "Channels", "Alerts", "Review"]
        for i, name in enumerate(step_names):
            step_widget = QWidget()
            step_layout = QHBoxLayout(step_widget)
            step_layout.setContentsMargins(0, 0, 0, 0)
            step_layout.setSpacing(8)

            # Step number circle
            num_label = QLabel(str(i + 1))
            num_label.setFixedSize(28, 28)
            num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_label.setStyleSheet(f"""
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
                border-radius: 14px;
                font-weight: 600;
            """)
            step_layout.addWidget(num_label)

            # Step name
            name_label = QLabel(name)
            name_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            step_layout.addWidget(name_label)

            self.step_labels.append((num_label, name_label))
            progress_layout.addWidget(step_widget)

            if i < len(step_names) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet(f"background-color: {COLORS['border']};")
                line.setFixedWidth(40)
                progress_layout.addWidget(line)

        layout.addWidget(progress_frame)

        # Step pages
        self.page_stack = QStackedWidget()

        self.data_step = DataSourceStep()
        self.model_step = ModelConfigStep()
        self.channel_step = ChannelGroupStep()
        self.alarm_step = AlarmConfigStep()
        self.review_step = ReviewStep()

        self.page_stack.addWidget(self.data_step)
        self.page_stack.addWidget(self.model_step)
        self.page_stack.addWidget(self.channel_step)
        self.page_stack.addWidget(self.alarm_step)
        self.page_stack.addWidget(self.review_step)

        layout.addWidget(self.page_stack)

        # Navigation buttons
        nav_frame = QFrame()
        nav_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(40, 16, 40, 16)

        self.back_btn = QPushButton("Back")
        self.back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_secondary']};
            }}
        """)
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)

        nav_layout.addStretch()

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_tertiary']};
                border: none;
            }}
            QPushButton:hover {{
                color: {COLORS['text_secondary']};
            }}
        """)
        self.skip_btn.clicked.connect(self._skip_step)
        nav_layout.addWidget(self.skip_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addWidget(nav_frame)

        # Initialize
        self._update_navigation()

    def _update_navigation(self):
        """Update navigation buttons and progress indicator"""
        current = self.page_stack.currentIndex()
        total = self.page_stack.count()

        # Back button
        self.back_btn.setVisible(current > 0)

        # Skip button (not on last step)
        self.skip_btn.setVisible(current < total - 1 and current != 0)

        # Next button text
        if current == total - 1:
            self.next_btn.setText("Start Monitoring")
        else:
            self.next_btn.setText("Next")

        # Update progress indicator
        for i, (num_label, name_label) in enumerate(self.step_labels):
            if i < current:
                # Completed step
                num_label.setStyleSheet(f"""
                    background-color: {COLORS['success']};
                    color: white;
                    border-radius: 14px;
                    font-weight: 600;
                """)
                name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            elif i == current:
                # Current step
                num_label.setStyleSheet(f"""
                    background-color: {COLORS['primary']};
                    color: white;
                    border-radius: 14px;
                    font-weight: 600;
                """)
                name_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600;")
            else:
                # Future step
                num_label.setStyleSheet(f"""
                    background-color: {COLORS['border']};
                    color: {COLORS['text_secondary']};
                    border-radius: 14px;
                    font-weight: 600;
                """)
                name_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

    def _go_back(self):
        """Go to previous step"""
        current = self.page_stack.currentIndex()
        if current > 0:
            self.page_stack.setCurrentIndex(current - 1)
            self._update_navigation()

    def _go_next(self):
        """Go to next step or finish"""
        current = self.page_stack.currentIndex()
        total = self.page_stack.count()

        # Collect data from current step
        self._collect_step_data(current)

        if current == total - 1:
            # Finish wizard
            self.wizard_completed.emit(self._config)
            self.accept()
        else:
            # Prepare next step
            if current == 0:
                # After data source, populate channels
                self._populate_channels()
            elif current == 3:
                # Before review, update summary
                self.review_step.set_summary(self._config)

            self.page_stack.setCurrentIndex(current + 1)
            self._update_navigation()

    def _skip_step(self):
        """Skip current step"""
        current = self.page_stack.currentIndex()
        total = self.page_stack.count()

        if current < total - 1:
            self.page_stack.setCurrentIndex(current + 1)
            self._update_navigation()

    def _collect_step_data(self, step_index: int):
        """Collect data from a step"""
        steps = [
            ('data_source', self.data_step),
            ('model', self.model_step),
            ('channels', self.channel_step),
            ('alarms', self.alarm_step),
        ]

        if step_index < len(steps):
            key, step = steps[step_index]
            self._config[key] = step.get_data()

    def _populate_channels(self):
        """Populate channel list from data source"""
        data_source = self._config.get('data_source', {})
        source_type = data_source.get('source_type', '')
        channels = []

        # Try to get channels from CSV file
        if 'CSV' in source_type:
            file_path = data_source.get('file_path')
            if file_path:
                try:
                    import csv
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        ts_col = data_source.get('timestamp_column', '')
                        channels = [h for h in headers if h != ts_col]
                except Exception:
                    pass

        # Try to get channels from API test
        elif 'REST' in source_type:
            channels = self.data_step.get_discovered_channels()

        # Fallback to defaults if no channels found
        if not channels:
            channels = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]

        self.channel_step.set_channels(channels)

    def get_configuration(self) -> dict:
        """Get the complete configuration"""
        return self._config
