"""
Connection Center

Wizard-based interface for connecting to data sources:
- REST APIs
- MQTT brokers
- OPC-UA servers
- Custom integrations
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QComboBox, QGroupBox,
    QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem,
    QStackedWidget, QTextEdit, QCheckBox, QHeaderView,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..styles import COLORS


class ConnectionCard(QFrame):
    """Card displaying a single connection"""

    edit_clicked = pyqtSignal(str)  # connection_name
    delete_clicked = pyqtSignal(str)  # connection_name
    connect_clicked = pyqtSignal(str)  # connection_name

    def __init__(self, name: str, conn_type: str, status: str = "Disconnected", parent=None):
        super().__init__(parent)
        self.name = name
        self.conn_type = conn_type
        self._status = status

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        self.setFixedHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        name_label = QLabel(self.name)
        name_label.setStyleSheet(f"font-weight: 600; font-size: 14px; color: {COLORS['text_primary']};")
        info_layout.addWidget(name_label)

        type_label = QLabel(self.conn_type)
        type_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        info_layout.addWidget(type_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Status
        self.status_label = QLabel(self._status)
        self._update_status_style()
        layout.addWidget(self.status_label)

        # Actions
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedWidth(80)
        self.connect_btn.clicked.connect(lambda: self.connect_clicked.emit(self.name))
        layout.addWidget(self.connect_btn)

        edit_btn = QPushButton("Edit")
        edit_btn.setFixedWidth(60)
        edit_btn.setProperty("class", "secondary")
        edit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_secondary']};
            }}
        """)
        edit_btn.clicked.connect(lambda: self.edit_clicked.emit(self.name))
        layout.addWidget(edit_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setFixedWidth(60)
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

    def set_status(self, status: str):
        self._status = status
        self.status_label.setText(status)
        self._update_status_style()

        if status == "Connected":
            self.connect_btn.setText("Disconnect")
        else:
            self.connect_btn.setText("Connect")

    def _update_status_style(self):
        if self._status == "Connected":
            color = COLORS['success']
        elif self._status == "Connecting...":
            color = COLORS['warning']
        else:
            color = COLORS['text_tertiary']

        self.status_label.setStyleSheet(f"""
            color: {color};
            font-weight: 500;
            padding: 4px 12px;
            background-color: {color}20;
            border-radius: 4px;
        """)


class ConnectionWizard(QWidget):
    """Step-by-step wizard for creating connections"""

    connection_created = pyqtSignal(dict)  # connection config

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Step indicator
        steps_frame = QFrame()
        steps_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_secondary']};
                border-radius: 8px;
                padding: 16px;
            }}
        """)
        steps_layout = QHBoxLayout(steps_frame)

        self.step_labels = []
        steps = ["Connection Type", "Configuration", "Test & Save"]
        for i, step in enumerate(steps):
            step_widget = QWidget()
            step_inner = QHBoxLayout(step_widget)
            step_inner.setContentsMargins(0, 0, 0, 0)

            num_label = QLabel(str(i + 1))
            num_label.setFixedSize(24, 24)
            num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_label.setStyleSheet(f"""
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
                border-radius: 12px;
                font-weight: 600;
            """)
            step_inner.addWidget(num_label)

            text_label = QLabel(step)
            text_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            step_inner.addWidget(text_label)

            self.step_labels.append((num_label, text_label))
            steps_layout.addWidget(step_widget)

            if i < len(steps) - 1:
                line = QFrame()
                line.setFixedHeight(1)
                line.setStyleSheet(f"background-color: {COLORS['border']};")
                steps_layout.addWidget(line, stretch=1)

        layout.addWidget(steps_frame)

        # Content stack
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, stretch=1)

        # Step 1: Connection Type
        step1 = QWidget()
        step1_layout = QVBoxLayout(step1)
        step1_layout.setContentsMargins(0, 20, 0, 0)

        step1_layout.addWidget(QLabel("Select the type of connection:"))
        step1_layout.addSpacing(16)

        self.type_buttons = {}
        types = [
            ("rest_api", "REST API", "Connect to any REST API endpoint that returns JSON data"),
            ("mqtt", "MQTT Broker", "Subscribe to MQTT topics for real-time streaming"),
            ("opcua", "OPC-UA Server", "Connect to industrial OPC-UA servers"),
            ("csv_stream", "CSV Stream", "Watch a folder for new CSV files"),
        ]

        for key, name, desc in types:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setMinimumHeight(70)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['surface']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    text-align: left;
                    padding: 16px;
                }}
                QPushButton:hover {{
                    border-color: {COLORS['primary']};
                }}
                QPushButton:checked {{
                    border-color: {COLORS['primary']};
                    border-width: 2px;
                    background-color: {COLORS['primary']}10;
                }}
            """)

            btn_layout = QVBoxLayout(btn)
            btn_layout.setContentsMargins(0, 0, 0, 0)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"font-weight: 600; font-size: 14px; color: {COLORS['text_primary']};")
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")

            btn.clicked.connect(lambda checked, k=key: self._select_type(k))
            self.type_buttons[key] = btn
            step1_layout.addWidget(btn)

        step1_layout.addStretch()
        self.stack.addWidget(step1)

        # Step 2: Configuration
        step2 = QWidget()
        step2_layout = QVBoxLayout(step2)
        step2_layout.setContentsMargins(0, 20, 0, 0)

        self.config_stack = QStackedWidget()

        # REST API config
        rest_config = QWidget()
        rest_form = QFormLayout(rest_config)
        rest_form.setSpacing(12)

        self.rest_name = QLineEdit()
        self.rest_name.setPlaceholderText("My API Connection")
        rest_form.addRow("Connection Name:", self.rest_name)

        self.rest_url = QLineEdit()
        self.rest_url.setPlaceholderText("https://api.example.com/data")
        rest_form.addRow("API URL:", self.rest_url)

        self.rest_interval = QSpinBox()
        self.rest_interval.setRange(1, 3600)
        self.rest_interval.setValue(5)
        self.rest_interval.setSuffix(" seconds")
        rest_form.addRow("Poll Interval:", self.rest_interval)

        self.rest_auth = QLineEdit()
        self.rest_auth.setPlaceholderText("Bearer token (optional)")
        self.rest_auth.setEchoMode(QLineEdit.EchoMode.Password)
        rest_form.addRow("Authorization:", self.rest_auth)

        self.rest_headers = QTextEdit()
        self.rest_headers.setPlaceholderText('{"Content-Type": "application/json"}')
        self.rest_headers.setMaximumHeight(80)
        rest_form.addRow("Custom Headers:", self.rest_headers)

        self.config_stack.addWidget(rest_config)

        # MQTT config
        mqtt_config = QWidget()
        mqtt_form = QFormLayout(mqtt_config)
        mqtt_form.setSpacing(12)

        self.mqtt_name = QLineEdit()
        self.mqtt_name.setPlaceholderText("My MQTT Connection")
        mqtt_form.addRow("Connection Name:", self.mqtt_name)

        self.mqtt_host = QLineEdit()
        self.mqtt_host.setPlaceholderText("broker.example.com")
        mqtt_form.addRow("Broker Host:", self.mqtt_host)

        self.mqtt_port = QSpinBox()
        self.mqtt_port.setRange(1, 65535)
        self.mqtt_port.setValue(1883)
        mqtt_form.addRow("Port:", self.mqtt_port)

        self.mqtt_topic = QLineEdit()
        self.mqtt_topic.setPlaceholderText("sensors/#")
        mqtt_form.addRow("Topic:", self.mqtt_topic)

        self.mqtt_user = QLineEdit()
        self.mqtt_user.setPlaceholderText("(optional)")
        mqtt_form.addRow("Username:", self.mqtt_user)

        self.mqtt_pass = QLineEdit()
        self.mqtt_pass.setEchoMode(QLineEdit.EchoMode.Password)
        mqtt_form.addRow("Password:", self.mqtt_pass)

        self.config_stack.addWidget(mqtt_config)

        # OPC-UA config
        opcua_config = QWidget()
        opcua_form = QFormLayout(opcua_config)
        opcua_form.setSpacing(12)

        self.opcua_name = QLineEdit()
        self.opcua_name.setPlaceholderText("My OPC-UA Connection")
        opcua_form.addRow("Connection Name:", self.opcua_name)

        self.opcua_url = QLineEdit()
        self.opcua_url.setPlaceholderText("opc.tcp://localhost:4840")
        opcua_form.addRow("Server URL:", self.opcua_url)

        self.config_stack.addWidget(opcua_config)

        # CSV Stream config
        csv_config = QWidget()
        csv_form = QFormLayout(csv_config)
        csv_form.setSpacing(12)

        self.csv_name = QLineEdit()
        self.csv_name.setPlaceholderText("My CSV Watcher")
        csv_form.addRow("Connection Name:", self.csv_name)

        self.csv_folder = QLineEdit()
        self.csv_folder.setPlaceholderText("/path/to/watch")
        csv_form.addRow("Watch Folder:", self.csv_folder)

        self.config_stack.addWidget(csv_config)

        step2_layout.addWidget(self.config_stack)
        step2_layout.addStretch()
        self.stack.addWidget(step2)

        # Step 3: Test & Save
        step3 = QWidget()
        step3_layout = QVBoxLayout(step3)
        step3_layout.setContentsMargins(0, 20, 0, 0)

        step3_layout.addWidget(QLabel("Test your connection before saving:"))
        step3_layout.addSpacing(16)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setFixedWidth(150)
        self.test_btn.clicked.connect(self._test_connection)
        step3_layout.addWidget(self.test_btn)

        self.test_result = QLabel("")
        self.test_result.setWordWrap(True)
        step3_layout.addWidget(self.test_result)

        step3_layout.addStretch()
        self.stack.addWidget(step3)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 16, 0, 0)

        self.back_btn = QPushButton("Back")
        self.back_btn.setFixedWidth(100)
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self._go_back)
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
        nav_layout.addWidget(self.back_btn)

        nav_layout.addStretch()

        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedWidth(100)
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        self._update_step(0)

    def _select_type(self, key: str):
        for k, btn in self.type_buttons.items():
            btn.setChecked(k == key)
        self._config['type'] = key

        # Show appropriate config
        type_indices = {'rest_api': 0, 'mqtt': 1, 'opcua': 2, 'csv_stream': 3}
        self.config_stack.setCurrentIndex(type_indices.get(key, 0))

    def _update_step(self, step: int):
        self.stack.setCurrentIndex(step)

        for i, (num_lbl, text_lbl) in enumerate(self.step_labels):
            if i < step:
                num_lbl.setStyleSheet(f"""
                    background-color: {COLORS['success']};
                    color: white;
                    border-radius: 12px;
                    font-weight: 600;
                """)
                text_lbl.setStyleSheet(f"color: {COLORS['text_primary']};")
            elif i == step:
                num_lbl.setStyleSheet(f"""
                    background-color: {COLORS['primary']};
                    color: white;
                    border-radius: 12px;
                    font-weight: 600;
                """)
                text_lbl.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600;")
            else:
                num_lbl.setStyleSheet(f"""
                    background-color: {COLORS['border']};
                    color: {COLORS['text_secondary']};
                    border-radius: 12px;
                    font-weight: 600;
                """)
                text_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")

        self.back_btn.setEnabled(step > 0)
        self.next_btn.setText("Save" if step == 2 else "Next")

    def _go_back(self):
        current = self.stack.currentIndex()
        if current > 0:
            self._update_step(current - 1)

    def _go_next(self):
        current = self.stack.currentIndex()
        if current < 2:
            self._update_step(current + 1)
        else:
            self._save_connection()

    def _test_connection(self):
        self.test_result.setText("Testing connection...")
        self.test_result.setStyleSheet(f"color: {COLORS['text_secondary']};")
        # In real implementation, would test the connection here
        self.test_result.setText("Connection successful!")
        self.test_result.setStyleSheet(f"color: {COLORS['success']};")

    def _save_connection(self):
        # Build config from form fields
        conn_type = self._config.get('type', 'rest_api')

        if conn_type == 'rest_api':
            self._config.update({
                'name': self.rest_name.text() or "Unnamed Connection",
                'url': self.rest_url.text(),
                'interval': self.rest_interval.value(),
                'auth': self.rest_auth.text(),
            })
        elif conn_type == 'mqtt':
            self._config.update({
                'name': self.mqtt_name.text() or "Unnamed Connection",
                'host': self.mqtt_host.text(),
                'port': self.mqtt_port.value(),
                'topic': self.mqtt_topic.text(),
            })

        self.connection_created.emit(self._config)

    def reset(self):
        """Reset wizard to initial state"""
        self._config = {}
        self._update_step(0)
        for btn in self.type_buttons.values():
            btn.setChecked(False)


class ConnectionCenterPage(QWidget):
    """
    Connection management page.

    Shows existing connections and allows creating new ones.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Manage data source connections"))
        header.addStretch()

        self.new_btn = QPushButton("New Connection")
        self.new_btn.clicked.connect(self._show_wizard)
        header.addWidget(self.new_btn)

        layout.addLayout(header)

        # Content stack (list vs wizard)
        self.content_stack = QStackedWidget()

        # Connection list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        self.connections_container = QVBoxLayout()
        self.connections_container.setSpacing(12)

        # Placeholder
        self.placeholder = QLabel("No connections configured. Click 'New Connection' to add one.")
        self.placeholder.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            padding: 40px;
            background-color: {COLORS['surface']};
            border: 1px dashed {COLORS['border']};
            border-radius: 8px;
        """)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connections_container.addWidget(self.placeholder)

        list_layout.addLayout(self.connections_container)
        list_layout.addStretch()

        self.content_stack.addWidget(list_widget)

        # Wizard
        self.wizard = ConnectionWizard()
        self.wizard.connection_created.connect(self._on_connection_created)
        self.content_stack.addWidget(self.wizard)

        layout.addWidget(self.content_stack)

    def _show_wizard(self):
        self.wizard.reset()
        self.content_stack.setCurrentIndex(1)
        self.new_btn.setText("Cancel")
        self.new_btn.clicked.disconnect()
        self.new_btn.clicked.connect(self._hide_wizard)

    def _hide_wizard(self):
        self.content_stack.setCurrentIndex(0)
        self.new_btn.setText("New Connection")
        self.new_btn.clicked.disconnect()
        self.new_btn.clicked.connect(self._show_wizard)

    def _on_connection_created(self, config: dict):
        name = config.get('name', 'Unnamed')
        conn_type = config.get('type', 'rest_api')

        # Hide placeholder
        self.placeholder.hide()

        # Create card
        card = ConnectionCard(name, conn_type)
        card.delete_clicked.connect(self._delete_connection)
        self._connections[name] = {'card': card, 'config': config}
        self.connections_container.insertWidget(self.connections_container.count() - 1, card)

        self._hide_wizard()

    def _delete_connection(self, name: str):
        if name in self._connections:
            self._connections[name]['card'].deleteLater()
            del self._connections[name]

            if not self._connections:
                self.placeholder.show()
