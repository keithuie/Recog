"""
Alarm Center Page

Configure alarm conditions and email notifications.
View and manage active alarms.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QComboBox, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QCheckBox, QListWidget,
    QListWidgetItem, QHeaderView, QSizePolicy, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..styles import COLORS
from ..core.alarm_manager import AlarmSeverity, AlarmConditionType


class AlarmConditionRow(QFrame):
    """Row widget for an alarm condition"""

    changed = pyqtSignal()
    delete_requested = pyqtSignal(str)  # condition name

    def __init__(self, name: str = "", parent=None):
        super().__init__(parent)
        self._name = name

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Enabled checkbox
        self.enabled_cb = QCheckBox()
        self.enabled_cb.setChecked(True)
        self.enabled_cb.stateChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.enabled_cb)

        # Name
        self.name_edit = QLineEdit(self._name)
        self.name_edit.setPlaceholderText("Condition name")
        self.name_edit.setMinimumWidth(150)
        self.name_edit.textChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.name_edit)

        # Condition type
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Match Below",
            "Match Above",
            "Channel Above",
            "Channel Below",
        ])
        self.type_combo.setMinimumWidth(120)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addWidget(self.type_combo)

        # Channel (for channel-specific conditions)
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(120)
        self.channel_combo.setVisible(False)
        self.channel_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.channel_combo)

        # Threshold
        layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setValue(85)
        self.threshold_spin.valueChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.threshold_spin)

        # Severity
        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["Info", "Warning", "Critical"])
        self.severity_combo.setCurrentText("Warning")
        self.severity_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.severity_combo)

        layout.addStretch()

        # Delete button
        delete_btn = QPushButton("Remove")
        delete_btn.setFixedWidth(70)
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['danger']};
                border: none;
            }}
            QPushButton:hover {{
                text-decoration: underline;
            }}
        """)
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._name))
        layout.addWidget(delete_btn)

    def _on_type_changed(self, type_text: str):
        # Show channel selector for channel-specific conditions
        is_channel_type = "Channel" in type_text
        self.channel_combo.setVisible(is_channel_type)
        self.changed.emit()

    def set_channels(self, channels: list):
        """Set available channels for channel-specific conditions"""
        self.channel_combo.clear()
        self.channel_combo.addItems(channels)

    def get_condition_data(self) -> dict:
        """Get condition configuration"""
        type_map = {
            "Match Below": "MATCH_BELOW",
            "Match Above": "MATCH_ABOVE",
            "Channel Above": "CHANNEL_ABOVE",
            "Channel Below": "CHANNEL_BELOW",
        }

        return {
            'name': self.name_edit.text(),
            'enabled': self.enabled_cb.isChecked(),
            'condition_type': type_map.get(self.type_combo.currentText(), "MATCH_BELOW"),
            'threshold': self.threshold_spin.value(),
            'channel': self.channel_combo.currentText() if self.channel_combo.isVisible() else None,
            'severity': self.severity_combo.currentText().upper(),
        }


class AlarmCenterPage(QWidget):
    """
    Alarm configuration and monitoring interface.

    Features:
    - Create/edit alarm conditions
    - Configure email notifications
    - View active alarms
    - Alarm history
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition_widgets = []
        self._available_channels = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Tabs
        tabs = QTabWidget()

        # Conditions Tab
        conditions_tab = QWidget()
        conditions_layout = QVBoxLayout(conditions_tab)
        conditions_layout.setContentsMargins(16, 16, 16, 16)

        # Header
        cond_header = QHBoxLayout()
        cond_header.addWidget(QLabel("Define conditions that will trigger alarms:"))
        cond_header.addStretch()

        self.add_condition_btn = QPushButton("Add Condition")
        self.add_condition_btn.clicked.connect(self._add_condition)
        cond_header.addWidget(self.add_condition_btn)

        conditions_layout.addLayout(cond_header)

        # Conditions container
        self.conditions_container = QVBoxLayout()
        self.conditions_container.setSpacing(8)
        conditions_layout.addLayout(self.conditions_container)

        # Default conditions hint
        self.cond_placeholder = QLabel("No conditions defined. Click 'Add Condition' to create one.")
        self.cond_placeholder.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            padding: 20px;
            background-color: {COLORS['surface_secondary']};
            border-radius: 6px;
        """)
        self.cond_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conditions_container.addWidget(self.cond_placeholder)

        conditions_layout.addStretch()

        tabs.addTab(conditions_tab, "Conditions")

        # Notifications Tab
        notifications_tab = QWidget()
        notif_layout = QVBoxLayout(notifications_tab)
        notif_layout.setContentsMargins(16, 16, 16, 16)

        # Email settings
        email_group = QGroupBox("Email Notifications")
        email_form = QFormLayout(email_group)
        email_form.setSpacing(12)

        self.email_enabled = QCheckBox("Enable email notifications")
        email_form.addRow("", self.email_enabled)

        self.smtp_server = QLineEdit()
        self.smtp_server.setPlaceholderText("smtp.gmail.com")
        self.smtp_server.setText("smtp.gmail.com")
        email_form.addRow("SMTP Server:", self.smtp_server)

        self.smtp_port = QSpinBox()
        self.smtp_port.setRange(1, 65535)
        self.smtp_port.setValue(587)
        email_form.addRow("Port:", self.smtp_port)

        self.smtp_user = QLineEdit()
        self.smtp_user.setPlaceholderText("your@email.com")
        email_form.addRow("Username:", self.smtp_user)

        self.smtp_pass = QLineEdit()
        self.smtp_pass.setEchoMode(QLineEdit.EchoMode.Password)
        email_form.addRow("Password:", self.smtp_pass)

        self.from_address = QLineEdit()
        self.from_address.setPlaceholderText("alerts@yourcompany.com")
        email_form.addRow("From Address:", self.from_address)

        notif_layout.addWidget(email_group)

        # Recipients
        recipients_group = QGroupBox("Notification Recipients")
        recipients_layout = QVBoxLayout(recipients_group)

        self.recipients_list = QListWidget()
        self.recipients_list.setMaximumHeight(150)
        recipients_layout.addWidget(self.recipients_list)

        add_row = QHBoxLayout()
        self.new_email = QLineEdit()
        self.new_email.setPlaceholderText("email@example.com")
        add_row.addWidget(self.new_email)

        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(80)
        add_btn.clicked.connect(self._add_recipient)
        add_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.setFixedWidth(120)
        remove_btn.clicked.connect(self._remove_recipient)
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
        """)
        add_row.addWidget(remove_btn)

        recipients_layout.addLayout(add_row)

        notif_layout.addWidget(recipients_group)

        # Test button
        test_btn = QPushButton("Send Test Email")
        test_btn.setFixedWidth(150)
        test_btn.clicked.connect(self._send_test_email)
        notif_layout.addWidget(test_btn)

        notif_layout.addStretch()

        tabs.addTab(notifications_tab, "Notifications")

        # Active Alarms Tab
        alarms_tab = QWidget()
        alarms_layout = QVBoxLayout(alarms_tab)
        alarms_layout.setContentsMargins(16, 16, 16, 16)

        # Actions
        actions_row = QHBoxLayout()
        actions_row.addWidget(QLabel("Active alarms requiring attention:"))
        actions_row.addStretch()

        ack_all_btn = QPushButton("Acknowledge All")
        ack_all_btn.clicked.connect(self._acknowledge_all)
        actions_row.addWidget(ack_all_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
            }}
        """)
        clear_all_btn.clicked.connect(self._clear_all)
        actions_row.addWidget(clear_all_btn)

        alarms_layout.addLayout(actions_row)

        # Alarms table
        self.alarms_table = QTableWidget()
        self.alarms_table.setColumnCount(6)
        self.alarms_table.setHorizontalHeaderLabels([
            "Time", "Severity", "Condition", "Message", "Score", "Status"
        ])
        self.alarms_table.horizontalHeader().setStretchLastSection(True)
        self.alarms_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.alarms_table.setAlternatingRowColors(True)
        alarms_layout.addWidget(self.alarms_table)

        tabs.addTab(alarms_tab, "Active Alarms")

        # History Tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(16, 16, 16, 16)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Severity", "Condition", "Message", "Acknowledged", "Cleared"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setAlternatingRowColors(True)
        history_layout.addWidget(self.history_table)

        tabs.addTab(history_tab, "History")

        layout.addWidget(tabs)

    def _add_condition(self):
        """Add a new alarm condition"""
        self.cond_placeholder.hide()

        name = f"Condition {len(self._condition_widgets) + 1}"
        widget = AlarmConditionRow(name)
        widget.set_channels(self._available_channels)
        widget.delete_requested.connect(self._remove_condition)
        self._condition_widgets.append(widget)
        self.conditions_container.addWidget(widget)

    def _remove_condition(self, name: str):
        """Remove an alarm condition"""
        for widget in self._condition_widgets:
            if widget._name == name or widget.name_edit.text() == name:
                widget.deleteLater()
                self._condition_widgets.remove(widget)
                break

        if not self._condition_widgets:
            self.cond_placeholder.show()

    def _add_recipient(self):
        """Add email recipient"""
        email = self.new_email.text().strip()
        if email and '@' in email:
            self.recipients_list.addItem(email)
            self.new_email.clear()

    def _remove_recipient(self):
        """Remove selected recipient"""
        for item in self.recipients_list.selectedItems():
            self.recipients_list.takeItem(self.recipients_list.row(item))

    def _send_test_email(self):
        """Send test email"""
        # In real implementation, would send test email
        pass

    def _acknowledge_all(self):
        """Acknowledge all active alarms"""
        pass

    def _clear_all(self):
        """Clear all active alarms"""
        self.alarms_table.setRowCount(0)

    def set_available_channels(self, channels: list):
        """Set available channels for channel-specific conditions"""
        self._available_channels = channels
        for widget in self._condition_widgets:
            widget.set_channels(channels)

    def add_alarm(self, alarm_data: dict):
        """Add an alarm to the active alarms table"""
        row = self.alarms_table.rowCount()
        self.alarms_table.insertRow(row)

        self.alarms_table.setItem(row, 0, QTableWidgetItem(alarm_data.get('time', '')))
        self.alarms_table.setItem(row, 1, QTableWidgetItem(alarm_data.get('severity', '')))
        self.alarms_table.setItem(row, 2, QTableWidgetItem(alarm_data.get('condition', '')))
        self.alarms_table.setItem(row, 3, QTableWidgetItem(alarm_data.get('message', '')))
        self.alarms_table.setItem(row, 4, QTableWidgetItem(str(alarm_data.get('score', ''))))
        self.alarms_table.setItem(row, 5, QTableWidgetItem(alarm_data.get('status', 'Active')))

    def get_conditions(self) -> list:
        """Get all condition configurations"""
        return [w.get_condition_data() for w in self._condition_widgets]

    def get_notification_config(self) -> dict:
        """Get notification configuration"""
        recipients = []
        for i in range(self.recipients_list.count()):
            recipients.append(self.recipients_list.item(i).text())

        return {
            'enabled': self.email_enabled.isChecked(),
            'smtp_server': self.smtp_server.text(),
            'smtp_port': self.smtp_port.value(),
            'username': self.smtp_user.text(),
            'password': self.smtp_pass.text(),
            'from_address': self.from_address.text(),
            'recipients': recipients,
        }
