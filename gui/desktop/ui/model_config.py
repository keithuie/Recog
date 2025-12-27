"""
Model Configuration Page

Configure ML model parameters:
- Kernel type and width
- Number of bins
- Channel groups
- Anomaly threshold
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QComboBox, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSlider,
    QListWidget, QListWidgetItem, QSplitter, QCheckBox,
    QProgressBar, QTextEdit, QColorDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..styles import COLORS, get_chart_colors


class ChannelGroupWidget(QFrame):
    """Widget for managing a single channel group"""

    group_changed = pyqtSignal()
    delete_requested = pyqtSignal(str)  # group_name

    def __init__(self, name: str, color: str = None, parent=None):
        super().__init__(parent)
        self.group_name = name
        self.color = color or COLORS['primary']
        self._channels = []

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        self.color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                border: none;
                border-radius: 4px;
            }}
        """)
        self.color_btn.clicked.connect(self._pick_color)
        header.addWidget(self.color_btn)

        self.name_label = QLabel(self.group_name)
        self.name_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.name_label)

        header.addStretch()

        self.enabled_cb = QCheckBox("Enabled")
        self.enabled_cb.setChecked(True)
        self.enabled_cb.stateChanged.connect(lambda: self.group_changed.emit())
        header.addWidget(self.enabled_cb)

        delete_btn = QPushButton("Delete")
        delete_btn.setFixedWidth(60)
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['danger']};
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{
                text-decoration: underline;
            }}
        """)
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.group_name))
        header.addWidget(delete_btn)

        layout.addLayout(header)

        # Channel list
        self.channel_list = QListWidget()
        self.channel_list.setMaximumHeight(120)
        self.channel_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['surface_secondary']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 4px 8px;
            }}
        """)
        layout.addWidget(self.channel_list)

        # Channel count
        self.count_label = QLabel("0 channels")
        self.count_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(self.count_label)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, "Select Group Color")
        if color.isValid():
            self.color = color.name()
            self.color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.color};
                    border: none;
                    border-radius: 4px;
                }}
            """)
            self.group_changed.emit()

    def set_channels(self, channels: list):
        self._channels = channels
        self.channel_list.clear()
        for ch in channels:
            self.channel_list.addItem(ch)
        self.count_label.setText(f"{len(channels)} channels")

    def get_channels(self) -> list:
        return self._channels

    def is_enabled(self) -> bool:
        return self.enabled_cb.isChecked()


class ModelConfigPage(QWidget):
    """
    Model configuration interface.

    Configure:
    - Kernel type (Triangular/Parabolic)
    - Bins per channel
    - Kernel width
    - Anomaly threshold
    - Channel groups
    """

    config_changed = pyqtSignal(dict)
    train_requested = pyqtSignal()
    training_started = pyqtSignal()
    training_completed = pyqtSignal(object)  # Emits detector

    def __init__(self, parent=None):
        super().__init__(parent)
        self._available_channels = []
        self._group_widgets = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Left column - Model parameters
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Model Parameters Group
        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(16)

        # Kernel type
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["Triangular", "Parabolic"])
        self.kernel_combo.setCurrentText("Parabolic")
        self.kernel_combo.currentTextChanged.connect(self._on_config_changed)
        params_layout.addRow("Kernel Type:", self.kernel_combo)

        # Help text
        kernel_help = QLabel("Parabolic kernel provides stronger match scoring for similar states")
        kernel_help.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        kernel_help.setWordWrap(True)
        params_layout.addRow("", kernel_help)

        # Bins per channel
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(8, 256)
        self.bins_spin.setValue(64)
        self.bins_spin.valueChanged.connect(self._on_config_changed)
        params_layout.addRow("Bins per Channel:", self.bins_spin)

        bins_help = QLabel("More bins = finer granularity, but requires more training data")
        bins_help.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        bins_help.setWordWrap(True)
        params_layout.addRow("", bins_help)

        # Kernel width
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 1.0)
        self.width_spin.setValue(0.5)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.valueChanged.connect(self._on_config_changed)
        params_layout.addRow("Kernel Width:", self.width_spin)

        width_help = QLabel("Controls how strictly the model matches (lower = stricter)")
        width_help.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        width_help.setWordWrap(True)
        params_layout.addRow("", width_help)

        left_layout.addWidget(params_group)

        # Threshold Settings Group
        threshold_group = QGroupBox("Detection Settings")
        threshold_layout = QFormLayout(threshold_group)
        threshold_layout.setSpacing(16)

        # Anomaly threshold
        threshold_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 99)
        self.threshold_slider.setValue(85)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_row.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("85%")
        self.threshold_label.setFixedWidth(50)
        threshold_row.addWidget(self.threshold_label)

        threshold_layout.addRow("Alarm Threshold:", threshold_row)

        threshold_help = QLabel("Match scores below this threshold will trigger alarms")
        threshold_help.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        threshold_help.setWordWrap(True)
        threshold_layout.addRow("", threshold_help)

        left_layout.addWidget(threshold_group)

        # Training Section
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout(training_group)

        training_info = QLabel("Train the model on loaded data to learn normal operating patterns.")
        training_info.setStyleSheet(f"color: {COLORS['text_secondary']};")
        training_info.setWordWrap(True)
        training_layout.addWidget(training_info)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(lambda: self.train_requested.emit())
        training_layout.addWidget(self.train_btn)

        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        training_layout.addWidget(self.training_progress)

        self.training_status = QLabel("")
        self.training_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        training_layout.addWidget(self.training_status)

        left_layout.addWidget(training_group)

        left_layout.addStretch()
        layout.addWidget(left_col)

        # Right column - Channel Groups
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Channel Groups Header
        groups_header = QHBoxLayout()
        groups_header.addWidget(QLabel("Channel Groups"))
        groups_header.addStretch()

        self.new_group_btn = QPushButton("New Group")
        self.new_group_btn.clicked.connect(self._create_group)
        groups_header.addWidget(self.new_group_btn)

        right_layout.addLayout(groups_header)

        # Groups help
        groups_help = QLabel(
            "Group related channels together for multivariate pattern matching. "
            "The model will learn joint patterns across all channels in a group."
        )
        groups_help.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        groups_help.setWordWrap(True)
        right_layout.addWidget(groups_help)

        # Available channels
        avail_group = QGroupBox("Available Channels")
        avail_layout = QVBoxLayout(avail_group)

        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.available_list.setMaximumHeight(150)
        avail_layout.addWidget(self.available_list)

        assign_row = QHBoxLayout()
        assign_row.addWidget(QLabel("Assign to:"))
        self.assign_combo = QComboBox()
        self.assign_combo.setMinimumWidth(150)
        assign_row.addWidget(self.assign_combo)

        self.assign_btn = QPushButton("Assign")
        self.assign_btn.clicked.connect(self._assign_channels)
        assign_row.addWidget(self.assign_btn)

        assign_row.addStretch()
        avail_layout.addLayout(assign_row)

        right_layout.addWidget(avail_group)

        # Group widgets container
        self.groups_container = QVBoxLayout()
        self.groups_container.setSpacing(12)
        right_layout.addLayout(self.groups_container)

        right_layout.addStretch()
        layout.addWidget(right_col)

    def _on_config_changed(self):
        config = self.get_config()
        self.config_changed.emit(config)

    def _on_threshold_changed(self, value: int):
        self.threshold_label.setText(f"{value}%")
        self._on_config_changed()

    def _create_group(self):
        # Generate unique name
        base_name = "Group"
        i = 1
        while f"{base_name} {i}" in self._group_widgets:
            i += 1
        name = f"{base_name} {i}"

        # Get color
        colors = get_chart_colors()
        color = colors[len(self._group_widgets) % len(colors)]

        # Create widget
        widget = ChannelGroupWidget(name, color)
        widget.group_changed.connect(self._on_config_changed)
        widget.delete_requested.connect(self._delete_group)
        self._group_widgets[name] = widget
        self.groups_container.addWidget(widget)

        # Update assign combo
        self.assign_combo.addItem(name)

    def _delete_group(self, name: str):
        if name in self._group_widgets:
            widget = self._group_widgets[name]
            # Return channels to available
            for ch in widget.get_channels():
                self.available_list.addItem(ch)
            widget.deleteLater()
            del self._group_widgets[name]

            # Update assign combo
            index = self.assign_combo.findText(name)
            if index >= 0:
                self.assign_combo.removeItem(index)

    def _assign_channels(self):
        group_name = self.assign_combo.currentText()
        if not group_name or group_name not in self._group_widgets:
            return

        # Get selected channels
        selected = self.available_list.selectedItems()
        channels = [item.text() for item in selected]

        # Remove from available
        for item in selected:
            self.available_list.takeItem(self.available_list.row(item))

        # Add to group
        widget = self._group_widgets[group_name]
        current = widget.get_channels()
        widget.set_channels(current + channels)

        self._on_config_changed()

    def set_available_channels(self, channels: list):
        """Set the list of available channels from data source"""
        self._available_channels = channels
        self.available_list.clear()
        for ch in channels:
            self.available_list.addItem(ch)

    def get_config(self) -> dict:
        """Get current configuration"""
        groups = {}
        for name, widget in self._group_widgets.items():
            if widget.is_enabled():
                groups[name] = {
                    'channels': widget.get_channels(),
                    'color': widget.color,
                }

        return {
            'kernel_type': self.kernel_combo.currentText().lower(),
            'bins_per_channel': self.bins_spin.value(),
            'kernel_width': self.width_spin.value(),
            'threshold': self.threshold_slider.value(),
            'channel_groups': groups,
        }

    def set_training_progress(self, progress: int, status: str = ""):
        """Update training progress"""
        self.training_progress.setVisible(True)
        self.training_progress.setValue(progress)
        self.training_status.setText(status)

        if progress >= 100:
            self.training_status.setStyleSheet(f"color: {COLORS['success']};")
        else:
            self.training_status.setStyleSheet(f"color: {COLORS['text_secondary']};")

    @property
    def threshold_spin(self):
        """Alias for threshold_slider for compatibility"""
        return self.threshold_slider

    def add_group_card(self, name: str, channels: list):
        """Add a channel group card from external source (e.g., wizard)"""
        if name in self._group_widgets:
            return

        colors = get_chart_colors()
        color = colors[len(self._group_widgets) % len(colors)]

        group_widget = ChannelGroupWidget(name, color)
        group_widget.set_channels(channels)

        group_widget.delete_requested.connect(self._delete_group)
        group_widget.group_changed.connect(self._on_config_changed)

        self._group_widgets[name] = group_widget
        self.groups_container.insertWidget(
            self.groups_container.count() - 1,
            group_widget
        )
