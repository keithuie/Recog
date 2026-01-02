"""
MachineIQ Dashboard

Grouped channel display with per-group training controls and match strength.
Each channel group has its own detector for multivariate pattern matching.
"""

import numpy as np
import pyqtgraph as pg
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy, QPushButton, QComboBox,
    QSpinBox, QProgressBar, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QFont, QColor, QCursor

from ..styles import COLORS, get_chart_colors, get_match_color


class HoverablePlotWidget(pg.PlotWidget):
    """PlotWidget with timestamp display on hover"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hover_label = None
        self._data_timestamps = []
        self._data_values = []

        # Enable mouse tracking
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._on_mouse_move)

        # Create hover label
        self._hover_label = pg.TextItem(color='#FFF', anchor=(0, 1))
        self._hover_label.setZValue(1000)
        self.addItem(self._hover_label, ignoreBounds=True)
        self._hover_label.hide()

    def set_data_for_hover(self, timestamps, values):
        """Store data for hover lookup"""
        self._data_timestamps = timestamps
        self._data_values = values

    def _on_mouse_move(self, pos):
        """Show timestamp and value on hover"""
        if not len(self._data_timestamps) or not len(self._data_values):
            self._hover_label.hide()
            return

        # Check if in plot area
        if not self.sceneBoundingRect().contains(pos):
            self._hover_label.hide()
            return

        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x = mouse_point.x()

        # Find nearest data point
        if len(self._data_timestamps) > 0:
            idx = np.searchsorted(self._data_timestamps, x)
            idx = min(max(0, idx), len(self._data_timestamps) - 1)

            ts = self._data_timestamps[idx]
            val = self._data_values[idx]

            # Format timestamp
            try:
                dt = datetime.fromtimestamp(ts)
                time_str = dt.strftime("%H:%M:%S.%f")[:-3]
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, OSError, OverflowError):
                time_str = f"{ts:.2f}"
                date_str = ""

            self._hover_label.setText(f"{date_str} {time_str}\nValue: {val:.4f}")
            self._hover_label.setPos(x, self.getPlotItem().vb.viewRange()[1][1])
            self._hover_label.show()


class ChannelStrip(QFrame):
    """Individual channel display strip with hover support"""

    def __init__(self, channel_name: str, color: str = None, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.color = color or COLORS['chart_1']
        self._current_value = 0.0
        self._min_value = 0.0
        self._max_value = 100.0

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1A1A1C;
                border: none;
                border-bottom: 1px solid #2A2A2C;
            }}
        """)
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left panel - Y-axis labels
        y_panel = QFrame()
        y_panel.setFixedWidth(60)
        y_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        y_layout = QVBoxLayout(y_panel)
        y_layout.setContentsMargins(4, 4, 4, 4)
        y_layout.setSpacing(0)

        self.max_label = QLabel(f"{self._max_value:.1f}")
        self.max_label.setStyleSheet("color: #666; font-size: 9px;")
        self.max_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.max_label)

        y_layout.addStretch()

        self.min_label = QLabel(f"{self._min_value:.1f}")
        self.min_label.setStyleSheet("color: #666; font-size: 9px;")
        self.min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.min_label)

        layout.addWidget(y_panel)

        # Center - Plot with hover
        self.plot_widget = HoverablePlotWidget(
            axisItems={'bottom': pg.DateAxisItem(orientation='bottom')}
        )
        self.plot_widget.setBackground('#1A1A1C')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.showAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getViewBox().setDefaultPadding(0)

        # Vertical hover line
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.v_line.setPen(pg.mkPen('#FFFFFF', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)

        # Plot curve
        pen = pg.mkPen(color=self.color, width=1.5)
        self.curve = self.plot_widget.plot([], [], pen=pen)

        layout.addWidget(self.plot_widget, stretch=1)

        # Right panel - Channel name and value
        info_panel = QFrame()
        info_panel.setFixedWidth(140)
        info_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(8, 4, 8, 4)
        info_layout.setSpacing(2)

        self.name_label = QLabel(self.channel_name)
        self.name_label.setStyleSheet(f"color: {self.color}; font-size: 11px; font-weight: 500;")
        info_layout.addWidget(self.name_label)

        info_layout.addStretch()

        self.value_label = QLabel("0.0000")
        self.value_label.setStyleSheet("color: #AAA; font-size: 12px; font-family: monospace;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        info_layout.addWidget(self.value_label)

        layout.addWidget(info_panel)

    def update_data(self, timestamps: np.ndarray, values: np.ndarray):
        """Update the plot with new data"""
        if len(values) > 0:
            self._current_value = values[-1]
            self._min_value = np.min(values)
            self._max_value = np.max(values)

            range_span = self._max_value - self._min_value
            if range_span == 0:
                range_span = 1.0
            margin = range_span * 0.1
            self._min_value -= margin
            self._max_value += margin

            self.curve.setData(timestamps, values)
            self.plot_widget.setYRange(self._min_value, self._max_value, padding=0)
            self.plot_widget.set_data_for_hover(timestamps, values)

            self.value_label.setText(f"{self._current_value:.4f}")
            self.max_label.setText(f"{self._max_value:.2f}")
            self.min_label.setText(f"{self._min_value:.2f}")

    def set_hover_line(self, x_val):
        """Update hover line position"""
        self.v_line.setPos(x_val)


class CombinedChannelPlot(QFrame):
    """Combined overlay plot showing all channels in a group"""

    def __init__(self, group_name: str, color: str, parent=None):
        super().__init__(parent)
        self.group_name = group_name
        self.color = color
        self._curves = {}
        self._channel_colors = {}

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1A1A1C;
                border: none;
                border-bottom: 1px solid #2A2A2C;
            }}
        """)
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Y-axis panel
        y_panel = QFrame()
        y_panel.setFixedWidth(60)
        y_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        y_layout = QVBoxLayout(y_panel)
        y_layout.setContentsMargins(4, 4, 4, 4)

        self.max_label = QLabel("1.0")
        self.max_label.setStyleSheet("color: #666; font-size: 9px;")
        self.max_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.max_label)

        y_layout.addStretch()

        self.min_label = QLabel("0.0")
        self.min_label.setStyleSheet("color: #666; font-size: 9px;")
        self.min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.min_label)

        layout.addWidget(y_panel)

        # Plot
        self.plot_widget = HoverablePlotWidget(
            axisItems={'bottom': pg.DateAxisItem(orientation='bottom')}
        )
        self.plot_widget.setBackground('#1A1A1C')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.showAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.addLegend(offset=(10, 10))

        layout.addWidget(self.plot_widget, stretch=1)

        # Info panel
        info_panel = QFrame()
        info_panel.setFixedWidth(140)
        info_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel(f"All Channels")
        title.setStyleSheet(f"color: {self.color}; font-size: 11px; font-weight: 600;")
        info_layout.addWidget(title)

        self.count_label = QLabel("0 channels")
        self.count_label.setStyleSheet("color: #888; font-size: 10px;")
        info_layout.addWidget(self.count_label)

        info_layout.addStretch()

        layout.addWidget(info_panel)

    def set_channels(self, channel_names: list, colors: list):
        """Set up curves for each channel"""
        self._curves.clear()
        self.plot_widget.clear()

        for i, name in enumerate(channel_names):
            color = colors[i % len(colors)]
            self._channel_colors[name] = color
            pen = pg.mkPen(color=color, width=1.2)
            curve = self.plot_widget.plot([], [], pen=pen, name=name[:12])
            self._curves[name] = curve

        self.count_label.setText(f"{len(channel_names)} channels")

    def update_channel(self, channel_name: str, timestamps: np.ndarray, values: np.ndarray):
        """Update data for a specific channel"""
        if channel_name in self._curves:
            # Normalize values to 0-1 range for overlay
            if len(values) > 0:
                vmin, vmax = np.min(values), np.max(values)
                if vmax > vmin:
                    normalized = (values - vmin) / (vmax - vmin)
                else:
                    normalized = np.zeros_like(values)
                self._curves[channel_name].setData(timestamps, normalized)


class MatchStrengthBar(QFrame):
    """Match strength indicator for a group"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = 100.0
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("background-color: #1A1A1C; border: none;")
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Y-axis
        y_panel = QFrame()
        y_panel.setFixedWidth(60)
        y_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        y_layout = QVBoxLayout(y_panel)
        y_layout.setContentsMargins(4, 4, 4, 4)

        max_label = QLabel("100")
        max_label.setStyleSheet("color: #666; font-size: 9px;")
        max_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(max_label)

        y_layout.addStretch()

        min_label = QLabel("0")
        min_label.setStyleSheet("color: #666; font-size: 9px;")
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(min_label)

        layout.addWidget(y_panel)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1A1A1C')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setYRange(0, 100, padding=0)

        pen = pg.mkPen(color=COLORS['success'], width=2)
        self.curve = self.plot_widget.plot([], [], pen=pen, fillLevel=0,
                                           brush=pg.mkBrush(color=(52, 199, 89, 50)))

        layout.addWidget(self.plot_widget, stretch=1)

        # Info panel
        info_panel = QFrame()
        info_panel.setFixedWidth(140)
        info_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(8, 4, 8, 4)

        self.name_label = QLabel("MATCH STRENGTH")
        self.name_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 10px; font-weight: 600;")
        info_layout.addWidget(self.name_label)

        info_layout.addStretch()

        self.value_label = QLabel("100.0%")
        self.value_label.setStyleSheet("color: #AAA; font-size: 12px; font-family: monospace;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        info_layout.addWidget(self.value_label)

        layout.addWidget(info_panel)

    def update_data(self, timestamps: np.ndarray, scores: np.ndarray):
        """Update match strength display"""
        if len(scores) > 0:
            self._score = scores[-1]
            color = get_match_color(self._score)
            self.curve.setPen(pg.mkPen(color=color, width=2))
            rgb = QColor(color).getRgb()
            self.curve.setBrush(pg.mkBrush(color=(rgb[0], rgb[1], rgb[2], 50)))
            self.curve.setData(timestamps, scores)
            self.value_label.setText(f"{self._score:.1f}%")
            self.name_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")


class ChannelGroupSection(QFrame):
    """
    A section displaying a channel group with training controls.

    Features:
    - Group header with name, training controls
    - Expandable/collapsible channel strips
    - Match strength indicator
    - Training duration selector
    """

    training_requested = pyqtSignal(str, int)  # group_name, duration_seconds
    stop_training_requested = pyqtSignal(str)  # group_name

    def __init__(self, group_name: str, channels: list, color: str = "#007AFF", parent=None):
        super().__init__(parent)
        self.group_name = group_name
        self.channels = channels
        self.color = color
        self._is_expanded = True
        self._is_training = False
        self._channel_strips = {}
        self._detector = None
        self._training_progress = 0

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame#group_section {{
                background-color: #1A1A1C;
                border: 1px solid {self.color}40;
                border-radius: 8px;
                margin: 4px;
            }}
        """)
        self.setObjectName("group_section")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color}20;
                border-bottom: 1px solid {self.color}40;
                border-radius: 8px 8px 0 0;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        # Color dot
        color_dot = QLabel("●")
        color_dot.setStyleSheet(f"color: {self.color}; font-size: 16px;")
        header_layout.addWidget(color_dot)

        # Group name
        name_label = QLabel(self.group_name)
        name_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: 600;")
        header_layout.addWidget(name_label)

        # Channel count
        count_label = QLabel(f"({len(self.channels)} channels)")
        count_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        header_layout.addWidget(count_label)

        header_layout.addStretch()

        # Training controls
        training_frame = QFrame()
        training_frame.setStyleSheet("background: transparent; border: none;")
        training_layout = QHBoxLayout(training_frame)
        training_layout.setContentsMargins(0, 0, 0, 0)
        training_layout.setSpacing(8)

        # Training duration
        train_label = QLabel("Train:")
        train_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        training_layout.addWidget(train_label)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 999)
        self.duration_spin.setValue(5)
        self.duration_spin.setFixedWidth(60)
        self.duration_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 2px 4px;
                color: {COLORS['text_primary']};
            }}
        """)
        training_layout.addWidget(self.duration_spin)

        self.duration_unit = QComboBox()
        self.duration_unit.addItems(["min", "hr", "day"])
        self.duration_unit.setFixedWidth(55)
        self.duration_unit.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 2px 4px;
                color: {COLORS['text_primary']};
            }}
        """)
        training_layout.addWidget(self.duration_unit)

        # Train button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setFixedWidth(100)
        self.train_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.train_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: 500;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: #2DB84D;
            }}
        """)
        self.train_btn.clicked.connect(self._on_train_clicked)
        training_layout.addWidget(self.train_btn)

        header_layout.addWidget(training_frame)

        # Expand/collapse button
        self.expand_btn = QPushButton("−")
        self.expand_btn.setFixedSize(24, 24)
        self.expand_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.expand_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #888;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #333;
                color: #FFF;
            }
        """)
        self.expand_btn.clicked.connect(self._toggle_expand)
        header_layout.addWidget(self.expand_btn)

        layout.addWidget(header)

        # Training progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #333;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['success']};
            }}
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Content container
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        # Combined plot (for collapsed view)
        self.combined_plot = CombinedChannelPlot(self.group_name, self.color)
        self.combined_plot.hide()
        self.content_layout.addWidget(self.combined_plot)

        # Individual channel strips (for expanded view)
        self.strips_widget = QWidget()
        self.strips_layout = QVBoxLayout(self.strips_widget)
        self.strips_layout.setContentsMargins(0, 0, 0, 0)
        self.strips_layout.setSpacing(0)

        colors = get_chart_colors()
        for i, channel in enumerate(self.channels):
            color = colors[i % len(colors)]
            strip = ChannelStrip(channel, color)
            self._channel_strips[channel] = strip
            self.strips_layout.addWidget(strip)

        self.content_layout.addWidget(self.strips_widget)

        # Match strength bar
        self.match_bar = MatchStrengthBar()
        self.content_layout.addWidget(self.match_bar)

        layout.addWidget(self.content_widget)

        # Set up combined plot channels
        self.combined_plot.set_channels(self.channels, colors)

    def _on_train_clicked(self):
        """Handle train button click"""
        if self._is_training:
            self._is_training = False
            self.train_btn.setText("Start Training")
            self.train_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['success']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: 500;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: #2DB84D;
                }}
            """)
            self.progress_bar.hide()
            self.stop_training_requested.emit(self.group_name)
        else:
            self._is_training = True
            self.train_btn.setText("Stop Training")
            self.train_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['danger']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: 500;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: #E04545;
                }}
            """)
            self.progress_bar.setValue(0)
            self.progress_bar.show()

            # Calculate duration in seconds
            value = self.duration_spin.value()
            unit = self.duration_unit.currentText()
            if unit == "min":
                duration = value * 60
            elif unit == "hr":
                duration = value * 3600
            else:  # day
                duration = value * 86400

            self.training_requested.emit(self.group_name, duration)

    def _toggle_expand(self):
        """Toggle between expanded and collapsed view"""
        self._is_expanded = not self._is_expanded

        if self._is_expanded:
            self.expand_btn.setText("−")
            self.strips_widget.show()
            self.combined_plot.hide()
        else:
            self.expand_btn.setText("+")
            self.strips_widget.hide()
            self.combined_plot.show()

    def update_channel_data(self, channel_name: str, timestamps: np.ndarray, values: np.ndarray):
        """Update data for a specific channel"""
        if channel_name in self._channel_strips:
            self._channel_strips[channel_name].update_data(timestamps, values)
            self.combined_plot.update_channel(channel_name, timestamps, values)

    def update_match_strength(self, timestamps: np.ndarray, scores: np.ndarray):
        """Update match strength display"""
        self.match_bar.update_data(timestamps, scores)

    def set_training_progress(self, progress: float):
        """Set training progress (0-100)"""
        self._training_progress = progress
        self.progress_bar.setValue(int(progress))

        if progress >= 100:
            self._is_training = False
            self.train_btn.setText("Start Training")
            self.train_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['success']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: 500;
                    font-size: 11px;
                }}
            """)
            self.progress_bar.hide()

    def set_detector(self, detector):
        """Set the detector for this group"""
        self._detector = detector


class DashboardPage(QWidget):
    """
    Main dashboard page with grouped channel display.

    Features:
    - Channel groups as collapsible sections
    - Per-group training controls
    - Match strength per group
    - Timestamp hover on data points
    """

    training_started = pyqtSignal(str, int)  # group_name, duration
    training_stopped = pyqtSignal(str)  # group_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups = {}  # name -> ChannelGroupSection
        self._ungrouped_channels = []
        self._ungrouped_section = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for groups
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0D0D0E;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1A1A1C;
                width: 16px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 40px;
                margin: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #888;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: #1A1A1C;
            }
        """)

        # Container for groups
        self.groups_container = QWidget()
        self.groups_container.setStyleSheet("background-color: #0D0D0E;")
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setContentsMargins(8, 8, 8, 8)
        self.groups_layout.setSpacing(12)
        self.groups_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Placeholder
        self.placeholder = QLabel("No data loaded.\n\nUse Dataflow to connect a data source,\nthen create channel groups in Model Config.")
        self.placeholder.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 14px;
            padding: 60px;
        """)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.groups_layout.addWidget(self.placeholder)

        self.scroll.setWidget(self.groups_container)
        layout.addWidget(self.scroll)

    def set_groups(self, groups: list):
        """
        Set up channel group sections.

        Args:
            groups: List of dicts with 'name', 'channels', 'color' keys
        """
        # Clear existing groups
        for section in self._groups.values():
            section.deleteLater()
        self._groups.clear()

        if self._ungrouped_section:
            self._ungrouped_section.deleteLater()
            self._ungrouped_section = None

        self.placeholder.hide()

        # Create group sections
        for group_info in groups:
            name = group_info['name']
            channels = group_info.get('channels', [])
            color = group_info.get('color', '#007AFF')

            if channels:  # Only create section if has channels
                section = ChannelGroupSection(name, channels, color)
                # Use default arguments to capture current values (avoid closure issue)
                section.training_requested.connect(
                    lambda n, d, s=section: self._on_training_requested(s.group_name, d)
                )
                section.stop_training_requested.connect(
                    lambda n, s=section: self._on_stop_training(s.group_name)
                )
                self._groups[name] = section
                self.groups_layout.insertWidget(self.groups_layout.count() - 1, section)

    def _on_training_requested(self, group_name: str, duration: int):
        """Forward training request signal"""
        self.training_started.emit(group_name, duration)

    def _on_stop_training(self, group_name: str):
        """Forward stop training signal"""
        self.training_stopped.emit(group_name)

    def set_channel_groups(self, groups: list):
        """Alias for set_groups for compatibility"""
        self.set_groups(groups)

    def set_ungrouped_channels(self, channels: list):
        """Set channels that aren't in any group"""
        self._ungrouped_channels = channels

        if self._ungrouped_section:
            self._ungrouped_section.deleteLater()
            self._ungrouped_section = None

        if channels:
            self.placeholder.hide()
            self._ungrouped_section = ChannelGroupSection(
                "Ungrouped Channels", channels, "#888888"
            )
            self._ungrouped_section.training_requested.connect(
                lambda n, d: self._on_training_requested("Ungrouped Channels", d)
            )
            self._ungrouped_section.stop_training_requested.connect(
                lambda n: self._on_stop_training("Ungrouped Channels")
            )
            self.groups_layout.addWidget(self._ungrouped_section)

    def update_channel_data(self, channel_name: str, timestamps: np.ndarray, values: np.ndarray):
        """Update data for a specific channel"""
        # Find which group contains this channel
        for section in self._groups.values():
            if channel_name in section._channel_strips:
                section.update_channel_data(channel_name, timestamps, values)
                return

        # Check ungrouped
        if self._ungrouped_section and channel_name in self._ungrouped_section._channel_strips:
            self._ungrouped_section.update_channel_data(channel_name, timestamps, values)

    def update_group_match_strength(self, group_name: str, timestamps: np.ndarray, scores: np.ndarray):
        """Update match strength for a specific group"""
        if group_name in self._groups:
            self._groups[group_name].update_match_strength(timestamps, scores)

    def set_group_training_progress(self, group_name: str, progress: float):
        """Set training progress for a group"""
        if group_name in self._groups:
            self._groups[group_name].set_training_progress(progress)

    def clear(self):
        """Clear all groups"""
        for section in self._groups.values():
            section.deleteLater()
        self._groups.clear()

        if self._ungrouped_section:
            self._ungrouped_section.deleteLater()
            self._ungrouped_section = None

        self.placeholder.show()

    # Legacy compatibility methods
    def set_channels(self, channel_names: list):
        """Legacy: Create single ungrouped section with all channels"""
        self.set_ungrouped_channels(channel_names)
        self.placeholder.hide()

    def update_match_strength(self, timestamps: np.ndarray, scores: np.ndarray):
        """Legacy: Update match strength for all groups"""
        for section in self._groups.values():
            section.update_match_strength(timestamps, scores)
        if self._ungrouped_section:
            self._ungrouped_section.update_match_strength(timestamps, scores)

    def get_current_match_score(self) -> float:
        """Get current match score from first group"""
        for section in self._groups.values():
            return section.match_bar._score
        if self._ungrouped_section:
            return self._ungrouped_section.match_bar._score
        return 100.0
