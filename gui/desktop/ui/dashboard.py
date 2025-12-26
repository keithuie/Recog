"""
MachineIQ Dashboard

AURA-style stacked channel display with match strength indicator.
Each channel displayed in its own strip with Y-axis scale and current value.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from ..styles import COLORS, get_chart_colors, get_match_color


class ChannelStrip(QFrame):
    """
    Individual channel display strip.

    Shows:
    - Channel name
    - Current value
    - Time series plot
    - Y-axis scale
    """

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
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left panel - Y-axis labels
        y_panel = QFrame()
        y_panel.setFixedWidth(70)
        y_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        y_layout = QVBoxLayout(y_panel)
        y_layout.setContentsMargins(8, 4, 4, 4)
        y_layout.setSpacing(0)

        self.max_label = QLabel(f"{self._max_value:.1f}")
        self.max_label.setStyleSheet("color: #666; font-size: 10px;")
        self.max_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.max_label)

        y_layout.addStretch()

        self.min_label = QLabel(f"{self._min_value:.1f}")
        self.min_label.setStyleSheet("color: #666; font-size: 10px;")
        self.min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.min_label)

        layout.addWidget(y_panel)

        # Center - Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1A1A1C')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getViewBox().setDefaultPadding(0)

        # Create plot curve
        pen = pg.mkPen(color=self.color, width=1.5)
        self.curve = self.plot_widget.plot([], [], pen=pen)

        layout.addWidget(self.plot_widget, stretch=1)

        # Right panel - Channel name and value
        info_panel = QFrame()
        info_panel.setFixedWidth(180)
        info_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(2)

        self.name_label = QLabel(self.channel_name)
        self.name_label.setStyleSheet(f"color: {self.color}; font-size: 12px; font-weight: 500;")
        info_layout.addWidget(self.name_label)

        info_layout.addStretch()

        self.value_label = QLabel("0.000000")
        self.value_label.setStyleSheet("color: #AAA; font-size: 14px; font-family: monospace;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        info_layout.addWidget(self.value_label)

        layout.addWidget(info_panel)

    def update_data(self, timestamps: np.ndarray, values: np.ndarray):
        """Update the plot with new data"""
        if len(values) > 0:
            self._current_value = values[-1]
            self._min_value = np.min(values)
            self._max_value = np.max(values)

            # Add margin to range
            range_span = self._max_value - self._min_value
            if range_span == 0:
                range_span = 1.0
            margin = range_span * 0.1
            self._min_value -= margin
            self._max_value += margin

            self.curve.setData(timestamps, values)
            self.plot_widget.setYRange(self._min_value, self._max_value, padding=0)

            self.value_label.setText(f"{self._current_value:.6f}")
            self.max_label.setText(f"{self._max_value:.4f}")
            self.min_label.setText(f"{self._min_value:.4f}")

    def set_color(self, color: str):
        """Set the channel color"""
        self.color = color
        self.name_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;")
        self.curve.setPen(pg.mkPen(color=color, width=1.5))


class MatchStrengthBar(QFrame):
    """
    Match strength indicator bar.

    Displays match score 0-100 with color coding.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = 100.0

        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1A1A1C;
                border: none;
            }}
        """)
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left panel - Y-axis style
        y_panel = QFrame()
        y_panel.setFixedWidth(70)
        y_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        y_layout = QVBoxLayout(y_panel)
        y_layout.setContentsMargins(8, 4, 4, 4)
        y_layout.setSpacing(0)

        max_label = QLabel("100")
        max_label.setStyleSheet("color: #666; font-size: 10px;")
        max_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(max_label)

        y_layout.addStretch()

        min_label = QLabel("0")
        min_label.setStyleSheet("color: #666; font-size: 10px;")
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(min_label)

        layout.addWidget(y_panel)

        # Center - Score bar
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1A1A1C')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getViewBox().setDefaultPadding(0)
        self.plot_widget.setYRange(0, 100, padding=0)

        # Score line
        pen = pg.mkPen(color=COLORS['success'], width=2)
        self.curve = self.plot_widget.plot([], [], pen=pen, fillLevel=0,
                                           brush=pg.mkBrush(color=(52, 199, 89, 50)))

        layout.addWidget(self.plot_widget, stretch=1)

        # Right panel - Label and value
        info_panel = QFrame()
        info_panel.setFixedWidth(180)
        info_panel.setStyleSheet("background-color: #1A1A1C; border: none;")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(2)

        self.name_label = QLabel("MATCH STRENGTH")
        self.name_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px; font-weight: 600;")
        info_layout.addWidget(self.name_label)

        info_layout.addStretch()

        self.value_label = QLabel("100.000000")
        self.value_label.setStyleSheet("color: #AAA; font-size: 14px; font-family: monospace;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        info_layout.addWidget(self.value_label)

        layout.addWidget(info_panel)

    def update_data(self, timestamps: np.ndarray, scores: np.ndarray):
        """Update match strength display"""
        if len(scores) > 0:
            self._score = scores[-1]

            # Color based on score
            color = get_match_color(self._score)
            self.curve.setPen(pg.mkPen(color=color, width=2))

            # Update fill color
            rgb = QColor(color).getRgb()
            self.curve.setBrush(pg.mkBrush(color=(rgb[0], rgb[1], rgb[2], 50)))

            self.curve.setData(timestamps, scores)
            self.value_label.setText(f"{self._score:.6f}")


class DashboardPage(QWidget):
    """
    Main dashboard page with stacked channel strips.
    """

    channel_clicked = pyqtSignal(str, float)  # channel_name, timestamp

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channels = {}
        self._channel_strips = {}
        self._match_bar = None
        self._timestamps = np.array([])
        self._match_scores = np.array([])

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for channels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: #1A1A1C;
                border: none;
            }}
        """)

        # Container for strips
        self.strips_container = QWidget()
        self.strips_container.setStyleSheet("background-color: #1A1A1C;")
        self.strips_layout = QVBoxLayout(self.strips_container)
        self.strips_layout.setContentsMargins(0, 0, 0, 0)
        self.strips_layout.setSpacing(0)

        # Placeholder when no channels
        self.placeholder = QLabel("No data loaded. Use Data Manager or Connection Center to load data.")
        self.placeholder.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 14px;
            padding: 40px;
        """)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.strips_layout.addWidget(self.placeholder)

        self.strips_layout.addStretch()

        scroll.setWidget(self.strips_container)
        layout.addWidget(scroll)

    def set_channels(self, channel_names: list):
        """Set up channel strips for the given channels"""
        # Clear existing strips
        for strip in self._channel_strips.values():
            strip.deleteLater()
        self._channel_strips.clear()

        # Hide placeholder
        self.placeholder.hide()

        # Get colors
        colors = get_chart_colors()

        # Create channel strips
        for i, name in enumerate(channel_names):
            color = colors[i % len(colors)]
            strip = ChannelStrip(name, color)
            self._channel_strips[name] = strip
            # Insert before the stretch
            self.strips_layout.insertWidget(self.strips_layout.count() - 1, strip)

        # Add match strength bar at the bottom
        if self._match_bar:
            self._match_bar.deleteLater()

        self._match_bar = MatchStrengthBar()
        self.strips_layout.insertWidget(self.strips_layout.count() - 1, self._match_bar)

    def update_channel_data(self, channel_name: str, timestamps: np.ndarray, values: np.ndarray):
        """Update data for a specific channel"""
        if channel_name in self._channel_strips:
            self._channel_strips[channel_name].update_data(timestamps, values)

    def update_match_strength(self, timestamps: np.ndarray, scores: np.ndarray):
        """Update match strength display"""
        self._timestamps = timestamps
        self._match_scores = scores
        if self._match_bar:
            self._match_bar.update_data(timestamps, scores)

    def get_current_match_score(self) -> float:
        """Get the current match score"""
        if len(self._match_scores) > 0:
            return self._match_scores[-1]
        return 100.0

    def clear(self):
        """Clear all channel strips"""
        for strip in self._channel_strips.values():
            strip.deleteLater()
        self._channel_strips.clear()

        if self._match_bar:
            self._match_bar.deleteLater()
            self._match_bar = None

        self.placeholder.show()
