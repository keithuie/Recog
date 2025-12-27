"""
MachineIQ Dashboard

AURA-style stacked channel display with match strength indicator.
Each channel displayed in its own strip with Y-axis scale and current value.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy, QSplitter, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QCursor

from ..styles import COLORS, get_chart_colors, get_match_color, get_auto_contrast_color


class ChannelGroupWidget(QFrame):
    """
    Collapsible container for a group of channel strips.
    """
    
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.is_expanded = True
        
        self._setup_ui()
        
    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame#group_container {{
                background-color: {COLORS['surface_secondary']};
                border-radius: 6px;
                margin-bottom: 8px;
            }}
            QPushButton {{
                border: none;
                text-align: left;
                font-weight: 600;
                color: {COLORS['text_primary']};
                padding: 8px;
                background-color: transparent;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface']};
                border-radius: 4px;
            }}
        """)
        self.setObjectName("group_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 0, 4, 4)
        
        self.toggle_btn = QPushButton(f"▼ {self.name}")
        self.toggle_btn.clicked.connect(self.toggle)
        header_layout.addWidget(self.toggle_btn)
        
        layout.addLayout(header_layout)
        
        # Content container
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(1)  # Space between strips
        
        layout.addWidget(self.content_widget)
        
    def add_strip(self, strip):
        """Add a channel strip to this group"""
        self.content_layout.addWidget(strip)
        
    def toggle(self):
        """Toggle collapsed state"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            self.content_widget.show()
            self.toggle_btn.setText(f"▼ {self.name}")
            self.setMaximumHeight(16777215) # Restore max height
        else:
            self.content_widget.hide()
            self.toggle_btn.setText(f"▶ {self.name}")
            self.setFixedHeight(40)


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
        self._is_minimized = False

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
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        self.plot_widget.setBackground('#1A1A1C')
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.showAxis('bottom')
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getViewBox().setDefaultPadding(0)

        # Add infinite line for hover
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.v_line.setPen(pg.mkPen('#FFFFFF', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)

        # Create plot curve
        pen = pg.mkPen(color=self.color, width=1.5)
        self.curve = self.plot_widget.plot([], [], pen=pen, symbol='o', symbolSize=3, symbolBrush=self.color, symbolPen=None)

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
        
        # Minimize button
        self.min_btn = QPushButton("−")
        self.min_btn.setFixedSize(20, 20)
        self.min_btn.setCheckable(True)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #666;
                border: 1px solid #333;
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #333;
                color: #FFF;
            }
            QPushButton:checked {
                background: #333;
                color: #FFF;
                content: "+";
            }
        """)
        self.min_btn.clicked.connect(self._toggle_minimize)
        
        # Reset View button
        self.reset_btn = QPushButton("⟲")
        self.reset_btn.setFixedSize(20, 20)
        self.reset_btn.setToolTip("Reset View")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #666;
                border: 1px solid #333;
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #333;
                color: #FFF;
            }
        """)
        self.reset_btn.clicked.connect(lambda: self.plot_widget.autoRange())

        
        # Add to header row in info panel
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.min_btn)
        info_layout.addLayout(btn_layout)

        # Add info panel to main layout
        layout.addWidget(info_panel)

    def _toggle_minimize(self, checked):
        """Toggle between minimized and expanded view"""
        self._is_minimized = checked
        if checked:
            self.plot_widget.hide()
            self.min_btn.setText("+")
            self.setMaximumHeight(50)
            self.setMinimumHeight(40)
        else:
            self.plot_widget.show()
            self.min_btn.setText("−")
            self.setMaximumHeight(120)
            self.setMinimumHeight(100)

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
        self.curve.setSymbolBrush(color)

    def set_hover_line(self, x_val):
        """Update hover line position"""
        self.v_line.setPos(x_val)


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
        self._groups_config = []  # List of {name, channels} dicts
        self._group_widgets = {}
        self._match_bar = None
        self._timestamps = np.array([])
        self._match_scores = np.array([])
        self._first_strip = None

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
        self.placeholder = QLabel("No data loaded. Use Dataflow to load data.")
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
        self._first_strip = None

        # Hide placeholder
        self.placeholder.hide()

        # Get colors
        colors = get_chart_colors()
        
        # Identify grouped channels
        grouped_channels = set()
        for group in self._groups_config:
            for ch in group['channels']:
                if ch in channel_names:
                    grouped_channels.add(ch)
        
        # 1. Create Group Widgets
        current_color_idx = 0
        
        for group in self._groups_config:
            group_name = group['name']
            group_chs = [ch for ch in group['channels'] if ch in channel_names]
            
            if not group_chs:
                continue
                
            group_widget = ChannelGroupWidget(group_name)
            self._group_widgets[group_name] = group_widget
            self.strips_layout.addWidget(group_widget)
            
            for ch_name in group_chs:
                color = colors[current_color_idx % len(colors)]
                current_color_idx += 1
                
                strip = ChannelStrip(ch_name, color)
                self._channel_strips[ch_name] = strip
                group_widget.add_strip(strip) # Add to group instead of main layout
                
                # Link X-axis
                if self._first_strip is None:
                    self._first_strip = strip
                else:
                    strip.plot_widget.setXLink(self._first_strip.plot_widget)
                    
                strip.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

        # 2. Create Ungrouped Channels
        ungrouped = [ch for ch in channel_names if ch not in grouped_channels]
        
        if ungrouped:
            # Check if we have groups, if so, put leftovers in "Ungrouped"
            target_layout = self.strips_layout
            if self._groups_config:
                ungrouped_widget = ChannelGroupWidget("Ungrouped Channels")
                self.strips_layout.addWidget(ungrouped_widget)
                target_layout = ungrouped_widget.content_layout
                
            for ch_name in ungrouped:
                color = colors[current_color_idx % len(colors)]
                current_color_idx += 1
                
                strip = ChannelStrip(ch_name, color)
                self._channel_strips[ch_name] = strip
                target_layout.addWidget(strip)
                
                if self._first_strip is None:
                    self._first_strip = strip
                else:
                    strip.plot_widget.setXLink(self._first_strip.plot_widget)
                    
                strip.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

        self.strips_layout.addStretch()

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

    def set_channel_groups(self, groups: list):
        """Set channel group configuration"""
        self._groups_config = groups
        # If we already have strips, we might need to refresh, 
        # but usually this is called before set_channels or we can rely on next set_channels

    def clear(self):
        """Clear all channel strips"""
        for strip in self._channel_strips.values():
            strip.deleteLater()
        self._channel_strips.clear()
        self._first_strip = None

        if self._match_bar:
            self._match_bar.deleteLater()
            self._match_bar = None

        self.placeholder.show()

    def _on_mouse_move(self, pos):
        """Handle mouse move to update hover lines across all strips"""
        if not self._channel_strips or not self._first_strip:
            return

        # Check if position is within the first strip's plot area
        if self._first_strip.plot_widget.sceneBoundingRect().contains(pos):
            # Map scene point to view coordinates
            mouse_point = self._first_strip.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x_val = mouse_point.x()

            # Update hover line on all strips
            for strip in self._channel_strips.values():
                strip.set_hover_line(x_val)