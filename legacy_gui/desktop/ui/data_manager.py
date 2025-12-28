"""
Data Manager Page

CSV import, playback controls, and data preview.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
    QFormLayout, QHeaderView, QProgressBar, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..styles import COLORS


class DataManagerPage(QWidget):
    """
    Data management interface for CSV files.

    Features:
    - CSV file import
    - Data preview table
    - Playback speed control
    - Timeline scrubbing
    """

    file_loaded = pyqtSignal(str)  # filepath
    playback_started = pyqtSignal()
    playback_paused = pyqtSignal()
    playback_stopped = pyqtSignal()
    seek_requested = pyqtSignal(int)  # index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filepath = None
        self._total_points = 0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # File Import Section
        import_group = QGroupBox("Data Source")
        import_layout = QVBoxLayout(import_group)

        file_row = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            padding: 10px 12px;
            background-color: {COLORS['surface_secondary']};
            border-radius: 6px;
        """)
        file_row.addWidget(self.file_label, stretch=1)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(100)
        self.browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.browse_btn)

        import_layout.addLayout(file_row)

        # Timestamp column selector
        ts_row = QHBoxLayout()
        ts_row.addWidget(QLabel("Timestamp Column:"))
        self.timestamp_combo = QComboBox()
        self.timestamp_combo.setMinimumWidth(200)
        self.timestamp_combo.addItem("timestamp")
        self.timestamp_combo.addItem("time")
        self.timestamp_combo.addItem("Time")
        self.timestamp_combo.addItem("datetime")
        self.timestamp_combo.setEditable(True)
        ts_row.addWidget(self.timestamp_combo)
        ts_row.addStretch()
        import_layout.addLayout(ts_row)

        layout.addWidget(import_group)

        # Playback Controls Section
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout(playback_group)

        # Control buttons
        btn_row = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(100)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_play)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #2DB84D;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
            }}
        """)
        btn_row.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setFixedWidth(100)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #E68600;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
            }}
        """)
        btn_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #E6352B;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
            }}
        """)
        btn_row.addWidget(self.stop_btn)

        btn_row.addStretch()

        # Speed control
        btn_row.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 100.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("x")
        self.speed_spin.setDecimals(1)
        self.speed_spin.setFixedWidth(80)
        btn_row.addWidget(self.speed_spin)

        playback_layout.addLayout(btn_row)

        # Timeline slider
        timeline_row = QHBoxLayout()

        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-family: monospace;")
        self.time_label.setFixedWidth(70)
        timeline_row.addWidget(self.time_label)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.sliderMoved.connect(self._on_seek)
        timeline_row.addWidget(self.timeline_slider)

        self.end_time_label = QLabel("00:00:00")
        self.end_time_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-family: monospace;")
        self.end_time_label.setFixedWidth(70)
        timeline_row.addWidget(self.end_time_label)

        playback_layout.addLayout(timeline_row)

        # Progress bar
        progress_row = QHBoxLayout()
        progress_row.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v / %m)")
        progress_row.addWidget(self.progress_bar)
        playback_layout.addLayout(progress_row)

        layout.addWidget(playback_group)

        # Data Preview Section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Info labels
        info_row = QHBoxLayout()

        self.rows_label = QLabel("Rows: 0")
        self.rows_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        info_row.addWidget(self.rows_label)

        self.cols_label = QLabel("Columns: 0")
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
        self.preview_table.setMaximumHeight(300)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)

        layout.addStretch()

    def _browse_file(self):
        """Open file browser dialog"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filepath:
            self._filepath = filepath
            self.file_label.setText(Path(filepath).name)
            self.file_label.setStyleSheet(f"""
                color: {COLORS['text_primary']};
                padding: 10px 12px;
                background-color: {COLORS['surface_secondary']};
                border-radius: 6px;
            """)
            self.file_loaded.emit(filepath)

    def set_file_info(self, rows: int, columns: list, start_time: str = None, end_time: str = None):
        """Update file info display"""
        self._total_points = rows
        self.rows_label.setText(f"Rows: {rows:,}")
        self.cols_label.setText(f"Columns: {len(columns)}")

        if start_time and end_time:
            self.time_range_label.setText(f"Time Range: {start_time} to {end_time}")
            self.end_time_label.setText(end_time.split()[-1] if ' ' in end_time else end_time)

        # Update slider
        self.timeline_slider.setRange(0, rows - 1)
        self.timeline_slider.setEnabled(True)

        # Update progress bar
        self.progress_bar.setRange(0, rows)
        self.progress_bar.setValue(0)

        # Enable playback
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        # Update table columns
        self.preview_table.setColumnCount(len(columns))
        self.preview_table.setHorizontalHeaderLabels(columns)

        # Auto-detect timestamp column
        for col in columns:
            if col.lower() in ['timestamp', 'time', 'datetime']:
                index = self.timestamp_combo.findText(col)
                if index >= 0:
                    self.timestamp_combo.setCurrentIndex(index)
                else:
                    self.timestamp_combo.setCurrentText(col)
                break

    def set_preview_data(self, rows: list):
        """Set preview table data"""
        self.preview_table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)

    def update_playback_position(self, index: int, timestamp: str = None):
        """Update current playback position"""
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(index)
        self.timeline_slider.blockSignals(False)

        if timestamp:
            time_part = timestamp.split()[-1] if ' ' in timestamp else timestamp
            self.time_label.setText(time_part)

        self.progress_bar.setValue(index)

    def get_playback_speed(self) -> float:
        """Get current playback speed"""
        return self.speed_spin.value()

    def get_timestamp_column(self) -> str:
        """Get selected timestamp column name"""
        return self.timestamp_combo.currentText()

    def _on_play(self):
        self.playback_started.emit()

    def _on_pause(self):
        self.playback_paused.emit()

    def _on_stop(self):
        self.playback_stopped.emit()
        self.update_playback_position(0, "00:00:00")

    def _on_seek(self, value: int):
        self.seek_requested.emit(value)
