"""
MachineIQ Desktop Application

Main entry point for the professional desktop application.
Provides industrial condition monitoring with multivariate anomaly detection.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont

from .styles import get_stylesheet, COLORS
from .ui import (
    MainWindow,
    DashboardPage,
    DataflowPage,
    ModelConfigPage,
    AlarmCenterPage,
    SetupWizard
)
from .core import (
    ChannelGroupManager,
    DataPlayer,
    AlarmManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MachineIQApp:
    """
    Main application controller for MachineIQ.

    Coordinates between UI components and core functionality:
    - Data playback and streaming
    - Anomaly detection
    - Alarm management
    - Channel groups
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("MachineIQ")
        self.app.setApplicationVersion("0.1.0")
        self.app.setOrganizationName("MachineIQ")

        # Apply global stylesheet
        self.app.setStyleSheet(get_stylesheet())

        # Set default font
        font = QFont("Inter", 10)
        if not font.exactMatch():
            font = QFont("Segoe UI", 10)
        self.app.setFont(font)

        # Core components
        self.channel_manager = ChannelGroupManager()
        self.alarm_manager = AlarmManager()
        self.data_player: Optional[DataPlayer] = None

        # UI components
        self.main_window: Optional[MainWindow] = None
        self.dashboard: Optional[DashboardPage] = None
        self.dataflow: Optional[DataflowPage] = None
        self.model_config: Optional[ModelConfigPage] = None
        self.alarm_center: Optional[AlarmCenterPage] = None

        # Detector reference
        self._detector = None

        # Initialize UI
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initialize all UI components"""
        logger.info("Initializing UI components...")

        # Create main window
        self.main_window = MainWindow()

        # Create pages
        self.dashboard = DashboardPage()
        self.dataflow = DataflowPage()
        self.model_config = ModelConfigPage()
        self.alarm_center = AlarmCenterPage()

        # Add pages to main window
        self.main_window.add_page("dashboard", self.dashboard, "Dashboard")
        self.main_window.add_page("dataflow", self.dataflow, "Dataflow")
        self.main_window.add_page("model", self.model_config, "Model Configuration")
        self.main_window.add_page("alarms", self.alarm_center, "Alarm Center")

        # Show dashboard by default
        self.main_window.show_page("dashboard")

        logger.info("UI initialization complete")

    def _connect_signals(self):
        """Connect signals between components"""
        # Setup wizard button
        self.main_window.sidebar.wizard_btn.clicked.connect(self._show_setup_wizard)

        # Header playback controls
        header = self.main_window.header
        header.play_btn.clicked.connect(self._on_play)
        header.pause_btn.clicked.connect(self._on_pause)

        # Dataflow signals
        self.dataflow.source_added.connect(self._on_source_added)
        self.dataflow.source_connected.connect(self._on_source_connect)
        self.dataflow.source_disconnected.connect(self._on_source_disconnect)
        self.dataflow.file_loaded.connect(self._on_csv_file_loaded)

        # Model config signals
        self.model_config.training_started.connect(self._on_training_started)
        self.model_config.training_completed.connect(self._on_training_completed)

        # Alarm manager callback
        self.alarm_manager.alarm_triggered_callback = self._on_alarm_triggered

    def _show_setup_wizard(self):
        """Show the setup wizard dialog"""
        wizard = SetupWizard(self.main_window)
        wizard.wizard_completed.connect(self._apply_wizard_config)
        wizard.exec()

    def _apply_wizard_config(self, config: dict):
        """Apply configuration from setup wizard"""
        logger.info("Applying wizard configuration...")

        # Apply data source config
        data_source = config.get('data_source', {})
        source_type = data_source.get('source_type', '')

        if source_type == 'CSV File Upload':
            file_path = data_source.get('file_path')
            if file_path:
                self._setup_csv_source(file_path, data_source.get('timestamp_column', 'timestamp'))

        elif source_type == 'REST API':
            url = data_source.get('url')
            interval = data_source.get('interval', 5)
            if url:
                self._setup_api_source(url, interval)

        # Apply model config
        model = config.get('model', {})
        if model:
            self.model_config.kernel_combo.setCurrentText(
                model.get('kernel_type', 'triangular').title()
            )
            self.model_config.bins_spin.setValue(model.get('num_bins', 100))
            self.model_config.width_spin.setValue(model.get('kernel_width', 0.1))
            self.model_config.threshold_spin.setValue(
                int(model.get('threshold', 0.3) * 100)
            )

        # Apply channel groups
        channels = config.get('channels', {})
        for group in channels.get('groups', []):
            self.channel_manager.create_group(
                name=group['name'],
                channels=group['channels']
            )
            self.model_config.add_group_card(
                group['name'],
                group['channels']
            )

        # Pass groups to dashboard
        self.dashboard.set_channel_groups(channels.get('groups', []))

        # Apply alarm config
        alarms = config.get('alarms', {})
        email_config = alarms.get('email', {})
        if email_config.get('enabled'):
            self.alarm_manager.configure_email(
                smtp_server=email_config.get('smtp_server', ''),
                smtp_port=email_config.get('smtp_port', 587),
                username=email_config.get('username', ''),
                password=email_config.get('password', ''),
                from_address=email_config.get('username', '')
            )
            for recipient in email_config.get('recipients', []):
                self.alarm_manager.add_recipient(recipient)

        for condition in alarms.get('conditions', []):
            self.alarm_manager.add_condition(
                name=f"{condition['severity']} Alert",
                condition_type=condition['type'],
                threshold=condition['threshold'],
                severity=condition['severity']
            )

        # Update status
        self.main_window.header.set_status("Configured", connected=True)
        self.main_window.header.set_playback_enabled(True)

        # Switch to dashboard
        self.main_window.show_page("dashboard")

        logger.info("Wizard configuration applied successfully")

    def _setup_csv_source(self, filepath: str, timestamp_col: str = 'timestamp'):
        """Set up CSV data source"""
        from .core import CSVDataSource, DataPlayer

        logger.info(f"Setting up CSV source: {filepath}")

        # Create data player if needed
        if not self.data_player:
            self.data_player = DataPlayer()
            self.data_player.add_data_callback(self._on_data_received)

        # Create CSV source
        source = CSVDataSource("CSV File", filepath)
        source.set_timestamp_column(timestamp_col)

        self.data_player.add_source(source)
        self.data_player.set_active_source("CSV File")

        if self.data_player.connect():
            channels = self.data_player.channels
            logger.info(f"CSV loaded with channels: {channels}")

            # Update dashboard with channels
            self.dashboard.set_channels(channels)

            # Update model config with available channels
            self.model_config.set_available_channels(channels)

            self.main_window.header.set_status("Data loaded", connected=True)
            self.main_window.header.set_playback_enabled(True)
        else:
            logger.error("Failed to load CSV file")
            self.main_window.header.set_status("Load failed", connected=False)

    def _setup_api_source(self, url: str, interval: int = 5):
        """Set up REST API data source"""
        from .core import APIDataSource, DataPlayer

        logger.info(f"Setting up API source: {url}")

        # Create data player if needed
        if not self.data_player:
            self.data_player = DataPlayer()
            self.data_player.add_data_callback(self._on_data_received)

        # Create API source
        source = APIDataSource("REST API", url, poll_interval=float(interval))
        source.set_status_callback(self._on_api_status)

        self.data_player.add_source(source)
        self.data_player.set_active_source("REST API")

        # Connect (this will test and discover channels)
        if self.data_player.connect():
            channels = self.data_player.channels
            logger.info(f"API connected with channels: {channels}")

            # Update dashboard with channels
            self.dashboard.set_channels(channels)

            # Update model config with available channels
            self.model_config.set_available_channels(channels)

            self.main_window.header.set_status("Monitoring", connected=True, monitoring=True)
            self.main_window.header.set_playback_enabled(True)
        else:
            logger.error("Failed to connect to API")
            self.main_window.header.set_status("Connection failed", connected=False)

    def _on_api_status(self, status: str):
        """Handle API status updates"""
        self.main_window.header.set_status(status, connected=True, monitoring=True)

    def _on_data_received(self, data_point):
        """Handle incoming data from data player"""
        import numpy as np

        # Get recent data from buffer
        buffer = self.data_player.buffer
        if not buffer:
            return

        # Update each channel on dashboard
        for channel in self.data_player.channels:
            timestamps, values = self.data_player.get_channel_data(channel, limit=500)
            if len(timestamps) > 0:
                self.dashboard.update_channel_data(channel, timestamps, values)

        # TODO: Run through detector and update match strength
        # For now, simulate match strength
        if len(buffer) > 10:
            timestamps = np.array([p.unix_timestamp for p in buffer[-100:]])
            # Simulate high match score (will be replaced with actual detector)
            scores = np.ones(len(timestamps)) * 95 + np.random.randn(len(timestamps)) * 2
            scores = np.clip(scores, 0, 100)
            self.dashboard.update_match_strength(timestamps, scores)

    def _on_play(self):
        """Handle play button click"""
        if self.data_player:
            source = self.data_player.active_source
            if source and hasattr(source, 'play'):
                source.play()
            self.main_window.header.set_status("Monitoring", connected=True, monitoring=True)

    def _on_pause(self):
        """Handle pause button click"""
        if self.data_player:
            source = self.data_player.active_source
            if source:
                if hasattr(source, 'pause'):
                    source.pause()
                elif hasattr(source, 'disconnect'):
                    source.disconnect()
            self.main_window.header.set_status("Paused", connected=True)

    def _on_source_added(self, config: dict):
        """Handle new data source added from Dataflow page"""
        source_type = config.get('type', '')
        name = config.get('name', 'Unknown')

        logger.info(f"Data source added: {name} ({source_type})")

        if source_type == 'csv':
            # CSV sources are auto-connected when added
            pass
        elif source_type == 'api':
            # Store config for later connection
            url = config.get('url', '')
            interval = config.get('interval', 5)
            if url:
                self._setup_api_source(url, interval)
                self.dataflow.set_source_status(name, "Monitoring")

    def _on_source_connect(self, name: str):
        """Handle source connect request from Dataflow page"""
        logger.info(f"Connecting to source: {name}")
        config = self.dataflow.get_source_config(name)

        if config.get('type') == 'api':
            self._setup_api_source(config.get('url', ''), config.get('interval', 5))
            self.dataflow.set_source_status(name, "Monitoring")
        elif config.get('type') == 'csv':
            self._setup_csv_source(config.get('path', ''), config.get('timestamp_column', 'timestamp'))
            self.dataflow.set_source_status(name, "Connected")

    def _on_source_disconnect(self, name: str):
        """Handle source disconnect request from Dataflow page"""
        logger.info(f"Disconnecting from source: {name}")

        if self.data_player:
            source = self.data_player.active_source
            if source:
                source.disconnect()
            self.dataflow.set_source_status(name, "Disconnected")
            self.main_window.header.set_status("Disconnected", connected=False)

    def _on_csv_file_loaded(self, filepath: str):
        """Handle CSV file loaded from Dataflow page"""
        logger.info(f"CSV file loaded: {filepath}")
        self._setup_csv_source(filepath, 'timestamp')

        # Update dataflow with file info
        if self.data_player and self.data_player.active_source:
            source = self.data_player.active_source
            if hasattr(source, 'data') and source.data is not None:
                df = source.data
                columns = list(df.columns)
                rows = len(df)
                start_time = str(df.iloc[0]['timestamp']) if 'timestamp' in df.columns else None
                end_time = str(df.iloc[-1]['timestamp']) if 'timestamp' in df.columns else None

                self.dataflow.set_file_info(rows, columns, start_time, end_time)

                # Set preview data (first 10 rows)
                preview_rows = []
                for i in range(min(10, len(df))):
                    preview_rows.append([str(df.iloc[i][col]) for col in columns])
                self.dataflow.set_preview_data(preview_rows)

    def _on_training_started(self):
        """Handle training started signal"""
        self.main_window.header.set_status("Training model...", connected=True)

    def _on_training_completed(self, detector):
        """Handle training completed signal"""
        self._detector = detector
        self.main_window.header.set_status("Model ready", connected=True)
        self.main_window.header.set_playback_enabled(True)
        logger.info("Model training completed")

    def _on_alarm_triggered(self, alarm):
        """Handle alarm triggered"""
        # Add to alarm center
        self.alarm_center.add_alarm({
            'time': alarm.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'severity': alarm.severity.name,
            'condition': alarm.condition_name,
            'message': alarm.message,
            'score': f"{alarm.value:.1f}",
            'status': 'Active' if not alarm.acknowledged else 'Acknowledged'
        })

        logger.warning(f"Alarm triggered: {alarm.condition_name} - {alarm.message}")

    def _process_data_point(self, data: dict):
        """Process incoming data point through detector"""
        if not self._detector:
            return

        # Update dashboard with raw data
        for channel, value in data.items():
            if channel != 'timestamp':
                self.dashboard.update_channel(channel, value)

        # Run through detector
        # This would integrate with the core ML system

    def run(self) -> int:
        """Run the application"""
        logger.info("Starting MachineIQ application...")

        # Show main window
        self.main_window.show()

        # Check if first run - show wizard
        # In a real app, we'd check a config file
        QTimer.singleShot(500, self._check_first_run)

        return self.app.exec()

    def _check_first_run(self):
        """Check if this is first run and offer wizard"""
        # For demo, always offer wizard
        result = QMessageBox.question(
            self.main_window,
            "Welcome to MachineIQ",
            "Would you like to run the Setup Wizard to configure your system?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if result == QMessageBox.StandardButton.Yes:
            self._show_setup_wizard()


def main():
    """Main entry point"""
    try:
        app = MachineIQApp()
        sys.exit(app.run())
    except Exception as e:
        logger.exception("Fatal error in application")
        QMessageBox.critical(
            None,
            "Application Error",
            f"A fatal error occurred:\n\n{str(e)}\n\nThe application will now close."
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
