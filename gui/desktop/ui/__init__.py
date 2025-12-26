"""
MachineIQ Desktop UI Components
"""

from .main_window import MainWindow
from .dashboard import DashboardPage
from .data_manager import DataManagerPage
from .connection_center import ConnectionCenterPage
from .model_config import ModelConfigPage
from .alarm_center import AlarmCenterPage
from .setup_wizard import SetupWizard

__all__ = [
    'MainWindow',
    'DashboardPage',
    'DataManagerPage',
    'ConnectionCenterPage',
    'ModelConfigPage',
    'AlarmCenterPage',
    'SetupWizard',
]
