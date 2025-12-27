"""
MachineIQ Desktop UI Components
"""

from .main_window import MainWindow
from .dashboard import DashboardPage
from .dataflow import DataflowPage
from .model_config import ModelConfigPage
from .alarm_center import AlarmCenterPage
from .setup_wizard import SetupWizard

__all__ = [
    'MainWindow',
    'DashboardPage',
    'DataflowPage',
    'ModelConfigPage',
    'AlarmCenterPage',
    'SetupWizard',
]
