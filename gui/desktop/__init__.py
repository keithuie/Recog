"""
MachineIQ Desktop Application

Professional desktop interface for industrial condition monitoring
with multivariate anomaly detection.
"""

__version__ = "0.1.0"

from .app import main, MachineIQApp

__all__ = ['main', 'MachineIQApp', '__version__']
