#!/usr/bin/env python3
"""
MachineIQ Desktop Application Launcher

Run this script to start the desktop application.

Usage:
    python run_desktop.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legacy_gui.desktop import main

if __name__ == '__main__':
    main()
