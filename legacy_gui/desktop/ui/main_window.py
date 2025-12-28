"""
MachineIQ Main Window

Professional desktop application with sidebar navigation.
"""

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame,
    QSizePolicy, QSpacerItem, QApplication
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon

from ..styles import COLORS, get_stylesheet


class SidebarButton(QPushButton):
    """Custom sidebar navigation button"""

    def __init__(self, text: str, icon_text: str = None, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setMinimumHeight(44)
        self._icon_text = icon_text

    def set_active(self, active: bool):
        self.setChecked(active)


class Sidebar(QFrame):
    """Sidebar navigation panel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setMinimumWidth(220)
        self.setMaximumWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(4)

        # Logo/Title
        title = QLabel("MachineIQ")
        title.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 700;
            color: {COLORS['sidebar_text']};
            padding: 8px 12px;
            margin-bottom: 16px;
        """)
        layout.addWidget(title)

        # Navigation buttons
        self.buttons = {}
        nav_items = [
            ("dashboard", "Dashboard"),
            ("dataflow", "Dataflow"),
            ("model", "Model Config"),
            ("alarms", "Alarms"),
        ]

        for key, label in nav_items:
            btn = SidebarButton(label)
            btn.setObjectName(f"nav_{key}")
            self.buttons[key] = btn
            layout.addWidget(btn)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Setup Wizard button
        self.wizard_btn = QPushButton("Setup Wizard")
        self.wizard_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['sidebar_text_secondary']};
                border: 1px solid {COLORS['sidebar_text_secondary']};
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['sidebar_hover']};
                color: {COLORS['sidebar_text']};
                border-color: {COLORS['sidebar_text']};
            }}
        """)
        layout.addWidget(self.wizard_btn)

        # Version info
        version = QLabel("v0.1.0")
        version.setStyleSheet(f"""
            color: {COLORS['sidebar_text_secondary']};
            font-size: 11px;
            padding: 8px 12px;
        """)
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

    def set_active_page(self, page_key: str):
        """Set the active navigation button"""
        for key, btn in self.buttons.items():
            btn.set_active(key == page_key)


class HeaderBar(QFrame):
    """Top header bar with status and controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        self.setFixedHeight(60)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)

        # Page title
        self.title_label = QLabel("Dashboard")
        self.title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Status indicator
        self.status_container = QWidget()
        status_layout = QHBoxLayout(self.status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_dot.setStyleSheet(f"""
            background-color: {COLORS['text_tertiary']};
            border-radius: 4px;
        """)
        status_layout.addWidget(self.status_dot)

        self.status_label = QLabel("Not Connected")
        self.status_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 13px;
        """)
        status_layout.addWidget(self.status_label)

        layout.addWidget(self.status_container)

        # Play/Pause controls
        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(80)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #2DB84D;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_tertiary']};
            }}
        """)
        self.play_btn.setEnabled(False)
        layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setFixedWidth(80)
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #E68600;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_tertiary']};
            }}
        """)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)

    def set_title(self, title: str):
        self.title_label.setText(title)

    def set_status(self, status: str, connected: bool = False, monitoring: bool = False):
        self.status_label.setText(status)

        if monitoring:
            color = COLORS['success']
        elif connected:
            color = COLORS['primary']
        else:
            color = COLORS['text_tertiary']

        self.status_dot.setStyleSheet(f"""
            background-color: {color};
            border-radius: 4px;
        """)

    def set_playback_enabled(self, enabled: bool):
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MachineIQ - Industrial Condition Monitoring")
        self.setMinimumSize(1280, 800)

        # Apply stylesheet
        self.setStyleSheet(get_stylesheet())

        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Content area
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Header
        self.header = HeaderBar()
        content_layout.addWidget(self.header)

        # Page stack
        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet(f"""
            QStackedWidget {{
                background-color: {COLORS['background']};
            }}
        """)
        content_layout.addWidget(self.page_stack)

        main_layout.addWidget(content_area)

        # Initialize pages (will be populated by app)
        self.pages = {}

        # Connect sidebar buttons
        for key, btn in self.sidebar.buttons.items():
            btn.clicked.connect(lambda checked, k=key: self._on_nav_click(k))

        # Set initial page
        self.sidebar.set_active_page("dashboard")

    def add_page(self, key: str, page: QWidget, title: str):
        """Add a page to the stack"""
        self.pages[key] = {'widget': page, 'title': title}
        self.page_stack.addWidget(page)

    def show_page(self, key: str):
        """Show a specific page"""
        if key in self.pages:
            page_info = self.pages[key]
            self.page_stack.setCurrentWidget(page_info['widget'])
            self.header.set_title(page_info['title'])
            self.sidebar.set_active_page(key)

    def _on_nav_click(self, key: str):
        """Handle navigation button click"""
        self.show_page(key)

    def get_sidebar(self) -> Sidebar:
        return self.sidebar

    def get_header(self) -> HeaderBar:
        return self.header
