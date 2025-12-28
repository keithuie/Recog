"""
MachineIQ Design System
Clean, professional, Apple-inspired styling
"""

# Color Palette
COLORS = {
    # Primary
    'primary': '#007AFF',
    'primary_hover': '#0056CC',
    'primary_pressed': '#004499',

    # Status
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'info': '#5AC8FA',

    # Neutrals
    'background': '#F5F5F7',
    'surface': '#FFFFFF',
    'surface_secondary': '#F9F9F9',
    'border': '#E5E5E7',
    'border_light': '#F0F0F2',

    # Text
    'text_primary': '#1D1D1F',
    'text_secondary': '#86868B',
    'text_tertiary': '#AEAEB2',
    'text_inverse': '#FFFFFF',

    # Chart colors
    'chart_1': '#007AFF',
    'chart_2': '#34C759',
    'chart_3': '#FF9500',
    'chart_4': '#AF52DE',
    'chart_5': '#FF3B30',
    'chart_6': '#5AC8FA',
    'chart_7': '#FFCC00',
    'chart_8': '#00C7BE',

    # Match strength gradient
    'match_high': '#34C759',
    'match_medium': '#FF9500',
    'match_low': '#FF3B30',

    # Sidebar
    'sidebar_bg': '#1D1D1F',
    'sidebar_text': '#FFFFFF',
    'sidebar_text_secondary': '#86868B',
    'sidebar_hover': '#2D2D2F',
    'sidebar_active': '#007AFF',
}

# Typography
FONTS = {
    'family': '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif',
    'mono': '"SF Mono", "Monaco", "Inconsolata", "Fira Mono", monospace',
}

# Spacing
SPACING = {
    'xs': 4,
    'sm': 8,
    'md': 16,
    'lg': 24,
    'xl': 32,
    'xxl': 48,
}

# Border radius
RADIUS = {
    'sm': 4,
    'md': 8,
    'lg': 12,
    'xl': 16,
    'round': 9999,
}


def get_auto_contrast_color(hex_color: str) -> str:
    """
    Return black or white text color based on background brightness.
    Calculates perceived brightness using standard formula.
    """
    try:
        h = hex_color.lstrip('#')
        # Handle 3-digit hex
        if len(h) == 3:
            h = ''.join([c*2 for c in h])
            
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        # Perceived brightness formula
        brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
        
        # Return black for light backgrounds, white for dark
        return '#000000' if brightness > 128 else '#FFFFFF'
    except Exception:
        # Fallback to black if parsing fails
        return '#000000'

def get_stylesheet():
    """Return the main application stylesheet"""
    return f"""
    /* Global */
    QWidget {{
        font-family: {FONTS['family']};
        font-size: 13px;
        color: {COLORS['text_primary']};
    }}

    QMainWindow {{
        background-color: {COLORS['background']};
    }}

    /* Sidebar Navigation */
    #sidebar {{
        background-color: {COLORS['sidebar_bg']};
        min-width: 220px;
        max-width: 220px;
    }}

    #sidebar QLabel {{
        color: {COLORS['sidebar_text']};
    }}

    #sidebar QPushButton {{
        background-color: transparent;
        color: {COLORS['sidebar_text_secondary']};
        border: none;
        border-radius: {RADIUS['md']}px;
        padding: 12px 16px;
        text-align: left;
        font-size: 14px;
        font-weight: 500;
    }}

    #sidebar QPushButton:hover {{
        background-color: {COLORS['sidebar_hover']};
        color: {COLORS['sidebar_text']};
    }}

    #sidebar QPushButton:checked {{
        background-color: {COLORS['sidebar_active']};
        color: {COLORS['sidebar_text']};
    }}

    /* Cards/Panels */
    .card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['lg']}px;
    }}

    QFrame[frameShape="4"] {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['lg']}px;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['text_inverse']};
        border: none;
        border-radius: {RADIUS['md']}px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 13px;
        font-weight: 600;
        font-size: 13px;
    }}

    QPushButton:hover {{
        background-color: {COLORS['primary_hover']};
    }}

    QPushButton:pressed {{
        background-color: {COLORS['primary_pressed']};
    }}

    QPushButton:disabled {{
        background-color: {COLORS['border']};
        color: {COLORS['text_tertiary']};
    }}

    QPushButton.secondary {{
        background-color: {COLORS['surface']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
    }}

    QPushButton.secondary:hover {{
        background-color: {COLORS['surface_secondary']};
        border-color: {COLORS['text_secondary']};
    }}

    QPushButton.danger {{
        background-color: {COLORS['danger']};
    }}

    QPushButton.danger:hover {{
        background-color: #E6352B;
    }}

    /* Input Fields */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {COLORS['surface']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['md']}px;
        padding: 8px 12px;
        min-height: 32px;
        font-size: 13px;
        selection-background-color: {COLORS['primary']};
        selection-color: {COLORS['text_inverse']};
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {COLORS['primary']};
        outline: none;
    }}

    QComboBox QAbstractItemView {{
        background-color: {COLORS['surface']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['primary']};
        selection-color: {COLORS['text_inverse']};
        outline: none;
    }}

    QComboBox::drop-down {{
        border: none;
        padding-right: 12px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {COLORS['text_secondary']};
        margin-right: 8px;
    }}

    /* Tables */
    QTableWidget {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['md']}px;
        gridline-color: {COLORS['border_light']};
    }}

    QTableWidget::item {{
        padding: 8px;
    }}

    QTableWidget::item:selected {{
        background-color: {COLORS['primary']};
        color: {COLORS['text_inverse']};
    }}

    QHeaderView::section {{
        background-color: {COLORS['surface_secondary']};
        color: {COLORS['text_secondary']};
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 10px;
        border: none;
        border-bottom: 1px solid {COLORS['border']};
    }}

    /* Scroll Bars */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 8px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {COLORS['border']};
        border-radius: 4px;
        min-height: 40px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['text_tertiary']};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background-color: transparent;
        height: 8px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {COLORS['border']};
        border-radius: 4px;
        min-width: 40px;
    }}

    /* Labels */
    QLabel.heading {{
        font-size: 24px;
        font-weight: 700;
        color: {COLORS['text_primary']};
    }}

    QLabel.subheading {{
        font-size: 17px;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    QLabel.caption {{
        font-size: 12px;
        color: {COLORS['text_secondary']};
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['md']}px;
        background-color: {COLORS['surface']};
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        padding: 10px 20px;
        border: none;
        font-weight: 500;
    }}

    QTabBar::tab:selected {{
        color: {COLORS['primary']};
        border-bottom: 2px solid {COLORS['primary']};
    }}

    QTabBar::tab:hover {{
        color: {COLORS['text_primary']};
    }}

    /* Progress Bar */
    QProgressBar {{
        background-color: {COLORS['border']};
        border: none;
        border-radius: {RADIUS['sm']}px;
        height: 8px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: {RADIUS['sm']}px;
    }}

    /* Checkbox */
    QCheckBox {{
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: {RADIUS['sm']}px;
        border: 2px solid {COLORS['border']};
        background-color: {COLORS['surface']};
    }}

    QCheckBox::indicator:checked {{
        background-color: {COLORS['primary']};
        border-color: {COLORS['primary']};
    }}

    /* Group Box */
    QGroupBox {{
        font-weight: 600;
        font-size: 13px;
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: {RADIUS['lg']}px;
        margin-top: 16px;
        padding-top: 16px;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        padding: 0 8px;
        background-color: {COLORS['background']};
    }}

    /* Slider */
    QSlider::groove:horizontal {{
        background-color: {COLORS['border']};
        height: 4px;
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background-color: {COLORS['primary']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:hover {{
        background-color: {COLORS['primary_hover']};
    }}

    /* Tool Tips */
    QToolTip {{
        background-color: {COLORS['sidebar_bg']};
        color: {COLORS['text_inverse']};
        border: none;
        border-radius: {RADIUS['sm']}px;
        padding: 6px 10px;
        font-size: 12px;
    }}

    /* Status indicators */
    .status-good {{
        color: {COLORS['success']};
    }}

    .status-warning {{
        color: {COLORS['warning']};
    }}

    .status-error {{
        color: {COLORS['danger']};
    }}
    """


def get_chart_colors():
    """Return list of colors for chart series"""
    return [
        COLORS['chart_1'],
        COLORS['chart_2'],
        COLORS['chart_3'],
        COLORS['chart_4'],
        COLORS['chart_5'],
        COLORS['chart_6'],
        COLORS['chart_7'],
        COLORS['chart_8'],
    ]


def get_match_color(score: float) -> str:
    """Get color based on match score (0-100)"""
    if score >= 90:
        return COLORS['match_high']
    elif score >= 70:
        return COLORS['match_medium']
    else:
        return COLORS['match_low']
