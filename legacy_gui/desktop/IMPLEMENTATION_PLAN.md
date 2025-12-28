# MachineIQ GUI Implementation Plan

## Executive Summary

This document outlines the improvements needed to transform the current GUI from a basic representation into a fully functional industrial monitoring application. The core ML mathematics remain untouched - all changes are UI/UX layer only.

---

## Current State Analysis

### Navigation Structure (Current)
```
Sidebar:
├── Dashboard         - Channel strips, match strength display
├── Data Manager      - CSV import, playback controls, preview table
├── Connections       - Wizard for API/MQTT/OPC-UA connections
├── Model Config      - Kernel, bins, width, threshold, groups, training
└── Alarms           - Alarm conditions, email config, history
```

### Identified Issues

#### 1. Duplicate/Redundant Controls
- **Playback controls appear twice**: Header bar AND Data Manager page
- Solution: Remove from Data Manager, keep only in header

#### 2. Poor Information Architecture
- **Data Manager and Connections are too similar** - both deal with data sources
- Users must switch between pages for related tasks
- Solution: Merge into unified "Dataflow" page

#### 3. Training Location
- **Training is buried in Model Config** - should be more prominent
- Training should be per-group and accessible from Dashboard
- Solution: Move to Dashboard with group-specific training

#### 4. Missing Features for New Users
- **No sample data** for testing without real data
- **No CSV template** to help format data correctly
- **No tooltips** explaining parameters
- Solution: Add sample generator, template download, tooltips

#### 5. UI Polish Issues
- Inconsistent padding/spacing
- Some text boxes need size adjustments
- Missing help text for connection types

---

## Proposed Navigation Structure

```
Sidebar:
├── Dashboard         - Channel strips, match strength, group training
├── Dataflow          - All data source management (merged)
├── Model Config      - Kernel, bins, width, channel groups
└── Alarms           - Alarm conditions, notifications
```

**Changes:**
- Remove "Data Manager" and "Connections"
- Add "Dataflow" (combines both)
- "Model Config" loses Training section (moves to Dashboard)

---

## Detailed Implementation Plan

### Phase 1: Dataflow Page (Priority: HIGH)

**New file:** `gui/desktop/ui/dataflow.py`

**Sections:**
```
┌─────────────────────────────────────────────────────────────┐
│  DATAFLOW                                    [+ Add Source] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─ Active Data Sources ──────────────────────────────────┐ │
│  │  [Card: CSV File - motor_data.csv - Playing]           │ │
│  │  [Card: REST API - sensor-api - Connected]             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Quick Actions ────────────────────────────────────────┐ │
│  │  [Upload CSV]  [Generate Sample Data]  [Download       │ │
│  │                                         Template]      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Data Preview ─────────────────────────────────────────┐ │
│  │  Rows: 10,000  |  Channels: 4  |  Time: 1h 23m         │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  timestamp  │ ch1    │ ch2    │ ch3    │ ch4     │  │ │
│  │  │  ...        │ ...    │ ...    │ ...    │ ...     │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
1. **Add Source Button** → Opens wizard (from current connection_center.py)
2. **Active Sources List** → Cards showing status, connect/disconnect
3. **Quick Actions:**
   - Upload CSV (from data_manager.py)
   - Generate Sample Data (new feature)
   - Download Template (new feature)
4. **Data Preview** → Shows currently active data

**Sample Data Generator Config:**
```
┌─ Generate Sample Data ─────────────────────────┐
│                                                │
│  Duration:  [____5____] [minutes ▼]            │
│                                                │
│  Channels:  [____4____]                        │
│                                                │
│  Include Anomaly: [x] Yes  [ ] No              │
│  Anomaly at: [___70___] % through data         │
│                                                │
│  [Generate]  [Cancel]                          │
└────────────────────────────────────────────────┘
```

**CSV Template Download:**
- Generates sample CSV with correct format
- Includes timestamp column and example channel columns
- Provides comments/documentation row

**Connection Type Help Text:**
```python
HELP_TEXT = {
    'rest_api': """
        REST API connections poll a URL at regular intervals.

        Requirements:
        • Endpoint must return JSON array or object
        • Each response should contain channel values
        • Timestamp can be in data or generated automatically

        Example response format:
        {"timestamp": "2024-01-01T00:00:00", "ch1": 45.2, "ch2": 12.1}
    """,
    'mqtt': """
        MQTT connections subscribe to topics on a broker.

        Requirements:
        • Broker must support MQTT 3.1.1 or 5.0
        • Topic wildcards (+, #) are supported
        • Payload should be JSON with channel values

        Example topic: sensors/machine1/+
    """,
    # ... etc
}
```

---

### Phase 2: Sample Data Generator (Priority: HIGH)

**Enhance:** `miq/utils/data_generator.py`

**New function:**
```python
def generate_vibration_data(
    duration_minutes: int,
    num_channels: int = 4,
    sample_rate_hz: float = 10.0,
    include_anomaly: bool = False,
    anomaly_position: float = 0.7,
    channel_names: List[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate realistic vibration monitoring data.

    Args:
        duration_minutes: Length of data in minutes
        num_channels: Number of sensor channels
        sample_rate_hz: Samples per second
        include_anomaly: Whether to inject anomaly
        anomaly_position: Where anomaly starts (0-1)
        channel_names: Optional custom names

    Returns:
        DataFrame with timestamp and channel columns
        Metadata dict with generation parameters
    """
```

**Vibration-like patterns:**
- Base sinusoidal with multiple frequencies (motor harmonics)
- Random noise overlay
- Slow drift (temperature effects)
- Optional fault injection (bearing defect frequencies)

---

### Phase 3: Dashboard Training Widget (Priority: HIGH)

**Modify:** `gui/desktop/ui/dashboard.py`

**Add Training Panel per Group:**
```
┌─ Channel Group: Motor Bearings ────────────────────────────┐
│                                                            │
│  [DE Radial Accel] [DE Axial Accel] [Temperature]         │
│                                                            │
│  Training Status: ● Not Trained                           │
│                                                            │
│  Training Period:                                          │
│  [ ] Last  [___5___] [minutes ▼]                          │
│  [x] Custom Range: [2024-01-01 00:00] to [2024-01-01 05:00]│
│                                                            │
│  [▶ Train This Group]                                      │
│                                                            │
│  Progress: [████████░░] 80%  Learning normal patterns...   │
└────────────────────────────────────────────────────────────┘
```

**Training Duration Options:**
```python
TRAINING_PERIODS = [
    ('minutes', [1, 5, 10, 15, 30]),
    ('hours', [1, 2, 4, 8, 12, 24]),
    ('days', [1, 2, 3, 7, 14]),
    ('weeks', [1, 2, 4]),
    ('months', [1, 2, 3, 6]),
]
```

**Integration with Core ML:**
- Training uses `MachineIQDetector.learn_pattern()`
- Training data selected by time range from loaded data
- Progress updates via signals
- Trained state persisted per group

---

### Phase 4: Tooltips System (Priority: MEDIUM)

**Create:** `gui/desktop/ui/tooltips.py`

**Tooltip Content:**
```python
TOOLTIPS = {
    # Model Config
    'kernel_type': """
        Kernel Type determines how the model scores pattern matches.

        Triangular: Linear falloff from center. Good for general use.
        Simple and fast, works well when states are well-separated.

        Parabolic: Quadratic falloff from center. More sensitive.
        Better at distinguishing similar states, but may need more bins.
    """,

    'bins_per_channel': """
        Bins divide each channel's value range into discrete states.

        More bins = finer granularity, captures subtle changes
        Fewer bins = coarser, more robust to noise

        Recommended: 32-128 depending on data complexity.
        Start with 64 and adjust based on results.
    """,

    'kernel_width': """
        Kernel Width controls matching strictness.

        Lower values (0.1-0.3): Stricter matching, fewer false matches
        Higher values (0.5-1.0): More tolerant, may miss subtle anomalies

        Typical range: 0.3-0.6 for industrial monitoring.
    """,

    'threshold': """
        Alarm Threshold sets when alerts trigger.

        Match scores below this value indicate the current state
        doesn't match learned normal patterns.

        Higher threshold = More sensitive, more alerts
        Lower threshold = Less sensitive, fewer alerts

        Start around 70-80% and tune based on false alarm rate.
    """,

    # Dataflow
    'sample_rate': """
        How frequently data points are recorded.

        Higher rates capture faster dynamics but use more storage.
        Typical industrial monitoring: 1-100 Hz
    """,
}
```

**Implementation:**
- Use `QToolTip.setToolTip()` with rich text
- Add `[?]` icons next to complex parameters
- Implement hover delay for better UX

---

### Phase 5: TagoCore Integration (Priority: LOW - Investigation)

**Research findings:**
- TagoCore is an edge IoT platform by TagoIO
- Supports plugins: MQTT, databases, custom integrations
- Has HTTPS and MQTT connectivity built-in
- GitHub repos: `mqtt-relay` (Rust), `tagocore` (TypeScript)
- Could serve as a data aggregation layer

**Integration approach:**
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Sensors   │────▶│   TagoCore   │────▶│  MachineIQ  │
│  (MQTT/API) │     │  (edge node) │     │  (analysis) │
└─────────────┘     └──────────────┘     └─────────────┘
```

**Benefits:**
- Unified data ingestion from multiple protocols
- Edge preprocessing
- Established ecosystem with good documentation

**Implementation:**
- Add TagoCore as optional data source type
- Connect via TagoCore's MQTT output or REST API
- Provide setup instructions in connection wizard

**References:**
- [TagoCore Features](https://tago.io/blog/tagocore-new-features-for-iot-edge-computing)
- [MQTT Protocol with TagoIO](https://tago.io/events/using-mqtt-protocol-with-the-tagoio-iot-platform)
- [TagoIO GitHub](https://github.com/tago-io)

---

### Phase 6: UI Polish (Priority: MEDIUM)

**Spacing/Padding Fixes:**

| Element | Current | Target |
|---------|---------|--------|
| GroupBox padding | 12px | 16px |
| Form field spacing | 12px | 16px |
| Card margins | 16px | 20px |
| Button padding | 10px 20px | 12px 24px |

**Text Box Sizing:**
- Spinboxes: min-width 80px → 100px
- Line edits: consistent 200px minimum
- ComboBoxes: match line edit widths

**Consistent Component Heights:**
- Input fields: 40px
- Buttons: 40px (primary), 36px (secondary)
- Cards: min-height 80px

---

## File Changes Summary

### New Files
- `gui/desktop/ui/dataflow.py` - Unified data source management
- `gui/desktop/ui/tooltips.py` - Tooltip content and helpers

### Modified Files
- `gui/desktop/ui/main_window.py` - Update sidebar navigation
- `gui/desktop/ui/dashboard.py` - Add group training widgets
- `gui/desktop/ui/model_config.py` - Remove training section
- `gui/desktop/ui/setup_wizard.py` - Update data source step
- `gui/desktop/app.py` - Update page routing
- `gui/desktop/styles.py` - Spacing/sizing adjustments
- `miq/utils/data_generator.py` - Enhanced sample generation

### Removed/Deprecated
- `gui/desktop/ui/data_manager.py` - Merged into dataflow
- `gui/desktop/ui/connection_center.py` - Merged into dataflow

---

## Implementation Order

1. **Phase 1: Dataflow Page** (Est. 3-4 hours)
   - Create unified dataflow.py
   - Migrate functionality from data_manager and connection_center
   - Update sidebar navigation

2. **Phase 2: Sample Generator** (Est. 1-2 hours)
   - Enhance data_generator.py
   - Add UI in dataflow page
   - Add CSV template download

3. **Phase 3: Dashboard Training** (Est. 2-3 hours)
   - Add training widget to dashboard
   - Implement time period selection
   - Connect to ML training functions

4. **Phase 4: Tooltips** (Est. 1-2 hours)
   - Create tooltip content
   - Add to all parameters
   - Test rendering

5. **Phase 5: TagoCore** (Est. 2-3 hours)
   - Add as connection type
   - Implement connector
   - Add setup documentation

6. **Phase 6: UI Polish** (Est. 1-2 hours)
   - Update styles.py
   - Fix spacing throughout
   - Test on different screen sizes

---

## Notes

- **Core ML is UNTOUCHED** - Only UI layer changes
- **Wizard remains** - Can be launched from Dataflow page
- **Backwards compatible** - Existing configs should still work
- **Progressive disclosure** - Simple by default, advanced options available
