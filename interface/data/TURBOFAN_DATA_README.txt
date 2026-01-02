NASA C-MAPSS Turbofan Engine Degradation Sample Data
=====================================================

Source: NASA Prognostics Center - C-MAPSS Dataset
Format: GeoJSON (FeatureCollection)
File: turbofan_sample.json

DATA LENGTH AND TIMING
----------------------
Total Data Points: 10 features
Simulated Unit: FD001_Unit_1
Time Span: 191 operational cycles (engine run-to-failure)
Data Interval: Variable cycles (showing degradation progression)

Cycle Progression:
- Cycle 1:   RUL=191 (Healthy - Start of life)
- Cycle 20:  RUL=171 (Healthy)
- Cycle 50:  RUL=141 (Healthy)
- Cycle 80:  RUL=111 (Healthy)
- Cycle 120: RUL=71  (Early degradation signs)
- Cycle 150: RUL=41  (Degradation progressing)
- Cycle 175: RUL=16  (WARNING - Approaching failure)
- Cycle 185: RUL=6   (CRITICAL - Imminent failure)
- Cycle 190: RUL=1   (CRITICAL - Last safe cycle)
- Cycle 191: RUL=0   (FAILURE - Engine failed)

FAILURE EVENTS
--------------
Failure Mode: High Pressure Compressor (HPC) Degradation

Key Degradation Indicators:
- T30 (HPC outlet temperature): Increases from 1400.6 to 1434.25
- Nc (Core speed): Increases from 9046 to 9198 RPM
- Ps30 (HPC static pressure): Increases from 47.47 to 49.12
- phi (Fuel flow ratio): Increases from 521.66 to 525.12
- BPR (Bypass ratio): Decreases from 8.4195 to 8.2225

Anomaly Detection Points:
- Cycle 120 onwards: Detectable degradation pattern
- Cycle 175: Warning threshold (RUL < 20)
- Cycle 185: Critical threshold (RUL < 10)

CHANNEL DESCRIPTIONS
--------------------
Operational Settings (typically constant):
- T2: Total temperature at fan inlet (°R)
- P2: Pressure at fan inlet (psia)
- P15: Total pressure in bypass-duct (psia)

Temperature Sensors:
- T24: Total temperature at LPC outlet (°R)
- T30: Total temperature at HPC outlet (°R) - KEY DEGRADATION INDICATOR
- T50: Total temperature at LPT outlet (°R)

Pressure Sensors:
- P30: Total pressure at HPC outlet (psia)
- Ps30: Static pressure at HPC outlet (psia) - KEY DEGRADATION INDICATOR

Speed Sensors:
- Nf: Physical fan speed (rpm)
- Nc: Physical core speed (rpm) - KEY DEGRADATION INDICATOR
- NRf: Corrected fan speed (rpm)
- NRc: Corrected core speed (rpm)

Performance Parameters:
- epr: Engine pressure ratio
- phi: Ratio of fuel flow to Ps30 - KEY DEGRADATION INDICATOR
- BPR: Bypass ratio - KEY DEGRADATION INDICATOR
- farB: Burner fuel-air ratio
- htBleed: Bleed enthalpy

Actuators:
- Nf_dmd: Demanded fan speed (rpm)
- W31: HPT coolant bleed (lbm/s)
- W32: LPT coolant bleed (lbm/s)

Remaining Useful Life:
- RUL: Remaining cycles until failure (0 = failed)

ANOMALY DETECTION THRESHOLDS
----------------------------
For testing MachineIQ anomaly detection:

CRITICAL (Immediate Alert - RUL < 10):
- T30 > 1425°R
- Nc > 9175 RPM
- BPR < 8.26

WARNING (Elevated Risk - RUL < 30):
- T30 > 1415°R
- Nc > 9120 RPM
- BPR < 8.32
- phi > 523

CAUTION (Monitor Closely - RUL < 50):
- T30 > 1410°R
- Nc > 9100 RPM
- BPR < 8.36

USAGE
-----
1. Open web interface: python3 interface/server.py
2. Navigate to Setup Wizard
3. Select "NASA Turbofan Sample" data source
4. Test connection to discover 21 sensor channels
5. Create channel groups (e.g., Temperature, Pressure, Speed)
6. Train model on early cycles (1-80) - healthy operation
7. Monitor for anomalies as degradation progresses (cycles 120+)

STREAMING BEHAVIOR
------------------
When used as a data source:
- Poll Interval: Default 5 seconds
- Data cycles through all 10 points continuously
- Simulates real-time degradation monitoring
- RUL countdown shown in data
