NASA Space Weather Sample Data Documentation
=============================================

Source: Simulated NASA Space Weather Telemetry
Format: GeoJSON (FeatureCollection)
File: nasa_sample.json

DATA LENGTH AND TIMING
----------------------
Total Data Points: 10 features
Time Span: 9 hours (from timestamp base)
Data Interval: 1 hour between each data point

Timestamps (milliseconds since epoch):
- Point 1:  1704067200000 (2024-01-01 00:00:00 UTC)
- Point 2:  1704070800000 (2024-01-01 01:00:00 UTC)
- Point 3:  1704074400000 (2024-01-01 02:00:00 UTC)
- Point 4:  1704078000000 (2024-01-01 03:00:00 UTC) - STORM ONSET
- Point 5:  1704081600000 (2024-01-01 04:00:00 UTC) - PEAK ACTIVITY
- Point 6:  1704085200000 (2024-01-01 05:00:00 UTC) - RECOVERY PHASE
- Point 7:  1704088800000 (2024-01-01 06:00:00 UTC)
- Point 8:  1704092400000 (2024-01-01 07:00:00 UTC)
- Point 9:  1704096000000 (2024-01-01 08:00:00 UTC)
- Point 10: 1704099600000 (2024-01-01 09:00:00 UTC) - QUIET CONDITIONS

FAILURE/ANOMALY EVENTS
----------------------
The data simulates a geomagnetic storm event with these phases:

1. QUIET BASELINE (Points 1-3): Normal solar conditions
   - Kp Index: 2.0-3.3
   - Dst Index: -12 to -22 nT

2. STORM ONSET (Point 4): Sudden commencement
   - Kp Index jumps to 5.0
   - Dst drops to -48 nT
   - Plasma speed increases to 512 km/s

3. STORM PEAK (Point 5): Maximum disturbance - FAILURE CONDITION
   - Kp Index: 6.7 (NOAA G2 storm level)
   - Dst Index: -85 nT (moderate storm)
   - Magnetic field Bz: -14.5 nT (strong southward)
   - Plasma temp: 198,000 K

4. RECOVERY PHASE (Points 6-10): Gradual return to quiet
   - Kp decreases: 5.7 -> 2.0
   - Dst recovers: -68 -> -14 nT

CHANNEL DESCRIPTIONS
--------------------
1. solar_flux (SFU - Solar Flux Units)
   Range: 144.8 - 182.1
   Description: 10.7 cm radio flux, indicator of solar activity
   Normal: 70-150 SFU
   Elevated: > 150 SFU

2. proton_density (particles/cm³)
   Range: 4.8 - 12.3
   Description: Solar wind proton density
   Normal: 2-10 particles/cm³
   High: > 10 particles/cm³

3. plasma_speed (km/s)
   Range: 412.5 - 598.7
   Description: Solar wind bulk speed
   Normal: 300-450 km/s
   Fast stream: > 500 km/s

4. plasma_temp (Kelvin)
   Range: 88,500 - 198,000
   Description: Solar wind proton temperature
   Normal: 50,000-150,000 K
   Storm: > 150,000 K

5. mag_field_bt (nT - nanotesla)
   Range: 6.2 - 18.6
   Description: Total interplanetary magnetic field
   Quiet: < 10 nT
   Active: > 10 nT

6. mag_field_bz (nT)
   Range: -14.5 to -1.5
   Description: North-South component of IMF (negative = southward)
   Quiet: > -5 nT
   Storm driver: < -10 nT

7. kp_index (0-9 scale)
   Range: 2.0 - 6.7
   Description: Planetary K-index, global geomagnetic activity
   Quiet: 0-2
   Active: 3-4
   Storm: 5+
   Severe Storm: 7+

8. dst_index (nT)
   Range: -85 to -12
   Description: Disturbance Storm Time index
   Quiet: > -20 nT
   Weak Storm: -30 to -50 nT
   Moderate Storm: -50 to -100 nT
   Intense Storm: < -100 nT

ANOMALY DETECTION THRESHOLDS
----------------------------
For testing MachineIQ anomaly detection, these parameters indicate failures:

CRITICAL (Immediate Alert):
- Kp Index >= 7.0
- Dst Index <= -100 nT
- Magnetic Bz <= -15 nT

WARNING (Elevated Risk):
- Kp Index >= 5.0
- Dst Index <= -50 nT
- Plasma Speed >= 600 km/s
- Proton Density >= 15 particles/cm³

INFO (Monitoring):
- Kp Index >= 4.0
- Solar Flux >= 150 SFU
- Plasma Temp >= 150,000 K

STREAMING BEHAVIOR
------------------
When used as a data source:
- Poll Interval: Default 5 seconds
- Data cycles through all 10 points continuously
- Timestamps are updated to current wall-clock time for real-time feel
- Channels are auto-discovered on connection test

USAGE
-----
1. Open Setup Wizard
2. Select "NASA Sample Data" as data source type
3. Choose "NASA Space Weather" from dataset dropdown
4. Click "Test Connection" to verify and discover channels
5. Proceed through wizard steps (same as API connection)
6. Train model during quiet phases (Points 1-3, 8-10)
7. Monitor for anomalies during storm phases (Points 4-6)
