# NASA CMAPSS Turbofan Engine Failure Reference

## Dataset Overview
- **Source**: NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)
- **Dataset**: FD001 (Train)
- **Total Engines**: 100
- **Sensors**: 21 measurements per cycle
- **Operating Conditions**: Single (sea level static)

## Engine Failure Times (Cycles to Failure)

| Engine # | Total Cycles | Failure Point | Category |
|----------|--------------|---------------|----------|
| 1 | 192 | Cycle 192 | Medium |
| 2 | 287 | Cycle 287 | Long |
| 3 | 179 | Cycle 179 | Medium |
| 4 | 189 | Cycle 189 | Medium |
| 5 | 128 | Cycle 128 | Short |
| 6 | 236 | Cycle 236 | Long |
| 7 | 134 | Cycle 134 | Short |
| 8 | 192 | Cycle 192 | Medium |
| 9 | 219 | Cycle 219 | Long |
| 10 | 156 | Cycle 156 | Medium |
| 11 | 182 | Cycle 182 | Medium |
| 12 | 169 | Cycle 169 | Medium |
| 13 | 170 | Cycle 170 | Medium |
| 14 | 199 | Cycle 199 | Medium |
| 15 | 205 | Cycle 205 | Long |
| 16 | 223 | Cycle 223 | Long |
| 17 | 149 | Cycle 149 | Short |
| 18 | 213 | Cycle 213 | Long |
| 19 | 235 | Cycle 235 | Long |
| 20 | 206 | Cycle 206 | Long |
| 21 | 147 | Cycle 147 | Short |
| 22 | 207 | Cycle 207 | Long |
| 23 | 155 | Cycle 155 | Medium |
| 24 | 134 | Cycle 134 | Short |
| 25 | 181 | Cycle 181 | Medium |
| 26 | 175 | Cycle 175 | Medium |
| 27 | 214 | Cycle 214 | Long |
| 28 | 143 | Cycle 143 | Short |
| 29 | 175 | Cycle 175 | Medium |
| 30 | 160 | Cycle 160 | Medium |
| 31 | 143 | Cycle 143 | Short |
| 32 | 203 | Cycle 203 | Long |
| 33 | 213 | Cycle 213 | Long |
| 34 | 133 | Cycle 133 | Short |
| 35 | 173 | Cycle 173 | Medium |
| 36 | 162 | Cycle 162 | Medium |
| 37 | 189 | Cycle 189 | Medium |
| 38 | 165 | Cycle 165 | Medium |
| 39 | 220 | Cycle 220 | Long |
| 40 | 159 | Cycle 159 | Medium |
| 41 | 173 | Cycle 173 | Medium |
| 42 | 214 | Cycle 214 | Long |
| 43 | 159 | Cycle 159 | Medium |
| 44 | 184 | Cycle 184 | Medium |
| 45 | 225 | Cycle 225 | Long |
| 46 | 168 | Cycle 168 | Medium |
| 47 | 151 | Cycle 151 | Short |
| 48 | 199 | Cycle 199 | Medium |
| 49 | 175 | Cycle 175 | Medium |
| 50 | 190 | Cycle 190 | Medium |
| 51 | 176 | Cycle 176 | Medium |
| 52 | 181 | Cycle 181 | Medium |
| 53 | 204 | Cycle 204 | Long |
| 54 | 175 | Cycle 175 | Medium |
| 55 | 191 | Cycle 191 | Medium |
| 56 | 146 | Cycle 146 | Short |
| 57 | 199 | Cycle 199 | Medium |
| 58 | 178 | Cycle 178 | Medium |
| 59 | 147 | Cycle 147 | Short |
| 60 | 146 | Cycle 146 | Short |
| 61 | 157 | Cycle 157 | Medium |
| 62 | 171 | Cycle 171 | Medium |
| 63 | 205 | Cycle 205 | Long |
| 64 | 186 | Cycle 186 | Medium |
| 65 | 179 | Cycle 179 | Medium |
| 66 | 175 | Cycle 175 | Medium |
| 67 | 191 | Cycle 191 | Medium |
| 68 | 185 | Cycle 185 | Medium |
| 69 | 202 | Cycle 202 | Long |
| 70 | 203 | Cycle 203 | Long |
| 71 | 193 | Cycle 193 | Medium |
| 72 | 220 | Cycle 220 | Long |
| 73 | 146 | Cycle 146 | Short |
| 74 | 164 | Cycle 164 | Medium |
| 75 | 168 | Cycle 168 | Medium |
| 76 | 156 | Cycle 156 | Medium |
| 77 | 176 | Cycle 176 | Medium |
| 78 | 192 | Cycle 192 | Medium |
| 79 | 183 | Cycle 183 | Medium |
| 80 | 224 | Cycle 224 | Long |
| 81 | 198 | Cycle 198 | Medium |
| 82 | 186 | Cycle 186 | Medium |
| 83 | 311 | Cycle 311 | Very Long |
| 84 | 186 | Cycle 186 | Medium |
| 85 | 166 | Cycle 166 | Medium |
| 86 | 178 | Cycle 178 | Medium |
| 87 | 160 | Cycle 160 | Medium |
| 88 | 175 | Cycle 175 | Medium |
| 89 | 246 | Cycle 246 | Long |
| 90 | 144 | Cycle 144 | Short |
| 91 | 163 | Cycle 163 | Medium |
| 92 | 173 | Cycle 173 | Medium |
| 93 | 153 | Cycle 153 | Medium |
| 94 | 186 | Cycle 186 | Medium |
| 95 | 219 | Cycle 219 | Long |
| 96 | 340 | Cycle 340 | Very Long |
| 97 | 125 | Cycle 125 | Short |
| 98 | 164 | Cycle 164 | Medium |
| 99 | 215 | Cycle 215 | Long |
| 100 | 200 | Cycle 200 | Long |

## Categories
- **Short** (< 150 cycles): 15 engines - Fast degradation, tests quick anomaly detection
- **Medium** (150-199 cycles): 55 engines - Typical degradation patterns
- **Long** (200-299 cycles): 28 engines - Slow degradation, tests sustained monitoring
- **Very Long** (300+ cycles): 2 engines (#83: 311, #96: 340) - Extended operation

## Recommended Test Engines

| Engine | Cycles | Why Use It |
|--------|--------|------------|
| **#5** | 128 | Fastest failure - quick test |
| **#97** | 125 | Shortest lifetime - stress test |
| **#1** | 192 | Classic medium - baseline |
| **#10** | 156 | Medium with clear degradation |
| **#20** | 206 | Slow burn - patience test |
| **#83** | 311 | Long-running stability test |
| **#96** | 340 | Longest lifetime - endurance test |

## Sensor Reference (21 Sensors)

| Sensor | Physical Measurement | Unit | Type |
|--------|---------------------|------|------|
| s_1 | Fan Inlet Temp (T2) | R | Temperature |
| s_2 | LPC Outlet Temp (T24) | R | Temperature |
| s_3 | HPC Outlet Temp (T30) | R | Temperature |
| s_4 | LPT Outlet Temp (T50) | R | Temperature |
| s_5 | Fan Inlet Pressure (P2) | psia | Pressure |
| s_6 | Bypass Duct Pressure (P15) | psia | Pressure |
| s_7 | HPC Outlet Pressure (P30) | psia | Pressure |
| s_8 | Physical Fan Speed (Nf) | rpm | Speed |
| s_9 | Physical Core Speed (Nc) | rpm | Speed |
| s_10 | Engine Pressure Ratio (epr) | - | Ratio |
| s_11 | Static Pressure (Ps30) | psia | Pressure |
| s_12 | Fuel Flow Ratio (phi) | pps/psi | Flow |
| s_13 | Corrected Fan Speed (NRf) | rpm | Speed |
| s_14 | Corrected Core Speed (NRc) | rpm | Speed |
| s_15 | Bypass Ratio (BPR) | - | Ratio |
| s_16 | Burner Fuel-Air Ratio (farB) | - | Ratio |
| s_17 | Bleed Enthalpy (htBleed) | - | Energy |
| s_18 | Demanded Fan Speed (Nf_dmd) | rpm | Speed |
| s_19 | Demanded Corrected Fan Speed (PCNfR_dmd) | rpm | Speed |
| s_20 | HPT Coolant Bleed (W31) | lbm/s | Flow |
| s_21 | LPT Coolant Bleed (W32) | lbm/s | Flow |

## Key Diagnostic Sensors (Recommended 8)
These show the most variation and are best for anomaly detection:
- **s_2** - LPC Outlet Temperature
- **s_3** - HPC Outlet Temperature
- **s_4** - LPT Outlet Temperature
- **s_7** - HPC Outlet Pressure
- **s_8** - Physical Fan Speed
- **s_9** - Physical Core Speed
- **s_11** - Static Pressure
- **s_12** - Fuel Flow Ratio

## Notes for ML Testing
- Each engine starts "healthy" at cycle 1
- Degradation is gradual but accelerates near failure
- The final cycle is the failure point (RUL = 0)
- Your ML model should detect anomalies BEFORE the failure point
- Goal: Catch degradation at 70-80% of lifetime (leave 20-30% margin)
