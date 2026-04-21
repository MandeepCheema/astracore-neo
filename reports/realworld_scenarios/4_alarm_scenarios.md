# Scenario 4 — Safety fusion alarm (US + lidar + CAN) across 4 driving scenarios

Fuses ultrasonic + lidar + CAN vehicle-speed into 4-level alarm (OFF / CAUTION / WARNING / CRITICAL). Pass criterion per subscenario listed below.

| Subscenario | OFF | CAU | WARN | CRIT | Min US m | Min lidar m | First CRIT | PASS |
|---|---:|---:|---:|---:|---:|---:|---|:---:|
| parking_crawl_5_to_0p5_kph | 6 | 2 | 1 | 1 | 0.20 | 0.04 | synthetic-s000-n006 | PASS |
| highway_cruise_100_kph_clear_road | 10 | 0 | 0 | 0 | inf | inf | — | PASS |
| emergency_brake_60_kph_to_35_kph | 4 | 0 | 0 | 2 | 0.25 | 0.07 | synthetic-s000-n002 | PASS |
| us_dropout_lidar_only_detection | 7 | 1 | 0 | 0 | inf | 0.31 | — | PASS |

- **parking_crawl_5_to_0p5_kph** — Decelerating 5->0.5 kph, 4 injected obstacles.
- **highway_cruise_100_kph_clear_road** — 10 samples at 100 kph, no obstacles — alarm must stay OFF.
  - Pass: `CRITICAL == 0 and WARNING == 0` → PASS
- **emergency_brake_60_kph_to_35_kph** — 60->35 kph cruise; obstacle appears at sample 2 at 0.25m.
  - Pass: `CRITICAL >= 1 AND first_critical at sample 2 or 3` → PASS
- **us_dropout_lidar_only_detection** — Every US sensor reports no-echo; lidar alone sees the 0.5m obstacle at sample 3.
  - Pass: `CAUTION >= 1 (US dead → alarm degrades to CAUTION)` → PASS