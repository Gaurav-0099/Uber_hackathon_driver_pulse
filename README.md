Driver Pulse —Popcorns

Demo Video: <https://drive.google.com/drive/folders/1XSVPZgTSUTMX_Lrcjv9UKrSNVhg1XOsK?usp=sharing>

Live Application: <https://apprealtimedemopy-ga6nxjkdbmnprgyoapji7e.streamlit.app/>

---

# Driver Pulse 🚗

> **Shift intelligence for rideshare drivers** — real-time motion, audio, and earnings analysis fused into a single mobile-first dashboard. One app. One command. Full shift visibility.

---

The unified app combines:
- The full shift analytics dashboard (home, trips, safety, charts, export)
- The live sensor simulation (replays `realtime_trip_simulation.csv` as a streaming trip)
- Driver login with per-driver data isolation
- The complete three-engine pipeline running inside a single cached session

Run everything with:

```bash
streamlit run app_realtime_demo.py
```

The three engine files have also moved into a `src/` subdirectory. All imports in the app reflect this.

---

## System Overview

Driver Pulse is an on-device analytics system for rideshare drivers that fuses three independent signal streams:

1. **Motion (Accelerometer)** — detects harsh braking, sharp turns, sudden acceleration, speed bumps, speeding, and jerk events in real time
2. **Cabin Audio Intensity** — detects elevated noise, argument-level sound, and sustained conflict signals; raw audio is never recorded or transmitted — only the dB intensity level is used
3. **Earnings & Trip Data** — calculates earnings velocity (₹/hr), forecasts goal completion, detects surge pricing, identifies dead zones, and surfaces per-trip profitability

These three streams are processed by three pure-function Python engines, whose outputs are merged into a mobile-first Streamlit dashboard. A live simulation tab replays a real sensor dataset frame-by-frame to demonstrate the intended streaming architecture.

---

## Architecture

```
Raw Sensor Data (CSV files in data/)
        │
        ├─────────────────────────┐
        ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│   sensor_engine.py  │   │  earnings_engine.py  │
│  (src/ directory)   │   │  (src/ directory)    │
│                     │   │                      │
│  7 accel checks     │   │  Velocity calc       │
│  4 audio checks     │   │  Goal forecasting    │
│  Stress fusion      │   │  Surge detection     │
│  Explainability log │   │  Dead zone detection │
└──────────┬──────────┘   └──────────┬───────────┘
           │  flagged_moments              │  trip_earnings
           │  trip_summaries_sensor        │  enriched_velocity
           └───────────────┬──────────────┘  enriched_goals
                           ▼
               ┌─────────────────────┐
               │   merge_engine.py   │
               │  (src/ directory)   │
               │                     │
               │  Left join on       │
               │  trip_id            │
               │  Trip quality       │
               │  Shift summary      │
               └──────────┬──────────┘
                          │  trip_summaries
                          │  driver_shift_summary
                          ▼
              ┌───────────────────────┐
              │   app_realtime_demo.py│   ← ONE unified app
              │                       │
              │  Login screen         │
              │  Live Sim tab         │
              │  Home tab             │
              │  Trips tab            │
              │  Safety tab           │
              │  Charts tab           │
              │  Export tab           │
              └───────────────────────┘
```

**Key architecture properties:**
- Sensor Engine and Earnings Engine run **in parallel** — neither depends on the other
- Merge Engine depends on both; it always runs last
- All engines are **pure functions**: DataFrame in → DataFrame out, no side effects, no shared state
- Each trip is processed independently — the pipeline is stateless and horizontally scalable
- Pipeline output is cached indefinitely per session (`@st.cache_data(ttl=None)`) — runs once on first load, never again unless the process restarts

---

## Project Structure

```
DrivePlus_Riddh_codes/
│
├── app_realtime_demo.py        ← THE ONLY APP — run this
│
├── src/                        ← All engine logic lives here
│   ├── sensor_engine.py        ← Motion + audio signal processing
│   ├── earnings_engine.py      ← Velocity, forecasts, goal tracking
│   └── merge_engine.py         ← Joins safety + earnings → final outputs
│
├── data/                       ← All input CSVs (read-only at runtime)
│   ├── trips.csv               ← Trip metadata and fare data
│   ├── accelerometer_data.csv  ← Per-reading motion sensor data
│   ├── audio_intensity_data.csv← Per-reading cabin audio levels
│   ├── drivers.csv             ← Driver profiles (login list)
│   ├── driver_goals.csv        ← Per-driver shift earnings targets
│   ├── earnings_velocity_log.csv  ← Cumulative earnings time-series
│   ├── flagged_moments.csv     ← Pre-run snapshot (regenerated at runtime)
│   ├── trip_summaries.csv      ← Pre-run snapshot (regenerated at runtime)
│   └── realtime_trip_simulation.csv ← Sensor replay data for Live Sim tab
│
└── outputs/                    ← Auto-generated on every pipeline run
    ├── flagged_moments.csv
    ├── trip_summaries.csv
    ├── driver_shift_summary.csv
    ├── earnings_velocity_log.csv
    └── goals_forecast.csv
```

**Navigation guide — where to look for what:**

| What you want to do | Where to look |
|---------------------|---------------|
| Run the app | `app_realtime_demo.py` — single entry point |
| Understand motion/audio detection logic | `src/sensor_engine.py` |
| Understand earnings/goal forecasting | `src/earnings_engine.py` |
| Understand trip quality and shift summaries | `src/merge_engine.py` |
| Change detection thresholds | Constants block at top of `src/sensor_engine.py` |
| Change earnings rating bands | Constants block at top of `src/earnings_engine.py` |
| Modify dashboard UI / tabs / charts | `app_realtime_demo.py` — each tab is a clearly labelled `with tab_X:` block |
| Add a new driver | `data/drivers.csv` + `data/driver_goals.csv` |
| Inspect last run outputs | `outputs/` directory |
| Swap in real sensor data | Replace CSVs in `data/` — no engine changes needed |
| Run pipeline without Streamlit | See headless example in Running the App section below |

---

## Dashboard Tabs — What Each One Does

The unified app opens on a **driver login screen**. A driver is selected from `drivers.csv`. After clicking **Enter Dashboard**, six tabs are available:

### Tab 1 — Live Sim
Replays `realtime_trip_simulation.csv` row-by-row to simulate a live sensor stream. Press **Play** to start. Each row is injected every ~2 seconds. Four live charts update in real time: speed (km/h), motion intensity (m/s²), cabin audio (dB), and live stress score. Events are detected inline as they occur and shown as alert cards. When replay finishes, the **full sensor engine pipeline runs** against the completed simulation trip and produces a real trip summary: flagged moments, peak stress score, and earnings velocity — exactly as it would for an actual trip.

The simulation is isolated in a `@st.fragment` — only this block reruns during playback; the other five tabs remain fully interactive.

### Tab 2 — Home
Shift KPI overview: total earned today, goal target, goal completion percentage, projected earnings at shift end, average ₹/hr, needed ₹/hr to hit goal. Goal progress bar coloured by forecast status. Phase-aware motivational message from the earnings engine, citing the exact ₹/hr gap. Best route card (highest velocity trip). Trip quality pills (Excellent / Good / Average / Poor counts). Safety snapshot (stress badge, total flags, high-severity count). Top 5 flagged moments with route, timestamp, and plain-English explanation.

### Tab 3 — Trips
One card per trip, sorted chronologically. Each card shows: trip ID, route, fare, duration, trip quality badge, alert count pill, stress severity dot, earnings velocity (₹/hr), and a **goal completion probability bar** — a sigmoid estimate of the driver's likelihood of hitting their shift goal based on the pace recorded after each trip. Probability colour: green ≥ 70%, amber ≥ 45%, red < 45%.

### Tab 4 — Safety
Severity count metrics (High / Medium / Low). Shift-level timeline scatter plot — all flagged moments plotted at (shift minute, stress score), bubble size proportional to score, coloured by severity. Severity filter and trip filter. Per-event detail cards showing: event type, severity badge, plain-English explanation, route, motion score, audio score, combined score. Downloadable explainability log (CSV) with structured fields: `timestamp`, `signal_type`, `raw_value`, `threshold`, `event_label`, `severity`, `driver_explanation`. Preview table (first 10 rows) in an expander.

### Tab 5 — Charts
Five chart modes selectable via radio button:

| Mode | What it shows |
|------|--------------|
| **Live Velocity** | Animated earnings velocity: current ₹/hr vs. required ₹/hr, ahead/behind fill zones, shift phase markers, per-trip velocity bar chart. Scrub slider or auto-play at ~16 fps via `@st.fragment`. |
| **Earnings Curve** | Cumulative earnings with goal line + projected earnings extension, velocity on secondary axis. |
| **Stress Worm** | Per-trip stress score timeline with motion magnitude background. Select a trip from dropdown. |
| **Acceleration** | Two-panel: motion magnitude + speed, with flagged moment highlight bands. |
| **Audio** | Per-trip cabin audio bar chart coloured by classification (quiet/normal/conversation/elevated/argument/mechanical noise). |

### Tab 6 — Export
Per-file download buttons for all five output CSVs with row count, column count, and file size. ZIP bundle download containing all five files, named `driver_pulse_{driver_id}_outputs.zip`.

---

## Data Sources

| File | Description | Key Fields |
|------|-------------|------------|
| `trips.csv` | Trip metadata and fare data | `trip_id`, `driver_id`, `fare`, `duration_min`, `pickup_location`, `dropoff_location`, `surge_multiplier` |
| `accelerometer_data.csv` | Per-reading motion sensor data | `trip_id`, `elapsed_seconds`, `timestamp`, `accel_x`, `accel_y`, `accel_z`, `speed_kmh` |
| `audio_intensity_data.csv` | Per-reading cabin audio intensity | `trip_id`, `elapsed_seconds`, `audio_level_db`, `sustained_duration_sec`, `audio_classification` |
| `drivers.csv` | Driver profiles for login | `driver_id`, `name`, `city`, `rating`, `shift_preference` |
| `driver_goals.csv` | Per-driver shift earnings targets | `driver_id`, `target_earnings`, `target_hours`, `shift_start_time`, `shift_end_time`, `current_earnings` |
| `earnings_velocity_log.csv` | Cumulative earnings time-series | `driver_id`, `log_id`, `timestamp`, `elapsed_hours`, `cumulative_earnings`, `current_velocity` |
| `realtime_trip_simulation.csv` | Sensor replay data for Live Sim tab | `trip_id`, `elapsed_seconds`, `speed_kmh`, `accel_x/y/z`, `audio_level_db`, `audio_classification`, `cumulative_earnings`, `earnings_delta` |

All files are validated via `validate_schema()` at every engine entry point. Missing columns emit `[WARN]` and trigger graceful degradation — the pipeline never crashes on missing data.

---

## Engine Architecture

### `src/sensor_engine.py`

All motion and audio signal processing. Responsible for everything from raw sensor readings to labeled, scored, explained safety events.

**Public functions:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `validate_schema` | `(df, required, context) → bool` | Warns on missing columns; returns False if any missing |
| `run_accel_checks` | `(df) → DataFrame` | Annotated accelerometer readings with event type and score |
| `run_audio_checks` | `(df) → DataFrame` | Annotated audio readings with event type and score; door slams filtered |
| `detect_flagged_moments` | `(accel, audio, trips) → DataFrame` | One row per flagged event with combined stress score, severity, explanation |
| `build_trip_summaries_sensor` | `(trips, flagged, accel, audio) → DataFrame` | Per-trip aggregated sensor metrics |
| `build_explainability_log` | `(flagged, accel) → DataFrame` | Structured log: `signal_type`, `raw_value`, `threshold`, `event_label`, `severity`, `driver_explanation` |
| `process_single_trip` | `(trip_id, accel, audio, trips) → dict` | Stateless single-trip processor — runs full pipeline for one trip independently |

**Key constants (edit thresholds here):**

```python
# src/sensor_engine.py — constants block near top of file

ACCEL_OVERALL       = 3.5   # √(x²+y²+|z-9.8|²) > this → harsh event
ACCEL_LONGITUDINAL  = 2.5   # m/s² → hard brake / sudden acceleration
ACCEL_LATERAL       = 2.0   # m/s² → sharp turn / swerve
ACCEL_ZBUMP         = 7.5   # z-axis drop below this → speed bump
ACCEL_SPEED_LIMIT   = 60    # km/h → speeding
ACCEL_JERK          = 3.0   # m/s³ → jerk event
ACCEL_TRIP_MARGIN   = 20    # seconds from trip start/end — soft events suppressed
MIN_SPEED_MOVING    = 5     # km/h — below this, all motion events suppressed
ZSCORE_CAP          = 3.5   # σ — sensor spike outlier cap (per trip)

AUDIO_DOOR_SLAM_MAX_SECS = 3    # sub-3s audio spike → mechanical_noise
AUDIO_HIGH_DB            = 80   # dB → very_loud classification
AUDIO_ARGUMENT_DB        = 90   # dB → argument classification
AUDIO_SUSTAINED_SECS     = 60   # seconds sustained → elevated classification

MOTION_WEIGHT  = 0.65           # weight in combined stress score
AUDIO_WEIGHT   = 0.35           # weight in combined stress score
BOTH_BOOST     = 1.20           # co-occurrence multiplier when both signals fire
ROAD_NOISE_BOOST_WINDOW = 5     # seconds — audio spike near speed bump is suppressed
FLAG_COOLDOWN  = 15             # seconds between consecutive flags per trip
END_TRIP_BUF   = 30             # seconds before trip end — dropoff braking excluded
```

### `src/earnings_engine.py`

All financial analysis and goal forecasting. No dashboard or display logic.

**Public functions:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `build_trip_earnings` | `(trips) → DataFrame` | Per-trip `earnings_velocity`, `earnings_rating`, `route` |
| `build_enriched_velocity` | `(velocity_log, goals) → DataFrame` | Time-series log enriched with `shift_phase`, `needed_velocity`, `forecast_status`, `message` |
| `build_enriched_goals` | `(goals, velocity_log, trips) → DataFrame` | Goal snapshot: `projected_earnings`, `effective_velocity`, `forecast`, `dead_zone`, `surge_adjusted`, `message` |

**Key constants:**

```python
# src/earnings_engine.py — constants block near top of file

EXCELLENT_VEL = 400   # ₹/hr → excellent trip rating
GOOD_VEL      = 280   # ₹/hr → good
AVERAGE_VEL   = 180   # ₹/hr → average  (below this = below_average)

PHASE_EARLY  = 0.25   # 0–25% of shift elapsed → "early"
PHASE_MID    = 0.60   # 25–60% → "mid"
PHASE_PEAK   = 0.80   # 60–80% → "peak"
# 80–100% → "late"

PHASE_TOLERANCE = {"early": 0.20, "mid": 0.15, "peak": 0.10, "late": 0.05}
# How far behind needed_velocity before status changes from on_track → at_risk
```

**Surge adjustment:** if `current_velocity > driver_baseline_median × 2.0`, the forecast uses `effective_velocity = 0.4 × current + 0.6 × baseline` — preventing surge spikes from producing falsely optimistic goal projections.

**Dead zone detection:** `effective_velocity < area_median × 0.5` AND `shift status == in_progress` AND `remaining_hours > 1` → `forecast = "dead_zone"`. Dashboard message prompts the driver to relocate.

### `src/merge_engine.py`

Joins safety and financial signals. Always runs after both other engines complete.

**Public functions:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `build_trip_summaries` | `(trip_earnings, sensor_summaries) → DataFrame` | Master trip table: earnings + sensor merged, `trip_quality` computed |
| `build_driver_shift_summary` | `(trip_summaries, enriched_goals, flagged) → DataFrame` | One row per driver: total earned, quality counts, best route, peak stress badge, goal forecast |

**Trip quality matrix:**

| | Excellent / Good earnings | Average earnings | Below Average earnings |
|---|---|---|---|
| **stress < 0.20** | excellent | good | average |
| **stress 0.20–0.35** | good | average | average |
| **stress 0.35–0.65** | average | average | poor |
| **stress ≥ 0.65** | poor | poor | poor |
| **stress = -1.0 (no sensor)** | average | average | average |

---

## Signal Processing Logic

### Accelerometer — 7 Checks (applied in priority order)

1. **Stationary suppression** — `speed < 5 km/h` → `stationary_noise`, score 0.0
2. **Dropoff brake filter** — `speed_delta < -8 km/h` within last 30s of trip → `dropoff_brake`, score 0.0
3. **Speeding** — `speed > 60 km/h` → score scales 0.30–0.70 with speed
4. **Speed bump** — `accel_z < 7.5 m/s²` at speed > 15 km/h → score 0.30
5. **Jerk** — `Δmagnitude/Δtime > 3.0 m/s³` (min 0.5s denominator to prevent sub-millisecond explosions) → score up to 0.65
6. **Sharp turn / swerve** — `lateral_magnitude > 2.0 m/s²` → score up to 0.80
7. **Harsh brake / acceleration** — `overall_magnitude > 3.5 m/s²` AND `longitudinal > 2.5 m/s²` → score up to 1.0

**Pre-processing applied before all checks:**
- 3-sample rolling max of `speed_kmh` (handles GPS lag — instantaneous speed may read zero briefly after the car starts moving)
- Gap-aware speed delta: if time gap between rows > 5s, `speed_delta = 0.0` (suppresses GPS-dropout false acceleration spikes)
- Z-score cap at 3.5σ per trip (suppresses sensor hardware malfunction outliers)

### Audio — 4 Checks (applied in order)

1. **Door slam filter** — `duration < 3s AND dB < 90` → `mechanical_noise`, score 0.0
2. **Argument** — `audio_classification == "argument"` OR `dB ≥ 90` → score scales with dB
3. **Very loud** — `dB ≥ 80` → score up to 0.75
4. **Sustained elevated** — `sustained_duration_sec ≥ 60` → score 0.50

### Stress Score Fusion

```
combined = (motion_score × 0.65) + (audio_score × 0.35)

If motion_score > 0.4 AND audio_score > 0.4:
    combined = min(combined × 1.20, 1.0)    ← co-occurrence boost
```

**Audio matching window** — adaptive: ±62s normally, tightens to ±15s within the last 5 minutes of a trip (prevents exit-noise from pairing with motion events near dropoff). An extra ±2s sync tolerance handles polling lag between accelerometer and microphone.

**Road noise suppression** — if an audio spike occurs within ±5s of a detected speed bump, the audio score for that event is set to 0.0 (`audio_type = "road_noise_filtered"`).

**Safety maneuver downweight** — if a `harsh_brake` is followed by near-zero speed (≤ 2 km/h) within 10 seconds, the event is reclassified as `safety_maneuver` and its score is multiplied by 0.35.

**Severity bands:**

| Band | Score Range |
|------|-------------|
| NONE | < 0.20 |
| LOW | 0.20 – 0.39 |
| MODERATE | 0.40 – 0.64 |
| HIGH | ≥ 0.65 |

---

## Output Files

Five files are written to `outputs/` on every run by `_save_outputs()`. Writes are wrapped in `try/except` — an I/O failure never crashes the dashboard.

| File | Written By | Rows | Key Columns |
|------|------------|------|-------------|
| `flagged_moments.csv` | sensor_engine | One per event | `flag_id`, `trip_id`, `driver_id`, `timestamp`, `elapsed_seconds`, `accel_type`, `audio_type`, `accel_score`, `audio_score`, `combined_score`, `severity`, `driver_explanation` |
| `trip_summaries.csv` | merge_engine | One per trip | `trip_id`, `driver_id`, `fare`, `earnings_velocity`, `earnings_rating`, `stress_score`, `flagged_count`, `trip_quality` |
| `driver_shift_summary.csv` | merge_engine | One per driver | `driver_id`, `total_earned`, `avg_velocity`, `peak_stress_score`, `stress_badge`, `n_excellent/good/average/poor_trips`, `best_route`, `forecast`, `goal_message`, `total_flags` |
| `earnings_velocity_log.csv` | earnings_engine | One per log entry | `driver_id`, `timestamp`, `elapsed_hours`, `recalc_velocity`, `needed_velocity`, `shift_phase`, `forecast_status`, `message` |
| `goals_forecast.csv` | earnings_engine | One per driver | `driver_id`, `current_earnings`, `projected_earnings`, `effective_velocity`, `forecast`, `dead_zone`, `surge_adjusted`, `message` |

---

## Processed Output Evidence

Every flagging decision is fully traceable from raw sensor value to plain-English driver explanation. The explainability log produced by `build_explainability_log()` matches the hackathon spec format exactly:

```
Timestamp:    2025-10-12 08:45:10
Signal Type:  ACCELEROMETER
Raw Value:    -2.1 m/s²
Threshold:    2.5 m/s²
Event Label:  HARSH_BRAKE
Severity:     medium
Stress Score: 0.52
Explanation:  Moderate event — Hard braking detected.
```

This log is downloadable from the **Safety tab** as `driver_pulse_{driver_id}_log.csv`. A 10-row preview is available in an expandable table in the same tab.

---

## Live Simulation — How It Works

`@st.fragment` isolates the simulation so only the sim block reruns during playback — switching to any other tab is instant.

**Replay loop (inside `_live_sim_fragment()`):**
1. `demo_row_idx` tracks the current position in `realtime_trip_simulation.csv`
2. Every 0.2s, the fragment checks wall-clock time against `_demo_last_tick`
3. If ≥ 2 seconds have elapsed, `demo_row_idx` advances by one row and charts rebuild from `sim_df.iloc[:demo_row_idx]`
4. `detect_live_event()` runs a lightweight inline threshold check against the current row — no full engine call during streaming
5. Detected events are appended to `demo_events` in session state and displayed as alert cards

**On trip completion:**
The full `sensor_engine` pipeline (`run_accel_checks` → `run_audio_checks` → `detect_flagged_moments` → `build_trip_summaries_sensor`) runs against the completed simulation trip. This is the same code path as a real trip — demonstrating that the streaming and batch pipelines are architecturally identical.

**Intended production architecture this demonstrates:**
```
Mobile device sensor stream
    → on-device SDK (stationary filter, z-score cap, door slam filter)
    → Kafka topic partitioned by driver_id
    → Flink stateless worker (sensor_engine logic)
    → ClickHouse + PostgreSQL output store
    → WebSocket API → React Native dashboard
```

---

## Setup Instructions

**Requirements:** Python 3.9+, pip

```bash
git clone <repo-url>
cd DrivePlus_Riddh_codes

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt` (minimum):**
```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
plotly>=5.18
```

Ensure all CSV files are present in `data/` before running. The `outputs/` directory is created automatically on first run.

---

## Running the App

```bash
# Everything — one command
streamlit run app_realtime_demo.py
```

The app opens on the driver login screen. Select a driver, click **Enter Dashboard**. All pipeline processing runs on first load and is cached for the session.

**To run the pipeline headlessly (without Streamlit):**

```python
import pandas as pd
import sys
sys.path.insert(0, "src")   # engines are in src/

from sensor_engine   import run_accel_checks, run_audio_checks, \
                            detect_flagged_moments, build_trip_summaries_sensor
from earnings_engine import build_trip_earnings, build_enriched_velocity, \
                            build_enriched_goals
from merge_engine    import build_trip_summaries, build_driver_shift_summary

# Load inputs
trips    = pd.read_csv("data/trips.csv")
accel    = pd.read_csv("data/accelerometer_data.csv")
audio    = pd.read_csv("data/audio_intensity_data.csv")
goals    = pd.read_csv("data/driver_goals.csv")
velocity = pd.read_csv("data/earnings_velocity_log.csv")

# Stage 1 — Sensor Engine
accel_p  = run_accel_checks(accel)
audio_p  = run_audio_checks(audio)
flagged  = detect_flagged_moments(accel_p, audio_p, trips)
sensor_s = build_trip_summaries_sensor(trips, flagged, accel_p, audio_p)

# Stage 2 — Earnings Engine  (parallel to Stage 1 — no dependency)
trip_earn  = build_trip_earnings(trips)
enrich_vel = build_enriched_velocity(velocity, goals)
enrich_g   = build_enriched_goals(goals, velocity, trips)

# Stage 3 — Merge Engine  (depends on both Stage 1 and Stage 2)
trip_sum  = build_trip_summaries(trip_earn, sensor_s)
shift_sum = build_driver_shift_summary(trip_sum, enrich_g, flagged)

print(shift_sum[["driver_id", "total_earned", "forecast", "stress_badge"]])
```

---

## Tradeoffs

### Rule-Based Detection vs. Machine Learning
Threshold rules are interpretable, debuggable, and require no training data. The tradeoff is reduced precision — a pothole and a harsh brake can produce similar accelerometer signatures. A production system would layer an ML classifier on top of rules (rules as fast pre-filter, model for ambiguous cases), trained on driver-labeled event data.

### Edge Processing vs. Cloud Processing
All processing runs locally in the prototype. In production, the stationary filter, door slam filter, z-score cap, and threshold checks run on-device. Only labeled events — not raw readings — are transmitted, reducing bandwidth by ~95% and ensuring raw audio values never leave the device.

### Sensor Sampling Rate vs. Battery Life
The pipeline is rate-agnostic: it uses `elapsed_seconds` for all time calculations and makes no fixed-interval assumptions. A gap-aware speed delta (EC-04) suppresses false spikes from GPS dropouts. Production would expose a configurable rate: 50 Hz for research, 5 Hz default, 1 Hz for battery-critical sessions.

### Accuracy vs. Responsiveness
Cooldown windows, trip margin buffers, and z-score capping reduce false positives at the cost of latency — an event may be confirmed up to 15 seconds after it occurs. For driver coaching and post-trip review this is acceptable. For real-time conflict alerts the cooldown would need to be tunable per use case.

### Single-Process Streamlit vs. Production Scale
Streamlit is suitable for prototyping and single-driver demos. Thousands of concurrent drivers require a stateless API layer with a React Native mobile frontend. The engine logic itself requires no changes — it is already pure and stateless.

---

## Handling Real-World Constraints

### Edge Device
- Rolling window design: maximum in-memory lookback is ~30 seconds (`FLAG_COOLDOWN + END_TRIP_BUF`). Memory is bounded regardless of trip duration.
- Offline-first: pipeline writes outputs locally before any network call. Syncs to cloud on reconnect.

### Sensor Noise

| Mechanism | Constant | Purpose |
|-----------|----------|---------|
| Trip boundary filter | `ACCEL_TRIP_MARGIN = 20s` | Suppresses soft events near trip start/end |
| Stationary suppression | `MIN_SPEED_MOVING = 5 km/h` | All motion events suppressed when vehicle idle |
| Outlier cap | `ZSCORE_CAP = 3.5σ` | Hardware spike outliers capped per trip |
| Door slam filter | `AUDIO_DOOR_SLAM_MAX_SECS = 3s` | Sub-3s audio spikes → mechanical_noise |
| Road noise filter | `ROAD_NOISE_BOOST_WINDOW = 5s` | Audio near speed bump → road_noise_filtered |
| Duplicate event filter | `FLAG_COOLDOWN = 15s` | No consecutive flags within 15s per trip |
| GPS dropout guard | `_tdiff.clip(lower=0.5)` | Min 0.5s time delta prevents jerk explosion |
| GPS lag guard | 3-sample rolling max of speed | Handles momentary zero-speed GPS readings |

### Data Gaps and Schema Failures
- `validate_schema()` at every engine entry point — missing columns log `[WARN]` and trigger graceful degradation
- Left join fills unmatched sensor rows with `stress_score = -1.0` sentinel — explicitly handled as "unknown" in quality scoring, not "calm"
- All division operations guarded against zero denominators
- Datetime parsing in goal engine wrapped in `try/except` with fallback to `target_hours`

### Privacy
- Audio intensity (dB level) only — no audio waveform captured, buffered, or stored anywhere
- On-device preprocessing: raw sensor values classified locally; only event labels transmitted
- No passenger identity data — `trip_id` is an opaque reference
- All dashboard data scoped to authenticated `driver_id`

---

## Edge Case Handling

| Edge Case | Status | Handling |
|-----------|--------|---------|
| Stationary phone / engine idle vibrations | ✅ Implemented | `speed_kmh < 5` → `stationary_noise`; 3-sample rolling max handles GPS lag |
| Door slam / car horn (brief audio spike) | ✅ Implemented | `duration < 3s AND dB < 90` → `mechanical_noise`, score 0.0 |
| Pothole road noise triggering audio flag | ✅ Implemented | Audio within ±5s of speed bump → `road_noise_filtered`, audio score 0.0 |
| Hard brake as legitimate safety maneuver | ✅ Implemented | Post-brake stop within 10s → `safety_maneuver`, score × 0.35 |
| GPS dropout creating false acceleration spike | ✅ Implemented | Time gap > 5s between rows → `speed_delta = 0.0` |
| Dropoff braking flagged as harsh brake | ✅ Implemented | Last 30s + speed delta < -8 km/h → `dropoff_brake`, score 0.0 |
| Single event producing duplicate flags | ✅ Implemented | 15-second cooldown between flags per trip |
| Sensor hardware spike (e.g. 450 m/s²) | ✅ Implemented | Z-score cap at 3.5σ per trip |
| Surge pricing inflating velocity forecast | ✅ Implemented | `current > baseline × 2.0` → `effective_vel = 0.4 × current + 0.6 × baseline` |
| Driver in dead zone with stalled earnings | ✅ Implemented | `effective_vel < 50% area median AND remaining > 1hr` → `dead_zone` forecast |
| Driver with no goal configured | ✅ Implemented | None-safe `gr` accessor — returns `"unknown"` gracefully, no crash |
| Trip with no sensor coverage | ✅ Implemented | `stress_score = -1.0` sentinel → `trip_quality = "average"` |
| below_average earnings rated as poor | ✅ Fixed | Explicit branch: `below_average + stress < 0.35 → "average"` |
| Jerk values exploding from sub-millisecond row gaps | ✅ Fixed | `_tdiff.clip(lower=0.5)` minimum 0.5s denominator |
| Driver goal row crash on empty data | ✅ Fixed | `gr = goal_row.iloc[0] if not goal_row.empty else None` |
| Phone rotation mid-trip changing axis meaning | ⚠️ Partial | Total magnitude check is orientation-independent; directional checks assume fixed mounting. Full fix requires gyroscope-based axis normalisation (Phase 2 SDK). |
| No network connectivity | 🗒️ Design | Write-local-first pattern designed; sync agent scoped to Phase 2. |

---

## Hackathon Limitations

| Gap | Current State | Production Approach |
|-----|---------------|---------------------|
| No live sensor stream | Pre-recorded CSV replay | Mobile SDK reading from device accelerometer and microphone |
| No real ride-hailing API | Synthetic trip data | Platform webhooks for trip events, fare updates, surge data |
| No mobile deployment | Streamlit web app | React Native component inside driver app |
| No personalised driver models | Global thresholds | Per-driver calibration from rolling historical baseline |
| No streaming infrastructure | File replay simulation | Apache Kafka + Apache Flink |
| No database | In-memory + CSV | ClickHouse (events) + PostgreSQL (summaries) + BigQuery (analytics) |
| Area median for dead zone | Current run's `velocity_df` only | Fleet-wide median by city/zone/time window from production database |
| Single-process Streamlit | `@st.cache_data` session cache | Stateless API + CDN-served frontend for concurrent users |

---

## Future Work

| Area | Description |
|------|-------------|
| ML-based stress classifier | Replace threshold rules with a trained model; rules become fast pre-filter |
| Deep audio classification | On-device model: conversation / argument / music / road noise |
| Real ride-hailing API | Live trip events, fare data, surge multipliers via platform webhooks |
| Personalised driver baselines | Per-driver threshold calibration from rolling historical data |
| Phase 2 — Streaming | Kafka + Flink; edge SDK for on-device preprocessing |
| Phase 3 — Platform integration | Real webhooks, surge data, driver goal sync |
| Phase 4 — Database | ClickHouse, PostgreSQL, BigQuery |
| Phase 5 — ML inference | Vertex AI serving for stress and audio classifiers |
| Phase 6 — Mobile native | React Native app; push notifications for stress events and earnings pacing |
| Route intelligence | High-value corridors and peak demand windows from historical trip data |
| Driver well-being | Longitudinal stress accumulation and recovery tracking across multiple shifts |
