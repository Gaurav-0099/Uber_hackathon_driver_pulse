# Driver Pulse — Session Handoff

Paste this into the next Claude session as the first message.

---

## Project

**Driver Pulse** — a Streamlit dashboard for rideshare driver shift analytics.
Built for an Uber hackathon. Backend is pure Python (rule-based, no ML).

## File Structure

```
project/
├── app.py                      ← Streamlit dashboard (1339 lines)
├── src/
│   ├── sensor_engine.py        ← Accelerometer + audio detection
│   ├── earnings_engine.py      ← Velocity, forecasts, goal tracking
│   └── merge_engine.py         ← Joins safety + earnings → summaries
├── data/
│   ├── trips.csv
│   ├── accelerometer_data.csv
│   ├── audio_intensity_data.csv
│   ├── driver_goals.csv
│   ├── earnings_velocity_log.csv
│   └── drivers.csv
└── outputs/                    ← Auto-generated on every pipeline run
    ├── flagged_moments.csv
    ├── trip_summaries.csv
    ├── driver_shift_summary.csv
    ├── earnings_velocity_log.csv
    └── goals_forecast.csv
```

## Pipeline (load_and_process, cached indefinitely)

```
trips.csv + accel + audio
  → sensor_engine   → flagged_moments, trip_summaries_sensor
  → earnings_engine → trip_earnings, enriched_velocity, enriched_goals
  → merge_engine    → trip_summaries, driver_shift_summary
  → _save_outputs() → writes all 5 CSVs to outputs/
```

## Dashboard Tabs

1. **Home** — KPI metrics, goal progress bar, motivational message, best route, trip quality pills, safety snapshot, top 5 flagged moments
2. **Trips** — per-trip cards with: route, fare, quality badge, alert count, stress dot + **goal completion probability bar + 1-line explanation** (new)
3. **Safety** — severity metrics, filterable flagged moment cards with motion/audio scores
4. **Charts** — radio: Live Velocity | Earnings Curve | Stress Worm | Acceleration | Audio
5. **Export** — per-driver CSV downloads + ZIP bundle

## Key Architecture Decisions

### app.py patterns
- `@st.cache_data(ttl=None)` on `load_and_process()` — runs once per process
- `@st.cache_data(ttl=None)` on `slice_for_driver()` — per-driver slice cached by DID
- `@st.fragment` + `st.rerun(scope="fragment")` on Live Velocity animation — only the chart reruns per frame, not the full page
- `card(html_string)` helper — wraps content in a dark glassmorphism card div
- All trip cards batched into one `st.markdown("".join(cards_html))` call — avoids N Streamlit round-trips

### Live Velocity chart (Charts tab)
- 200 dense interpolated frames from 3-5 raw velocity checkpoints
- Ghost lines (full shift preview) pre-built outside fragment — constant, no rebuild
- Fill area: **2 vectorized numpy polygon traces** (was: up to 199 per-segment traces)
- Fragment-scoped rerun at 0.06s interval ≈ 16 fps
- Two-key pattern: `_vel_frame` (code-owned) + slider without key (widget-owned) to avoid session_state conflict

### Goal completion probability (Trips tab)
- Computed per-trip after cumulative fare sums built from `my_trips.sort_values("start_time")`
- Formula: `ratio = current_velocity / needed_velocity` → sigmoid `100 / (1 + exp(-4*(ratio-1)))`
- Clamped to 3–97% to avoid false certainty mid-shift
- Color: green ≥70%, amber ≥45%, red <45%
- One-line explanation based on: earnings_rating, stress_score, trip_quality

### Trip quality pills fix (Home tab)
- Pills were defined as a local `pill()` f-string function passed inside another f-string
- Streamlit was escaping the nested HTML in complex f-strings
- Fix: build pill HTML as plain Python string concatenation, pass clean string to `card()`

### outputs/ folder
- `_save_outputs()` called at the end of `load_and_process()` — writes all 5 CSVs
- Uses `os.makedirs(OUTPUT_DIR, exist_ok=True)` — safe on first run
- Wrapped in try/except — never crashes the dashboard on I/O error

## Removed in last session
- **Stress vs Earnings scatter** chart (Charts tab)
- **Shift Heatmap** chart (Charts tab)

## What's left to build
- README.md for engineering handoff
- System architecture diagram
- Threshold calibration notes (from `_documenttion.docx` — see issues list)
- Deployment config (requirements.txt, Dockerfile or Cloud Run)
- Chronological progress log (hackathon deliverable)
- Design document (product vision + algorithmic reasoning)

## Known Data Issues (from _documenttion.docx)
- Sensor data covers only 30/220 trips → handle DATA_MISSING gracefully
- Harsh braking threshold: use ≤3.5 m/s² (max in dataset is 6.8)
- All earnings in ₹ INR (₹142–₹468 per trip, goals ₹1,100–₹1,800)
- Accelerometer is ~30s intervals, not dense timeseries — no rolling windows
- velocity_delta outlier: max +₹3,392/hr — use median, not mean
- Three different earnings_velocity fields across files — each measures different scope

## Run command
```
streamlit run app.py
```
