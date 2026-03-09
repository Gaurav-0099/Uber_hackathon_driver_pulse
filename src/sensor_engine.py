"""
Driver Pulse — Sensor Engine
Runs 7 accelerometer checks + 4 audio checks.
Produces: flagged_moments, trip_summaries_sensor
"""

import pandas as pd
import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────
ACCEL_OVERALL       = 3.5   # √(x²+y²+|z-9.8|²) > this → harsh event
ACCEL_LONGITUDINAL  = 2.5   # Δspeed/Δtime m/s²   → hard brake / sudden accel
ACCEL_LATERAL       = 2.0   # √(x²+y²)            → sharp turn / swerve
ACCEL_ZBUMP         = 7.5   # z < this at speed    → speed bump
ACCEL_SPEED_LIMIT   = 60    # km/h                 → speeding
ACCEL_JERK          = 3.0   # Δmag/Δtime m/s³      → jerk
ACCEL_TRIP_MARGIN   = 20    # seconds from trip start/end → soft events ignored
MIN_SPEED_MOVING    = 5     # km/h below which motion ignored (EC-06)
ZSCORE_CAP          = 3.5   # EC-10 outlier cap

AUDIO_DOOR_SLAM_MAX_SECS = 3    # EC-02: sub-3s spike = mechanical noise
AUDIO_HIGH_DB            = 80
AUDIO_ARGUMENT_DB        = 90
AUDIO_SUSTAINED_SECS     = 60

MOTION_WEIGHT  = 0.65
AUDIO_WEIGHT   = 0.35
BOTH_BOOST     = 1.20       # 20% boost when both signals fire
ROAD_NOISE_BOOST_WINDOW = 5 # seconds: if audio spike coincides with bump → filter

FLAG_COOLDOWN  = 15         # EC-08
END_TRIP_BUF   = 30         # EC-07
STATIONARY_WIN = 10         # EC-01
STATIONARY_SPD = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore_cap(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return series
    return series.clip(upper=mu + ZSCORE_CAP * sigma)


def _severity(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "medium"
    elif score >= 0.30:
        return "low"
    return "none"


def _plain_english(accel_type: str, audio_type: str, combined: float) -> str:
    sev = _severity(combined)
    sev_word = {"high": "Serious", "medium": "Moderate", "low": "Minor", "none": ""}[sev]

    motion_phrases = {
        "harsh_brake":      "Hard braking detected",
        "harsh_accel":      "Sudden acceleration",
        "sharp_turn":       "Sharp turn or swerve",
        "speed_bump":       "Speed bump hit at speed",
        "speeding":         "Speed limit exceeded",
        "jerk":             "Jerky movement detected",
        "safety_maneuver":  "Emergency stop (safety maneuver)",
        "normal":           "",
        "dropoff_brake":    "",
        "stationary_noise": "",
    }
    audio_phrases = {
        "argument":    "loud argument in cabin",
        "very_loud":   "very loud cabin noise",
        "elevated":    "elevated noise levels",
        "conversation":"conversation detected",
        "normal":      "",
        "quiet":       "",
    }

    m = motion_phrases.get(accel_type, "Motion event")
    a = audio_phrases.get(audio_type, "")

    if m and a:
        return f"{sev_word} event — {m} with {a}."
    elif m:
        return f"{sev_word} event — {m}."
    elif a:
        return f"{sev_word} event — {a.capitalize()}."
    return "Event detected."


# ── 7 Accelerometer Checks ────────────────────────────────────────────────────

def run_accel_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["trip_id", "elapsed_seconds"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # EC-10: cap outliers
    df["magnitude"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + (df["accel_z"] - 9.8)**2)
    df["magnitude"] = df.groupby("trip_id")["magnitude"].transform(_zscore_cap)

    # EC-04: gap-aware speed delta
    df["_tdiff"] = df.groupby("trip_id")["elapsed_seconds"].diff().fillna(0)
    df["_sdiff"] = df.groupby("trip_id")["speed_kmh"].diff().fillna(0)
    df["speed_delta"] = df["_sdiff"].where(df["_tdiff"] <= 5, 0.0)
    df["accel_longitudinal"] = df["speed_delta"] / df["_tdiff"].replace(0, np.nan).fillna(1) / 3.6  # km/h→m/s per s
    df["lateral_magnitude"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2)

    # Δmagnitude/Δtime = jerk proxy
    df["_magdiff"] = df.groupby("trip_id")["magnitude"].diff().fillna(0)
    df["jerk"] = (df["_magdiff"] / df["_tdiff"].replace(0, np.nan).fillna(1)).abs()

    # Trip duration for EC-07
    df["_tmax"] = df.groupby("trip_id")["elapsed_seconds"].transform("max")

    def classify(row):
        spd   = row["speed_kmh"]
        t     = row["elapsed_seconds"]
        tmax  = row["_tmax"]
        mag   = row["magnitude"]
        lon   = abs(row["accel_longitudinal"])
        lat   = row["lateral_magnitude"]
        jerk  = row["jerk"]
        sdelta= row["speed_delta"]

        # EC-06: stationary phone noise
        if spd < MIN_SPEED_MOVING:
            return "stationary_noise", 0.0

        # EC-07: end-of-trip drop-off
        if (tmax - t) <= END_TRIP_BUF and sdelta < -8:
            return "dropoff_brake", 0.0

        # Near trip start (soft events ignored) — Check 5
        if t < ACCEL_TRIP_MARGIN:
            if mag < ACCEL_OVERALL * 1.3:
                return "normal", 0.1

        # Check 6: speeding
        if spd > ACCEL_SPEED_LIMIT:
            score = min(0.3 + (spd - ACCEL_SPEED_LIMIT) / 60.0, 0.7)
            return "speeding", round(score, 3)

        # Check 4: speed bump (z-drop at speed)
        if row["accel_z"] < ACCEL_ZBUMP and spd > 15:
            return "speed_bump", 0.30

        # Check 7: jerk
        if jerk > ACCEL_JERK:
            return "jerk", min(jerk / 8.0, 0.65)

        # Check 3: sharp turn/swerve
        if lat > ACCEL_LATERAL:
            return "sharp_turn", min(lat / 5.0, 0.80)

        # Check 1 + 2 combined: overall harsh + longitudinal
        if mag > ACCEL_OVERALL:
            if lon > ACCEL_LONGITUDINAL and sdelta < -5:
                return "harsh_brake", min(mag / 8.0, 1.0)
            elif lon > ACCEL_LONGITUDINAL and sdelta > 5:
                return "harsh_accel", min(mag / 8.0, 1.0)
            return "harsh_brake", min(mag / 8.0, 0.85)

        return "normal", 0.1

    result = df.apply(classify, axis=1)
    df["accel_type"]  = result.apply(lambda x: x[0])
    df["accel_score"] = result.apply(lambda x: x[1])

    df.drop(columns=["_tdiff","_sdiff","_magdiff","_tmax"], inplace=True)
    return df


# ── 4 Audio Checks ────────────────────────────────────────────────────────────

def run_audio_checks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def classify_audio(row):
        dur = row["sustained_duration_sec"]
        db  = row["audio_level_db"]
        cls = row["audio_classification"]

        # EC-02: sub-3s below argument level → door slam / honk
        if dur < AUDIO_DOOR_SLAM_MAX_SECS and db < AUDIO_ARGUMENT_DB:
            return "mechanical_noise", 0.0

        # Check A1: sustained argument → strong override
        if cls == "argument" or db >= AUDIO_ARGUMENT_DB:
            score = min((db - 70) / 30.0, 1.0)
            return "argument", round(score, 3)

        # Check A2: high db
        if db >= AUDIO_HIGH_DB:
            return "very_loud", round(min((db - 70) / 30.0, 0.75), 3)

        # Check A3: sustained elevated
        if dur >= AUDIO_SUSTAINED_SECS:
            return "elevated", 0.50

        # Check A4: baseline
        score = max((db - 50) / 60.0, 0.0)
        if cls in ("conversation",):
            return "conversation", round(min(score + 0.1, 0.35), 3)
        return "normal", round(score, 3)

    result = df.apply(classify_audio, axis=1)
    df["audio_type"]  = result.apply(lambda x: x[0])
    df["audio_score"] = result.apply(lambda x: x[1])
    return df


# ── Combine + Flag ────────────────────────────────────────────────────────────

def detect_flagged_moments(accel_df, audio_df, trips_df) -> pd.DataFrame:
    flagged = []
    SKIP = {"stationary_noise", "dropoff_brake", "normal"}

    # build trip→driver map
    trip_driver = trips_df.set_index("trip_id")["driver_id"].to_dict()
    trip_start  = trips_df.set_index("trip_id")["start_time"].to_dict()

    for trip_id in trips_df["trip_id"].unique():
        a_data  = accel_df[accel_df["trip_id"] == trip_id].sort_values("elapsed_seconds")
        au_data = audio_df[audio_df["trip_id"] == trip_id].sort_values("elapsed_seconds")

        if a_data.empty:
            continue

        events = a_data[~a_data["accel_type"].isin(SKIP)]
        last_flag_t = -999

        # Road noise filter: timestamps of speed bumps
        bump_times = set(a_data[a_data["accel_type"] == "speed_bump"]["elapsed_seconds"].tolist())

        for _, row in events.iterrows():
            t = row["elapsed_seconds"]

            # EC-08 cooldown
            if t - last_flag_t < FLAG_COOLDOWN:
                continue

            # Audio window: ±60s + 2s sync tolerance (EC-05)
            nearby = au_data[
                (au_data["elapsed_seconds"] >= t - 62) &
                (au_data["elapsed_seconds"] <= t + 62)
            ]

            accel_sc = row["accel_score"]
            accel_t  = row["accel_type"]

            # Road noise filter: if audio spike coincides with bump, downweight audio
            road_noise = any(abs(t - bt) <= ROAD_NOISE_BOOST_WINDOW for bt in bump_times)
            if road_noise:
                audio_sc = 0.0
                audio_t  = "road_noise_filtered"
            elif not nearby.empty:
                best_idx = nearby["audio_score"].idxmax()
                audio_sc = nearby.loc[best_idx, "audio_score"]
                audio_t  = nearby.loc[best_idx, "audio_type"]
            else:
                audio_sc = 0.0
                audio_t  = "none"

            # EC-01: safety maneuver
            post_rows = a_data[
                (a_data["elapsed_seconds"] > t) &
                (a_data["elapsed_seconds"] <= t + STATIONARY_WIN)
            ]
            if accel_t == "harsh_brake" and not post_rows.empty:
                if (post_rows["speed_kmh"] <= STATIONARY_SPD).all():
                    accel_sc *= 0.35
                    accel_t   = "safety_maneuver"

            combined = MOTION_WEIGHT * accel_sc + AUDIO_WEIGHT * audio_sc

            # Boost if both signals fire
            if accel_sc > 0.4 and audio_sc > 0.4:
                combined = min(combined * BOTH_BOOST, 1.0)

            if combined < 0.30:
                continue

            sev = _severity(combined)
            explanation = _plain_english(accel_t, audio_t, combined)

            flagged.append({
                "trip_id":         trip_id,
                "driver_id":       trip_driver.get(trip_id, ""),
                "timestamp":       row["timestamp"],
                "elapsed_seconds": int(t),
                "accel_type":      accel_t,
                "audio_type":      audio_t,
                "accel_score":     round(accel_sc, 3),
                "audio_score":     round(audio_sc, 3),
                "combined_score":  round(combined, 3),
                "severity":        sev,
                "driver_explanation": explanation,
            })
            last_flag_t = t

    out = pd.DataFrame(flagged)
    if not out.empty:
        out.insert(0, "flag_id", [f"FLAG{str(i+1).zfill(3)}" for i in range(len(out))])
    return out


# ── Trip Summaries (sensor) ───────────────────────────────────────────────────

def build_trip_summaries_sensor(trips_df, flagged_df, accel_df, audio_df) -> pd.DataFrame:
    rows = []
    AGGRESSIVE = {"harsh_brake","harsh_accel","sharp_turn","speeding","jerk","conflict_moment"}

    for _, trip in trips_df.iterrows():
        tid = trip["trip_id"]
        flags = flagged_df[flagged_df["trip_id"] == tid] if not flagged_df.empty else pd.DataFrame()
        real_flags = flags[flags["accel_type"] != "safety_maneuver"] if not flags.empty else flags

        n_motion = len(accel_df[
            (accel_df["trip_id"] == tid) &
            (accel_df["accel_type"].isin(AGGRESSIVE))
        ]) if "accel_type" in accel_df.columns else 0

        n_audio = len(audio_df[
            (audio_df["trip_id"] == tid) &
            (audio_df["audio_score"] > 0.4)
        ]) if "audio_score" in audio_df.columns else 0

        stress_score = round(real_flags["combined_score"].max(), 3) if not real_flags.empty else 0.0

        worst = ""
        if not real_flags.empty:
            worst = real_flags.loc[real_flags["combined_score"].idxmax(), "accel_type"]

        rows.append({
            "trip_id":            tid,
            "driver_id":          trip["driver_id"],
            "motion_events_count": n_motion,
            "audio_events_count":  n_audio,
            "flagged_count":       len(real_flags),
            "stress_score":        stress_score,
            "worst_event_type":    worst,
        })

    return pd.DataFrame(rows)
