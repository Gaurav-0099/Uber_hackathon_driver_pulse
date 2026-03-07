
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT  = Path(r"")
SENSOR     = DATA_ROOT / "sensor_data"
TRIPS_DIR = DATA_ROOT / "trips"
OUT        = DATA_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

GRAVITY = 9.8

THRESH = {
    "magnitude_limit"    : 3.5,   # m/s²  total resultant
    "longitudinal_limit" : 2.5,   # m/s²  hard brake / sudden accel
    "lateral_limit"      : 2.0,   # m/s²  harsh turn
    "breaker_z_drop"     : 7.5,   # m/s²  Z below this = speed bump
    "speed_limit_kmh"    : 60,    # km/h
    "jerk_limit"         : 3.0,   # m/s³
    "proximity_radius_m" : 150,   # metres from trip start/end
    "soft_event_mag"     : 4.5,
    "loud_db"            : 72,
    "very_loud_db"       : 82,
    "sustain_min_sec"    : 8,
    "brief_spike_sec"    : 3,
    "accel_weight"       : 0.65,
    "audio_weight"       : 0.35,
    "combined_high"      : 0.65,
    "combined_medium"    : 0.35,
}


# ── Load ───────────────────────────────────────────────────────────────────────
def load_data():
    accel = pd.read_csv(SENSOR / "accelerometer_data.csv", parse_dates=["timestamp"])
    audio = pd.read_csv(SENSOR / "audio_intensity_data.csv", parse_dates=["timestamp"])
    trips = pd.read_csv(TRIPS_DIR / "trips.csv")
    return accel, audio, trips


# ── Accelerometer processing (7 checks) ───────────────────────────────────────
def process_accelerometer(accel_df, trips_df):
    df = accel_df.copy().sort_values(["trip_id", "timestamp"])

    # Check 1: Total magnitude
    df["magnitude"] = np.sqrt(
        df["accel_x"]**2 + df["accel_y"]**2 + (df["accel_z"] - GRAVITY).abs()**2
    ).round(3)

    # Check 2: Longitudinal (braking / acceleration)
    df["delta_speed"] = df.groupby("trip_id")["speed_kmh"].diff().fillna(0)
    df["delta_time"]  = df.groupby("trip_id")["elapsed_seconds"].diff().fillna(1).replace(0, 1)
    df["long_accel"]  = ((df["delta_speed"] / 3.6) / df["delta_time"]).abs().round(3)
    df["abs_y"]       = df["accel_y"].abs()
    df["longitudinal"]= df[["long_accel", "abs_y"]].max(axis=1)

    # Check 3: Lateral (turn / swerve)
    df["lateral"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2).round(3)

    # Check 4: Z-drop (speed bump)
    df["z_drop"] = (GRAVITY - df["accel_z"]).clip(lower=0).round(3)

    # Check 5: GPS proximity to trip start/end
    trip_gps = df.groupby("trip_id").agg(
        start_lat=("gps_lat", "first"), start_lon=("gps_lon", "first"),
        end_lat  =("gps_lat", "last"),  end_lon  =("gps_lon", "last"),
    ).to_dict("index")

    def near_boundary(row):
        tid = row["trip_id"]
        if tid not in trip_gps: return False
        sg = trip_gps[tid]
        def dist_m(la1, lo1, la2, lo2):
            return np.sqrt(((la2-la1)*111000)**2 + ((lo2-lo1)*111000*np.cos(np.radians(la1)))**2)
        return min(dist_m(row["gps_lat"],row["gps_lon"],sg["start_lat"],sg["start_lon"]),
                   dist_m(row["gps_lat"],row["gps_lon"],sg["end_lat"],  sg["end_lon"])) < THRESH["proximity_radius_m"]

    df["near_boundary"] = df.apply(near_boundary, axis=1)

    # Check 6: Speed limit
    df["speeding"] = df["speed_kmh"] > THRESH["speed_limit_kmh"]

    # Check 7: Jerk
    df["jerk"] = (df.groupby("trip_id")["magnitude"].diff().fillna(0) / df["delta_time"]).abs().round(3)

    # Score each row
    df["accel_score"]      = 0.0
    df["accel_event_type"] = "none"

    for idx, row in df.iterrows():
        # Check 5: near stop + soft event → skip
        if row["near_boundary"] and row["magnitude"] < THRESH["soft_event_mag"]:
            df.at[idx, "accel_event_type"] = "ignored_boundary"
            continue

        score = 0.0
        event = "none"

        if row["magnitude"] > THRESH["magnitude_limit"]:
            ratio = min((row["magnitude"] - THRESH["magnitude_limit"]) / THRESH["magnitude_limit"], 1.0)
            score = max(score, 0.3 + 0.4 * ratio)

        if row["longitudinal"] > THRESH["longitudinal_limit"]:
            ratio = min((row["longitudinal"] - THRESH["longitudinal_limit"]) / THRESH["longitudinal_limit"], 1.0)
            score = max(score, 0.35 + 0.4 * ratio)
            event = "hard_brake_or_accel"

        if row["lateral"] > THRESH["lateral_limit"]:
            ratio = min((row["lateral"] - THRESH["lateral_limit"]) / THRESH["lateral_limit"], 1.0)
            score = max(score, 0.3 + 0.35 * ratio)
            event = "sharp_turn"

        if row["z_drop"] > (GRAVITY - THRESH["breaker_z_drop"]):
            ratio = min(row["z_drop"] / 3.0, 1.0)
            score = max(score, 0.2 + 0.3 * ratio)
            if event == "none": event = "speed_bump_or_pothole"

        if row["speeding"]:
            excess = min((row["speed_kmh"] - THRESH["speed_limit_kmh"]) / THRESH["speed_limit_kmh"], 1.0)
            score  = max(score, 0.25 + 0.35 * excess)
            if event == "none": event = "speeding"

        if row["jerk"] > THRESH["jerk_limit"]:
            ratio = min((row["jerk"] - THRESH["jerk_limit"]) / THRESH["jerk_limit"], 1.0)
            score = max(score, 0.2 + 0.3 * ratio)
            if event == "none": event = "jerky_movement"

        df.at[idx, "accel_score"]      = round(min(score, 1.0), 3)
        df.at[idx, "accel_event_type"] = event

    return df


# ── Audio processing (4 checks) ───────────────────────────────────────────────
def process_audio(audio_df):
    df = audio_df.copy().sort_values(["trip_id", "timestamp"])
    CLASS_SCORE = {"quiet":0.0,"normal":0.0,"conversation":0.05,
                   "loud":0.30,"very_loud":0.55,"argument":0.85}

    df["audio_score"]      = 0.0
    df["audio_event_type"] = "none"

    for idx, row in df.iterrows():
        score = 0.0
        event = "none"
        db    = row["audio_level_db"]
        cls   = row["audio_classification"]
        dur   = row.get("sustained_duration_sec", 0)

        # Check 1: Raw dB
        if db >= THRESH["very_loud_db"]:
            score = max(score, 0.5 + 0.3 * min((db - THRESH["very_loud_db"]) / 20.0, 1.0))
            event = "loud_cabin"
        elif db >= THRESH["loud_db"]:
            score = max(score, 0.2 + 0.3 * min((db - THRESH["loud_db"]) / (THRESH["very_loud_db"] - THRESH["loud_db"]), 1.0))
            event = "elevated_noise"

        # Check 2: Brief spike = probably honk → downweight
        if score > 0 and dur < THRESH["brief_spike_sec"]:
            score *= 0.2

        # Check 3: Argument = strong override
        if cls == "argument":
            score = max(score, CLASS_SCORE["argument"])
            event = "cabin_argument"

        # Check 4: Sustained loud/argument
        cls_base = CLASS_SCORE.get(cls, 0.0)
        score = max(score, cls_base if dur >= THRESH["sustain_min_sec"] else cls_base * 0.5)

        df.at[idx, "audio_score"]      = round(min(score, 1.0), 3)
        df.at[idx, "audio_event_type"] = event

    return df


# ── Driver explanation builder ─────────────────────────────────────────────────
def build_explanation(accel_score, audio_score, combined,
                      accel_event, audio_event, speed, both_fired, severity):
    event_phrases = {
        "hard_brake_or_accel"  : "sudden braking or acceleration",
        "sharp_turn"           : "a sharp turn or swerve",
        "speed_bump_or_pothole": "a speed bump or rough road",
        "speeding"             : f"speeding at {speed:.0f} km/h (limit: 60)",
        "jerky_movement"       : "erratic jerky movement",
    }
    audio_phrases = {
        "cabin_argument" : "an argument inside the cabin",
        "loud_cabin"     : "unusually loud cabin noise",
        "elevated_noise" : "elevated cabin noise",
    }
    a = event_phrases.get(accel_event)
    b = audio_phrases.get(audio_event)

    if both_fired and a and b:
        base = f"{a} and {b} both detected simultaneously."
    elif a:
        base = f"{a} detected."
    elif b:
        base = f"{b} detected."
    else:
        base = "Minor event."

    sev_text = {"high":"High-severity event.","medium":"Moderate event.","low":"Minor event, logged for records."}.get(severity,"")
    return f"{base} {sev_text} Motion: {accel_score:.0%} · Noise: {audio_score:.0%} · Stress: {combined:.0%}."


# ── Combine signals ────────────────────────────────────────────────────────────
def combine_signals(accel_df, audio_df):
    accel_flags = accel_df[accel_df["accel_score"] > 0][
        ["trip_id","timestamp","speed_kmh","gps_lat","gps_lon",
         "accel_score","accel_event_type"]
    ].copy()

    audio_flags = audio_df[audio_df["audio_score"] > 0][
        ["trip_id","timestamp","audio_score","audio_event_type"]
    ].copy()

    rows = []
    for _, ar in accel_flags.iterrows():
        window = audio_flags[
            (audio_flags["trip_id"] == ar["trip_id"]) &
            (audio_flags["timestamp"] - ar["timestamp"]).abs().dt.total_seconds() <= 30
        ]
        if len(window):
            best        = window.loc[window["audio_score"].idxmax()]
            audio_score = best["audio_score"]
            audio_event = best["audio_event_type"]
            both_fired  = True
        else:
            audio_score = 0.0
            audio_event = "none"
            both_fired  = False

        # Road noise filter: speed bump + small audio spike = road noise
        if ar["accel_event_type"] == "speed_bump_or_pothole" and both_fired and audio_score < 0.4:
            audio_score = 0.0
            both_fired  = False

        combined = round(
            min(ar["accel_score"] * THRESH["accel_weight"] +
                audio_score       * THRESH["audio_weight"], 1.0) * (1.2 if both_fired else 1.0), 3
        )
        combined = min(combined, 1.0)

        severity = "high"   if combined >= THRESH["combined_high"]   else \
                   "medium" if combined >= THRESH["combined_medium"]  else "low"

        rows.append({
            "trip_id"           : ar["trip_id"],
            "timestamp"         : ar["timestamp"],
            "speed_kmh"         : ar["speed_kmh"],
            "gps_lat"           : ar["gps_lat"],
            "gps_lon"           : ar["gps_lon"],
            "accel_score"       : ar["accel_score"],
            "audio_score"       : audio_score,
            "combined_score"    : combined,
            "severity"          : severity,
            "accel_event_type"  : ar["accel_event_type"],
            "audio_event_type"  : audio_event,
            "both_signals_fired": both_fired,
            "driver_explanation": build_explanation(
                ar["accel_score"], audio_score, combined,
                ar["accel_event_type"], audio_event,
                ar["speed_kmh"] if not pd.isna(ar["speed_kmh"]) else 0,
                both_fired, severity
            ),
        })

    # Pure audio events (no accel counterpart)
    matched = {(r["trip_id"], r["timestamp"]) for r in rows}
    for _, aud in audio_flags.iterrows():
        if (aud["trip_id"], aud["timestamp"]) not in matched:
            combined = round(aud["audio_score"] * THRESH["audio_weight"], 3)
            severity = "high" if combined >= THRESH["combined_high"] else \
                       "medium" if combined >= THRESH["combined_medium"] else "low"
            rows.append({
                "trip_id":"none" ,"timestamp":aud["timestamp"],"speed_kmh":None,
                "gps_lat":None,"gps_lon":None,"accel_score":0.0,
                "audio_score":aud["audio_score"],"combined_score":combined,
                "severity":severity,"accel_event_type":"none",
                "audio_event_type":aud["audio_event_type"],"both_signals_fired":False,
                "driver_explanation":build_explanation(
                    0.0,aud["audio_score"],combined,"none",
                    aud["audio_event_type"],0,False,severity),
            })
            # fix trip_id
            rows[-1]["trip_id"] = aud["trip_id"]

    result = pd.DataFrame(rows).sort_values(["trip_id","timestamp"]).reset_index(drop=True)
    result.insert(0, "event_id", ["EVT"+str(i+1).zfill(3) for i in range(len(result))])
    return result


# ── Trip summary aggregation ───────────────────────────────────────────────────
def build_trip_summaries_sensor(flagged_df, trips_df):
    if len(flagged_df) == 0:
        return pd.DataFrame()

    agg = flagged_df.groupby("trip_id").agg(
        motion_events_count  =("accel_score",   lambda x: (x>0).sum()),
        audio_events_count   =("audio_score",   lambda x: (x>0).sum()),
        flagged_moments_count=("event_id",       "count"),
        max_severity         =("severity",       lambda x:
                                "high"   if "high"   in x.values else
                                "medium" if "medium" in x.values else "low"),
        stress_score         =("combined_score", "max"),
        worst_event_type     =("accel_event_type",lambda x:
                                x[x!="none"].mode()[0] if len(x[x!="none"])>0 else "none"),
    ).reset_index()

    result = trips_df[["trip_id","driver_id"]].merge(agg, on="trip_id", how="left")
    result["stress_score"]          = result["stress_score"].fillna(0.0).round(3)
    result["motion_events_count"]   = result["motion_events_count"].fillna(0).astype(int)
    result["audio_events_count"]    = result["audio_events_count"].fillna(0).astype(int)
    result["flagged_moments_count"] = result["flagged_moments_count"].fillna(0).astype(int)
    result["max_severity"]          = result["max_severity"].fillna("none")
    result["worst_event_type"]      = result["worst_event_type"].fillna("none")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    accel, audio, trips = load_data()

    print("Running 7 accelerometer checks...")
    accel_p = process_accelerometer(accel, trips)

    print("Running 4 audio checks...")
    audio_p = process_audio(audio)

    print("Combining signals...")
    flagged = combine_signals(accel_p, audio_p)

    print("Building sensor trip summaries...")
    sensor_summary = build_trip_summaries_sensor(flagged, trips)

    flagged.to_csv(OUT / "flagged_moments.csv", index=False)
    sensor_summary.to_csv(OUT / "trip_summaries_sensor.csv", index=False)

    print(f"\n✅ sensor_engine done")
    print(f"   flagged_moments.csv       → {len(flagged)} events, {len(flagged.columns)} columns")
    print(f"   trip_summaries_sensor.csv → {len(sensor_summary)} trips")

if __name__ == "__main__":
    main()