"""
merge_engine.py — Driver Pulse  (Run this THIRD)
=================================================
Reads  : flagged_moments.csv, trip_earnings.csv, enriched_goals.csv,
         trips.csv, drivers.csv
Outputs: trip_summaries.csv, driver_shift_summary.csv,
         driver_reports/report_{driver_id}.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path(r"data")
OUT       = DATA_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
def load_all():
    flagged        = pd.read_csv(OUT / "flagged_moments.csv")
    trip_earnings  = pd.read_csv(OUT / "trip_earnings.csv")
    enriched_goals = pd.read_csv(OUT / "enriched_goals.csv")
    drivers        = pd.read_csv(DATA_ROOT / "drivers" / "drivers.csv")
    trips_raw      = pd.read_csv(DATA_ROOT / "trips" / "trips.csv")
    return flagged, trip_earnings, enriched_goals, drivers, trips_raw


# ── trip_summaries.csv ─────────────────────────────────────────────────────────
# Dashboard use: trips table, per-trip earnings vs stress view
def build_trip_summaries(flagged, trip_earnings, trips_raw):

    # Aggregate sensor events per trip
    sensor_agg = flagged.groupby("trip_id").agg(
        motion_events_count  =("accel_score",     lambda x: (x > 0).sum()),
        audio_events_count   =("audio_score",     lambda x: (x > 0).sum()),
        flagged_moments_count=("event_id",         "count"),
        max_severity         =("severity",         lambda x:
                                "high"   if "high"   in x.values else
                                "medium" if "medium" in x.values else "low"),
        stress_score         =("combined_score",   "max"),
        worst_event_type     =("accel_event_type", lambda x:
                                x[x != "none"].mode()[0] if len(x[x != "none"]) > 0 else "none"),
    ).reset_index()

    # Merge earnings + sensor
    df = trip_earnings.merge(sensor_agg, on="trip_id", how="left")

    df["stress_score"]          = df["stress_score"].fillna(0.0).round(3)
    df["motion_events_count"]   = df["motion_events_count"].fillna(0).astype(int)
    df["audio_events_count"]    = df["audio_events_count"].fillna(0).astype(int)
    df["flagged_moments_count"] = df["flagged_moments_count"].fillna(0).astype(int)
    df["max_severity"]          = df["max_severity"].fillna("none")
    df["worst_event_type"]      = df["worst_event_type"].fillna("none")

    # Overall trip quality combining earnings + stress
    df["trip_quality_rating"] = df.apply(lambda r:
        "poor"      if r["stress_score"] >= 0.65 else
        "fair"      if r["stress_score"] >= 0.35 and r["earnings_rating"] != "excellent" else
        "good"      if r["stress_score"] >= 0.35 else
        "excellent" if r["earnings_rating"] == "excellent" else
        "good"      if r["earnings_rating"] == "good" else "average",
        axis=1
    )

    # Add route info from raw trips
    trip_meta = trips_raw[["trip_id", "start_time", "duration_min",
                            "distance_km", "pickup_location", "dropoff_location"]]
    df = df.merge(trip_meta, on="trip_id", how="left")

    # Final column order
    cols = [
        # Identity
        "trip_id", "driver_id", "start_time",
        # Trip facts
        "duration_min", "distance_km", "pickup_location", "dropoff_location",
        # Earnings
        "fare", "earnings_velocity", "earnings_rating",
        # Safety
        "motion_events_count", "audio_events_count", "flagged_moments_count",
        "max_severity", "stress_score", "worst_event_type",
        # Combined
        "trip_quality_rating",
    ]
    return df[[c for c in cols if c in df.columns]]


# ── driver_shift_summary.csv ───────────────────────────────────────────────────
# Dashboard use: driver cards, leaderboard, goal progress bars
def build_shift_summary(trip_summaries, enriched_goals, drivers):

    # Roll up all trips per driver
    trip_agg = trip_summaries.groupby("driver_id").agg(
        total_trips          =("trip_id",             "count"),
        total_earned         =("fare",                "sum"),
        best_trip_velocity   =("earnings_velocity",   "max"),
        avg_trip_velocity    =("earnings_velocity",   "mean"),
        high_stress_trips    =("max_severity",        lambda x: (x == "high").sum()),
        total_flagged_moments=("flagged_moments_count","sum"),
        stress_score         =("stress_score",        "max"),
        most_common_issue    =("worst_event_type",    lambda x:
                                x[x != "none"].mode()[0] if len(x[x != "none"]) > 0 else "none"),
        excellent_trips      =("trip_quality_rating", lambda x: (x == "excellent").sum()),
        poor_trips           =("trip_quality_rating", lambda x: (x == "poor").sum()),
    ).reset_index().round(2)

    # Best earning route of the day
    def best_route(grp):
        row = grp.loc[grp["earnings_velocity"].idxmax()]
        return f"{row['pickup_location']} to {row['dropoff_location']}"

    route_map = (trip_summaries
                 .groupby("driver_id")
                 .apply(best_route, include_groups=False)
                 .reset_index())
    route_map.columns = ["driver_id", "best_route_today"]
    trip_agg = trip_agg.merge(route_map, on="driver_id", how="left")

    # Latest goal snapshot per driver
    latest_goal = (
        enriched_goals
        .sort_values("current_hours", ascending=False)
        .groupby("driver_id").first()
        .reset_index()
    )[["driver_id", "target_earnings", "current_earnings", "goal_pct_complete",
       "projected_earnings", "forecast_status", "shift_phase", "message"]]
    latest_goal.rename(columns={"message": "goal_message"}, inplace=True)

    # Stress label for dashboard badge
    trip_agg["overall_stress_label"] = trip_agg["stress_score"].apply(
        lambda s: "HIGH" if s >= 0.65 else "MODERATE" if s >= 0.35 else "LOW" if s > 0 else "NONE"
    )

    df = trip_agg.merge(latest_goal, on="driver_id", how="left")
    df = df.merge(drivers[["driver_id", "name", "city", "rating"]],
                  on="driver_id", how="left")

    # Final column order
    cols = [
        # Identity
        "driver_id", "name", "city", "rating",
        # Earnings summary
        "total_trips", "total_earned", "best_trip_velocity", "avg_trip_velocity",
        "best_route_today", "excellent_trips", "poor_trips",
        # Safety summary
        "high_stress_trips", "total_flagged_moments",
        "stress_score", "most_common_issue", "overall_stress_label",
        # Goal progress (feeds progress bar + forecast card on dashboard)
        "target_earnings", "current_earnings", "goal_pct_complete",
        "projected_earnings", "forecast_status", "shift_phase", "goal_message",
    ]
    return df[[c for c in cols if c in df.columns]]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    flagged, trip_earnings, enriched_goals, drivers, trips_raw = load_all()

    print("Building trip_summaries.csv...")
    trip_summaries = build_trip_summaries(flagged, trip_earnings, trips_raw)
    trip_summaries.to_csv(OUT / "trip_summaries.csv", index=False)

    print("Building driver_shift_summary.csv...")
    shift_summary = build_shift_summary(trip_summaries, enriched_goals, drivers)
    shift_summary.to_csv(OUT / "driver_shift_summary.csv", index=False)

    print(f"\n--- trip_summaries.csv ({len(trip_summaries)} rows, {len(trip_summaries.columns)} cols) ---")
    print(trip_summaries.head(3).to_string())

    print(f"\n--- driver_shift_summary.csv ({len(shift_summary)} rows, {len(shift_summary.columns)} cols) ---")
    print(shift_summary.head(3).to_string())

    print("\nDone. Files saved to:", OUT)

if __name__ == "__main__":
    main()
