"""
Driver Pulse — Earnings Engine
Calculates earning velocity, forecasts, phase-aware messages, trip ratings.
Produces: enriched_velocity, enriched_goals, trip_earnings
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ── Phase thresholds (% of shift elapsed) ────────────────────────────────────
PHASE_EARLY  = 0.25
PHASE_MID    = 0.60
PHASE_PEAK   = 0.80

# Tolerance for on_track: tightens as shift progresses
PHASE_TOLERANCE = {
    "early": 0.20,   # within 20% of target → on_track
    "mid":   0.15,
    "peak":  0.10,
    "late":  0.05,
}

# Trip earnings rating thresholds (₹/hr)
EXCELLENT_VEL = 400
GOOD_VEL      = 280
AVERAGE_VEL   = 180


def _shift_phase(elapsed_frac: float) -> str:
    if elapsed_frac < PHASE_EARLY:
        return "early"
    elif elapsed_frac < PHASE_MID:
        return "mid"
    elif elapsed_frac < PHASE_PEAK:
        return "peak"
    return "late"


def _forecast_status(current_vel, needed_vel, phase: str, goal_status: str="") -> str:
    if goal_status == "achieved":
        return "achieved"
    if needed_vel <= 0:
        return "achieved"
    tol = PHASE_TOLERANCE.get(phase, 0.15)
    ratio = current_vel / needed_vel if needed_vel > 0 else 1.0
    if ratio >= (1 - tol):
        return "on_track"
    elif ratio >= (1 - tol - 0.20):
        return "at_risk"
    return "behind"


def _motivational_message(phase: str, status: str, pct: float, remaining_h: float,
                           current_vel: float, needed_vel: float) -> str:
    short_by = round(needed_vel - current_vel, 0) if needed_vel > current_vel else 0

    if status == "achieved":
        return "🎯 Goal reached! Any more trips today are a bonus."

    if status == "on_track":
        msgs = {
            "early": "Good start! Keep this pace and you'll hit your goal comfortably.",
            "mid":   f"You're on track — just {100 - pct:.0f}% left to go.",
            "peak":  "Almost there. Stay consistent for the last stretch.",
            "late":  "Final push! You're very close.",
        }
        return msgs.get(phase, "You're on track. Keep going!")

    if status == "at_risk":
        if short_by > 0:
            return (f"⚠️ You need ₹{int(short_by)}/hr more than your current pace. "
                    f"Try to fit in {max(1,int(remaining_h*1.5))} more trips.")
        return "⚠️ Slightly behind pace. Pick up speed if you can."

    # behind
    return (f"📉 You're behind goal — currently ₹{int(short_by)}/hr short of needed pace. "
            f"Focus on {int(remaining_h * 1.8)} more trips in the next hour.")


def _trip_rating(velocity: float) -> str:
    if velocity >= EXCELLENT_VEL:
        return "excellent"
    elif velocity >= GOOD_VEL:
        return "good"
    elif velocity >= AVERAGE_VEL:
        return "average"
    return "below_average"


# ── Main functions ────────────────────────────────────────────────────────────

def build_trip_earnings(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Per-trip earnings velocity and rating."""
    df = trips_df.copy()
    df["earnings_velocity"] = (
        df["fare"] / (df["duration_min"] / 60)
    ).round(2)
    df["earnings_rating"] = df["earnings_velocity"].apply(_trip_rating)
    df["route"] = df["pickup_location"] + " → " + df["dropoff_location"]
    return df[[
        "trip_id","driver_id","date","start_time","end_time",
        "duration_min","distance_km","fare","surge_multiplier",
        "pickup_location","dropoff_location","route",
        "earnings_velocity","earnings_rating",
    ]]


def build_enriched_velocity(velocity_df: pd.DataFrame, goals_df: pd.DataFrame) -> pd.DataFrame:
    """Time-series velocity log with phase, forecast, message."""
    rows = []
    goal_map = goals_df.drop_duplicates("driver_id").set_index("driver_id").to_dict("index")

    for _, row in velocity_df.iterrows():
        did   = row["driver_id"]
        goal  = goal_map.get(did, {})
        if not goal:
            continue

        target_hrs   = float(goal.get("target_hours", 8))
        elapsed_frac = row["elapsed_hours"] / target_hrs if target_hrs > 0 else 0
        phase        = _shift_phase(elapsed_frac)

        target_earn  = float(goal.get("target_earnings", 0))
        current_earn = row["cumulative_earnings"]
        elapsed_hrs  = row["elapsed_hours"]
        remaining_h  = max(target_hrs - elapsed_hrs, 0)

        current_vel  = round(current_earn / elapsed_hrs, 2) if elapsed_hrs > 0 else 0
        needed_vel   = round((target_earn - current_earn) / remaining_h, 2) if remaining_h > 0 else 0

        pct    = round(current_earn / target_earn * 100, 1) if target_earn > 0 else 0
        status = _forecast_status(current_vel, needed_vel, phase)
        msg    = _motivational_message(phase, status, pct, remaining_h, current_vel, needed_vel)

        rows.append({
            "log_id":           row["log_id"],
            "driver_id":        did,
            "timestamp":        row["timestamp"],
            "elapsed_hours":    elapsed_hrs,
            "cumulative_earnings": current_earn,
            "recalc_velocity":  current_vel,
            "target_velocity":  row.get("target_velocity", 175),
            "needed_velocity":  needed_vel,
            "shift_phase":      phase,
            "forecast_status":  status,
            "pct_complete":     pct,
            "message":          msg,
        })

    return pd.DataFrame(rows)


def build_enriched_goals(goals_df: pd.DataFrame, velocity_df: pd.DataFrame,
                         trips_df: pd.DataFrame = None) -> pd.DataFrame:
    """Goal snapshot with projection, gap, surge detection."""
    # Compute actual earnings per driver from real trips (ignore stale CSV field)
    if trips_df is not None:
        actual_earnings = trips_df.groupby("driver_id")["fare"].sum().to_dict()
        actual_trips    = trips_df.groupby("driver_id").size().to_dict()
    else:
        actual_earnings = {}
        actual_trips    = {}

    # Median baseline per driver
    baseline = (
        velocity_df.groupby("driver_id")["current_velocity"]
        .median().rename("baseline_vel").reset_index()
    )
    area_median = float(velocity_df["current_velocity"].median())

    rows = []
    for _, goal in goals_df.iterrows():
        did    = goal["driver_id"]
        target = float(goal["target_earnings"])
        # Use actual trip earnings — if trips_df provided, trust it fully (even if 0)
        if trips_df is not None:
            curr = float(actual_earnings.get(did, 0.0))
        else:
            curr = float(goal["current_earnings"])
        hrs    = float(goal["current_hours"])
        status = goal["status"]

        try:
            s_end   = datetime.strptime(f"2024-02-06 {goal['shift_end_time']}", "%Y-%m-%d %H:%M:%S")
            s_start = datetime.strptime(f"2024-02-06 {goal['shift_start_time']}", "%Y-%m-%d %H:%M:%S")
            total_h = (s_end - s_start).seconds / 3600
        except Exception:
            total_h = float(goal.get("target_hours", 8))

        remaining_h  = max(total_h - hrs, 0)
        current_vel  = round(curr / hrs, 2) if hrs > 0 else 0
        needed_vel   = round((target - curr) / remaining_h, 2) if remaining_h > 0 else 0

        # EC-09 surge adjustment
        brow = baseline[baseline["driver_id"] == did]
        base_vel = float(brow["baseline_vel"].iloc[0]) if not brow.empty else current_vel
        surge_adj = current_vel > base_vel * 2.0
        effective_vel = round(0.4 * current_vel + 0.6 * base_vel, 2) if surge_adj else current_vel

        projected = round(curr + effective_vel * remaining_h, 2)
        pct       = round(curr / target * 100, 1) if target > 0 else 0

        # EC-03 dead zone
        dead_zone = (effective_vel < area_median * 0.5 and status == "in_progress" and remaining_h > 1)

        elapsed_frac = hrs / total_h if total_h > 0 else 0
        phase = _shift_phase(elapsed_frac)

        if curr >= target:
            forecast = "achieved"
        elif remaining_h <= 0:
            # Shift is over and goal wasn't met
            forecast = "behind"
        elif dead_zone:
            forecast = "dead_zone"
        else:
            forecast = _forecast_status(effective_vel, needed_vel, phase)

        msg = _motivational_message(phase, forecast, pct, remaining_h, effective_vel, needed_vel)

        rows.append({
            "driver_id":           did,
            "target_earnings":     target,
            "current_earnings":    curr,
            "remaining_earnings":  round(max(target - curr, 0), 2),
            "current_hours":       hrs,
            "remaining_hours":     round(remaining_h, 2),
            "current_velocity":    current_vel,
            "effective_velocity":  effective_vel,
            "needed_velocity":     needed_vel,
            "projected_earnings":  projected,
            "goal_pct_complete":   pct,
            "shift_phase":         phase,
            "forecast":            forecast,
            "message":             msg,
            "surge_adjusted":      surge_adj,
            "dead_zone":           dead_zone,
        })

    return pd.DataFrame(rows)
