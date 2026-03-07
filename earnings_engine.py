
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("data")
EARNINGS  = DATA_ROOT / "earnings"
TRIPS_DIR    = DATA_ROOT / "trips"
OUT       = DATA_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

AVG_TRIP_FARE = 200  # ₹ used to estimate "trips needed"

PHASE_TOLERANCE = {
    "early": 0.25,  # 0–25% of shift  — forgiving, lots of time left
    "mid"  : 0.15,  # 25–60%
    "peak" : 0.10,  # 60–80%  — rush hour window
    "late" : 0.05,  # 80%+    — strict, almost no time left
}


# ── Load ───────────────────────────────────────────────────────────────────────
def load_data():
    goals    = pd.read_csv(EARNINGS / "driver_goals.csv")
    velocity = pd.read_csv(EARNINGS / "earnings_velocity_log.csv")
    trips    = pd.read_csv(TRIPS_DIR / "trips.csv")
    return goals, velocity, trips


# ── Shift phase ────────────────────────────────────────────────────────────────
def get_shift_phase(elapsed_hours, target_hours):
    if target_hours == 0: return "late"
    pct = elapsed_hours / target_hours
    if pct < 0.25:   return "early"
    elif pct < 0.60: return "mid"
    elif pct < 0.80: return "peak"
    else:            return "late"


# ── Time-aware forecast ────────────────────────────────────────────────────────
def time_aware_forecast(current_velocity, target_velocity,
                        elapsed_hours, target_hours,
                        current_earnings, target_earnings):
    phase = get_shift_phase(elapsed_hours, target_hours)

    if current_earnings >= target_earnings:
        return "achieved", phase

    delta = current_velocity - target_velocity
    slack = target_velocity * PHASE_TOLERANCE[phase]

    if delta >= 0:        return "ahead",    phase
    elif delta >= -slack: return "on_track", phase
    else:                 return "at_risk",  phase


# ── Driver message ─────────────────────────────────────────────────────────────
def build_message(status, phase, current_velocity, target_velocity,
                  current_earnings, target_earnings,
                  elapsed_hours, target_hours):

    remaining_hours    = max(target_hours - elapsed_hours, 0)
    remaining_earnings = float(max(target_earnings - current_earnings, 0))
    if remaining_earnings != remaining_earnings: remaining_earnings = 0.0
    projected_earnings = current_earnings + current_velocity * remaining_hours
    gap                = abs(current_velocity - target_velocity)
    trips_needed       = max(round(remaining_earnings / AVG_TRIP_FARE), 0)

    phase_context = {
        "early": "You're early in your shift — plenty of time to build momentum.",
        "mid"  : "You're in the middle of your shift.",
        "peak" : "You're in peak hours — your best window to earn.",
        "late" : "You're in the final stretch of your shift.",
    }

    if status == "achieved":
        return (f"Goal achieved! You've hit ₹{current_earnings:.0f} — "
                f"past your target of ₹{target_earnings:.0f}. "
                f"Any more trips are pure bonus.")

    if status == "ahead":
        return (f"You're ahead of pace by ₹{gap:.0f}/hr. "
                f"On track to finish around ₹{projected_earnings:.0f} — "
                f"₹{projected_earnings - target_earnings:.0f} above your goal. "
                f"{phase_context[phase]}")

    if status == "on_track":
        return (f"You're on track. Earning ₹{current_velocity:.0f}/hr, "
                f"need ₹{target_velocity:.0f}/hr. "
                f"About {trips_needed} more trips should get you there. "
                f"{phase_context[phase]}")

    # at_risk
    if phase == "late":
        return (f"You're behind and time is short. "
                f"At this pace you'll reach ₹{projected_earnings:.0f} "
                f"out of your ₹{target_earnings:.0f} goal. "
                f"Push for {trips_needed} more trips — every one counts now.")
    if phase == "peak":
        return (f"You're ₹{gap:.0f}/hr behind but peak hours are here. "
                f"Complete {trips_needed} more trips in {remaining_hours:.1f}h to recover. "
                f"Surge pricing could help close the gap.")
    return (f"You're ₹{gap:.0f}/hr behind pace. "
            f"Aim for {trips_needed} more trips. {phase_context[phase]}")


# ── Velocity engine → enriched_velocity.csv ───────────────────────────────────
def run_velocity_engine(velocity_df, goals_df):
    df = velocity_df.copy()

    goal_lookup = (
        goals_df
        .assign(
            _target_vel  = lambda g: g["target_earnings"] / g["target_hours"],
            _target_hrs  = lambda g: g["target_hours"],
            _target_earn = lambda g: g["target_earnings"],
        )
        .groupby("driver_id")[["_target_vel", "_target_hrs", "_target_earn"]]
        .mean()
    )
    df = df.join(goal_lookup, on="driver_id")

    df["recalc_velocity"] = (df["cumulative_earnings"] / df["elapsed_hours"]).round(2)

    results = df.apply(lambda r: time_aware_forecast(
        r["recalc_velocity"],     r.get("_target_vel",  r["target_velocity"]),
        r["elapsed_hours"],       r.get("_target_hrs",  8),
        r["cumulative_earnings"], r.get("_target_earn", 1400),
    ), axis=1)

    df["forecast_status"] = results.apply(lambda x: x[0])
    df["shift_phase"]     = results.apply(lambda x: x[1])
    df["message"]         = df.apply(lambda r: build_message(
        r["forecast_status"], r["shift_phase"],
        r["recalc_velocity"],     r.get("_target_vel",  r["target_velocity"]),
        r["cumulative_earnings"], r.get("_target_earn", 1400),
        r["elapsed_hours"],       r.get("_target_hrs",  8),
    ), axis=1)

    keep = ["driver_id", "timestamp", "cumulative_earnings", "elapsed_hours",
            "trips_completed", "recalc_velocity", "shift_phase",
            "forecast_status", "message"]
    return df[[c for c in keep if c in df.columns]]


# ── Goal tracker → enriched_goals.csv ─────────────────────────────────────────
def run_goal_tracker(goals_df):
    df = goals_df.copy()

    df["remaining_earnings"] = (df["target_earnings"] - df["current_earnings"]).clip(lower=0)
    df["remaining_hours"]    = (df["target_hours"]    - df["current_hours"]   ).clip(lower=0)
    df["projected_earnings"] = (
        df["current_earnings"] + df["earnings_velocity"] * df["remaining_hours"]
    ).round(2)
    df["goal_pct_complete"]  = (
        (df["current_earnings"] / df["target_earnings"]) * 100
    ).round(1)

    results = df.apply(lambda r: time_aware_forecast(
        r["earnings_velocity"], r["target_earnings"] / r["target_hours"],
        r["current_hours"],     r["target_hours"],
        r["current_earnings"],  r["target_earnings"],
    ), axis=1)

    df["forecast_status"] = results.apply(lambda x: x[0])
    df["shift_phase"]     = results.apply(lambda x: x[1])
    df["message"]         = df.apply(lambda r: build_message(
        r["forecast_status"], r["shift_phase"],
        r["earnings_velocity"], r["target_earnings"] / r["target_hours"],
        r["current_earnings"],  r["target_earnings"],
        r["current_hours"],     r["target_hours"],
    ), axis=1)

    keep = ["driver_id", "target_earnings", "target_hours",
            "current_earnings", "current_hours",
            "remaining_earnings", "remaining_hours",
            "projected_earnings", "goal_pct_complete",
            "shift_phase", "forecast_status", "message"]
    return df[[c for c in keep if c in df.columns]]


# ── Trip velocity → trip_earnings.csv ─────────────────────────────────────────
def run_trip_velocity(trips_df):
    df = trips_df[trips_df["trip_status"] == "completed"].copy()
    df["earnings_velocity"] = (df["fare"] / (df["duration_min"] / 60)).round(2)
    df["earnings_rating"]   = df["earnings_velocity"].apply(
        lambda v: "excellent" if v >= 400 else
                  "good"      if v >= 250 else
                  "average"   if v >= 150 else "below_average"
    )
    return df[["trip_id", "driver_id", "fare", "earnings_velocity", "earnings_rating"]]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    goals, velocity, trips = load_data()

    print("Running velocity engine...")
    enriched_velocity = run_velocity_engine(velocity, goals)

    print("Running goal tracker...")
    enriched_goals = run_goal_tracker(goals)

    print("Computing trip velocity...")
    trip_earnings = run_trip_velocity(trips)

    enriched_velocity.to_csv(OUT / "enriched_velocity.csv", index=False)
    enriched_goals.to_csv(OUT / "enriched_goals.csv",       index=False)
    trip_earnings.to_csv(OUT / "trip_earnings.csv",         index=False)

    print(f"\n✅ earnings_engine done")
    print(f"   enriched_velocity.csv → {len(enriched_velocity)} rows, {len(enriched_velocity.columns)} columns")
    print(f"   enriched_goals.csv    → {len(enriched_goals)} rows, {len(enriched_goals.columns)} columns")
    print(f"   trip_earnings.csv     → {len(trip_earnings)} rows, {len(trip_earnings.columns)} columns")
    print(f"\n   Sample: {enriched_goals['message'].iloc[0]}")

if __name__ == "__main__":
    main()
