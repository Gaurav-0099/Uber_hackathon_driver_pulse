"""
Driver Pulse — Merge Engine
Joins financial + safety data → final dashboard files.
Produces: trip_summaries, driver_shift_summary
"""

import pandas as pd


STRESS_BADGE = {
    (0.65, 1.01): ("HIGH",     "🔴"),
    (0.40, 0.65): ("MODERATE", "🟡"),
    (0.20, 0.40): ("LOW",      "🟢"),
    (0.00, 0.20): ("NONE",     "✅"),
}


def _stress_badge(peak_score: float):
    for (lo, hi), (label, icon) in STRESS_BADGE.items():
        if lo <= peak_score < hi:
            return label, icon
    return "NONE", "✅"


def _trip_quality(stress: float, earnings_rating: str) -> str:
    """Balance profit and safety."""
    if stress >= 0.65:
        return "poor"
    if stress < 0.20 and earnings_rating in ("excellent", "good"):
        return "excellent"
    if stress < 0.35 and earnings_rating in ("excellent", "good", "average"):
        return "good"
    if stress < 0.50:
        return "average"
    return "poor"


def build_trip_summaries(trip_earnings_df, sensor_summaries_df) -> pd.DataFrame:
    """Master trip table combining earnings + sensor."""
    merged = trip_earnings_df.merge(
        sensor_summaries_df[[
            "trip_id","motion_events_count","audio_events_count",
            "flagged_count","stress_score","worst_event_type"
        ]],
        on="trip_id", how="left"
    )
    merged["stress_score"]       = merged["stress_score"].fillna(0.0)
    merged["flagged_count"]      = merged["flagged_count"].fillna(0).astype(int)
    merged["worst_event_type"]   = merged["worst_event_type"].fillna("")
    merged["motion_events_count"]= merged["motion_events_count"].fillna(0).astype(int)
    merged["audio_events_count"] = merged["audio_events_count"].fillna(0).astype(int)

    merged["trip_quality"] = merged.apply(
        lambda r: _trip_quality(r["stress_score"], r["earnings_rating"]), axis=1
    )

    return merged


def build_driver_shift_summary(trip_summaries_df, enriched_goals_df, flagged_df) -> pd.DataFrame:
    """One row per driver — full shift summary."""
    rows = []

    for did, trips in trip_summaries_df.groupby("driver_id"):
        goal_row = enriched_goals_df[enriched_goals_df["driver_id"] == did]
        flags    = flagged_df[flagged_df["driver_id"] == did] if not flagged_df.empty else pd.DataFrame()

        peak_stress   = trips["stress_score"].max()
        badge, icon   = _stress_badge(peak_stress)
        total_earned  = trips["fare"].sum()
        total_trips   = len(trips)
        avg_velocity  = trips["earnings_velocity"].mean().round(0)

        quality_counts = trips["trip_quality"].value_counts().to_dict()
        n_excellent = quality_counts.get("excellent", 0)
        n_good      = quality_counts.get("good", 0)
        n_average   = quality_counts.get("average", 0)
        n_poor      = quality_counts.get("poor", 0)

        # Best route = highest earnings_velocity trip
        best_trip = trips.loc[trips["earnings_velocity"].idxmax()]
        best_route = best_trip.get("route", "")
        best_vel   = best_trip["earnings_velocity"]

        # Goal data
        forecast   = goal_row["forecast"].iloc[0]       if not goal_row.empty else "unknown"
        message    = goal_row["message"].iloc[0]        if not goal_row.empty else ""
        target     = goal_row["target_earnings"].iloc[0] if not goal_row.empty else 0
        projected  = goal_row["projected_earnings"].iloc[0] if not goal_row.empty else total_earned
        remaining_h= goal_row["remaining_hours"].iloc[0] if not goal_row.empty else 0
        pct        = goal_row["goal_pct_complete"].iloc[0] if not goal_row.empty else 0

        rows.append({
            "driver_id":         did,
            "total_earned":      total_earned,
            "total_trips":       total_trips,
            "avg_velocity":      avg_velocity,
            "peak_stress_score": round(peak_stress, 3),
            "stress_badge":      badge,
            "stress_icon":       icon,
            "n_excellent_trips": n_excellent,
            "n_good_trips":      n_good,
            "n_average_trips":   n_average,
            "n_poor_trips":      n_poor,
            "best_route":        best_route,
            "best_route_vel":    round(best_vel, 0),
            "target_earnings":   target,
            "projected_earnings":projected,
            "remaining_hours":   remaining_h,
            "goal_pct_complete": pct,
            "forecast":          forecast,
            "goal_message":      message,
            "total_flags":       len(flags),
            "high_flags":        len(flags[flags["severity"]=="high"])  if not flags.empty else 0,
        })

    return pd.DataFrame(rows)
