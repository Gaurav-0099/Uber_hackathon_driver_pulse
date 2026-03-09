"""
Driver Pulse — Streamlit App
Simple, driver-facing dashboard.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sensor_engine   import run_accel_checks, run_audio_checks, detect_flagged_moments, build_trip_summaries_sensor
from earnings_engine import build_trip_earnings, build_enriched_velocity, build_enriched_goals
from merge_engine    import build_trip_summaries, build_driver_shift_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

st.set_page_config(page_title="Driver Pulse", page_icon="🚗", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;background:#f5f6fa!important;color:#1a1d23!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.5rem 2rem 3rem!important;max-width:1100px!important}
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid #e8eaed!important}
[data-testid="metric-container"]{background:#fff!important;border:1px solid #e8eaed!important;border-radius:16px!important;padding:20px 22px!important;box-shadow:0 1px 3px rgba(0,0,0,.06)!important}
[data-testid="metric-container"] label{font-size:12px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.05em!important;color:#6b7280!important}
[data-testid="stMetricValue"]{font-size:1.75rem!important;font-weight:800!important;color:#1a1d23!important}
.stTabs [data-baseweb="tab-list"]{background:#fff!important;border-radius:12px!important;padding:4px!important;gap:2px!important;border:1px solid #e8eaed!important;width:fit-content!important}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;font-weight:600!important;font-size:13px!important;color:#6b7280!important;padding:8px 18px!important}
.stTabs [aria-selected="true"]{background:#1a1d23!important;color:#fff!important}
.stTabs [aria-selected="true"] p,.stTabs [aria-selected="true"] span,.stTabs [aria-selected="true"] div{color:#fff!important}
.stButton>button{border-radius:10px!important;font-weight:600!important;border:1px solid #e8eaed!important;background:#fff!important;color:#1a1d23!important}
.stSelectbox>div>div{border-radius:10px!important;border-color:#e8eaed!important;background:#fff!important}
[data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden!important;border:1px solid #e8eaed!important}
hr{border-color:#e8eaed!important;margin:1.2rem 0!important}
p,span,label,caption,.stMarkdown{color:#1a1d23!important}
.stCaption,.stCaption p{color:#6b7280!important;font-size:13px!important}
h1,h2,h3,h4,h5,h6{color:#1a1d23!important}
.stRadio label{color:#1a1d23!important;font-weight:600!important;font-size:13px!important}
.stRadio [data-testid="stMarkdownContainer"] p{color:#1a1d23!important}
.stRadio div[role="radiogroup"]{background:#fff!important;border:1px solid #e8eaed!important;border-radius:12px!important;padding:6px 8px!important;display:inline-flex!important;gap:4px!important}
.stRadio label{padding:7px 16px!important;border-radius:8px!important;cursor:pointer!important}
.stRadio input[type="radio"]{display:none!important}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:#1a1d23!important}
[data-testid="stMetricDelta"] p{color:#6b7280!important}
.stAlert p{color:#1a1d23!important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#ffffff!important;border:1px solid #e8eaed!important;border-radius:12px!important;box-shadow:0 4px 16px rgba(0,0,0,.08)!important}
[data-baseweb="menu"] ul{background:#ffffff!important;padding:6px!important}
[data-baseweb="menu"] li{background:#ffffff!important;color:#1a1d23!important;border-radius:8px!important;padding:8px 12px!important;font-size:14px!important;font-weight:500!important}
[data-baseweb="menu"] li:hover{background:#f0f1f3!important;color:#1a1d23!important}
[data-baseweb="menu"] [aria-selected="true"]{background:#f0f1f3!important;color:#1a1d23!important;font-weight:600!important}
[role="listbox"]{background:#ffffff!important}
[role="option"]{background:#ffffff!important;color:#1a1d23!important}
[role="option"]:hover,[role="option"]:focus{background:#f0f1f3!important;color:#1a1d23!important}
</style>
""", unsafe_allow_html=True)

FORECAST_COLOR = {"achieved":"#16a34a","on_track":"#2563eb","at_risk":"#d97706","behind":"#dc2626","dead_zone":"#7c3aed"}
QUALITY_COLOR  = {"excellent":"#16a34a","good":"#2563eb","average":"#d97706","below_average":"#dc2626","poor":"#dc2626"}
SEVERITY_COLOR = {"high":"#dc2626","medium":"#d97706","low":"#16a34a","none":"#9ca3af"}
EVENT_LABEL    = {
    "harsh_brake":"Hard Brake","harsh_accel":"Sudden Acceleration","sharp_turn":"Sharp Turn",
    "speed_bump":"Speed Bump Hit","speeding":"Speeding","jerk":"Jerky Movement",
    "safety_maneuver":"Emergency Stop","conflict_moment":"Conflict Moment","argument":"Loud Argument",
}
PLOTLY_BASE = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#ffffff",
                   font=dict(family="Inter, sans-serif",color="#1a1d23",size=12),
                   margin=dict(l=8,r=8,t=28,b=8))

# ── Pipeline (cached 30s) ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def load_and_process():
    raw_trips    = pd.read_csv(os.path.join(DATA_DIR,"trips.csv"))
    raw_accel    = pd.read_csv(os.path.join(DATA_DIR,"accelerometer_data.csv"))
    raw_audio    = pd.read_csv(os.path.join(DATA_DIR,"audio_intensity_data.csv"))
    raw_goals    = pd.read_csv(os.path.join(DATA_DIR,"driver_goals.csv"))
    raw_velocity = pd.read_csv(os.path.join(DATA_DIR,"earnings_velocity_log.csv"))
    raw_drivers  = pd.read_csv(os.path.join(DATA_DIR,"drivers.csv"))
    accel_p  = run_accel_checks(raw_accel)
    audio_p  = run_audio_checks(raw_audio)
    flagged  = detect_flagged_moments(accel_p, audio_p, raw_trips)
    sensor_s = build_trip_summaries_sensor(raw_trips, flagged, accel_p, audio_p)
    trip_earn  = build_trip_earnings(raw_trips)
    enrich_vel = build_enriched_velocity(raw_velocity, raw_goals)
    enrich_g   = build_enriched_goals(raw_goals, raw_velocity, raw_trips)
    trip_sum   = build_trip_summaries(trip_earn, sensor_s)
    shift_sum  = build_driver_shift_summary(trip_sum, enrich_g, flagged)
    return dict(drivers=raw_drivers, accel=accel_p, audio=audio_p, flagged=flagged,
                trip_sum=trip_sum, shift_sum=shift_sum, enrich_vel=enrich_vel, enrich_goals=enrich_g)

with st.spinner("Loading…"):
    data = load_and_process()

drivers_df  = data["drivers"]
flagged_df  = data["flagged"]
trip_sum_df = data["trip_sum"]
shift_sum_df= data["shift_sum"]
vel_df      = data["enrich_vel"]
goals_df    = data["enrich_goals"]
accel_df    = data["accel"]
audio_df    = data["audio"]

# ── Header ────────────────────────────────────────────────────────────────────
driver_options = drivers_df[["driver_id","name"]].copy()
driver_options["label"] = driver_options["name"] + "  ·  " + driver_options["driver_id"]

h1, h2, h3 = st.columns([2,3,1])
with h1: st.markdown("### 🚗 Driver Pulse")
with h2:
    selected_label = st.selectbox("Driver", driver_options["label"].tolist(),
                                  label_visibility="collapsed")
with h3:
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

DID   = driver_options.loc[driver_options["label"]==selected_label,"driver_id"].iloc[0]
dinfo = drivers_df[drivers_df["driver_id"]==DID].iloc[0]
st.divider()

# ── Filter ────────────────────────────────────────────────────────────────────
my_trips  = trip_sum_df[trip_sum_df["driver_id"]==DID].copy()
my_flags  = flagged_df[flagged_df["driver_id"]==DID].copy() if not flagged_df.empty else pd.DataFrame()
my_goal   = goals_df[goals_df["driver_id"]==DID]
my_shift  = shift_sum_df[shift_sum_df["driver_id"]==DID]
my_vel    = vel_df[vel_df["driver_id"]==DID].sort_values("elapsed_hours")
my_accel  = accel_df[accel_df["trip_id"].isin(my_trips["trip_id"])]
my_audio  = audio_df[audio_df["trip_id"].isin(my_trips["trip_id"])]

gc = my_goal.iloc[0]  if not my_goal.empty  else None
sc = my_shift.iloc[0] if not my_shift.empty else None

# ── Driver name + badge ───────────────────────────────────────────────────────
forecast_label = {
    "achieved":"Goal Achieved 🎯","on_track":"On Track ✅",
    "at_risk":"At Risk ⚠️","behind":"Behind 📉","dead_zone":"Low Demand 🗺️",
}.get(gc["forecast"] if gc is not None else "","")

na, nb = st.columns([4,1])
with na:
    st.markdown(f"## {dinfo['name']}")
    st.caption(f"{dinfo['city']}  ·  ⭐ {dinfo['rating']}  ·  {dinfo['shift_preference'].title()} shift")
with nb:
    if gc is not None:
        fc_col = FORECAST_COLOR.get(gc["forecast"],"#6b7280")
        st.markdown(f'<div style="text-align:right;padding-top:16px"><span style="background:{fc_col}18;color:{fc_col};border:1.5px solid {fc_col}55;padding:7px 16px;border-radius:999px;font-size:13px;font-weight:700">{forecast_label}</span></div>',unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_home, tab_trips, tab_safety, tab_charts = st.tabs([
    "  Home  ", f"  Trips ({len(my_trips)})  ", "  Safety  ", "  Charts  "])

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    earned_today = int(my_trips["fare"].sum()) if not my_trips.empty else 0
    target       = int(gc["target_earnings"]) if gc is not None else 0
    projected    = int(gc["projected_earnings"]) if gc is not None else earned_today
    avg_vel      = int(my_trips["earnings_velocity"].mean()) if not my_trips.empty else 0
    needed_vel   = int(gc["needed_velocity"]) if gc is not None else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Earned Today",     f"₹{earned_today:,}", f"of ₹{target:,} goal")
    c2.metric("Goal",             f"₹{target:,}",       f"{gc['goal_pct_complete']:.0f}% done" if gc is not None else "")
    c3.metric("End-of-shift est", f"₹{projected:,}",    "▲ Above goal" if projected>=target else "▼ Below goal")
    c4.metric("Avg ₹/hr",         f"₹{avg_vel:,}",      f"Need ₹{needed_vel}/hr" if needed_vel>0 else "Goal met!")

    st.markdown("<br>", unsafe_allow_html=True)

    # Goal bar
    if gc is not None:
        pct = min(gc["goal_pct_complete"],100)
        bar_col = FORECAST_COLOR.get(gc["forecast"],"#2563eb")
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e8eaed;border-radius:16px;padding:22px 24px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <span style="font-weight:700;font-size:15px">Goal Progress</span>
            <span style="font-weight:800;font-size:18px;color:{bar_col}">{pct:.0f}%</span>
          </div>
          <div style="background:#f0f1f3;border-radius:999px;height:14px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{bar_col};border-radius:999px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:12px;color:#6b7280">
            <span>₹0</span><span>₹{earned_today:,} earned</span><span>₹{target:,} goal</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Message
    if gc is not None and gc["message"]:
        mc = FORECAST_COLOR.get(gc["forecast"],"#2563eb")
        st.markdown(f'<div style="background:{mc}0d;border-left:4px solid {mc};border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:16px"><div style="font-size:14px;color:#1a1d23;line-height:1.6">{gc["message"]}</div></div>',unsafe_allow_html=True)
        if gc.get("dead_zone"):
            st.info("🗺️ **Low demand in your area.** Try moving to a busier zone — this isn't a driving issue.")
        if gc.get("surge_adjusted"):
            st.info("⚡ **Surge trips included.** Your forecast uses a realistic average, not just the surge rate.")

    # Best route + trip breakdown
    bc, qc = st.columns(2)
    with bc:
        br  = sc["best_route"]    if sc is not None else "—"
        brv = int(sc["best_route_vel"]) if sc is not None else 0
        st.markdown(f'<div style="background:#fff;border:1px solid #e8eaed;border-radius:16px;padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,.05)"><div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#6b7280;margin-bottom:8px">Best Route Today</div><div style="font-size:16px;font-weight:700;margin-bottom:4px">{br}</div><div style="font-size:13px;color:#16a34a;font-weight:600">₹{brv}/hr earnings rate</div></div>',unsafe_allow_html=True)

    with qc:
        if sc is not None:
            ne,ng,nav,np_ = sc["n_excellent_trips"],sc["n_good_trips"],sc["n_average_trips"],sc["n_poor_trips"]
            def pill(l,n,c): return f'<span style="background:{c}18;color:{c};border:1px solid {c}44;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:700;margin-right:5px">{n} {l}</span>' if n>0 else ""
            st.markdown(f'<div style="background:#fff;border:1px solid #e8eaed;border-radius:16px;padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,.05)"><div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#6b7280;margin-bottom:10px">Trip Breakdown</div><div style="margin-bottom:8px">{pill("Excellent",ne,"#16a34a")}{pill("Good",ng,"#2563eb")}{pill("Average",nav,"#d97706")}{pill("Poor",np_,"#dc2626")}</div><div style="font-size:12px;color:#6b7280">{ne+ng+nav+np_} trips total</div></div>',unsafe_allow_html=True)

    # Safety snapshot
    st.markdown("<br>", unsafe_allow_html=True)
    n_flags = len(my_flags) if not my_flags.empty else 0
    n_high  = len(my_flags[my_flags["severity"]=="high"]) if not my_flags.empty else 0
    bl      = sc["stress_badge"] if sc is not None else "NONE"
    bi      = sc["stress_icon"]  if sc is not None else "✅"
    bc2     = {"HIGH":"#dc2626","MODERATE":"#d97706","LOW":"#16a34a","NONE":"#16a34a"}.get(bl,"#6b7280")
    high_txt= f'&nbsp;·&nbsp;<span style="color:#dc2626;font-weight:600">{n_high} high severity</span>' if n_high>0 else ""
    st.markdown(f'<div style="background:#fff;border:1px solid #e8eaed;border-radius:16px;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.05)"><div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px"><span style="font-size:15px;font-weight:700">Safety &amp; Stress</span><span style="background:{bc2}18;color:{bc2};border:1.5px solid {bc2}55;padding:5px 14px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.05em">{bi} {bl} STRESS</span></div><div style="font-size:13px;color:#6b7280">{n_flags} {"moment" if n_flags==1 else "moments"} flagged today{high_txt}</div></div>',unsafe_allow_html=True)

    # Top 5 worst moments
    if not my_flags.empty:
        top5 = my_flags.nlargest(5,"combined_score")
        st.markdown("<br>**Top flagged moments**")
        for _,flag in top5.iterrows():
            sc2  = SEVERITY_COLOR.get(flag["severity"],"#9ca3af")
            evl  = EVENT_LABEL.get(flag["accel_type"],flag["accel_type"].replace("_"," ").title())
            scr  = int(flag["combined_score"]*100)
            tr   = my_trips[my_trips["trip_id"]==flag["trip_id"]]
            rt   = tr["route"].iloc[0] if not tr.empty and "route" in tr.columns else flag["trip_id"]
            ts   = str(flag["timestamp"]).split(" ")[-1][:5] if " " in str(flag["timestamp"]) else str(flag["timestamp"])
            st.markdown(f'<div style="display:flex;align-items:center;padding:12px 16px;background:#fff;border:1px solid #e8eaed;border-radius:12px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,.04)"><div style="width:10px;height:10px;border-radius:50%;background:{sc2};flex-shrink:0;margin-right:14px"></div><div style="flex:1"><div style="font-size:14px;font-weight:600">{evl}</div><div style="font-size:12px;color:#6b7280;margin-top:1px">{rt}&nbsp;·&nbsp;{ts}</div><div style="font-size:12px;color:#6b7280;margin-top:2px">{flag["driver_explanation"]}</div></div><div style="text-align:right;flex-shrink:0;margin-left:12px"><div style="font-size:18px;font-weight:800;color:{sc2}">{scr}</div><div style="font-size:10px;color:#9ca3af">/ 100</div></div></div>',unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("✅ No stress moments flagged today. Clean shift!")

# ══════════════════════════════════════════════════════════════════════════════
# TRIPS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trips:
    if my_trips.empty:
        st.info("No trips yet.")
    else:
        st.markdown("### All Trips")
        st.caption("Every trip with route, earnings, and safety rating.")
        st.markdown("<br>", unsafe_allow_html=True)
        for _,trip in my_trips.iterrows():
            qc2  = QUALITY_COLOR.get(trip["trip_quality"],"#6b7280")
            qlbl = trip["trip_quality"].replace("_"," ").title()
            elbl = trip["earnings_rating"].replace("_"," ").title()
            fn   = int(trip["flagged_count"])
            ev2  = int(trip["earnings_velocity"])
            rt2  = trip.get("route",f"{trip.get('pickup_location','')} → {trip.get('dropoff_location','')}")
            sev_dot = SEVERITY_COLOR.get("high" if trip["stress_score"]>=0.65 else "medium" if trip["stress_score"]>=0.4 else "low" if trip["stress_score"]>=0.2 else "none","#9ca3af")
            alert_pill = f'<span style="background:#dc262618;color:#dc2626;border:1px solid #dc262644;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700">{fn} alert{"s" if fn>1 else ""}</span>' if fn>0 else ""
            st.markdown(f'<div style="background:#fff;border:1px solid #e8eaed;border-radius:14px;padding:16px 20px;margin-bottom:10px;box-shadow:0 1px 2px rgba(0,0,0,.04)"><div style="display:flex;align-items:flex-start;justify-content:space-between"><div style="flex:1"><div style="display:flex;align-items:center;gap:10px;margin-bottom:5px"><span style="font-size:13px;color:#9ca3af;font-weight:500">{trip["trip_id"]}</span><span style="background:{qc2}18;color:{qc2};border:1px solid {qc2}44;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700">{qlbl}</span>{alert_pill}</div><div style="font-size:15px;font-weight:700;margin-bottom:4px">{rt2}</div><div style="font-size:12px;color:#6b7280">{trip["start_time"]} – {trip["end_time"]}&nbsp;·&nbsp;{trip["duration_min"]} min&nbsp;·&nbsp;{trip["distance_km"]} km</div></div><div style="text-align:right;flex-shrink:0;margin-left:20px"><div style="font-size:20px;font-weight:800;color:#16a34a">₹{int(trip["fare"])}</div><div style="font-size:12px;color:#6b7280">₹{ev2}/hr</div><div style="display:flex;align-items:center;justify-content:flex-end;gap:5px;margin-top:5px"><div style="width:8px;height:8px;border-radius:50%;background:{sev_dot}"></div><span style="font-size:11px;color:#6b7280">{elbl}</span></div></div></div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SAFETY
# ══════════════════════════════════════════════════════════════════════════════
with tab_safety:
    st.markdown("### Safety Details")
    st.caption("Full breakdown of every flagged moment this shift.")
    st.markdown("<br>", unsafe_allow_html=True)

    if my_flags.empty:
        st.success("✅ Nothing flagged today — great shift!")
    else:
        n_h = len(my_flags[my_flags["severity"]=="high"])
        n_m = len(my_flags[my_flags["severity"]=="medium"])
        n_l = len(my_flags[my_flags["severity"]=="low"])
        s1,s2,s3 = st.columns(3)
        s1.metric("High Severity",n_h)
        s2.metric("Medium Severity",n_m)
        s3.metric("Low / Minor",n_l)
        st.markdown("<br>", unsafe_allow_html=True)

        trip_opts = ["All trips"] + sorted(my_flags["trip_id"].unique().tolist())
        sel_t = st.selectbox("Filter by trip", trip_opts)
        vf = my_flags if sel_t=="All trips" else my_flags[my_flags["trip_id"]==sel_t]
        vf = vf.sort_values("combined_score",ascending=False)

        for _,flag in vf.iterrows():
            sc3 = SEVERITY_COLOR.get(flag["severity"],"#9ca3af")
            evl = EVENT_LABEL.get(flag["accel_type"],flag["accel_type"].replace("_"," ").title())
            scr = int(flag["combined_score"]*100)
            tr2 = my_trips[my_trips["trip_id"]==flag["trip_id"]]
            rt3 = tr2["route"].iloc[0] if not tr2.empty and "route" in tr2.columns else flag["trip_id"]
            ts2 = str(flag["timestamp"]).split(" ")[-1][:5] if " " in str(flag["timestamp"]) else ""
            mp  = int(flag["accel_score"]*100)
            ap  = int(flag["audio_score"]*100)
            st.markdown(f'<div style="background:#fff;border:1px solid #e8eaed;border-left:4px solid {sc3};border-radius:0 14px 14px 0;padding:18px 20px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.05)"><div style="display:flex;align-items:flex-start;justify-content:space-between"><div style="flex:1"><div style="display:flex;align-items:center;gap:8px;margin-bottom:6px"><span style="font-size:15px;font-weight:700">{evl}</span><span style="background:{sc3}18;color:{sc3};border:1px solid {sc3}44;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase">{flag["severity"]}</span></div><div style="font-size:13px;color:#374151;margin-bottom:8px;line-height:1.5">{flag["driver_explanation"]}</div><div style="font-size:12px;color:#9ca3af">{rt3}&nbsp;·&nbsp;{ts2}&nbsp;·&nbsp;{flag["trip_id"]}</div></div><div style="text-align:right;margin-left:20px;flex-shrink:0"><div style="font-size:24px;font-weight:800;color:{sc3}">{scr}</div><div style="font-size:10px;color:#9ca3af;margin-bottom:6px">stress score</div><div style="font-size:11px;color:#6b7280">Motion: {mp}</div><div style="font-size:11px;color:#6b7280">Audio: {ap}</div></div></div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown("### Detailed Charts")
    st.caption("Dig into the data behind your shift.")
    st.markdown("<br>", unsafe_allow_html=True)

    section = st.radio("Show", ["Earnings Curve","Acceleration","Audio"],
                       horizontal=True, label_visibility="collapsed")

    if section == "Earnings Curve":
        if my_vel.empty:
            st.info("No velocity data.")
        else:
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=my_vel["elapsed_hours"],y=my_vel["cumulative_earnings"],
                fill="tozeroy",fillcolor="rgba(22,163,74,.08)",line=dict(color="#16a34a",width=2.5),
                name="Cumulative ₹",hovertemplate="Hour %{x:.1f}<br>₹%{y:,.0f}<extra></extra>"),secondary_y=False)
            if gc is not None:
                fig.add_hline(y=gc["target_earnings"],line_dash="dot",line_color="#d97706",line_width=2,
                              annotation_text=f"Goal ₹{int(gc['target_earnings']):,}",annotation_font_color="#d97706")
                lh = my_vel["elapsed_hours"].max(); le = my_vel["cumulative_earnings"].max()
                fig.add_trace(go.Scatter(x=[lh,lh+gc["remaining_hours"]],y=[le,gc["projected_earnings"]],
                    line=dict(color="#16a34a",width=2,dash="dash"),name="Projected",
                    hovertemplate="Projected: ₹%{y:,.0f}<extra></extra>"),secondary_y=False)
            fig.add_trace(go.Scatter(x=my_vel["elapsed_hours"],y=my_vel["recalc_velocity"],
                line=dict(color="#2563eb",width=2),name="₹/hr",
                hovertemplate="Hour %{x:.1f}<br>₹%{y:.0f}/hr<extra></extra>"),secondary_y=True)
            if gc is not None and gc["needed_velocity"]>0:
                fig.add_hline(y=gc["needed_velocity"],line_dash="dot",line_color="#dc2626",line_width=1.5,
                              annotation_text=f"Need ₹{int(gc['needed_velocity'])}/hr",
                              annotation_font_color="#dc2626",secondary_y=True)
            fig.update_layout(**PLOTLY_BASE,height=360,
                yaxis=dict(title="Cumulative ₹",gridcolor="#f0f1f3"),
                yaxis2=dict(title="₹/hr",gridcolor="rgba(0,0,0,0)"),
                xaxis=dict(title="Hours into shift",gridcolor="#f0f1f3"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified")
            st.plotly_chart(fig,use_container_width=True)

    elif section == "Acceleration":
        tlist = my_accel["trip_id"].unique().tolist()
        if not tlist:
            st.info("No accelerometer data.")
        else:
            sel = st.selectbox("Trip",tlist)
            ta  = my_accel[my_accel["trip_id"]==sel].sort_values("elapsed_seconds")
            tf  = my_flags[my_flags["trip_id"]==sel] if not my_flags.empty else pd.DataFrame()
            fig2 = make_subplots(rows=2,cols=1,shared_xaxes=True,
                                 subplot_titles=("Motion Intensity (m/s²)","Speed (km/h)"),vertical_spacing=0.1)
            fig2.add_trace(go.Scatter(x=ta["elapsed_seconds"],y=ta["magnitude"],
                fill="tozeroy",fillcolor="rgba(124,58,237,.08)",line=dict(color="#7c3aed",width=2),
                name="Magnitude",hovertemplate="t=%{x}s  |  %{y:.2f} m/s²<extra></extra>"),row=1,col=1)
            fig2.add_hline(y=3.5,line_dash="dot",line_color="#d97706",
                           annotation_text="Alert",row=1,col=1,annotation_font_color="#d97706")
            fig2.add_hline(y=4.0,line_dash="dot",line_color="#dc2626",
                           annotation_text="Brake threshold",row=1,col=1,annotation_font_color="#dc2626")
            fig2.add_trace(go.Scatter(x=ta["elapsed_seconds"],y=ta["speed_kmh"],
                fill="tozeroy",fillcolor="rgba(37,99,235,.08)",line=dict(color="#2563eb",width=2),
                name="Speed",hovertemplate="t=%{x}s  |  %{y} km/h<extra></extra>"),row=2,col=1)
            for _,fl in tf.iterrows():
                c = SEVERITY_COLOR.get(fl["severity"],"#9ca3af")
                for r in [1,2]:
                    fig2.add_vrect(x0=fl["elapsed_seconds"]-4,x1=fl["elapsed_seconds"]+4,
                                   fillcolor=c,opacity=0.12,line_width=0,row=r,col=1)
            fig2.update_layout(**PLOTLY_BASE,height=400,
                yaxis=dict(gridcolor="#f0f1f3"),yaxis2=dict(gridcolor="#f0f1f3"),
                xaxis2=dict(title="Elapsed seconds",gridcolor="#f0f1f3"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2,use_container_width=True)
            if not tf.empty:
                st.caption(f"Shaded areas = {len(tf)} flagged moments on this trip")

    elif section == "Audio":
        atlist = my_audio["trip_id"].unique().tolist()
        if not atlist:
            st.info("No audio data.")
        else:
            sel_a = st.selectbox("Trip",atlist)
            ta2   = my_audio[my_audio["trip_id"]==sel_a].sort_values("elapsed_seconds")
            CLC   = {"quiet":"#16a34a","normal":"#2563eb","conversation":"#7c3aed",
                     "elevated":"#d97706","very_loud":"#d97706","argument":"#dc2626","mechanical_noise":"#9ca3af"}
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=ta2["elapsed_seconds"],y=ta2["audio_level_db"],
                marker_color=[CLC.get(c,"#9ca3af") for c in ta2["audio_classification"]],
                name="Audio dB",hovertemplate="t=%{x}s<br><b>%{y} dB</b><br>%{customdata}<extra></extra>",
                customdata=ta2["audio_classification"]))
            fig3.add_hline(y=80,line_dash="dot",line_color="#d97706",
                           annotation_text="High (80 dB)",annotation_font_color="#d97706")
            fig3.add_hline(y=90,line_dash="dot",line_color="#dc2626",
                           annotation_text="Argument (90 dB)",annotation_font_color="#dc2626")
            fig3.update_layout(**PLOTLY_BASE,height=320,
                yaxis=dict(title="dB",gridcolor="#f0f1f3"),
                xaxis=dict(title="Elapsed seconds",gridcolor="#f0f1f3"),bargap=0.15)
            st.plotly_chart(fig3,use_container_width=True)
            st.caption("Grey = filtered out (door slams / mechanical noise under 3 seconds)")
