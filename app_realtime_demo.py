"""
Driver Pulse — Unified Real-Time Dashboard
Combines live trip simulation with full shift analytics.
Run: streamlit run app_realtime_demo.py

Engines imported as-is — no modifications.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time, os, sys, math, io, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sensor_engine import (run_accel_checks, run_audio_checks, detect_flagged_moments,
                            build_trip_summaries_sensor, build_explainability_log)
from earnings_engine import build_trip_earnings, build_enriched_velocity, build_enriched_goals
from merge_engine    import build_trip_summaries, build_driver_shift_summary

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
SIM_CSV     = os.path.join(DATA_DIR, "realtime_trip_simulation.csv")
DRIVERS_CSV = os.path.join(DATA_DIR, "drivers.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

st.set_page_config(page_title="Driver Pulse", page_icon="DP", layout="centered",
                   initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{
  font-family:'Inter',sans-serif!important;
  color:#ffffff!important;
  background:#0f172a!important;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1rem 1rem 3rem!important;max-width:520px!important;margin:0 auto!important}

[data-testid="metric-container"]{
  background:rgba(255,255,255,0.04)!important;
  border:1px solid rgba(255,255,255,0.08)!important;
  border-radius:14px!important;padding:14px 12px!important;
}
[data-testid="metric-container"] label{
  font-size:10px!important;font-weight:700!important;
  text-transform:uppercase!important;letter-spacing:.06em!important;
  color:rgba(255,255,255,0.4)!important
}
[data-testid="stMetricValue"]{font-size:1.6rem!important;font-weight:800!important;color:#fff!important}
[data-testid="stMetricDelta"] p{color:rgba(255,255,255,0.35)!important;font-size:11px!important}

.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.05)!important;border-radius:12px!important;padding:3px!important;gap:1px!important;border:1px solid rgba(255,255,255,0.08)!important;width:100%!important}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;font-weight:600!important;font-size:11px!important;color:rgba(255,255,255,0.5)!important;padding:7px 8px!important;flex:1!important;text-align:center!important}
.stTabs [aria-selected="true"]{background:rgba(255,255,255,0.12)!important;color:#fff!important}
.stTabs [aria-selected="true"] p,.stTabs [aria-selected="true"] span{color:#fff!important}

.stButton>button{border-radius:12px!important;font-weight:700!important;font-size:14px!important;border:none!important;padding:12px 24px!important;width:100%!important;cursor:pointer!important;transition:all .15s ease!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(0,0,0,.3)!important}

.stSelectbox>div>div{border-radius:12px!important;border-color:rgba(255,255,255,0.15)!important;background:rgba(255,255,255,0.06)!important;color:#fff!important;font-size:14px!important}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:#fff!important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#1e293b!important;border:1px solid rgba(255,255,255,0.12)!important;border-radius:12px!important}
[data-baseweb="menu"] ul{background:#1e293b!important;padding:4px!important}
[data-baseweb="menu"] li{background:#1e293b!important;color:#fff!important;border-radius:8px!important;padding:8px 12px!important;font-size:13px!important}
[data-baseweb="menu"] li:hover{background:rgba(99,102,241,0.18)!important;color:#fff!important}
[data-baseweb="menu"] [aria-selected="true"]{background:rgba(99,102,241,0.25)!important;font-weight:700!important;color:#fff!important}
[role="listbox"]{background:#1e293b!important}
[role="option"]{background:#1e293b!important;color:#fff!important}
[role="option"]:hover,[role="option"]:focus{background:rgba(99,102,241,0.18)!important;color:#fff!important}
[data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden!important;border:1px solid rgba(255,255,255,0.08)!important}
hr{border-color:rgba(255,255,255,0.06)!important;margin:.8rem 0!important}
p,span,label,caption,.stMarkdown{color:#fff!important}
h1,h2,h3,h4,h5,h6{color:#fff!important}
.stCaption,.stCaption p{color:rgba(255,255,255,0.35)!important;font-size:11px!important}
.stRadio label{color:rgba(255,255,255,0.7)!important;font-weight:600!important;font-size:11px!important}
.stRadio [data-testid="stMarkdownContainer"] p{color:#fff!important}
.stRadio div[role="radiogroup"]{background:rgba(255,255,255,0.05)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:10px!important;padding:4px 6px!important;display:inline-flex!important;gap:3px!important}
.stRadio label{padding:6px 10px!important;border-radius:7px!important;cursor:pointer!important}
.stRadio input[type="radio"]{display:none!important}
.stAlert p{color:#1a1d23!important}
.stProgress>div>div{border-radius:999px!important}
</style>
""", unsafe_allow_html=True)

# ── Color Maps ────────────────────────────────────────────────────────────────
FORECAST_COLOR = {"achieved":"#34d399","on_track":"#60a5fa","at_risk":"#fbbf24","behind":"#f87171","dead_zone":"#a78bfa"}
QUALITY_COLOR  = {"excellent":"#34d399","good":"#60a5fa","average":"#fbbf24","below_average":"#f87171","poor":"#f87171"}
SEVERITY_COLOR = {"high":"#f87171","medium":"#fbbf24","low":"#34d399","none":"#9ca3af"}
EVENT_LABEL    = {
    "harsh_brake":"Hard Brake","harsh_accel":"Sudden Acceleration","sharp_turn":"Sharp Turn",
    "speed_bump":"Speed Bump Hit","speeding":"Speeding","jerk":"Jerky Movement",
    "safety_maneuver":"Emergency Stop","conflict_moment":"Conflict Moment","argument":"Loud Argument",
}
PLOTLY_BASE = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,0.02)",
                   font=dict(family="Inter, sans-serif",color="#ffffff",size=11),
                   margin=dict(l=8,r=8,t=28,b=8))

# ── Helpers ───────────────────────────────────────────────────────────────────
def card(content_html, extra_style=""):
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:14px;padding:16px 14px;margin-bottom:10px;{extra_style}">'
        f'{content_html}</div>',
        unsafe_allow_html=True
    )

def section_header(title):
    st.markdown(f'<div style="font-size:16px;font-weight:700;color:#fff;margin:8px 0 12px">{title}</div>', unsafe_allow_html=True)

def _save_outputs(flagged, trip_sum, shift_sum, enrich_vel, enrich_g):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saves = {"flagged_moments.csv":flagged,"trip_summaries.csv":trip_sum,
             "driver_shift_summary.csv":shift_sum,"earnings_velocity_log.csv":enrich_vel,
             "goals_forecast.csv":enrich_g}
    for fname, df in saves.items():
        try: df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        except Exception: pass

# ── Data Pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=None)
def load_and_process():
    raw_trips    = pd.read_csv(os.path.join(DATA_DIR,"trips.csv"))
    raw_accel    = pd.read_csv(os.path.join(DATA_DIR,"accelerometer_data.csv"))
    raw_audio    = pd.read_csv(os.path.join(DATA_DIR,"audio_intensity_data.csv"))
    raw_goals    = pd.read_csv(os.path.join(DATA_DIR,"driver_goals.csv"))
    raw_velocity = pd.read_csv(os.path.join(DATA_DIR,"earnings_velocity_log.csv"))
    raw_drivers  = pd.read_csv(DRIVERS_CSV)
    accel_p  = run_accel_checks(raw_accel)
    audio_p  = run_audio_checks(raw_audio)
    flagged  = detect_flagged_moments(accel_p, audio_p, raw_trips)
    sensor_s = build_trip_summaries_sensor(raw_trips, flagged, accel_p, audio_p)
    trip_earn  = build_trip_earnings(raw_trips)
    enrich_vel = build_enriched_velocity(raw_velocity, raw_goals)
    enrich_g   = build_enriched_goals(raw_goals, raw_velocity, raw_trips)
    trip_sum   = build_trip_summaries(trip_earn, sensor_s)
    shift_sum  = build_driver_shift_summary(trip_sum, enrich_g, flagged)
    _save_outputs(flagged, trip_sum, shift_sum, enrich_vel, enrich_g)
    return dict(drivers=raw_drivers, accel=accel_p, audio=audio_p, flagged=flagged,
                trip_sum=trip_sum, shift_sum=shift_sum, enrich_vel=enrich_vel, enrich_goals=enrich_g)

@st.cache_data(show_spinner=False, ttl=None)
def slice_for_driver(data: dict, did: str) -> dict:
    flagged_df  = data["flagged"]
    trip_sum_df = data["trip_sum"]
    shift_sum_df= data["shift_sum"]
    vel_df      = data["enrich_vel"]
    goals_df    = data["enrich_goals"]
    accel_df    = data["accel"]
    audio_df    = data["audio"]
    my_trips = trip_sum_df[trip_sum_df["driver_id"] == did].copy()
    trip_ids = set(my_trips["trip_id"])
    return dict(
        dinfo     = data["drivers"][data["drivers"]["driver_id"] == did].iloc[0],
        my_trips  = my_trips,
        my_flags  = flagged_df[flagged_df["driver_id"] == did].copy() if not flagged_df.empty else pd.DataFrame(),
        my_goal   = goals_df[goals_df["driver_id"] == did],
        my_shift  = shift_sum_df[shift_sum_df["driver_id"] == did],
        my_vel    = vel_df[vel_df["driver_id"] == did].sort_values("elapsed_hours"),
        my_accel  = accel_df[accel_df["trip_id"].isin(trip_ids)],
        my_audio  = audio_df[audio_df["trip_id"].isin(trip_ids)],
    )

@st.cache_data(show_spinner=False)
def load_sim_data():
    return pd.read_csv(SIM_CSV)

@st.cache_data(show_spinner=False)
def load_drivers():
    return pd.read_csv(DRIVERS_CSV)

def compute_accel_magnitude(row):
    return np.sqrt(row["accel_x"]**2 + row["accel_y"]**2 + (row["accel_z"] - 9.8)**2)

def quick_stress(magnitude: float, audio_db: float) -> float:
    m_score = min(magnitude / 8.0, 1.0) if magnitude > 3.5 else magnitude / 30.0
    a_score = min((audio_db - 70) / 30.0, 1.0) if audio_db >= 80 else max((audio_db - 50) / 60.0, 0.0)
    return round(0.65 * m_score + 0.35 * a_score, 3)

def detect_live_event(row):
    mag = compute_accel_magnitude(row)
    lon = abs(row["accel_x"]); lat = abs(row["accel_y"])
    az = row["accel_z"]; db = row["audio_level_db"]; spd = row["speed_kmh"]
    if lon > 2.5 and row["accel_x"] < -2.5 and spd > 15: return "harsh_brake", "Harsh braking detected"
    if lon > 2.5 and row["accel_x"] > 2.5: return "harsh_accel", "Sudden acceleration detected"
    if lat > 2.0: return "sharp_turn", "Sharp turn / swerve detected"
    if spd > 60: return "speeding", f"Speeding — {spd:.0f} km/h"
    if az < 7.5 and spd > 15: return "speed_bump", "Speed bump hit"
    if db >= 90: return "argument", "Passenger conflict / loud argument"
    if db >= 80: return "very_loud", "Very loud cabin noise detected"
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def show_login():
    drivers_df = load_drivers()

    st.markdown("""
    <style>
    @keyframes riseIn{
      0%{opacity:0;transform:translateY(60px) scale(0.95)}
      60%{opacity:1;transform:translateY(-8px) scale(1.02)}
      80%{transform:translateY(3px) scale(0.995)}
      100%{opacity:1;transform:translateY(0) scale(1)}
    }
    @keyframes fadeSlideIn{
      0%{opacity:0;transform:translateY(24px)}
      100%{opacity:1;transform:translateY(0)}
    }
    @keyframes accentDraw{
      0%{width:0;opacity:0}
      100%{width:60px;opacity:1}
    }
    @keyframes softPulse{
      0%,100%{box-shadow:0 4px 20px rgba(99,102,241,0.30)}
      50%{box-shadow:0 4px 32px rgba(99,102,241,0.55)}
    }
    @keyframes gradientShift{
      0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}
    }
    @keyframes bgZoom{
      0%{transform:scale(1.08)}
      100%{transform:scale(1)}
    }

    .login-bg{
      position:fixed;top:0;left:0;right:0;bottom:0;z-index:0;overflow:hidden;
    }
    .login-bg::before{
      content:'';position:absolute;inset:0;
      background:url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1400&q=80') center/cover no-repeat;
      animation:bgZoom 20s ease-out forwards;
    }
    .login-bg::after{
      content:'';position:absolute;inset:0;
      background:linear-gradient(180deg,
        rgba(15,23,42,0.50) 0%, rgba(15,23,42,0.75) 30%,
        rgba(15,23,42,0.92) 70%, rgba(15,23,42,0.98) 100%);
    }

    .login-hero{
      text-align:center;
      animation:riseIn 1s cubic-bezier(0.34,1.56,0.64,1) both;
      animation-delay:0.1s;
    }
    .login-logo{
      font-size:48px;font-weight:900;color:#fff;
      letter-spacing:-2px;line-height:1;margin-bottom:0;
    }
    .login-accent{
      height:3px;border-radius:2px;margin:12px auto 0;
      background:linear-gradient(90deg,#6366f1,#a78bfa,#6366f1);
      background-size:200% 100%;
      animation:accentDraw 0.8s ease-out 0.6s both, gradientShift 4s linear infinite;
    }
    .login-subtitle{
      font-size:12px;font-weight:500;color:rgba(255,255,255,0.35);
      letter-spacing:1.5px;text-transform:uppercase;margin-top:14px;
      animation:fadeSlideIn 0.7s ease-out 0.5s both;
    }

    .login-card{
      background:rgba(255,255,255,0.035);
      backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
      border:1px solid rgba(255,255,255,0.08);border-radius:20px;
      padding:22px 24px 20px;width:100%;max-width:380px;margin:0 auto;
      box-shadow:0 24px 48px rgba(0,0,0,0.3),
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 1px 0 rgba(255,255,255,0.06) inset;
      animation:riseIn 0.9s cubic-bezier(0.34,1.56,0.64,1) both;
      animation-delay:0.35s;
    }
    .login-card-label{
      font-size:10px;font-weight:700;color:rgba(255,255,255,0.28);
      text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px;
    }
    .login-sep{
      display:flex;align-items:center;gap:14px;margin:14px 0 12px;
      animation:fadeSlideIn 0.6s ease-out 0.8s both;
    }
    .login-sep::before,.login-sep::after{
      content:'';flex:1;height:1px;
      background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);
    }

    .stButton>button{
      background:linear-gradient(135deg,#6366f1 0%,#818cf8 50%,#6366f1 100%)!important;
      background-size:200% 200%!important;
      animation:gradientShift 3s ease infinite, softPulse 3s ease infinite!important;
      color:#fff!important;font-weight:700!important;font-size:14px!important;
      padding:14px 28px!important;border-radius:14px!important;
      border:none!important;letter-spacing:0.4px!important;
      transition:all 0.25s cubic-bezier(0.34,1.56,0.64,1)!important;
    }
    .stButton>button:hover{
      transform:translateY(-3px) scale(1.01)!important;
      box-shadow:0 12px 36px rgba(99,102,241,0.50)!important;
    }
    .stButton>button:active{
      transform:translateY(0) scale(0.98)!important;
    }

    .login-footer{
      text-align:center;margin-top:28px;
      animation:fadeSlideIn 0.6s ease-out 1.1s both;
    }
    .login-footer-text{
      font-size:11px;color:rgba(255,255,255,0.12);font-weight:400;letter-spacing:0.3px;
    }
    .login-footer-dots{
      display:flex;justify-content:center;gap:8px;margin-top:14px;
    }
    .login-footer-dots span{
      width:6px;height:6px;border-radius:50%;background:rgba(255,255,255,0.10);
    }
    .login-footer-dots span:nth-child(2){background:rgba(99,102,241,0.35)}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-bg"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:12vh"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="login-hero">
      <div class="login-logo">Driver Pulse</div>
      <div class="login-accent"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="login-card">
      <div class="login-card-label">Select Driver</div>
    """, unsafe_allow_html=True)

    driver_options = drivers_df[["driver_id","name"]].copy()
    driver_options["label"] = driver_options["name"] + "  —  " + driver_options["driver_id"]
    selected_label = st.selectbox("Driver", driver_options["label"].tolist(),
                                  label_visibility="collapsed", key="login_select")
    did   = driver_options.loc[driver_options["label"] == selected_label, "driver_id"].iloc[0]
    dname = driver_options.loc[driver_options["label"] == selected_label, "name"].iloc[0]

    st.markdown('<div class="login-sep"></div>', unsafe_allow_html=True)

    if st.button("Enter Dashboard", use_container_width=True, key="login_btn"):
        st.session_state["driver_id"]   = did
        st.session_state["driver_name"] = dname
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="login-footer">
      <div class="login-footer-text">Edge-device mode — no password required</div>
      <div class="login-footer-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    data = load_and_process()
    DID   = st.session_state["driver_id"]
    DNAME = st.session_state.get("driver_name", DID)

    sliced    = slice_for_driver(data, DID)
    dinfo     = sliced["dinfo"]
    my_trips  = sliced["my_trips"]
    my_flags  = sliced["my_flags"]
    my_goal   = sliced["my_goal"]
    my_shift  = sliced["my_shift"]
    my_vel    = sliced["my_vel"]
    my_accel  = sliced["my_accel"]
    my_audio  = sliced["my_audio"]
    trip_sum_df = data["trip_sum"]

    gc = my_goal.iloc[0]  if not my_goal.empty  else None
    sc = my_shift.iloc[0] if not my_shift.empty else None

    # ── Header ────────────────────────────────────────────────────────────────
    forecast_label = {
        "achieved":"Goal Achieved","on_track":"On Track",
        "at_risk":"At Risk","behind":"Behind","dead_zone":"Low Demand",
    }.get(gc["forecast"] if gc is not None else "","")
    fc_col = FORECAST_COLOR.get(gc["forecast"], "#6b7280") if gc is not None else "#6b7280"

    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f"""
        <div>
          <div style="font-size:18px;font-weight:800;color:#fff">{dinfo['name']}</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.4);margin-top:2px">
            {dinfo['city']} · {dinfo['rating']} · {dinfo['shift_preference'].title()} shift
          </div>
        </div>
        """, unsafe_allow_html=True)
    with hcol2:
        if st.button("Sign Out", key="logout_btn", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    if gc is not None:
        st.markdown(
            f'<div style="margin:8px 0 4px">'
            f'<span style="background:{fc_col}20;color:{fc_col};border:1px solid {fc_col}44;'
            f'padding:4px 12px;border-radius:999px;font-size:11px;font-weight:700">'
            f'{forecast_label}</span></div>',
            unsafe_allow_html=True
        )
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_live, tab_home, tab_trips, tab_safety, tab_charts, tab_export = st.tabs([
        "Live Sim", "Home", f"Trips ({len(my_trips)})", "Safety", "Charts", "Export"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB: LIVE SIMULATION  (isolated fragment — only this block reruns during sim)
    # ══════════════════════════════════════════════════════════════════════════

    @st.fragment
    def _live_sim_fragment():
        sim_df = load_sim_data()
        TOTAL  = len(sim_df)

        # Init session state
        for k, v in [("demo_running",False),("demo_row_idx",0),("demo_events",[]),("demo_done",False)]:
            if k not in st.session_state: st.session_state[k] = v

        # Controls
        if not st.session_state["demo_done"]:
            cb1, cb2 = st.columns(2)
            with cb1:
                if not st.session_state["demo_running"]:
                    if st.button("Play", use_container_width=True, key="play_btn"):
                        st.session_state["demo_running"] = True
                        st.rerun(scope="fragment")
                else:
                    if st.button("Pause", use_container_width=True, key="pause_btn"):
                        st.session_state["demo_running"] = False
                        st.rerun(scope="fragment")
            with cb2:
                if st.button("Restart", use_container_width=True, key="restart_btn"):
                    st.session_state["demo_row_idx"] = 0
                    st.session_state["demo_events"] = []
                    st.session_state["demo_done"] = False
                    st.session_state["demo_running"] = True
                    st.session_state["_demo_last_tick"] = 0.0
                    st.rerun(scope="fragment")
        else:
            if st.button("Run Again", use_container_width=True, key="again_btn"):
                st.session_state["demo_row_idx"] = 0
                st.session_state["demo_events"] = []
                st.session_state["demo_done"] = False
                st.session_state["demo_running"] = True
                st.session_state["_demo_last_tick"] = 0.0
                st.rerun(scope="fragment")

        # Progress
        idx = st.session_state["demo_row_idx"]
        pct = int(idx / TOTAL * 100)
        elapsed_s = idx * 3
        elapsed_str = f"{elapsed_s // 60}m {elapsed_s % 60:02d}s"

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin:12px 0 6px">
          <span style="font-size:11px;color:rgba(255,255,255,0.4);font-weight:600">TRIP PROGRESS</span>
          <span style="font-size:11px;color:rgba(255,255,255,0.4)">{elapsed_str} / ~5m</span>
        </div>
        <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:6px;overflow:hidden;margin-bottom:14px">
          <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#34d399,#60a5fa);
               border-radius:999px;transition:width .4s ease"></div>
        </div>""", unsafe_allow_html=True)

        # Live metrics
        visible = sim_df.iloc[:max(idx, 1)].copy()
        current = sim_df.iloc[max(idx-1, 0)]
        mag     = compute_accel_magnitude(current)
        stress  = quick_stress(mag, current["audio_level_db"])
        cum_earn= float(current["cumulative_earnings"])
        spd     = float(current["speed_kmh"])
        db_now  = float(current["audio_level_db"])
        stress_col = "#f87171" if stress >= 0.65 else "#fbbf24" if stress >= 0.4 else "#34d399"

        m1, m2 = st.columns(2)
        m1.metric("Trip Time", elapsed_str, f"{pct}% complete")
        m2.metric("Earnings", f"₹{cum_earn:.2f}", f"+₹{current['earnings_delta']:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("Speed", f"{spd:.0f} km/h", "Speeding" if spd > 60 else "Normal")
        m4.metric("Cabin Audio", f"{db_now:.0f} dB", "Loud" if db_now >= 80 else "Normal")

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
             border-radius:14px;padding:12px 16px;margin:8px 0 10px">
          <div>
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                 letter-spacing:.06em;color:rgba(255,255,255,0.35)">Live Stress</div>
            <div style="font-size:1.8rem;font-weight:800;color:{stress_col};margin-top:2px">{stress:.2f}</div>
          </div>
          <div style="font-size:28px">{"HIGH" if stress >= 0.65 else "MED" if stress >= 0.4 else "LOW"}</div>
        </div>""", unsafe_allow_html=True)

        # Event detection
        ev_key, ev_msg = detect_live_event(current)
        if ev_key and idx > 0:
            ev_events = st.session_state.setdefault("demo_events", [])
            last = ev_events[-1]["key"] if ev_events else None
            if ev_key != last:
                ev_events.append({"key":ev_key,"msg":ev_msg,"t":elapsed_str,"idx":idx})

        events_so_far = st.session_state.get("demo_events", [])
        if events_so_far:
            latest = events_so_far[-1]
            is_conflict = latest["key"] in ("argument","conflict_moment")
            alert_bg = "rgba(248,113,113,0.1)" if is_conflict else "rgba(251,191,36,0.08)"
            alert_border = "#f87171" if is_conflict else "#fbbf24"
            st.markdown(f"""
            <div style="background:{alert_bg};border:1px solid {alert_border};
                 border-radius:12px;padding:12px 16px;margin-bottom:10px">
              <div style="font-size:14px;font-weight:700;color:#fff">{latest['msg']}</div>
              <div style="font-size:10px;color:rgba(255,255,255,0.35);margin-top:3px">at {latest['t']}</div>
            </div>""", unsafe_allow_html=True)

        # Live charts
        section_header("Live Sensor Charts")
        t_axis = visible["elapsed_seconds"].tolist()

        # Speed chart
        fig_spd = go.Figure()
        fig_spd.add_trace(go.Scatter(x=t_axis, y=visible["speed_kmh"].tolist(),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
            line=dict(color="#60a5fa", width=2), name="Speed",
            hovertemplate="t=%{x}s | %{y:.1f} km/h<extra></extra>"))
        fig_spd.add_hline(y=60, line_dash="dot", line_color="#f87171",
                          annotation_text="Limit", annotation_font_color="#f87171")
        fig_spd.update_layout(**PLOTLY_BASE, height=160,
            title=dict(text="Speed (km/h)", font=dict(size=11), x=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 85]))
        st.plotly_chart(fig_spd, use_container_width=True)

        # Accel chart
        visible["magnitude"] = visible.apply(compute_accel_magnitude, axis=1)
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=t_axis, y=visible["magnitude"].tolist(),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
            line=dict(color="#a78bfa", width=2), name="Magnitude",
            hovertemplate="t=%{x}s | %{y:.2f} m/s2<extra></extra>"))
        fig_acc.add_hline(y=3.5, line_dash="dot", line_color="#fbbf24",
                          annotation_text="Threshold", annotation_font_color="#fbbf24")
        fig_acc.update_layout(**PLOTLY_BASE, height=160,
            title=dict(text="Motion Intensity (m/s2)", font=dict(size=11), x=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 8]))
        st.plotly_chart(fig_acc, use_container_width=True)

        # Audio chart
        def audio_color(db):
            if db >= 90: return "#f87171"
            if db >= 80: return "#fbbf24"
            if db >= 65: return "#a78bfa"
            return "#60a5fa"
        fig_aud = go.Figure()
        fig_aud.add_trace(go.Bar(x=t_axis, y=visible["audio_level_db"].tolist(),
            marker_color=[audio_color(d) for d in visible["audio_level_db"]],
            name="Audio dB", hovertemplate="t=%{x}s | %{y:.0f} dB<extra></extra>"))
        fig_aud.add_hline(y=80, line_dash="dot", line_color="#fbbf24",
                          annotation_text="80 dB", annotation_font_color="#fbbf24")
        fig_aud.add_hline(y=90, line_dash="dot", line_color="#f87171",
                          annotation_text="90 dB", annotation_font_color="#f87171")
        fig_aud.update_layout(**PLOTLY_BASE, height=160,
            title=dict(text="Cabin Audio (dB)", font=dict(size=11), x=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[30, 105]), bargap=0.1)
        st.plotly_chart(fig_aud, use_container_width=True)

        # Stress timeline chart
        visible["stress_live"] = visible.apply(
            lambda r: quick_stress(compute_accel_magnitude(r), r["audio_level_db"]), axis=1)
        stress_colors = ["#f87171" if s >= 0.65 else "#fbbf24" if s >= 0.40 else "#34d399"
                         for s in visible["stress_live"]]
        fig_st = go.Figure()
        fig_st.add_trace(go.Scatter(x=t_axis, y=visible["stress_live"].tolist(),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.06)",
            line=dict(color="#f87171", width=2), mode="lines+markers",
            marker=dict(color=stress_colors, size=5), name="Stress",
            hovertemplate="t=%{x}s | stress=%{y:.2f}<extra></extra>"))
        fig_st.add_hline(y=0.65, line_dash="dot", line_color="#f87171",
                         annotation_text="High", annotation_font_color="#f87171")
        fig_st.add_hline(y=0.40, line_dash="dot", line_color="#fbbf24",
                         annotation_text="Med", annotation_font_color="#fbbf24")
        fig_st.update_layout(**PLOTLY_BASE, height=160,
            title=dict(text="Stress Score", font=dict(size=11), x=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 1.1]))
        st.plotly_chart(fig_st, use_container_width=True)

        # Events log
        if events_so_far:
            section_header("Events Detected")
            for ev in reversed(events_so_far[-6:]):
                is_high = ev["key"] in ("argument","conflict_moment","harsh_brake","speeding")
                pill_c = "#f87171" if is_high else "#fbbf24"
                st.markdown(
                    f'<div style="display:flex;align-items:center;'
                    f'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
                    f'border-left:3px solid {pill_c};border-radius:0 10px 10px 0;'
                    f'padding:8px 12px;margin-bottom:5px">'
                    f'<div style="flex:1">'
                    f'<div style="font-size:12px;font-weight:700;color:#fff">{ev["msg"]}</div>'
                    f'<div style="font-size:10px;color:rgba(255,255,255,0.3);margin-top:2px">at {ev["t"]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

        # ── Trip Complete: run engine ─────────────────────────────────────────
        if st.session_state.get("demo_done", False) and idx >= TOTAL:
            st.divider()
            section_header("Trip Completed")
            with st.spinner("Running sensor analysis..."):
                fare_est = float(sim_df["cumulative_earnings"].iloc[-1])
                dur_min = round((TOTAL * 3) / 60, 1)
                dist_km = round(sim_df["speed_kmh"].mean() * (dur_min / 60), 2)
                trips_meta = pd.DataFrame([{
                    "trip_id":sim_df["trip_id"].iloc[0],"driver_id":DID,"date":"2024-02-06",
                    "start_time":sim_df["timestamp"].iloc[0],"end_time":sim_df["timestamp"].iloc[-1],
                    "duration_min":dur_min,"distance_km":dist_km,"fare":fare_est,
                    "surge_multiplier":1.0,"pickup_location":"Sim Start","dropoff_location":"Sim End",
                }])
                accel_in = sim_df[["trip_id","elapsed_seconds","timestamp","accel_x","accel_y","accel_z","speed_kmh"]].copy()
                audio_in = sim_df[["trip_id","elapsed_seconds","audio_level_db","audio_classification"]].copy()
                audio_in["sustained_duration_sec"] = sim_df.get("sustained_duration_sec",
                    sim_df["audio_level_db"].apply(lambda _: 10))
                accel_p = run_accel_checks(accel_in)
                audio_p = run_audio_checks(audio_in)
                flagged = detect_flagged_moments(accel_p, audio_p, trips_meta)
                sensor_summary = build_trip_summaries_sensor(trips_meta, flagged, accel_p, audio_p)

            n_events    = len(flagged) if not flagged.empty else 0
            peak_stress = float(sensor_summary["stress_score"].iloc[0]) if not sensor_summary.empty else 0.0
            ev_vel      = round(fare_est / (dur_min / 60), 0)
            stress_lbl  = "HIGH" if peak_stress >= 0.65 else "MODERATE" if peak_stress >= 0.4 else "LOW"
            stress_col2 = "#f87171" if peak_stress >= 0.65 else "#fbbf24" if peak_stress >= 0.4 else "#34d399"

            card(f"""
            <div style="text-align:center;margin-bottom:16px">
              <div style="font-size:18px;font-weight:800;color:#fff">Trip Summary</div>
              <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:4px">{DID} — Simulation</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
              <div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.15);
                   border-radius:12px;padding:12px;text-align:center">
                <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);
                     text-transform:uppercase;letter-spacing:.05em">Fare</div>
                <div style="font-size:22px;font-weight:800;color:#34d399;margin-top:4px">₹{fare_est:.2f}</div>
              </div>
              <div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.15);
                   border-radius:12px;padding:12px;text-align:center">
                <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);
                     text-transform:uppercase;letter-spacing:.05em">Duration</div>
                <div style="font-size:22px;font-weight:800;color:#60a5fa;margin-top:4px">{dur_min:.1f} min</div>
              </div>
              <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                   border-radius:12px;padding:12px;text-align:center">
                <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);
                     text-transform:uppercase;letter-spacing:.05em">Peak Stress</div>
                <div style="font-size:18px;font-weight:800;color:{stress_col2};margin-top:4px">{peak_stress:.2f}</div>
                <div style="font-size:10px;color:{stress_col2};margin-top:2px">{stress_lbl}</div>
              </div>
              <div style="background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.12);
                   border-radius:12px;padding:12px;text-align:center">
                <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);
                     text-transform:uppercase;letter-spacing:.05em">Events</div>
                <div style="font-size:22px;font-weight:800;color:#f87171;margin-top:4px">{n_events}</div>
              </div>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:12px;
                 display:flex;justify-content:space-between;align-items:center">
              <div style="font-size:11px;color:rgba(255,255,255,0.3)">Earnings velocity</div>
              <div style="font-size:15px;font-weight:700;color:#34d399">₹{ev_vel:.0f}/hr</div>
            </div>
            """)

            if not flagged.empty:
                section_header("Flagged Moments (Engine Output)")
                for _, fl in flagged.iterrows():
                    sc_  = SEVERITY_COLOR.get(fl["severity"], "#9ca3af")
                    evl  = EVENT_LABEL.get(fl["accel_type"], fl["accel_type"].replace("_"," ").title())
                    scr  = int(fl["combined_score"] * 100)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;'
                        f'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
                        f'border-left:3px solid {sc_};border-radius:0 10px 10px 0;'
                        f'padding:8px 12px;margin-bottom:5px">'
                        f'<div style="flex:1">'
                        f'<div style="font-size:12px;font-weight:700;color:#fff">{evl}</div>'
                        f'<div style="font-size:10px;color:rgba(255,255,255,0.3);margin-top:2px">{fl["driver_explanation"]}</div>'
                        f'</div>'
                        f'<div style="font-size:16px;font-weight:800;color:{sc_};margin-left:10px">{scr}</div>'
                        f'</div>',
                        unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align:center;margin-top:16px;font-size:11px;color:rgba(255,255,255,0.2)">
              Full pipeline ran via sensor_engine.py — no engines modified
            </div>""", unsafe_allow_html=True)
            return  # don't auto-advance after completion

        # Auto-advance — short polling (0.2 s) against wall-clock so other tabs
        # are never blocked for more than 0.2 s regardless of sim speed.
        if st.session_state.get("demo_running", False) and idx < TOTAL:
            now = time.time()
            last_tick = st.session_state.get("_demo_last_tick", 0.0)
            if now - last_tick >= 2.0:
                # 2 seconds elapsed — advance one row
                st.session_state["demo_row_idx"] = idx + 1
                st.session_state["_demo_last_tick"] = now
                if idx + 1 >= TOTAL:
                    st.session_state["demo_done"] = True
                    st.session_state["demo_running"] = False
            # Always stay fragment-scoped — other tabs are fully independent
            time.sleep(0.2)
            st.rerun(scope="fragment")

    with tab_live:
        _live_sim_fragment()



    # ══════════════════════════════════════════════════════════════════════════
    # TAB: HOME
    # ══════════════════════════════════════════════════════════════════════════
    with tab_home:
        earned_today = int(my_trips["fare"].sum()) if not my_trips.empty else 0
        target       = int(gc["target_earnings"]) if gc is not None else 0
        projected    = int(gc["projected_earnings"]) if gc is not None else earned_today
        avg_vel      = int(my_trips["earnings_velocity"].mean()) if not my_trips.empty else 0
        needed_vel   = int(gc["needed_velocity"]) if gc is not None else 0

        m1, m2 = st.columns(2)
        m1.metric("Earned Today", f"₹{earned_today:,}", f"of ₹{target:,} goal")
        m2.metric("Goal", f"₹{target:,}", f"{gc['goal_pct_complete']:.0f}% done" if gc is not None else "")
        m3, m4 = st.columns(2)
        m3.metric("Projected", f"₹{projected:,}", "Above goal" if projected >= target else "Below goal")
        m4.metric("Avg ₹/hr", f"₹{avg_vel:,}", f"Need ₹{needed_vel}/hr" if needed_vel > 0 else "Goal met")

        # Goal progress bar
        if gc is not None:
            pct2    = min(gc["goal_pct_complete"], 100)
            bar_col = FORECAST_COLOR.get(gc["forecast"], "#60a5fa")
            card(f"""
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <span style="font-weight:700;font-size:14px;color:#fff">Goal Progress</span>
                <span style="font-weight:800;font-size:18px;color:{bar_col}">{pct2:.0f}%</span>
              </div>
              <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:12px;overflow:hidden">
                <div style="width:{pct2}%;height:100%;background:{bar_col};border-radius:999px;
                     transition:width .5s ease"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:6px;
                   font-size:11px;color:rgba(255,255,255,0.3)">
                <span>₹0</span><span>₹{earned_today:,} earned</span><span>₹{target:,} goal</span>
              </div>
            """)

        # Motivational message
        if gc is not None and gc["message"]:
            mc = FORECAST_COLOR.get(gc["forecast"], "#60a5fa")
            card(f'<div style="font-size:13px;color:#fff;line-height:1.5">{gc["message"]}</div>',
                 f"border-left:3px solid {mc}!important")

        # Best route + Trip breakdown
        bc2_col, qc_col = st.columns(2)
        with bc2_col:
            br  = sc["best_route"]       if sc is not None else "—"
            brv = int(sc["best_route_vel"]) if sc is not None else 0
            card(f"""
              <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                   letter-spacing:.06em;color:rgba(255,255,255,0.3);margin-bottom:5px">Best Route</div>
              <div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:3px">{br}</div>
              <div style="font-size:12px;color:#34d399;font-weight:600">₹{brv}/hr</div>
            """)
        with qc_col:
            if sc is not None:
                ne  = int(sc["n_excellent_trips"])
                ng  = int(sc["n_good_trips"])
                nav = int(sc["n_average_trips"])
                np_ = int(sc["n_poor_trips"])
                total_trips = ne + ng + nav + np_
                def _pill(label, count, color):
                    if count == 0: return ""
                    return ('<span style="background:' + color + '20;color:' + color +
                            ';border:1px solid ' + color + '44;padding:2px 8px;'
                            'border-radius:999px;font-size:10px;font-weight:700;margin-right:3px">'
                            + str(count) + " " + label + "</span>")
                pills_html = (
                    _pill("Excellent", ne, "#34d399") + _pill("Good", ng, "#60a5fa") +
                    _pill("Average", nav, "#fbbf24") + _pill("Poor", np_, "#f87171")
                ) or '<span style="color:rgba(255,255,255,0.25);font-size:11px">No trips</span>'
                card(
                    '<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:.06em;color:rgba(255,255,255,0.3);margin-bottom:6px">Trip Quality</div>'
                    '<div style="margin-bottom:5px">' + pills_html + "</div>"
                    '<div style="font-size:11px;color:rgba(255,255,255,0.3)">' + str(total_trips) + " trips</div>")

        # Safety snapshot
        n_flags = len(my_flags) if not my_flags.empty else 0
        n_high  = len(my_flags[my_flags["severity"] == "high"]) if not my_flags.empty else 0
        bl      = sc["stress_badge"] if sc is not None else "NONE"
        bi_col  = {"HIGH":"#f87171","MODERATE":"#fbbf24","LOW":"#34d399","NONE":"#34d399"}.get(bl,"#9ca3af")
        high_txt= f'&nbsp;·&nbsp;<span style="color:#f87171;font-weight:700">{n_high} high severity</span>' if n_high > 0 else ""
        card(f"""
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:14px;font-weight:700;color:#fff">Safety & Stress</span>
            <span style="background:{bi_col}20;color:{bi_col};border:1px solid {bi_col}44;padding:4px 10px;
                 border-radius:999px;font-size:10px;font-weight:800">{bl}</span>
          </div>
          <div style="font-size:12px;color:rgba(255,255,255,0.4)">
            {n_flags} {"moment" if n_flags==1 else "moments"} flagged{high_txt}
          </div>
        """)

        # Top 5 flagged
        if not my_flags.empty:
            top5 = my_flags.nlargest(5, "combined_score")
            section_header("Top Flagged Moments")
            for _, flag in top5.iterrows():
                sc2  = SEVERITY_COLOR.get(flag["severity"], "#9ca3af")
                evl  = EVENT_LABEL.get(flag["accel_type"], flag["accel_type"].replace("_"," ").title())
                scr  = int(flag["combined_score"] * 100)
                tr   = my_trips[my_trips["trip_id"] == flag["trip_id"]]
                rt   = tr["route"].iloc[0] if not tr.empty and "route" in tr.columns else flag["trip_id"]
                ts   = str(flag["timestamp"]).split(" ")[-1][:5] if " " in str(flag["timestamp"]) else str(flag["timestamp"])
                st.markdown(
                    f'<div style="display:flex;align-items:center;padding:10px 12px;'
                    f'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
                    f'border-radius:12px;margin-bottom:6px">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{sc2};'
                    f'flex-shrink:0;margin-right:10px"></div>'
                    f'<div style="flex:1">'
                    f'<div style="font-size:13px;font-weight:700;color:#fff">{evl}</div>'
                    f'<div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:1px">{rt} · {ts}</div>'
                    f'<div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:1px">{flag["driver_explanation"]}</div>'
                    f'</div>'
                    f'<div style="text-align:right;flex-shrink:0;margin-left:8px">'
                    f'<div style="font-size:18px;font-weight:800;color:{sc2}">{scr}</div>'
                    f'<div style="font-size:9px;color:rgba(255,255,255,0.2)">/ 100</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
        else:
            st.success("No stress moments flagged — clean shift.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB: TRIPS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_trips:
        if my_trips.empty:
            st.info("No trips yet.")
        else:
            target_earn = float(gc["target_earnings"]) if gc is not None else None
            total_shift_h = float(gc["remaining_hours"] + gc["current_hours"]) if gc is not None else 8.0

            def _goal_prob(cumulative, target, elapsed_h, remaining_h, trip_row):
                if target is None or target <= 0: return None, None
                if cumulative >= target: return 100, "Goal reached — extra trip is bonus."
                if remaining_h <= 0:
                    pct_d = round(cumulative / target * 100)
                    return pct_d, f"Shift over — {pct_d}% of goal."
                needed_vel = (target - cumulative) / remaining_h
                current_vel = cumulative / elapsed_h if elapsed_h > 0 else 0
                ratio = current_vel / needed_vel if needed_vel > 0 else 1.0
                prob = int(round(100 / (1 + math.exp(-4 * (ratio - 1)))))
                prob = max(3, min(97, prob))
                vel = int(trip_row.earnings_velocity)
                gap_word = "ahead of" if ratio >= 1.0 else "behind"
                msg = f"₹{int(trip_row.fare)} earned — {gap_word} pace at ₹{vel}/hr."
                return prob, msg

            def _parse_h(t):
                try:
                    parts = str(t).split(":")
                    return int(parts[0]) + int(parts[1]) / 60
                except Exception: return None

            trips_sorted = my_trips.sort_values("start_time").reset_index(drop=True)
            cum_fares = trips_sorted["fare"].cumsum().values

            cards_html = []
            for idx2, trip in enumerate(trips_sorted.itertuples()):
                qc2  = QUALITY_COLOR.get(trip.trip_quality, "#6b7280")
                qlbl = trip.trip_quality.replace("_"," ").title()
                fn   = int(trip.flagged_count)
                ev2  = int(trip.earnings_velocity)
                rt2  = trip.route if hasattr(trip, "route") and trip.route else f"{trip.pickup_location} > {trip.dropoff_location}"
                ss   = trip.stress_score
                sev_dot = SEVERITY_COLOR.get(
                    "high" if ss >= 0.65 else "medium" if ss >= 0.4 else "low" if ss >= 0.2 else "none", "#9ca3af")
                alert_pill = (
                    f'<span style="background:#f8717120;color:#f87171;border:1px solid #f8717144;'
                    f'padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700">'
                    f'{fn} alert{"s" if fn>1 else ""}</span>'
                ) if fn > 0 else ""

                cum_now = float(cum_fares[idx2])
                elapsed_h_now = _parse_h(trip.end_time) or ((idx2 + 1) / max(len(trips_sorted), 1) * total_shift_h)
                remaining_h_now = max(total_shift_h - elapsed_h_now, 0)
                prob, prob_msg = _goal_prob(cum_now, target_earn, elapsed_h_now, remaining_h_now, trip)

                prob_html = ""
                if prob is not None:
                    pc = "#34d399" if prob >= 70 else "#fbbf24" if prob >= 45 else "#f87171"
                    prob_html = (
                        '<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05)">'
                        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">'
                        '<span style="font-size:9px;font-weight:700;text-transform:uppercase;'
                        'letter-spacing:.06em;color:rgba(255,255,255,0.25)">Goal probability</span>'
                        '<span style="font-size:13px;font-weight:800;color:' + pc + '">' + str(prob) + '%</span></div>'
                        '<div style="background:rgba(255,255,255,0.06);border-radius:999px;height:4px;overflow:hidden;margin-bottom:4px">'
                        '<div style="width:' + str(prob) + '%;height:100%;background:' + pc + ';border-radius:999px"></div></div>'
                        '<div style="font-size:10px;color:rgba(255,255,255,0.3)">' + (prob_msg or "") + '</div></div>'
                    )

                cards_html.append(
                    '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
                    'border-radius:14px;padding:14px;margin-bottom:8px">'
                    '<div style="display:flex;align-items:flex-start;justify-content:space-between">'
                    '<div style="flex:1">'
                    '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
                    '<span style="font-size:10px;color:rgba(255,255,255,0.25)">' + trip.trip_id + '</span>'
                    '<span style="background:' + qc2 + '20;color:' + qc2 + ';border:1px solid ' + qc2 + '44;'
                    'padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700">' + qlbl + '</span>'
                    + alert_pill + '</div>'
                    '<div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:3px">' + rt2 + '</div>'
                    '<div style="font-size:11px;color:rgba(255,255,255,0.3)">'
                    + str(trip.start_time) + ' – ' + str(trip.end_time) + ' · ' + str(trip.duration_min) + ' min</div></div>'
                    '<div style="text-align:right;flex-shrink:0;margin-left:14px">'
                    '<div style="font-size:20px;font-weight:800;color:#34d399">₹' + str(int(trip.fare)) + '</div>'
                    '<div style="font-size:11px;color:rgba(255,255,255,0.3)">₹' + str(ev2) + '/hr</div>'
                    '</div></div>' + prob_html + '</div>'
                )
            st.markdown("".join(cards_html), unsafe_allow_html=True)



    # ══════════════════════════════════════════════════════════════════════════
    # TAB: SAFETY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_safety:
        if my_flags.empty:
            st.success("Nothing flagged — great shift.")
        else:
            n_h = len(my_flags[my_flags["severity"] == "high"])
            n_m = len(my_flags[my_flags["severity"] == "medium"])
            n_l = len(my_flags[my_flags["severity"] == "low"])
            s1, s2, s3 = st.columns(3)
            s1.metric("High", n_h)
            s2.metric("Medium", n_m)
            s3.metric("Low", n_l)

            # Safety Timeline
            if "start_time" in trip_sum_df.columns:
                try:
                    my_trips_sorted = trip_sum_df[trip_sum_df["driver_id"] == DID].sort_values("start_time").copy()
                    my_trips_sorted["_t"] = pd.to_datetime(my_trips_sorted["start_time"], errors="coerce")
                    base_t = my_trips_sorted["_t"].iloc[0]
                    tstart_map = (my_trips_sorted["_t"] - base_t).dt.total_seconds().div(60).set_axis(my_trips_sorted["trip_id"]).to_dict()
                    timeline_flags = my_flags.copy()
                    timeline_flags["shift_minute"] = (
                        timeline_flags["trip_id"].map(tstart_map).fillna(0)
                        + timeline_flags["elapsed_seconds"] / 60.0
                    )
                    fig_tl = go.Figure()
                    for sev, col in SEVERITY_COLOR.items():
                        sub = timeline_flags[timeline_flags["severity"] == sev]
                        if sub.empty: continue
                        fig_tl.add_trace(go.Scatter(
                            x=sub["shift_minute"], y=sub["combined_score"],
                            mode="markers",
                            marker=dict(color=col, size=sub["combined_score"]*24+5,
                                        symbol="circle", opacity=0.8,
                                        line=dict(color="white", width=1)),
                            name=sev.title(),
                            hovertemplate="Minute %{x:.0f}<br>Score: %{y:.2f}<extra></extra>"
                        ))
                    fig_tl.add_hline(y=0.65, line_dash="dot", line_color="#f87171",
                                     annotation_text="High", annotation_font_color="#f87171")
                    fig_tl.add_hline(y=0.40, line_dash="dot", line_color="#fbbf24",
                                     annotation_text="Med", annotation_font_color="#fbbf24")
                    fig_tl.update_layout(**PLOTLY_BASE, height=200,
                        xaxis=dict(title="Minutes into shift", gridcolor="rgba(255,255,255,0.04)"),
                        yaxis=dict(title="Score", range=[0,1.05], gridcolor="rgba(255,255,255,0.04)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                        hovermode="closest")
                    st.plotly_chart(fig_tl, use_container_width=True)
                except Exception:
                    pass

            # Filters
            sev_filter = st.radio("Severity", ["All","High","Medium","Low"],
                                  horizontal=True, label_visibility="visible")
            trip_opts  = ["All trips"] + sorted(my_flags["trip_id"].unique().tolist())
            sel_t = st.selectbox("Trip", trip_opts)
            vf = my_flags.copy()
            if sev_filter != "All": vf = vf[vf["severity"] == sev_filter.lower()]
            if sel_t != "All trips": vf = vf[vf["trip_id"] == sel_t]
            vf = vf.sort_values("combined_score", ascending=False)

            if vf.empty:
                st.info("No flagged moments match this filter.")
            else:
                route_map = my_trips.set_index("trip_id")["route"].to_dict() if "route" in my_trips.columns else {}
                flag_cards = []
                for flag in vf.itertuples():
                    sc3 = SEVERITY_COLOR.get(flag.severity, "#9ca3af")
                    evl = EVENT_LABEL.get(flag.accel_type, flag.accel_type.replace("_"," ").title())
                    scr = int(flag.combined_score * 100)
                    rt3 = route_map.get(flag.trip_id, flag.trip_id)
                    mp  = int(flag.accel_score * 100)
                    ap  = int(flag.audio_score * 100)
                    flag_cards.append(
                        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
                        f'border-left:3px solid {sc3};border-radius:0 14px 14px 0;padding:14px;margin-bottom:8px">'
                        f'<div style="display:flex;align-items:flex-start;justify-content:space-between">'
                        f'<div style="flex:1">'
                        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
                        f'<span style="font-size:14px;font-weight:700;color:#fff">{evl}</span>'
                        f'<span style="background:{sc3}20;color:{sc3};border:1px solid {sc3}44;'
                        f'padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;text-transform:uppercase">{flag.severity}</span>'
                        f'</div>'
                        f'<div style="font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:6px;line-height:1.4">{flag.driver_explanation}</div>'
                        f'<div style="font-size:11px;color:rgba(255,255,255,0.25)">{rt3} · {flag.trip_id}</div>'
                        f'</div>'
                        f'<div style="text-align:right;margin-left:14px;flex-shrink:0">'
                        f'<div style="font-size:22px;font-weight:800;color:{sc3}">{scr}</div>'
                        f'<div style="font-size:9px;color:rgba(255,255,255,0.25)">score</div>'
                        f'<div style="font-size:10px;color:rgba(255,255,255,0.3);margin-top:4px">M:{mp} A:{ap}</div>'
                        f'</div></div></div>'
                    )
                st.markdown("".join(flag_cards), unsafe_allow_html=True)

            # Explainability log
            @st.cache_data(show_spinner=False, ttl=None)
            def _cached_exp_log(flags_hash, accel_hash):
                return build_explainability_log(my_flags, my_accel)
            exp_log = _cached_exp_log(
                hash(tuple(my_flags["flag_id"].tolist())) if not my_flags.empty else 0, id(my_accel))
            if not exp_log.empty:
                csv_bytes = exp_log.to_csv(index=False).encode("utf-8")
                st.download_button(label="Download Shift Log (CSV)", data=csv_bytes,
                    file_name=f"driver_pulse_log_{DID}.csv", mime="text/csv")
                with st.expander("Preview (first 10 rows)"):
                    st.dataframe(exp_log[["timestamp","signal_type","raw_value",
                                          "threshold","event_label","severity",
                                          "driver_explanation"]].head(10),
                                 use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB: CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_charts:
        section_sel = st.radio("Chart", ["Live Velocity", "Earnings Curve", "Stress Worm",
                                         "Acceleration", "Audio"],
                               horizontal=True, label_visibility="collapsed")

        if section_sel == "Live Velocity":
            if my_vel.empty:
                st.info("No velocity data.")
            else:
                N_FRAMES  = 200
                vel_sorted = my_vel.sort_values("elapsed_hours").reset_index(drop=True)
                t_raw  = vel_sorted["elapsed_hours"].values.astype(float)
                cv_raw = vel_sorted["recalc_velocity"].values.astype(float)
                nv_raw = vel_sorted["needed_velocity"].values.astype(float)
                ce_raw = vel_sorted["cumulative_earnings"].values.astype(float)
                t_dense  = np.linspace(t_raw[0], t_raw[-1], N_FRAMES)
                cv_dense = np.interp(t_dense, t_raw, cv_raw)
                nv_dense = np.interp(t_dense, t_raw, nv_raw)
                ce_dense = np.interp(t_dense, t_raw, ce_raw)
                total_h  = float(t_raw[-1])
                y_max    = max(float(np.max(cv_dense)), float(np.max(nv_dense))) * 1.18
                ph_dense = ["early" if p < 0.25 else "mid" if p < 0.60 else "peak" if p < 0.80 else "late"
                            for p in t_dense / total_h]
                ghost_nv = go.Scatter(x=t_dense, y=nv_dense, mode="lines",
                    line=dict(color="rgba(248,113,113,0.10)", width=1.5), hoverinfo="skip", showlegend=False)
                ghost_cv = go.Scatter(x=t_dense, y=cv_dense, mode="lines",
                    line=dict(color="rgba(52,211,153,0.10)", width=1.5), hoverinfo="skip", showlegend=False)
                target_earn_vel = float(gc["target_earnings"]) if gc is not None else 1e9

                @st.fragment
                def _vel_animation():
                    import time as _time
                    if "_vel_frame" not in st.session_state:
                        st.session_state["_vel_frame"] = N_FRAMES
                    ctrl_l, ctrl_r = st.columns([3, 1])
                    with ctrl_r:
                        auto_play = st.toggle("Play", key="vel_autoplay")
                    if auto_play:
                        nxt = st.session_state["_vel_frame"] + 1
                        st.session_state["_vel_frame"] = nxt if nxt <= N_FRAMES else 1
                    with ctrl_l:
                        frame = st.slider("Scrub", 1, N_FRAMES,
                            value=int(st.session_state["_vel_frame"]), label_visibility="collapsed")
                    if not auto_play:
                        st.session_state["_vel_frame"] = frame

                    f   = int(st.session_state["_vel_frame"])
                    xs  = t_dense[:f]; cvs = cv_dense[:f]; nvs = nv_dense[:f]; ces = ce_dense[:f]
                    cur_vel  = float(cvs[-1]); need_vel = float(nvs[-1])
                    earned_v = float(ces[-1]); elapsed_v = float(xs[-1])
                    gap      = cur_vel - need_vel; phase = ph_dense[f - 1]
                    pct_done = int(f / N_FRAMES * 100)

                    tol   = {"early":0.20,"mid":0.15,"peak":0.10,"late":0.05}.get(phase, 0.15)
                    ratio = cur_vel / need_vel if need_vel > 0 else 1.0
                    if earned_v >= target_earn_vel or need_vel <= 0: status = "achieved"
                    elif ratio >= (1 - tol): status = "on_track"
                    elif ratio >= (1 - tol - 0.20): status = "at_risk"
                    else: status = "behind"

                    sc_v = FORECAST_COLOR.get(status, "#9ca3af")
                    sl_v = {"achieved":"Goal Achieved","on_track":"On Track",
                            "at_risk":"At Risk","behind":"Behind"}.get(status, status)
                    gap_color = "#34d399" if gap >= 0 else "#f87171"
                    gap_sign  = "+" if gap >= 0 else ""
                    phase_lbl = {"early":"Early","mid":"Mid","peak":"Peak","late":"Late"}.get(phase,"")

                    st.markdown(
                        f'<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">'
                        f'<div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);'
                        f'border-radius:12px;padding:10px 12px;flex:1;min-width:70px">'
                        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;color:rgba(52,211,153,0.5)">Current</div>'
                        f'<div style="font-size:18px;font-weight:800;color:#34d399">₹{cur_vel:,.0f}</div></div>'
                        f'<div style="background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.2);'
                        f'border-radius:12px;padding:10px 12px;flex:1;min-width:70px">'
                        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;color:rgba(248,113,113,0.5)">Required</div>'
                        f'<div style="font-size:18px;font-weight:800;color:#f87171">₹{need_vel:,.0f}</div></div>'
                        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
                        f'border-radius:12px;padding:10px 12px;flex:1;min-width:70px">'
                        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;color:rgba(255,255,255,0.3)">Gap</div>'
                        f'<div style="font-size:18px;font-weight:800;color:{gap_color}">{gap_sign}₹{gap:,.0f}</div></div>'
                        f'<div style="background:{sc_v}12;border:1px solid {sc_v}35;'
                        f'border-radius:12px;padding:10px 12px;flex:1;min-width:70px">'
                        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;color:{sc_v}80">{phase_lbl}</div>'
                        f'<div style="font-size:11px;font-weight:800;color:{sc_v}">{sl_v}</div></div></div>',
                        unsafe_allow_html=True)

                    fig_lv = go.Figure()
                    fig_lv.add_trace(ghost_nv); fig_lv.add_trace(ghost_cv)
                    ahead_mask = cvs >= nvs
                    g_top = np.where(ahead_mask, cvs, nvs); g_bot = nvs
                    fig_lv.add_trace(go.Scatter(x=np.concatenate([xs, xs[::-1]]),
                        y=np.concatenate([g_top, g_bot[::-1]]), fill="toself",
                        fillcolor="rgba(52,211,153,0.12)", line=dict(width=0),
                        hoverinfo="skip", showlegend=True, name="Ahead"))
                    r_top = nvs; r_bot = np.where(~ahead_mask, cvs, nvs)
                    fig_lv.add_trace(go.Scatter(x=np.concatenate([xs, xs[::-1]]),
                        y=np.concatenate([r_top, r_bot[::-1]]), fill="toself",
                        fillcolor="rgba(248,113,113,0.12)", line=dict(width=0),
                        hoverinfo="skip", showlegend=True, name="Behind"))
                    fig_lv.add_trace(go.Scatter(x=xs, y=nvs, mode="lines",
                        line=dict(color="#f87171", width=2, dash="dot"), name="Required ₹/hr"))
                    fig_lv.add_trace(go.Scatter(x=xs, y=cvs, mode="lines",
                        line=dict(color="#34d399", width=2.5), name="Current ₹/hr"))
                    fig_lv.add_trace(go.Scatter(x=[xs[-1]], y=[cvs[-1]], mode="markers",
                        marker=dict(size=20, color=gap_color, opacity=0.15),
                        hoverinfo="skip", showlegend=False))
                    fig_lv.add_trace(go.Scatter(x=[xs[-1]], y=[cvs[-1]], mode="markers+text",
                        marker=dict(size=9, color=gap_color, line=dict(color="white", width=1)),
                        text=[f"  ₹{cur_vel:,.0f}"], textposition="middle right",
                        textfont=dict(color=gap_color, size=10, family="Inter"), name="Now"))
                    for ph_frac, ph_name in [(0.25,"Mid"),(0.60,"Peak"),(0.80,"Late")]:
                        px_val = total_h * ph_frac
                        if px_val <= xs[-1]:
                            fig_lv.add_vline(x=px_val, line_dash="dot",
                                line_color="rgba(255,255,255,0.1)", line_width=1,
                                annotation_text=ph_name, annotation_font_color="rgba(255,255,255,0.2)",
                                annotation_font_size=9)
                    fig_lv.update_layout(**PLOTLY_BASE, height=300,
                        xaxis=dict(title="Hours", range=[t_dense[0], t_dense[-1]],
                                   gridcolor="rgba(255,255,255,0.04)"),
                        yaxis=dict(title="₹/hr", range=[0, max(y_max, 100)],
                                   gridcolor="rgba(255,255,255,0.04)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                        hovermode="x unified")
                    st.plotly_chart(fig_lv, use_container_width=True)

                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;font-size:10px;'
                        f'color:rgba(255,255,255,0.25);margin-bottom:4px">'
                        f'<span>Start</span>'
                        f'<span style="color:{sc_v};font-weight:700">{pct_done}% · ₹{earned_v:,.0f} earned</span>'
                        f'<span>End</span></div>'
                        f'<div style="background:rgba(255,255,255,0.06);border-radius:999px;height:5px;overflow:hidden">'
                        f'<div style="width:{pct_done}%;height:100%;background:{sc_v};border-radius:999px"></div></div>',
                        unsafe_allow_html=True)

                    if not my_trips.empty and "earnings_velocity" in my_trips.columns:
                        fig_bars = go.Figure(go.Bar(
                            x=my_trips["trip_id"], y=my_trips["earnings_velocity"],
                            marker_color=[QUALITY_COLOR.get(q,"#9ca3af") for q in my_trips["trip_quality"]],
                            text=[f"₹{int(v)}" for v in my_trips["earnings_velocity"]],
                            textposition="outside",
                            textfont=dict(color="rgba(255,255,255,0.5)", size=9),
                            hovertemplate="<b>%{x}</b><br>₹/hr: %{y:,.0f}<extra></extra>"))
                        for thresh, col, lbl in [(400,"#34d399","Excellent"),(280,"#60a5fa","Good"),(180,"#fbbf24","Avg")]:
                            fig_bars.add_hline(y=thresh, line_dash="dot", line_color=col,
                                               line_width=1, annotation_text=lbl,
                                               annotation_font_color=col, annotation_font_size=9)
                        fig_bars.update_layout(**PLOTLY_BASE, height=180,
                            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickangle=-30),
                            yaxis=dict(title="₹/hr", gridcolor="rgba(255,255,255,0.04)"),
                            bargap=0.35, showlegend=False)
                        st.plotly_chart(fig_bars, use_container_width=True)

                    if auto_play:
                        _time.sleep(0.06)
                        st.rerun(scope="fragment")

                _vel_animation()

        elif section_sel == "Earnings Curve":
            if my_vel.empty:
                st.info("No velocity data.")
            else:
                fig = make_subplots(specs=[[{"secondary_y":True}]])
                fig.add_trace(go.Scatter(x=my_vel["elapsed_hours"],y=my_vel["cumulative_earnings"],
                    fill="tozeroy",fillcolor="rgba(52,211,153,.08)",line=dict(color="#34d399",width=2.5),
                    name="Cumulative ₹",hovertemplate="Hour %{x:.1f}<br>₹%{y:,.0f}<extra></extra>"),secondary_y=False)
                if gc is not None:
                    fig.add_hline(y=gc["target_earnings"],line_dash="dot",line_color="#fbbf24",line_width=2,
                                  annotation_text=f"Goal ₹{int(gc['target_earnings']):,}",annotation_font_color="#fbbf24")
                    lh = my_vel["elapsed_hours"].max(); le = my_vel["cumulative_earnings"].max()
                    fig.add_trace(go.Scatter(x=[lh,lh+gc["remaining_hours"]],y=[le,gc["projected_earnings"]],
                        line=dict(color="#34d399",width=2,dash="dash"),name="Projected"),secondary_y=False)
                fig.add_trace(go.Scatter(x=my_vel["elapsed_hours"],y=my_vel["recalc_velocity"],
                    line=dict(color="#60a5fa",width=2),name="₹/hr"),secondary_y=True)
                if gc is not None and gc["needed_velocity"] > 0:
                    fig.add_hline(y=gc["needed_velocity"],line_dash="dot",line_color="#f87171",line_width=1.5,
                                  annotation_text=f"Need ₹{int(gc['needed_velocity'])}/hr",
                                  annotation_font_color="#f87171",secondary_y=True)
                fig.update_layout(**PLOTLY_BASE, height=280,
                    yaxis=dict(title="Cumulative ₹", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis2=dict(title="₹/hr", gridcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title="Hours", gridcolor="rgba(255,255,255,0.04)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                    hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        elif section_sel == "Stress Worm":
            if my_flags.empty:
                st.info("No flagged moments.")
            else:
                trip_list_w = sorted(my_flags["trip_id"].unique().tolist())
                sel_w = st.selectbox("Trip", trip_list_w, key="worm_trip")
                trip_flags_w = my_flags[my_flags["trip_id"] == sel_w].sort_values("elapsed_seconds")
                trip_accel_w = my_accel[my_accel["trip_id"] == sel_w].sort_values("elapsed_seconds")
                fig_w = go.Figure()
                if not trip_accel_w.empty and "magnitude" in trip_accel_w.columns:
                    fig_w.add_trace(go.Scatter(x=trip_accel_w["elapsed_seconds"],
                        y=trip_accel_w["magnitude"] / 10.0, fill="tozeroy",
                        fillcolor="rgba(167,139,250,.05)", line=dict(color="rgba(167,139,250,.15)", width=1),
                        name="Motion (scaled)", hoverinfo="skip"))
                if not trip_flags_w.empty:
                    fig_w.add_trace(go.Scatter(x=trip_flags_w["elapsed_seconds"],
                        y=trip_flags_w["combined_score"], mode="lines+markers",
                        line=dict(color="#a78bfa", width=2.5),
                        marker=dict(color=[SEVERITY_COLOR.get(s,"#9ca3af") for s in trip_flags_w["severity"]],
                                    size=8, symbol="circle", line=dict(color="white", width=1)),
                        name="Stress", hovertemplate="t=%{x}s<br>Score: %{y:.2f}<extra></extra>"))
                fig_w.add_hline(y=0.65, line_dash="dot", line_color="#f87171",
                                annotation_text="High", annotation_font_color="#f87171")
                fig_w.add_hline(y=0.40, line_dash="dot", line_color="#fbbf24",
                                annotation_text="Med", annotation_font_color="#fbbf24")
                fig_w.update_layout(**PLOTLY_BASE, height=250,
                    xaxis=dict(title="Elapsed seconds", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="Score", range=[0, 1.05], gridcolor="rgba(255,255,255,0.04)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_w, use_container_width=True)

        elif section_sel == "Acceleration":
            tlist = my_accel["trip_id"].unique().tolist()
            if not tlist:
                st.info("No accelerometer data.")
            else:
                sel = st.selectbox("Trip", tlist)
                ta  = my_accel[my_accel["trip_id"] == sel].sort_values("elapsed_seconds")
                tf  = my_flags[my_flags["trip_id"] == sel] if not my_flags.empty else pd.DataFrame()
                fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=("Motion Intensity (m/s2)","Speed (km/h)"),
                                     vertical_spacing=0.1)
                fig2.add_trace(go.Scatter(x=ta["elapsed_seconds"],y=ta["magnitude"],
                    fill="tozeroy",fillcolor="rgba(167,139,250,.06)",line=dict(color="#a78bfa",width=2),
                    name="Magnitude"),row=1,col=1)
                fig2.add_hline(y=3.5,line_dash="dot",line_color="#fbbf24",
                               annotation_text="Alert",row=1,col=1,annotation_font_color="#fbbf24")
                fig2.add_trace(go.Scatter(x=ta["elapsed_seconds"],y=ta["speed_kmh"],
                    fill="tozeroy",fillcolor="rgba(96,165,250,.06)",line=dict(color="#60a5fa",width=2),
                    name="Speed"),row=2,col=1)
                for _, fl in tf.iterrows():
                    c = SEVERITY_COLOR.get(fl["severity"],"#9ca3af")
                    for r in [1,2]:
                        fig2.add_vrect(x0=fl["elapsed_seconds"]-4,x1=fl["elapsed_seconds"]+4,
                                       fillcolor=c,opacity=0.1,line_width=0,row=r,col=1)
                fig2.update_layout(**PLOTLY_BASE,height=320,
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis2=dict(gridcolor="rgba(255,255,255,0.04)"),
                    xaxis2=dict(title="Elapsed seconds",gridcolor="rgba(255,255,255,0.04)"),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig2, use_container_width=True)

        elif section_sel == "Audio":
            atlist = my_audio["trip_id"].unique().tolist()
            if not atlist:
                st.info("No audio data.")
            else:
                sel_a = st.selectbox("Trip", atlist)
                ta2   = my_audio[my_audio["trip_id"] == sel_a].sort_values("elapsed_seconds")
                CLC   = {"quiet":"#34d399","normal":"#60a5fa","conversation":"#a78bfa",
                         "elevated":"#fbbf24","very_loud":"#fbbf24","argument":"#f87171","mechanical_noise":"#9ca3af"}
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=ta2["elapsed_seconds"],y=ta2["audio_level_db"],
                    marker_color=[CLC.get(c,"#9ca3af") for c in ta2["audio_classification"]],
                    name="Audio dB",
                    hovertemplate="t=%{x}s<br>%{y} dB<br>%{customdata}<extra></extra>",
                    customdata=ta2["audio_classification"]))
                fig3.add_hline(y=80,line_dash="dot",line_color="#fbbf24",
                               annotation_text="80 dB",annotation_font_color="#fbbf24")
                fig3.add_hline(y=90,line_dash="dot",line_color="#f87171",
                               annotation_text="90 dB",annotation_font_color="#f87171")
                fig3.update_layout(**PLOTLY_BASE,height=250,
                    yaxis=dict(title="dB",gridcolor="rgba(255,255,255,0.04)"),
                    xaxis=dict(title="Elapsed seconds",gridcolor="rgba(255,255,255,0.04)"),
                    bargap=0.15)
                st.plotly_chart(fig3, use_container_width=True)



    # ══════════════════════════════════════════════════════════════════════════
    # TAB: EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_export:
        def _export_card(title, df, filename):
            if df is None or df.empty:
                card(f"""
                  <div style="font-size:14px;font-weight:700;color:#fff">{title}</div>
                  <div style="font-size:11px;color:#fbbf24;margin-top:4px">No data available.</div>
                """)
                return
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            card(f"""
              <div style="display:flex;align-items:center;justify-content:space-between">
                <div>
                  <div style="font-size:14px;font-weight:700;color:#fff">{title}</div>
                  <div style="font-size:10px;color:rgba(255,255,255,0.25);margin-top:3px">
                    {len(df):,} rows · {len(df.columns)} cols · {len(csv_bytes)/1024:.1f} KB
                  </div>
                </div>
              </div>
            """)
            st.download_button(label=f"Download {filename}", data=csv_bytes,
                file_name=filename, mime="text/csv", key=f"dl_{filename}", use_container_width=True)

        _export_card("Flagged Moments", my_flags, "flagged_moments.csv")
        _export_card("Trip Summaries", my_trips, "trip_summaries.csv")
        _export_card("Shift Summary",
                     my_shift.reset_index(drop=True) if not my_shift.empty else my_shift,
                     "driver_shift_summary.csv")
        _export_card("Earnings Velocity Log", my_vel.reset_index(drop=True), "earnings_velocity_log.csv")
        _export_card("Goals Forecast",
                     my_goal.reset_index(drop=True) if not my_goal.empty else my_goal,
                     "goals_forecast.csv")

        st.divider()
        section_header("Export All as ZIP")
        all_exports = {
            "flagged_moments.csv":       my_flags,
            "trip_summaries.csv":        my_trips,
            "driver_shift_summary.csv":  my_shift.reset_index(drop=True) if not my_shift.empty else my_shift,
            "earnings_velocity_log.csv": my_vel.reset_index(drop=True),
            "goals_forecast.csv":        my_goal.reset_index(drop=True) if not my_goal.empty else my_goal,
        }
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, df in all_exports.items():
                if df is not None and not df.empty:
                    zf.writestr(fname, df.to_csv(index=False))
        zip_buf.seek(0)
        st.download_button(
            label=f"Download driver_pulse_{DID}_outputs.zip",
            data=zip_buf.getvalue(),
            file_name=f"driver_pulse_{DID}_outputs.zip",
            mime="application/zip", key="dl_all_zip", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if "driver_id" not in st.session_state:
    show_login()
else:
    with st.spinner("Loading..."):
        show_dashboard()
