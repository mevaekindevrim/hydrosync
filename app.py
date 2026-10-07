"""
app.py
======
HydroSync AI - Streamlit Dashboard (Xylem Global Student Innovation Challenge 2026)

Calistirma:
    pip install streamlit plotly pandas numpy
    streamlit run app."""
app.py
======
HydroSync AI - Streamlit Dashboard (Xylem Global Student Innovation Challenge 2026)

Calistirma:
    pip install streamlit plotly pandas numpy
    streamlit run app.py

Ayni klasorde su dosyalar bulunmali: physics_engine.py, data_generator.py, optimizer.py
Arayuz metinleri juri icin Ingilizcedir; kod yorumlari Turkcedir.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from string import Template

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_generator import generate_dataset, spike_windows
from optimizer import OptimizerConfig, run_comparison
from physics_engine import PlantConfig

# set_page_config ilk Streamlit komutu olmali
st.set_page_config(
    page_title="HydroSync AI | Xylem Innovation Challenge",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Tasarim sabitleri (Xylem Cyan / Deep Navy, koyu tema)
# Grafik renkleri dataviz dogrulayicisindan gecirildi (koyu yuzey #0A192F).
# ----------------------------------------------------------------------
NAVY = "#0A192F"        # sayfa zemini
PANEL = "#0F2442"       # kart zemini
BORDER = "#1C3A5E"
CYAN = "#00A3E0"        # Xylem Cyan (arayuz vurgusu)
CYAN_CHART = "#009FDB"  # HydroSync serisi (dogrulanmis grafik tonu)
AMBER_CHART = "#CF7A26" # Baseline serisi (dogrulanmis)
SLATE = "#8FA3BF"       # notr seri (isi yuku, tahmin)
TEXT1 = "#E6EDF7"
TEXT2 = "#A9B8CF"
TEXT3 = "#7389A6"
GRID = "rgba(143,163,191,0.14)"
OK, WARN, IDLE = "#3FB68B", "#E0A030", "#6B7F99"

SERIES_BL = "Baseline (reactive)"
SERIES_HS = "HydroSync AI"
COLOR_MAP = {SERIES_BL: AMBER_CHART, SERIES_HS: CYAN_CHART}
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

CSS = Template(
    """
<style>
.stApp { background: linear-gradient(180deg, $NAVY 0%, #0C2038 100%); color: $TEXT1; font-family: $FONT; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #071324; border-right: 1px solid $BORDER; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: $TEXT2; }
h1, h2, h3, h4 { color: $TEXT1 !important; font-family: $FONT; letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: $TEXT2; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }

.hero { border: 1px solid $BORDER; border-radius: 12px; padding: 20px 26px; margin-bottom: 18px;
        background: linear-gradient(120deg, #0F2A4D 0%, #0A192F 60%); position: relative; overflow: hidden; }
.hero:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: $CYAN; }
.hero h1 { margin: 0; font-size: 2rem; font-weight: 750; }
.hero h1 span { color: $CYAN; }
.hero p { margin: 6px 0 10px 0; color: $TEXT2; font-size: 1rem; }
.chip { display: inline-block; padding: 3px 10px; margin-right: 8px; border: 1px solid $BORDER; border-radius: 999px;
        font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: $TEXT2; background: rgba(0,163,224,.08); }

.kpi { background: $PANEL; border: 1px solid $BORDER; border-left: 4px solid $CYAN; border-radius: 10px;
       padding: 16px 18px; height: 100%; }
.kpi.small { border-left-width: 3px; padding: 12px 16px; border-left-color: $SLATE; }
.kpi .label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: $TEXT3; margin-bottom: 4px; }
.kpi .value { font-size: 1.95rem; font-weight: 750; color: $TEXT1; font-variant-numeric: tabular-nums; line-height: 1.15; }
.kpi.small .value { font-size: 1.35rem; }
.kpi .sub { font-size: .84rem; color: $TEXT2; margin-top: 4px; }

.note { border: 1px solid $BORDER; border-radius: 8px; padding: 10px 14px; margin: 14px 0 6px 0;
        background: rgba(143,163,191,.07); color: $TEXT2; font-size: .9rem; }
.note b { color: $TEXT1; }
.section-title { margin: 26px 0 2px 0; font-size: 1.25rem; font-weight: 700; color: $TEXT1; }
.section-sub { color: $TEXT3; font-size: .88rem; margin-bottom: 8px; }

.vue-panel { background: $PANEL; border: 1px solid $BORDER; border-radius: 10px; padding: 14px 18px; }
.vue-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.vue-head b { color: $TEXT1; }
.tag { font-size: .68rem; letter-spacing: .08em; padding: 2px 8px; border-radius: 4px; border: 1px solid $BORDER; color: $TEXT2; }
.vue-row { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-top: 1px solid $BORDER; font-size: .9rem; }
.vue-row .name { color: $TEXT1; min-width: 150px; font-weight: 600; }
.vue-row .state { color: $TEXT2; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }

div.stButton > button, div.stDownloadButton > button { width: 100%; border-radius: 8px; font-weight: 650; }
div.stButton > button[kind="primary"] { background: $CYAN; border: none; color: #04121f; }
div.stButton > button[kind="primary"]:hover { background: #2BB8EE; color: #04121f; }
</style>
"""
).substitute(
    NAVY=NAVY, PANEL=PANEL, BORDER=BORDER, CYAN=CYAN, SLATE=SLATE,
    TEXT1=TEXT1, TEXT2=TEXT2, TEXT3=TEXT3, FONT=FONT,
)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Yardimcilar
# ----------------------------------------------------------------------
def _rgba(hex_color: str, alpha: float) -> str:
    """#RRGGBB -> rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def kpi_card(label: str, value: str, sub: str, small: bool = False) -> str:
    """KPI karti HTML'i."""
    cls = "kpi small" if small else "kpi"
    return (
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def px_trace(df: pd.DataFrame, y: str, name: str, color: str, unit: str,
             width: float = 2.0, dash: str = "solid", fill: bool = False):
    """Plotly Express ile tek seri uretip stilini ayarlar (subplot'lara aktarmak icin)."""
    tr = px.line(df, x="Timestamp", y=y).data[0]
    tr.update(
        name=name,
        showlegend=True,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate="%{y:.2f} " + unit,
    )
    if fill:
        tr.update(fill="tozeroy", fillcolor=_rgba(color, 0.18))
    return tr


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    """Tum grafikler icin ortak koyu tema ve ince, sessiz izgara."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT2, size=12),
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT1)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title_text="", font=dict(color=TEXT2)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT3), title_text="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT3))
    return fig


def shade_spikes(fig: go.Figure, windows, **kw) -> None:
    """LLM spike pencerelerini arka planda hafifce gri bantla isaretler."""
    for a, b in windows:
        fig.add_vrect(
            x0=a.to_pydatetime(), x1=(b + timedelta(minutes=15)).to_pydatetime(),
            fillcolor="rgba(169,184,207,0.08)", line_width=0, layer="below", **kw,
        )


# ----------------------------------------------------------------------
# Simulasyon (onbellekli)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate(capacity_mw, water_price, target_coc, pue, baseline_coc, tds_limit,
             elec_price, dry_assist, noise_pct, seed):
    """Veri uret + baseline / HydroSync karsilastirmasini kos."""
    df = generate_dataset(seed=seed, base_water_price=water_price)
    plant = PlantConfig(capacity_mw=capacity_mw, pue_baseline=pue)
    opt = OptimizerConfig(
        baseline_coc=baseline_coc, ai_target_coc=target_coc, ai_tds_limit_ppm=float(tds_limit),
        electricity_price_usd_kwh=elec_price, enable_dry_assist=dry_assist,
        forecast_noise_pct=noise_pct,
    )
    return run_comparison(df, plant, opt)


# ----------------------------------------------------------------------
# Grafikler
# ----------------------------------------------------------------------
def chart_water(res: pd.DataFrame, windows) -> go.Figure:
    """1) Su tuketimi: Baseline vs HydroSync (yanal alan dolgulu)."""
    long = pd.concat(
        [
            pd.DataFrame({"Timestamp": res["Timestamp"], "Scenario": SERIES_BL,
                          "Water intake (L/s)": res["BL_Makeup_Lps"]}),
            pd.DataFrame({"Timestamp": res["Timestamp"], "Scenario": SERIES_HS,
                          "Water intake (L/s)": res["HS_Makeup_Lps"]}),
        ],
        ignore_index=True,
    )
    fig = px.line(long, x="Timestamp", y="Water intake (L/s)", color="Scenario", color_discrete_map=COLOR_MAP)
    for tr in fig.data:
        color = COLOR_MAP[tr.name]
        tr.update(line=dict(color=color, width=2), fill="tozeroy", fillcolor=_rgba(color, 0.16),
                  hovertemplate="%{y:.2f} L/s")
    shade_spikes(fig, windows)
    style_fig(fig, 380)
    fig.update_yaxes(title_text="Cooling-tower water intake (L/s)", rangemode="tozero")
    return fig


def chart_predictive(res: pd.DataFrame, windows, window_idx: int) -> go.Figure:
    """2) Ongoru gorunumu: isi yuku (sol eksen) vs blowdown (sag eksen)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sol eksen (MW): gercek isi yuku, 30 dk ongoru, HydroSync'in hazirladigi sogutma kapasitesi
    fig.add_trace(px_trace(res, "Heat_Load_MW", "Heat load · MW (left)", SLATE, "MW", 1.5, fill=True),
                  secondary_y=False)
    fig.add_trace(px_trace(res, "Heat_Load_Pred_MW", "30-min forecast · MW (left)", SLATE, "MW", 1.2, "dot"),
                  secondary_y=False)
    fig.add_trace(px_trace(res, "HS_Cooling_Staged_MW", "HydroSync staged cooling · MW (left)",
                           CYAN_CHART, "MW", 2.0), secondary_y=False)

    # Sag eksen (L/s): blowdown debileri
    fig.add_trace(px_trace(res, "BL_Blowdown_Lps", "Baseline blowdown · L/s (right)", AMBER_CHART, "L/s", 1.6),
                  secondary_y=True)
    fig.add_trace(px_trace(res, "HS_Blowdown_Lps", "HydroSync blowdown · L/s (right)", CYAN_CHART, "L/s",
                           1.6, "dash"), secondary_y=True)

    shade_spikes(fig, windows, row=1, col=1)
    for i, (a, b) in enumerate(windows, start=1):
        mid = (a + (b - a) / 2).to_pydatetime()
        fig.add_annotation(x=mid, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"LLM spike {i}", font=dict(color=TEXT3, size=11))

    style_fig(fig, 470)
    fig.update_layout(margin=dict(l=8, r=8, t=70, b=8), legend=dict(y=1.12))
    fig.update_yaxes(title_text="Heat load / cooling (MW)", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Blowdown (L/s)", rangemode="tozero", showgrid=False, secondary_y=True)

    if window_idx > 0:  # secilen spike'a yakinlas
        a, b = windows[window_idx - 1]
        fig.update_xaxes(range=[(a - timedelta(hours=3)).isoformat(), (b + timedelta(hours=3)).isoformat()])
    return fig


def chart_tds_coc(res: pd.DataFrame, windows, baseline_limit: float, hs_limit: float, max_coc: float) -> go.Figure:
    """3) TDS ve CoC degisimi (iki ayri panel, ortak zaman ekseni)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.6, 0.4])
    fig.add_trace(px_trace(res, "BL_TDS_ppm", f"{SERIES_BL} · TDS", AMBER_CHART, "ppm"), row=1, col=1)
    fig.add_trace(px_trace(res, "HS_TDS_ppm", f"{SERIES_HS} · TDS", CYAN_CHART, "ppm"), row=1, col=1)
    fig.add_trace(px_trace(res, "BL_CoC", f"{SERIES_BL} · CoC", AMBER_CHART, "×"), row=2, col=1)
    fig.add_trace(px_trace(res, "HS_CoC", f"{SERIES_HS} · CoC", CYAN_CHART, "×"), row=2, col=1)
    # Ikinci paneldeki izler icin ayri legend girdisi acma
    for tr in fig.data[2:]:
        tr.update(showlegend=False)

    fig.add_hline(y=baseline_limit, line_dash="dash", line_width=1, line_color=AMBER_CHART, row=1, col=1,
                  annotation_text=f"Baseline scaling limit · {baseline_limit:,.0f} ppm",
                  annotation_position="top left", annotation_font=dict(color=TEXT2, size=11))
    fig.add_hline(y=hs_limit, line_dash="dash", line_width=1, line_color=CYAN_CHART, row=1, col=1,
                  annotation_text=f"HydroSync conditioned limit · {hs_limit:,.0f} ppm",
                  annotation_position="top left", annotation_font=dict(color=TEXT2, size=11))
    fig.add_hline(y=max_coc, line_dash="dot", line_width=1, line_color=SLATE, row=2, col=1,
                  annotation_text=f"Max CoC · {max_coc:.1f}", annotation_position="top left",
                  annotation_font=dict(color=TEXT2, size=11))

    shade_spikes(fig, windows, row="all", col=1)
    style_fig(fig, 560)
    fig.update_yaxes(title_text="Tower water TDS (ppm)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="CoC (C_tower / C_makeup)", rangemode="tozero", row=2, col=1)
    return fig


# ----------------------------------------------------------------------
# Xylem Vue entegrasyonu (SIMULASYON - hicbir dis cagri yapilmaz)
# ----------------------------------------------------------------------
def build_vue_payload(params: dict, m: dict, res: pd.DataFrame) -> dict:
    """HydroSync parametrelerini Xylem Vue'ya gonderilecekmis gibi JSON payload'a cevirir."""
    by_hour = (
        res.assign(h=pd.to_datetime(res["Timestamp"]).dt.hour)
        .groupby("h")["HS_Mode"].agg(lambda s: s.mode().iat[0])
    )
    schedule = {
        "concentrate_hours": [int(h) for h, v in by_hour.items() if v == "CONCENTRATE"],
        "flush_hours": [int(h) for h, v in by_hour.items() if v == "FLUSH"],
        "normal_hours": [int(h) for h, v in by_hour.items() if v == "NORMAL"],
    }
    return {
        "schema": "hydrosync.setpoints/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "site": {"name": "Demo AI Data Center", "asset_id": "CT-01", "capacity_mw": params["capacity_mw"]},
        "parameters": {
            "target_coc": params["target_coc"],
            "max_coc": 7.5,
            "tds_limit_ppm": params["tds_limit"],
            "forecast_horizon_min": 30,
            "spike_jump_threshold_pct": 15,
            "dry_assist_enabled": params["dry_assist"],
            "water_price_usd_m3": params["water_price"],
        },
        "blowdown_schedule": schedule,
        "expected_kpis": {
            "water_saved_pct": round(m["water_saved_pct"], 2),
            "wue_l_per_kwh": round(m["wue_hydrosync_l_per_kwh"], 3),
            "avg_coc": round(m["avg_coc_hydrosync"], 2),
            "cost_saved_usd_per_week": round(m["cost_saved_usd"], 2),
        },
        "note": "SIMULATED PAYLOAD - not transmitted to any external service",
    }


def payload_signature(payload: dict) -> str:
    """Zaman damgasi haric payload imzasi (parametre degisti mi kontrolu icin)."""
    body = {k: v for k, v in payload.items() if k != "generated_at"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


VUE_STEPS = [
    ("Authenticating with Xylem Vue gateway", "Access token issued (simulated)"),
    ("Validating parameter schema", "hydrosync.setpoints/v1 · OK"),
    ("Mapping assets", "Data center → Cooling tower CT-01 · 1 asset mapped"),
    ("Pushing setpoints and blowdown schedule", "Payload accepted · 24 hourly modes"),
    ("Waiting for controller acknowledgement", "Controller ACK received (simulated)"),
]


def vue_row(color: str, name: str, state: str) -> str:
    """Durum paneli satiri: renkli nokta + metin etiketi (renk tek basina anlam tasimaz)."""
    return (f'<div class="vue-row"><span class="dot" style="background:{color}"></span>'
            f'<span class="name">{name}</span><span class="state">{state}</span></div>')


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scenario controls")
    capacity_mw = st.slider("Data center capacity (MW)", min_value=1.0, max_value=10.0, value=8.0, step=0.5,
                            help="Maximum IT power. Actual IT load follows GPU utilization (idle ≈ 25%).")
    water_price = st.slider("Base water price ($/m³)", min_value=0.5, max_value=10.0, value=2.5, step=0.1,
                            help="Base tariff. A time-of-use profile (night ×0.75 … peak ×1.6) is applied on top.")
    target_coc = st.slider("HydroSync target CoC", min_value=3.0, max_value=7.5, value=6.0, step=0.1,
                           help="Normal-hour setpoint. During expensive peak hours HydroSync may rise to CoC 7.5.")
    with st.expander("Advanced assumptions"):
        pue = st.slider("Baseline PUE", min_value=1.10, max_value=1.80, value=1.35, step=0.01)
        baseline_coc = st.slider("Baseline design CoC", min_value=3.0, max_value=5.0, value=4.0, step=0.1)
        tds_limit = st.slider("HydroSync conditioned TDS limit (ppm)", min_value=1800, max_value=3200,
                              value=2800, step=50,
                              help="Assumption: predictive antiscalant dosing & water conditioning raises the "
                                   "safe TDS limit above the 1,500 ppm of a standard program.")
        elec_price = st.slider("Electricity price ($/kWh)", min_value=0.03, max_value=0.30, value=0.08, step=0.01)
        dry_assist = st.checkbox("Dynamic PUE↔WUE dry-assist", value=True,
                                 help="Shifts part of the heat rejection to the dry path when water is worth more "
                                      "than the extra fan/pump electricity.")
        noise_pct = st.slider("Forecast error (GPU %-points, 1σ)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        seed = st.number_input("Synthetic data seed", min_value=0, max_value=9999, value=42, step=1)
    st.caption("All data is synthetic (672 points · 7 days · 15-min). Simulation only.")

results, m = simulate(capacity_mw, water_price, target_coc, pue, baseline_coc, tds_limit,
                      elec_price, dry_assist, noise_pct, int(seed))
windows = spike_windows(results)

# ----------------------------------------------------------------------
# Baslik
# ----------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Hydro<span>Sync</span> AI</h1>
  <p>Predictive water optimization for AI data-center cooling towers: forecast the heat shock,
     shift blowdown to cheap hours, and raise cycles of concentration safely.</p>
  <span class="chip">Water Quantity</span><span class="chip">B2B Platform</span>
  <span class="chip">Xylem Innovation Challenge 2026</span>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# KPI kartlari
# ----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi_card("Water saved", f"{m['water_saved_l']:,.0f} L",
                     f"{m['water_saved_pct']:.1f}% less than baseline · {m['sim_days']:.0f}-day run"),
            unsafe_allow_html=True)
c2.markdown(kpi_card("Cost saved", f"${m['cost_saved_usd']:,.0f}",
                     f"{m['cost_saved_pct']:.1f}% · net of ${m['extra_electricity_cost_usd']:,.0f} extra electricity"),
            unsafe_allow_html=True)
c3.markdown(kpi_card("WUE improvement", f"{m['wue_improvement_pct']:.1f}%",
                     f"{m['wue_baseline_l_per_kwh']:.2f} → {m['wue_hydrosync_l_per_kwh']:.2f} L/kWh"),
            unsafe_allow_html=True)
c4.markdown(kpi_card("Average CoC", f"{m['avg_coc_hydrosync']:.2f}",
                     f"Baseline {m['avg_coc_baseline']:.2f} · +{m['avg_coc_hydrosync'] - m['avg_coc_baseline']:.2f}"),
            unsafe_allow_html=True)

# Tasarrufun kaynagi (seffaflik)
dry_pts = m["water_saved_pct"] - m["water_saved_pct_without_dry_assist"]
if dry_assist and dry_pts > 0.05:
    breakdown = (f"<b>Where the water savings come from:</b> {m['water_saved_pct_without_dry_assist']:.1f}% from "
                 f"CoC, tariff-aware blowdown and spike anticipation, plus {dry_pts:.1f} pts from dynamic "
                 f"dry-assist, which uses {m['extra_electricity_kwh']:,.0f} kWh of extra electricity "
                 f"(PUE {m['pue_baseline']:.3f} → {m['pue_hydrosync']:.3f}).")
else:
    breakdown = (f"<b>Where the water savings come from:</b> {m['water_saved_pct_without_dry_assist']:.1f}% from "
                 f"CoC, tariff-aware blowdown and spike anticipation. Dry-assist is "
                 f"{'off' if not dry_assist else 'not economical at this water price'}.")
st.markdown(f'<div class="note">{breakdown}</div>', unsafe_allow_html=True)

# Ikincil gostergeler
carbon = m["carbon_net_avoided_kg"]
carbon_sub = ("Net avoided: water-related emissions exceed added power"
              if carbon >= 0 else "Net increase: dry-assist power exceeds water-related emissions avoided")
s1, s2, s3, s4 = st.columns(4)
s1.markdown(kpi_card("LLM spikes anticipated", f"{m['spikes_anticipated']} / {m['spikes_total']}",
                     "Detected 30 min before the load rise", small=True), unsafe_allow_html=True)
s2.markdown(kpi_card("Peak TDS headroom to limit", f"{m['scaling_headroom_hydrosync_pct']:.1f}%",
                     f"Baseline {m['scaling_headroom_baseline_pct']:.1f}% (each vs. its own limit)", small=True),
            unsafe_allow_html=True)
s3.markdown(kpi_card("Effective PUE", f"{m['pue_hydrosync']:.3f}",
                     f"Baseline {m['pue_baseline']:.3f}", small=True), unsafe_allow_html=True)
s4.markdown(kpi_card("Net carbon impact", f"{carbon:+,.0f} kg CO₂e", carbon_sub, small=True),
            unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Grafik 1
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">1 · Water consumption: Baseline vs HydroSync AI</div>'
            '<div class="section-sub">Cooling-tower makeup water over 7 days. Shaded bands mark LLM training spikes.</div>',
            unsafe_allow_html=True)
st.plotly_chart(chart_water(results, windows), theme=None)

# ----------------------------------------------------------------------
# Grafik 2
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">2 · Predictive action view</div>'
            '<div class="section-sub">HydroSync forecasts the heat load 30 minutes ahead and stages cooling before the '
            'spike lands. Its blowdown follows the tariff schedule rather than the spike: held back in the daytime, '
            'flushed at cheap night hours. Left axis: MW. Right axis: L/s.</div>',
            unsafe_allow_html=True)
mode_at = results.set_index("Timestamp")["HS_Mode"]
mode_label = {"CONCENTRATE": "peak hours", "FLUSH": "night flush", "NORMAL": "standard hours"}
options = ["Full week"] + [
    f"Zoom: spike {i} · {a:%a %d %b %H:%M} · {mode_label[mode_at.loc[a]]}"
    for i, (a, _) in enumerate(windows, start=1)
]
# Varsayilan: gunduz (flush olmayan) ilk spike; yoksa ilk spike
default_idx = next((i for i, (a, _) in enumerate(windows, start=1) if mode_at.loc[a] != "FLUSH"),
                   1 if windows else 0)
choice = st.selectbox("View", options, index=default_idx, label_visibility="collapsed")
st.plotly_chart(chart_predictive(results, windows, options.index(choice)), theme=None)

# ----------------------------------------------------------------------
# Grafik 3
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">3 · TDS and CoC dynamics</div>'
            '<div class="section-sub">HydroSync concentrates water during expensive peak hours and flushes during cheap '
            'night hours, staying inside its conditioned TDS limit.</div>', unsafe_allow_html=True)
st.plotly_chart(chart_tds_coc(results, windows, 1500.0, float(tds_limit), 7.5), theme=None)

with st.expander("Data table view"):
    show_cols = ["Timestamp", "GPU_Utilization", "Ambient_WetBulb_Temp", "Water_Tariff_Price", "Incoming_Water_TDS",
                 "BL_Makeup_Lps", "HS_Makeup_Lps", "BL_Blowdown_Lps", "HS_Blowdown_Lps", "BL_TDS_ppm", "HS_TDS_ppm",
                 "BL_CoC", "HS_CoC", "HS_Mode", "HS_Dry_Fraction"]
    table = results[show_cols].copy()
    num_cols = table.select_dtypes("number").columns
    table[num_cols] = table[num_cols].round(2)
    st.dataframe(table, height=300)
    st.download_button("Download full results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       file_name="hydrosync_results.csv", mime="text/csv")

# ----------------------------------------------------------------------
# Xylem Vue entegrasyonu
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">Xylem Vue integration</div>'
            '<div class="section-sub">Simulated B2B hand-off: HydroSync parameters are packaged and pushed to the '
            'Xylem Vue platform. No external call is made in this demo.</div>', unsafe_allow_html=True)

params = {"capacity_mw": capacity_mw, "target_coc": target_coc, "tds_limit": int(tds_limit),
          "dry_assist": bool(dry_assist), "water_price": water_price}
payload = build_vue_payload(params, m, results)
sig = payload_signature(payload)

if "vue" not in st.session_state:
    st.session_state["vue"] = {"synced": False, "sig": None, "ts": None}

left, right = st.columns([1, 1])
with left:
    if st.button("Export Parameters to Xylem Vue API", type="primary"):
        with st.status("Exporting to Xylem Vue (simulated)…", expanded=True) as status:
            for step, detail in VUE_STEPS:
                st.write(f"{step}…")
                time.sleep(0.35)
                st.write(f"✓ {detail}")
            status.update(label="Export complete · controller acknowledged", state="complete", expanded=False)
        st.session_state["vue"] = {"synced": True, "sig": sig, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    st.download_button("Download payload (JSON)", json.dumps(payload, indent=2).encode("utf-8"),
                       file_name="hydrosync_xylem_vue_payload.json", mime="application/json")
    st.code(json.dumps(payload, indent=2), language="json")

with right:
    vue = st.session_state["vue"]
    if not vue["synced"]:
        rows = [
            vue_row(IDLE, "API gateway", "Ready · simulated endpoint"),
            vue_row(IDLE, "Asset registry", "1 site · 1 cooling tower (CT-01) mapped"),
            vue_row(IDLE, "Setpoint push", "Not exported yet"),
            vue_row(IDLE, "Telemetry stream", "Idle"),
        ]
        headline, tag = "Not synced", "SIMULATED"
    elif vue["sig"] == sig:
        rows = [
            vue_row(OK, "API gateway", "Connected · simulated"),
            vue_row(OK, "Asset registry", "1 site · 1 cooling tower (CT-01) mapped"),
            vue_row(OK, "Setpoint push", f"Acknowledged · {vue['ts']} · rev {vue['sig']}"),
            vue_row(OK, "Telemetry stream", "Live · 15-min cadence (simulated)"),
        ]
        headline, tag = "In sync", "SIMULATED"
    else:
        rows = [
            vue_row(OK, "API gateway", "Connected · simulated"),
            vue_row(OK, "Asset registry", "1 site · 1 cooling tower (CT-01) mapped"),
            vue_row(WARN, "Setpoint push", f"Out of date · parameters changed since {vue['ts']}"),
            vue_row(OK, "Telemetry stream", "Live · 15-min cadence (simulated)"),
        ]
        headline, tag = "Out of date: re-export to update", "SIMULATED"
    st.markdown(
        f'<div class="vue-panel"><div class="vue-head"><b>Integration status · {headline}</b>'
        f'<span class="tag">{tag}</span></div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Model notlari
# ----------------------------------------------------------------------
with st.expander("Model notes and assumptions"):
    st.markdown(
        f"""
- **Synthetic data.** 7 days at 15-minute resolution (672 points). GPU load, wet-bulb temperature, time-of-use water
  tariff and incoming water TDS are generated, not measured.
- **Physics.** Q = P_IT · PUE, E = 0.0018 · Q_tons (L/s), B = E / (CoC − 1), M = E + B, and the TDS mass balance
  C(t+1) = C(t) + Δt · (M·C_makeup − B·C_tower) / V.
- **Baseline.** Fixed-rate blowdown valve opens above 1,500 ppm and closes at 1,450 ppm. It ignores GPU load,
  spikes and tariffs.
- **HydroSync AI.** Forecasts GPU load 30 min ahead (a noisy look-ahead stands in for a model trained on Slurm or
  Kubernetes queue data), stages cooling before spikes, concentrates to CoC {7.5} in expensive hours and flushes in cheap
  or cool hours.
- **Conditioned TDS limit.** The {tds_limit:,} ppm limit is a design assumption (predictive antiscalant dosing and water
  conditioning). It needs field validation before it is claimed in practice.
- **Dry-assist (PUE↔WUE).** Moves part of the heat rejection to a dry path when the water saved is worth more than the
  extra electricity. It lowers water use but raises PUE and can raise carbon.
- **Carbon.** Water-related emissions use {OptimizerConfig().water_embedded_kwh_per_m3:.1f} kWh/m³ embedded energy and
  {OptimizerConfig().grid_emission_kg_per_kwh:.2f} kg CO₂e/kWh. Both are adjustable assumptions.
- **Annualized (indicative):** ≈ {m['annual_water_saved_m3']:,.0f} m³ of water and ${m['annual_cost_saved_usd']:,.0f} per
  year for this site, scaled linearly from the 7-day run.
"""
    )
py

Ayni klasorde su dosyalar bulunmali: physics_engine.py, data_generator.py, optimizer.py
Arayuz metinleri juri icin Ingilizcedir; kod yorumlari Turkcedir.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from string import Template

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_generator import generate_dataset, spike_windows
from optimizer import OptimizerConfig, run_comparison
from physics_engine import PlantConfig

# set_page_config ilk Streamlit komutu olmali
st.set_page_config(
    page_title="HydroSync AI | Xylem Innovation Challenge",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Tasarim sabitleri (Xylem Cyan / Deep Navy, koyu tema)
# Grafik renkleri dataviz dogrulayicisindan gecirildi (koyu yuzey #0A192F).
# ----------------------------------------------------------------------
NAVY = "#0A192F"        # sayfa zemini
PANEL = "#0F2442"       # kart zemini
BORDER = "#1C3A5E"
CYAN = "#00A3E0"        # Xylem Cyan (arayuz vurgusu)
CYAN_CHART = "#009FDB"  # HydroSync serisi (dogrulanmis grafik tonu)
AMBER_CHART = "#CF7A26" # Baseline serisi (dogrulanmis)
SLATE = "#8FA3BF"       # notr seri (isi yuku, tahmin)
TEXT1 = "#E6EDF7"
TEXT2 = "#A9B8CF"
TEXT3 = "#7389A6"
GRID = "rgba(143,163,191,0.14)"
OK, WARN, IDLE = "#3FB68B", "#E0A030", "#6B7F99"

SERIES_BL = "Baseline (reactive)"
SERIES_HS = "HydroSync AI"
COLOR_MAP = {SERIES_BL: AMBER_CHART, SERIES_HS: CYAN_CHART}
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

CSS = Template(
    """
<style>
.stApp { background: linear-gradient(180deg, $NAVY 0%, #0C2038 100%); color: $TEXT1; font-family: $FONT; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #071324; border-right: 1px solid $BORDER; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: $TEXT2; }
h1, h2, h3, h4 { color: $TEXT1 !important; font-family: $FONT; letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: $TEXT2; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }

.hero { border: 1px solid $BORDER; border-radius: 12px; padding: 20px 26px; margin-bottom: 18px;
        background: linear-gradient(120deg, #0F2A4D 0%, #0A192F 60%); position: relative; overflow: hidden; }
.hero:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: $CYAN; }
.hero h1 { margin: 0; font-size: 2rem; font-weight: 750; }
.hero h1 span { color: $CYAN; }
.hero p { margin: 6px 0 10px 0; color: $TEXT2; font-size: 1rem; }
.chip { display: inline-block; padding: 3px 10px; margin-right: 8px; border: 1px solid $BORDER; border-radius: 999px;
        font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: $TEXT2; background: rgba(0,163,224,.08); }

.kpi { background: $PANEL; border: 1px solid $BORDER; border-left: 4px solid $CYAN; border-radius: 10px;
       padding: 16px 18px; height: 100%; }
.kpi.small { border-left-width: 3px; padding: 12px 16px; border-left-color: $SLATE; }
.kpi .label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: $TEXT3; margin-bottom: 4px; }
.kpi .value { font-size: 1.95rem; font-weight: 750; color: $TEXT1; font-variant-numeric: tabular-nums; line-height: 1.15; }
.kpi.small .value { font-size: 1.35rem; }
.kpi .sub { font-size: .84rem; color: $TEXT2; margin-top: 4px; }

.note { border: 1px solid $BORDER; border-radius: 8px; padding: 10px 14px; margin: 14px 0 6px 0;
        background: rgba(143,163,191,.07); color: $TEXT2; font-size: .9rem; }
.note b { color: $TEXT1; }
.section-title { margin: 26px 0 2px 0; font-size: 1.25rem; font-weight: 700; color: $TEXT1; }
.section-sub { color: $TEXT3; font-size: .88rem; margin-bottom: 8px; }

.vue-panel { background: $PANEL; border: 1px solid $BORDER; border-radius: 10px; padding: 14px 18px; }
.vue-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.vue-head b { color: $TEXT1; }
.tag { font-size: .68rem; letter-spacing: .08em; padding: 2px 8px; border-radius: 4px; border: 1px solid $BORDER; color: $TEXT2; }
.vue-row { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-top: 1px solid $BORDER; font-size: .9rem; }
.vue-row .name { color: $TEXT1; min-width: 150px; font-weight: 600; }
.vue-row .state { color: $TEXT2; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }

div.stButton > button, div.stDownloadButton > button { width: 100%; border-radius: 8px; font-weight: 650; }
div.stButton > button[kind="primary"] { background: $CYAN; border: none; color: #04121f; }
div.stButton > button[kind="primary"]:hover { background: #2BB8EE; color: #04121f; }
</style>
"""
).substitute(
    NAVY=NAVY, PANEL=PANEL, BORDER=BORDER, CYAN=CYAN, SLATE=SLATE,
    TEXT1=TEXT1, TEXT2=TEXT2, TEXT3=TEXT3, FONT=FONT,
)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Yardimcilar
# ----------------------------------------------------------------------
def _rgba(hex_color: str, alpha: float) -> str:
    """#RRGGBB -> rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def kpi_card(label: str, value: str, sub: str, small: bool = False) -> str:
    """KPI karti HTML'i."""
    cls = "kpi small" if small else "kpi"
    return (
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def px_trace(df: pd.DataFrame, y: str, name: str, color: str, unit: str,
             width: float = 2.0, dash: str = "solid", fill: bool = False):
    """Plotly Express ile tek seri uretip stilini ayarlar (subplot'lara aktarmak icin)."""
    tr = px.line(df, x="Timestamp", y=y).data[0]
    tr.update(
        name=name,
        showlegend=True,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate="%{y:.2f} " + unit,
    )
    if fill:
        tr.update(fill="tozeroy", fillcolor=_rgba(color, 0.18))
    return tr


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    """Tum grafikler icin ortak koyu tema ve ince, sessiz izgara."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT2, size=12),
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT1)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title_text="", font=dict(color=TEXT2)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT3), title_text="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT3))
    return fig


def shade_spikes(fig: go.Figure, windows, **kw) -> None:
    """LLM spike pencerelerini arka planda hafifce gri bantla isaretler."""
    for a, b in windows:
        fig.add_vrect(
            x0=a.to_pydatetime(), x1=(b + timedelta(minutes=15)).to_pydatetime(),
            fillcolor="rgba(169,184,207,0.08)", line_width=0, layer="below", **kw,
        )


# ----------------------------------------------------------------------
# Simulasyon (onbellekli)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate(capacity_mw, water_price, target_coc, pue, baseline_coc, tds_limit,
             elec_price, dry_assist, noise_pct, seed):
    """Veri uret + baseline / HydroSync karsilastirmasini kos."""
    df = generate_dataset(seed=seed, base_water_price=water_price)
    plant = PlantConfig(capacity_mw=capacity_mw, pue_baseline=pue)
    opt = OptimizerConfig(
        baseline_coc=baseline_coc, ai_target_coc=target_coc, ai_tds_limit_ppm=float(tds_limit),
        electricity_price_usd_kwh=elec_price, enable_dry_assist=dry_assist,
        forecast_noise_pct=noise_pct,
    )
    return run_comparison(df, plant, opt)


# ----------------------------------------------------------------------
# Grafikler
# ----------------------------------------------------------------------
def chart_water(res: pd.DataFrame, windows) -> go.Figure:
    """1) Su tuketimi: Baseline vs HydroSync (yanal alan dolgulu)."""
    long = pd.concat(
        [
            pd.DataFrame({"Timestamp": res["Timestamp"], "Scenario": SERIES_BL,
                          "Water intake (L/s)": res["BL_Makeup_Lps"]}),
            pd.DataFrame({"Timestamp": res["Timestamp"], "Scenario": SERIES_HS,
                          "Water intake (L/s)": res["HS_Makeup_Lps"]}),
        ],
        ignore_index=True,
    )
    fig = px.line(long, x="Timestamp", y="Water intake (L/s)", color="Scenario", color_discrete_map=COLOR_MAP)
    for tr in fig.data:
        color = COLOR_MAP[tr.name]
        tr.update(line=dict(color=color, width=2), fill="tozeroy", fillcolor=_rgba(color, 0.16),
                  hovertemplate="%{y:.2f} L/s")
    shade_spikes(fig, windows)
    style_fig(fig, 380)
    fig.update_yaxes(title_text="Cooling-tower water intake (L/s)", rangemode="tozero")
    return fig


def chart_predictive(res: pd.DataFrame, windows, window_idx: int) -> go.Figure:
    """2) Ongoru gorunumu: isi yuku (sol eksen) vs blowdown (sag eksen)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sol eksen (MW): gercek isi yuku, 30 dk ongoru, HydroSync'in hazirladigi sogutma kapasitesi
    fig.add_trace(px_trace(res, "Heat_Load_MW", "Heat load · MW (left)", SLATE, "MW", 1.5, fill=True),
                  secondary_y=False)
    fig.add_trace(px_trace(res, "Heat_Load_Pred_MW", "30-min forecast · MW (left)", SLATE, "MW", 1.2, "dot"),
                  secondary_y=False)
    fig.add_trace(px_trace(res, "HS_Cooling_Staged_MW", "HydroSync staged cooling · MW (left)",
                           CYAN_CHART, "MW", 2.0), secondary_y=False)

    # Sag eksen (L/s): blowdown debileri
    fig.add_trace(px_trace(res, "BL_Blowdown_Lps", "Baseline blowdown · L/s (right)", AMBER_CHART, "L/s", 1.6),
                  secondary_y=True)
    fig.add_trace(px_trace(res, "HS_Blowdown_Lps", "HydroSync blowdown · L/s (right)", CYAN_CHART, "L/s",
                           1.6, "dash"), secondary_y=True)

    shade_spikes(fig, windows, row=1, col=1)
    for i, (a, b) in enumerate(windows, start=1):
        mid = (a + (b - a) / 2).to_pydatetime()
        fig.add_annotation(x=mid, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"LLM spike {i}", font=dict(color=TEXT3, size=11))

    style_fig(fig, 470)
    fig.update_layout(margin=dict(l=8, r=8, t=70, b=8), legend=dict(y=1.12))
    fig.update_yaxes(title_text="Heat load / cooling (MW)", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Blowdown (L/s)", rangemode="tozero", showgrid=False, secondary_y=True)

    if window_idx > 0:  # secilen spike'a yakinlas
        a, b = windows[window_idx - 1]
        fig.update_xaxes(range=[(a - timedelta(hours=3)).isoformat(), (b + timedelta(hours=3)).isoformat()])
    return fig


def chart_tds_coc(res: pd.DataFrame, windows, baseline_limit: float, hs_limit: float, max_coc: float) -> go.Figure:
    """3) TDS ve CoC degisimi (iki ayri panel, ortak zaman ekseni)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.6, 0.4])
    fig.add_trace(px_trace(res, "BL_TDS_ppm", f"{SERIES_BL} · TDS", AMBER_CHART, "ppm"), row=1, col=1)
    fig.add_trace(px_trace(res, "HS_TDS_ppm", f"{SERIES_HS} · TDS", CYAN_CHART, "ppm"), row=1, col=1)
    fig.add_trace(px_trace(res, "BL_CoC", f"{SERIES_BL} · CoC", AMBER_CHART, "×"), row=2, col=1)
    fig.add_trace(px_trace(res, "HS_CoC", f"{SERIES_HS} · CoC", CYAN_CHART, "×"), row=2, col=1)
    # Ikinci paneldeki izler icin ayri legend girdisi acma
    for tr in fig.data[2:]:
        tr.update(showlegend=False)

    fig.add_hline(y=baseline_limit, line_dash="dash", line_width=1, line_color=AMBER_CHART, row=1, col=1,
                  annotation_text=f"Baseline scaling limit · {baseline_limit:,.0f} ppm",
                  annotation_position="top left", annotation_font=dict(color=TEXT2, size=11))
    fig.add_hline(y=hs_limit, line_dash="dash", line_width=1, line_color=CYAN_CHART, row=1, col=1,
                  annotation_text=f"HydroSync conditioned limit · {hs_limit:,.0f} ppm",
                  annotation_position="top left", annotation_font=dict(color=TEXT2, size=11))
    fig.add_hline(y=max_coc, line_dash="dot", line_width=1, line_color=SLATE, row=2, col=1,
                  annotation_text=f"Max CoC · {max_coc:.1f}", annotation_position="top left",
                  annotation_font=dict(color=TEXT2, size=11))

    shade_spikes(fig, windows, row="all", col=1)
    style_fig(fig, 560)
    fig.update_yaxes(title_text="Tower water TDS (ppm)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="CoC (C_tower / C_makeup)", rangemode="tozero", row=2, col=1)
    return fig


# ----------------------------------------------------------------------
# Xylem Vue entegrasyonu (SIMULASYON - hicbir dis cagri yapilmaz)
# ----------------------------------------------------------------------
def build_vue_payload(params: dict, m: dict, res: pd.DataFrame) -> dict:
    """HydroSync parametrelerini Xylem Vue'ya gonderilecekmis gibi JSON payload'a cevirir."""
    by_hour = (
        res.assign(h=pd.to_datetime(res["Timestamp"]).dt.hour)
        .groupby("h")["HS_Mode"].agg(lambda s: s.mode().iat[0])
    )
    schedule = {
        "concentrate_hours": [int(h) for h, v in by_hour.items() if v == "CONCENTRATE"],
        "flush_hours": [int(h) for h, v in by_hour.items() if v == "FLUSH"],
        "normal_hours": [int(h) for h, v in by_hour.items() if v == "NORMAL"],
    }
    return {
        "schema": "hydrosync.setpoints/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "site": {"name": "Demo AI Data Center", "asset_id": "CT-01", "capacity_mw": params["capacity_mw"]},
        "parameters": {
            "target_coc": params["target_coc"],
            "max_coc": 7.5,
            "tds_limit_ppm": params["tds_limit"],
            "forecast_horizon_min": 30,
            "spike_jump_threshold_pct": 15,
            "dry_assist_enabled": params["dry_assist"],
            "water_price_usd_m3": params["water_price"],
        },
        "blowdown_schedule": schedule,
        "expected_kpis": {
            "water_saved_pct": round(m["water_saved_pct"], 2),
            "wue_l_per_kwh": round(m["wue_hydrosync_l_per_kwh"], 3),
            "avg_coc": round(m["avg_coc_hydrosync"], 2),
            "cost_saved_usd_per_week": round(m["cost_saved_usd"], 2),
        },
        "note": "SIMULATED PAYLOAD - not transmitted to any external service",
    }


def payload_signature(payload: dict) -> str:
    """Zaman damgasi haric payload imzasi (parametre degisti mi kontrolu icin)."""
    body = {k: v for k, v in payload.items() if k != "generated_at"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


VUE_STEPS = [
    ("Authenticating with Xylem Vue gateway", "Access token issued (simulated)"),
    ("Validating parameter schema", "hydrosync.setpoints/v1 · OK"),
    ("Mapping assets", "Data center → Cooling tower CT-01 · 1 asset mapped"),
    ("Pushing setpoints and blowdown schedule", "Payload accepted · 24 hourly modes"),
    ("Waiting for controller acknowledgement", "Controller ACK received (simulated)"),
]


def vue_row(color: str, name: str, state: str) -> str:
    """Durum paneli satiri: renkli nokta + metin etiketi (renk tek basina anlam tasimaz)."""
    return (f'<div class="vue-row"><span class="dot" style="background:{color}"></span>'
            f'<span class="name">{name}</span><span class="state">{state}</span></div>')


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scenario controls")
    capacity_mw = st.slider("Data center capacity (MW)", min_value=1.0, max_value=10.0, value=8.0, step=0.5,
                            help="Maximum IT power. Actual IT load follows GPU utilization (idle ≈ 25%).")
    water_price = st.slider("Base water price ($/m³)", min_value=0.5, max_value=10.0, value=2.5, step=0.1,
                            help="Base tariff. A time-of-use profile (night ×0.75 … peak ×1.6) is applied on top.")
    target_coc = st.slider("HydroSync target CoC", min_value=3.0, max_value=7.5, value=6.0, step=0.1,
                           help="Normal-hour setpoint. During expensive peak hours HydroSync may rise to CoC 7.5.")
    with st.expander("Advanced assumptions"):
        pue = st.slider("Baseline PUE", min_value=1.10, max_value=1.80, value=1.35, step=0.01)
        baseline_coc = st.slider("Baseline design CoC", min_value=3.0, max_value=5.0, value=4.0, step=0.1)
        tds_limit = st.slider("HydroSync conditioned TDS limit (ppm)", min_value=1800, max_value=3200,
                              value=2800, step=50,
                              help="Assumption: predictive antiscalant dosing & water conditioning raises the "
                                   "safe TDS limit above the 1,500 ppm of a standard program.")
        elec_price = st.slider("Electricity price ($/kWh)", min_value=0.03, max_value=0.30, value=0.08, step=0.01)
        dry_assist = st.checkbox("Dynamic PUE↔WUE dry-assist", value=True,
                                 help="Shifts part of the heat rejection to the dry path when water is worth more "
                                      "than the extra fan/pump electricity.")
        noise_pct = st.slider("Forecast error (GPU %-points, 1σ)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        seed = st.number_input("Synthetic data seed", min_value=0, max_value=9999, value=42, step=1)
    st.caption("All data is synthetic (672 points · 7 days · 15-min). Simulation only.")

results, m = simulate(capacity_mw, water_price, target_coc, pue, baseline_coc, tds_limit,
                      elec_price, dry_assist, noise_pct, int(seed))
windows = spike_windows(results)

# ----------------------------------------------------------------------
# Baslik
# ----------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Hydro<span>Sync</span> AI</h1>
  <p>Predictive water optimization for AI data-center cooling towers: forecast the heat shock,
     shift blowdown to cheap hours, and raise cycles of concentration safely.</p>
  <span class="chip">Water Quantity</span><span class="chip">B2B Platform</span>
  <span class="chip">Xylem Innovation Challenge 2026</span>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# KPI kartlari
# ----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi_card("Water saved", f"{m['water_saved_l']:,.0f} L",
                     f"{m['water_saved_pct']:.1f}% less than baseline · {m['sim_days']:.0f}-day run"),
            unsafe_allow_html=True)
c2.markdown(kpi_card("Cost saved", f"${m['cost_saved_usd']:,.0f}",
                     f"{m['cost_saved_pct']:.1f}% · net of ${m['extra_electricity_cost_usd']:,.0f} extra electricity"),
            unsafe_allow_html=True)
c3.markdown(kpi_card("WUE improvement", f"{m['wue_improvement_pct']:.1f}%",
                     f"{m['wue_baseline_l_per_kwh']:.2f} → {m['wue_hydrosync_l_per_kwh']:.2f} L/kWh"),
            unsafe_allow_html=True)
c4.markdown(kpi_card("Average CoC", f"{m['avg_coc_hydrosync']:.2f}",
                     f"Baseline {m['avg_coc_baseline']:.2f} · +{m['avg_coc_hydrosync'] - m['avg_coc_baseline']:.2f}"),
            unsafe_allow_html=True)

# Tasarrufun kaynagi (seffaflik)
dry_pts = m["water_saved_pct"] - m["water_saved_pct_without_dry_assist"]
if dry_assist and dry_pts > 0.05:
    breakdown = (f"<b>Where the water savings come from:</b> {m['water_saved_pct_without_dry_assist']:.1f}% from "
                 f"CoC, tariff-aware blowdown and spike anticipation, plus {dry_pts:.1f} pts from dynamic "
                 f"dry-assist, which uses {m['extra_electricity_kwh']:,.0f} kWh of extra electricity "
                 f"(PUE {m['pue_baseline']:.3f} → {m['pue_hydrosync']:.3f}).")
else:
    breakdown = (f"<b>Where the water savings come from:</b> {m['water_saved_pct_without_dry_assist']:.1f}% from "
                 f"CoC, tariff-aware blowdown and spike anticipation. Dry-assist is "
                 f"{'off' if not dry_assist else 'not economical at this water price'}.")
st.markdown(f'<div class="note">{breakdown}</div>', unsafe_allow_html=True)

# Ikincil gostergeler
carbon = m["carbon_net_avoided_kg"]
carbon_sub = ("Net avoided: water-related emissions exceed added power"
              if carbon >= 0 else "Net increase: dry-assist power exceeds water-related emissions avoided")
s1, s2, s3, s4 = st.columns(4)
s1.markdown(kpi_card("LLM spikes anticipated", f"{m['spikes_anticipated']} / {m['spikes_total']}",
                     "Detected 30 min before the load rise", small=True), unsafe_allow_html=True)
s2.markdown(kpi_card("Peak TDS headroom to limit", f"{m['scaling_headroom_hydrosync_pct']:.1f}%",
                     f"Baseline {m['scaling_headroom_baseline_pct']:.1f}% (each vs. its own limit)", small=True),
            unsafe_allow_html=True)
s3.markdown(kpi_card("Effective PUE", f"{m['pue_hydrosync']:.3f}",
                     f"Baseline {m['pue_baseline']:.3f}", small=True), unsafe_allow_html=True)
s4.markdown(kpi_card("Net carbon impact", f"{carbon:+,.0f} kg CO₂e", carbon_sub, small=True),
            unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Grafik 1
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">1 · Water consumption: Baseline vs HydroSync AI</div>'
            '<div class="section-sub">Cooling-tower makeup water over 7 days. Shaded bands mark LLM training spikes.</div>',
            unsafe_allow_html=True)
st.plotly_chart(chart_water(results, windows), theme=None)

# ----------------------------------------------------------------------
# Grafik 2
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">2 · Predictive action view</div>'
            '<div class="section-sub">HydroSync forecasts the heat load 30 minutes ahead and stages cooling before the '
            'spike lands, while its blowdown stays deferred. Left axis: MW. Right axis: L/s.</div>',
            unsafe_allow_html=True)
options = ["Full week"] + [f"Zoom: spike {i} · {a:%a %d %b %H:%M}" for i, (a, _) in enumerate(windows, start=1)]
choice = st.selectbox("View", options, index=min(1, len(options) - 1) if len(options) > 1 else 0,
                      label_visibility="collapsed")
st.plotly_chart(chart_predictive(results, windows, options.index(choice)), theme=None)

# ----------------------------------------------------------------------
# Grafik 3
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">3 · TDS and CoC dynamics</div>'
            '<div class="section-sub">HydroSync concentrates water during expensive peak hours and flushes during cheap '
            'night hours, staying inside its conditioned TDS limit.</div>', unsafe_allow_html=True)
st.plotly_chart(chart_tds_coc(results, windows, 1500.0, float(tds_limit), 7.5), theme=None)

with st.expander("Data table view"):
    show_cols = ["Timestamp", "GPU_Utilization", "Ambient_WetBulb_Temp", "Water_Tariff_Price", "Incoming_Water_TDS",
                 "BL_Makeup_Lps", "HS_Makeup_Lps", "BL_Blowdown_Lps", "HS_Blowdown_Lps", "BL_TDS_ppm", "HS_TDS_ppm",
                 "BL_CoC", "HS_CoC", "HS_Mode", "HS_Dry_Fraction"]
    table = results[show_cols].copy()
    num_cols = table.select_dtypes("number").columns
    table[num_cols] = table[num_cols].round(2)
    st.dataframe(table, height=300)
    st.download_button("Download full results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       file_name="hydrosync_results.csv", mime="text/csv")

# ----------------------------------------------------------------------
# Xylem Vue entegrasyonu
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">Xylem Vue integration</div>'
            '<div class="section-sub">Simulated B2B hand-off: HydroSync parameters are packaged and pushed to the '
            'Xylem Vue platform. No external call is made in this demo.</div>', unsafe_allow_html=True)

params = {"capacity_mw": capacity_mw, "target_coc": target_coc, "tds_limit": int(tds_limit),
          "dry_assist": bool(dry_assist), "water_price": water_price}
payload = build_vue_payload(params, m, results)
sig = payload_signature(payload)

if "vue" not in st.session_state:
    st.session_state["vue"] = {"synced": False, "sig": None, "ts": None}

left, right = st.columns([1, 1])
with left:
    if st.button("Export Parameters to Xylem Vue API", type="primary"):
        with st.status("Exporting to Xylem Vue (simulated)…", expanded=True) as status:
            for step, detail in VUE_STEPS:
                st.write(f"{step}…")
                time.sleep(0.35)
                st.write(f"✓ {detail}")
            status.update(label="Export complete · controller acknowledged", state="complete", expanded=False)
        st.session_state["vue"] = {"synced": True, "sig": sig, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    st.download_button("Download payload (JSON)", json.dumps(payload, indent=2).encode("utf-8"),
                       file_name="hydrosync_xylem_vue_payload.json", mime="application/json")
    st.code(json.dumps(payload, indent=2), language="json")

with right:
    vue = st.session_state["vue"]
    if not vue["synced"]:
        rows = [
            vue_row(IDLE, "API gateway", "Ready · simulated endpoint"),
            vue_row(IDLE, "Asset registry", "1 site · 1 cooling tower (CT-01) mapped"),
            vue_row(IDLE, "Setpoint push", "Not exported yet"),
            vue_row(IDLE, "Telemetry stream", "Idle"),
        ]
        headline, tag = "Not synced", "SIMULATED"
    elif vue["sig"] == sig:
        rows = [
            vue_row(OK, "API gateway", "Connected · simulated"),
            vue_row(OK, "Asset registry", "1 site · 1 cooling tower (CT-01) mapped"),
            vue_row(OK, "Setpoint push", f"Acknowledged · {vue['ts']} · rev {vue['sig']}"),
            vue_row(OK, "Telemetry stream", "Live · 15-min cadence (simulated)"),
        ]
        headline, tag = "In sync", "SIMULATED"
    else:
        rows = [
            vue_row(OK, "API gateway", "Connected · simulated"),
            vue_row(OK, "Asset registry", "1 site · 1 cooling tower (CT-01) mapped"),
            vue_row(WARN, "Setpoint push", f"Out of date · parameters changed since {vue['ts']}"),
            vue_row(OK, "Telemetry stream", "Live · 15-min cadence (simulated)"),
        ]
        headline, tag = "Out of date: re-export to update", "SIMULATED"
    st.markdown(
        f'<div class="vue-panel"><div class="vue-head"><b>Integration status · {headline}</b>'
        f'<span class="tag">{tag}</span></div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Model notlari
# ----------------------------------------------------------------------
with st.expander("Model notes and assumptions"):
    st.markdown(
        f"""
- **Synthetic data.** 7 days at 15-minute resolution (672 points). GPU load, wet-bulb temperature, time-of-use water
  tariff and incoming water TDS are generated, not measured.
- **Physics.** Q = P_IT · PUE, E = 0.0018 · Q_tons (L/s), B = E / (CoC − 1), M = E + B, and the TDS mass balance
  C(t+1) = C(t) + Δt · (M·C_makeup − B·C_tower) / V.
- **Baseline.** Fixed-rate blowdown valve opens above 1,500 ppm and closes at 1,450 ppm. It ignores GPU load,
  spikes and tariffs.
- **HydroSync AI.** Forecasts GPU load 30 min ahead (a noisy look-ahead stands in for a model trained on Slurm or
  Kubernetes queue data), stages cooling before spikes, concentrates to CoC {7.5} in expensive hours and flushes in cheap
  or cool hours.
- **Conditioned TDS limit.** The {tds_limit:,} ppm limit is a design assumption (predictive antiscalant dosing and water
  conditioning). It needs field validation before it is claimed in practice.
- **Dry-assist (PUE↔WUE).** Moves part of the heat rejection to a dry path when the water saved is worth more than the
  extra electricity. It lowers water use but raises PUE and can raise carbon.
- **Carbon.** Water-related emissions use {OptimizerConfig().water_embedded_kwh_per_m3:.1f} kWh/m³ embedded energy and
  {OptimizerConfig().grid_emission_kg_per_kwh:.2f} kg CO₂e/kWh. Both are adjustable assumptions.
- **Annualized (indicative):** ≈ {m['annual_water_saved_m3']:,.0f} m³ of water and ${m['annual_cost_saved_usd']:,.0f} per
  year for this site, scaled linearly from the 7-day run.
"""
    )
