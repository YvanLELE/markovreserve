from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import generate_demo_portfolio, validate_portfolio, standardize_columns
from src.states import build_state_panel
from src.transitions import TransitionModel
from src.payments import PaymentModel
from src.reserving import simulate_reserves, risk_summary
from src.chainladder import cumulative_triangle, chain_ladder_reserve, benchmark_reserves
from src.reporting import to_excel_bytes, build_pdf_report

st.set_page_config(page_title="MarkovReserve", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# Style professionnel : fond blanc, palette navy / actuarial analytics
# -----------------------------------------------------------------------------
NAVY = "#0B1F3A"
NAVY_2 = "#12355B"
BLUE = "#1F77B4"
CYAN = "#17A2B8"
TEAL = "#0E9384"
GREEN = "#039855"
ORANGE = "#F79009"
RED = "#D92D20"
GREY = "#667085"
LIGHT = "#F8FAFC"
GRID = "#E5E7EB"

PALETTE = [NAVY, BLUE, CYAN, TEAL, ORANGE, RED, "#6941C6", "#475467", "#2E90FA"]
STATE_COLORS = {
    "IBNR": "#BFD7EA",
    "RBNP": "#7FB3D5",
    "RBNS1": "#2878B5",
    "RBNS2": "#1F5F99",
    "RBNS3": "#174C7E",
    "RBNS4": "#123B63",
    "RBNS5+": NAVY,
    "Closed0": "#98A2B3",
    "Closed+": GREEN,
}
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.markdown(f"""
<style>
:root {{ --navy:{NAVY}; --line:#E5E7EB; --muted:#667085; }}
.stApp {{ background: #FFFFFF; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #E5E7EB; }}
[data-testid="stSidebar"] * {{ color: #182230; }}
.block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.02em; }}
[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
    padding: 14px 16px; box-shadow: 0 8px 22px rgba(11,31,58,.06);
}}
[data-testid="stMetricLabel"] {{ color:#667085; }}
[data-testid="stMetricValue"] {{ color:{NAVY}; font-weight:800; }}
div[data-testid="stTabs"] button p {{ font-weight: 700; color: {NAVY}; }}
.info-card {{ background:#F8FAFC; border:1px solid #E5E7EB; border-radius:16px; padding:16px; }}
.hero-card {{
    background: linear-gradient(135deg, #FFFFFF 0%, #F5F8FF 100%);
    border: 1px solid #D9E2F3; border-radius: 22px; padding: 24px;
    box-shadow: 0 12px 32px rgba(11,31,58,.08); margin-bottom: 16px;
}}
.small-note {{ color:#667085; font-size:0.90rem; }}

.kpi-grid {{display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 12px 0 18px 0;}}
.kpi-card {{background:#FFFFFF; border:1px solid #E5E7EB; border-radius:16px; padding:14px 16px; box-shadow:0 8px 22px rgba(11,31,58,.06);}}
.kpi-label {{color:#667085; font-size:.86rem; font-weight:700; margin-bottom:6px;}}
.kpi-value {{color:#0B1F3A; font-size:1.24rem; line-height:1.22; font-weight:850; word-break:normal; overflow-wrap:anywhere;}}
.kpi-sub {{color:#667085; font-size:.80rem; margin-top:4px;}}

.brand-row {{display:flex; align-items:center; gap:14px;}}
.brand-logo {{width:58px; min-width:58px; height:58px;}}
.sidebar-brand {{padding:14px; border:1px solid #D9E2F3; border-radius:18px; background:linear-gradient(135deg,#FFFFFF,#F5F8FF); margin-bottom:14px;}}
.sidebar-brand-title {{font-size:1.35rem; line-height:1; font-weight:900; color:#0B1F3A; letter-spacing:-0.04em; white-space:nowrap;}}
.sidebar-brand-sub {{font-size:.78rem; color:#667085; font-weight:700; margin-top:4px;}}
.score-badge {{display:inline-block; border-radius:999px; padding:6px 10px; font-weight:900;}}
.score-green {{background:#ECFDF3; color:#027A48; border:1px solid #ABEFC6;}}
.score-orange {{background:#FFFAEB; color:#B54708; border:1px solid #FEDF89;}}
.score-red {{background:#FEF3F2; color:#B42318; border:1px solid #FECDCA;}}
</style>
""", unsafe_allow_html=True)


MARKOVRESERVE_LOGO = """
<svg class="brand-logo" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <rect x="14" y="18" width="92" height="84" rx="22" fill="#0B1F3A"/>
  <path d="M35 78 L35 42 L52 62 L69 42 L86 78" stroke="#FFFFFF" stroke-width="9" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="35" cy="42" r="7" fill="#17A2B8"/>
  <circle cx="69" cy="42" r="7" fill="#F79009"/>
  <circle cx="86" cy="78" r="7" fill="#17A2B8"/>
</svg>
"""


def data_quality_score(df: pd.DataFrame, info: dict) -> tuple[int, list[str]]:
    score = 100
    findings = []
    required = ['AY']
    if not any(c in df.columns for c in ['ID', 'ClNr']):
        score -= 10; findings.append('Claim identifier was generated or missing from the imported file.')
    for c in required:
        if c not in df.columns:
            score -= 35; findings.append(f'Mandatory column {c} is missing.')
    if info.get('pay_columns', 0) == 0:
        score -= 40; findings.append('No payment development columns were detected.')
    if info.get('issues'):
        score -= min(35, 10 * len(info['issues'])); findings.extend(info['issues'])
    pay_cols_detected = [c for c in df.columns if c.startswith('Pay')]
    if pay_cols_detected:
        miss = df[pay_cols_detected].isna().mean().mean()
        if miss > 0.05:
            score -= 10; findings.append('More than 5% missing payment cells detected.')
    if 'RepDel' not in df.columns:
        score -= 5; findings.append('Reporting delay column RepDel is missing; default values may be used.')
    if not any(c.startswith('Open') for c in df.columns):
        score -= 5; findings.append('Open/closed indicators are not available; closure is inferred from payments.')
    score = max(0, min(100, int(score)))
    if not findings:
        findings.append('No material structural issue detected in the imported data.')
    return score, findings


def score_class(score: int) -> str:
    if score >= 85:
        return 'score-green'
    if score >= 65:
        return 'score-orange'
    return 'score-red'


def executive_insights(reserve_mean, ibnr_mean, cl_reserve, cdr_mean, snap: pd.DataFrame, sims: pd.DataFrame) -> list[str]:
    insights = []
    ibnr_share = ibnr_mean / reserve_mean if reserve_mean else 0
    gap = (cl_reserve - reserve_mean) / reserve_mean if reserve_mean else np.nan
    rbns5_share = 0
    if snap is not None and len(snap):
        rbns5_share = (snap['State'].eq('RBNS5+').sum() / len(snap))
    insights.append(f"The individual semi-Markov Best Estimate is {reserve_mean:,.0f}, with IBNR representing {ibnr_share:.1%} of the total reserve.".replace(',', ' '))
    if np.isfinite(gap):
        if gap > 0.10:
            insights.append(f"Chain-Ladder is {gap:.1%} higher than the individual estimate, suggesting that aggregated development factors may be conservative for this portfolio.")
        elif gap < -0.10:
            insights.append(f"Chain-Ladder is {abs(gap):.1%} lower than the individual estimate; long-running claims and payment assumptions should be reviewed.")
        else:
            insights.append(f"Chain-Ladder and individual estimates are broadly aligned ({gap:.1%} difference), which supports model consistency.")
    if cdr_mean < 0:
        insights.append("The mean one-year CDR is negative, indicating adverse expected one-year development under the selected convention.")
    else:
        insights.append("The mean one-year CDR is positive, indicating favourable expected one-year development under the selected convention.")
    if rbns5_share > 0.08:
        insights.append(f"RBNS5+ claims account for {rbns5_share:.1%} of the valuation snapshot, highlighting the importance of long-duration claim monitoring.")
    if sims is not None and len(sims):
        ratio = np.quantile(sims['TotalReserve'], .995) / max(sims['TotalReserve'].mean(), 1)
        insights.append(f"The reserve VaR-to-Best-Estimate ratio is {ratio:.2f}, providing a compact view of reserve tail risk.")
    return insights

def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#182230", size=13),
        title=dict(font=dict(color=NAVY, size=18)),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=95),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    if height:
        fig.update_layout(height=height)
    return fig


def add_vline(fig: go.Figure, x: float, text: str, color: str = NAVY):
    fig.add_vline(x=x, line_width=2, line_dash="dash", line_color=color)
    fig.add_annotation(x=x, y=1.02, yref="paper", text=text, showarrow=False,
                       font=dict(color=color, size=12), bgcolor="white")


def empirical_cdf(values: pd.Series | np.ndarray) -> pd.DataFrame:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    if len(x) == 0:
        return pd.DataFrame({"x": [], "cdf": []})
    return pd.DataFrame({"x": x, "cdf": np.arange(1, len(x) + 1) / len(x)})


def pp_dataframe(values, model) -> pd.DataFrame:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0 or model is None:
        return pd.DataFrame({"Theoretical": [], "Empirical": []})
    u = model.cdf(x)
    u = np.sort(u[np.isfinite(u)])
    return pd.DataFrame({"Theoretical": u, "Empirical": np.arange(1, len(u) + 1) / len(u)})


def compute_payment_events(panel: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1_rows, lr_rows = [], []
    for cid, sub in panel.sort_values("DY").groupby("ID"):
        sub = sub.sort_values("DY")
        pays = sub["Pay"].to_numpy(float)
        cum = np.cumsum(pays)
        idx = np.where(pays > 0)[0]
        if len(idx) == 0:
            continue
        p1 = float(pays[idx[0]])
        p1_rows.append({"ID": cid, "P1": p1, "Class": "P1 ≥ threshold" if p1 >= threshold else "P1 < threshold"})
        for k, j in enumerate(idx[1:], start=1):
            prev = float(cum[idx[k - 1]])
            cur = float(cum[j])
            if prev > 0 and cur > prev:
                lr_rows.append({
                    "ID": cid, "LambdaIndex": min(k, 5), "LR": cur / prev,
                    "P1": p1, "Class": "P1 ≥ threshold" if p1 >= threshold else "P1 < threshold"
                })
    return pd.DataFrame(p1_rows), pd.DataFrame(lr_rows)


def transition_heatmap(trans_table: pd.DataFrame) -> go.Figure:
    if trans_table.empty:
        return go.Figure()
    tmp = trans_table.copy()
    tmp["From"] = tmp["Previous_State"] + " | d=" + tmp["DurationCat"].astype(str)
    pivot = tmp.pivot_table(index="From", columns="Current_State", values="Probability", aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale="Blues", aspect="auto",
                    title="Transition probability heatmap")
    fig.update_coloraxes(colorbar_title="Prob.")
    return style_fig(fig, height=520)


def sankey_transitions(aug: pd.DataFrame) -> go.Figure:
    if aug.empty:
        return go.Figure()
    flows = aug.groupby(["Previous_State", "Current_State"]).size().reset_index(name="Count")
    flows = flows[flows["Previous_State"] != flows["Current_State"]]
    flows = flows.sort_values("Count", ascending=False).head(25)
    labels = sorted(set(flows["Previous_State"]).union(flows["Current_State"]), key=lambda x: list(STATE_COLORS).index(x) if x in STATE_COLORS else 99)
    idx = {v: i for i, v in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=18,
                  color=[STATE_COLORS.get(x, GREY) for x in labels]),
        link=dict(source=[idx[x] for x in flows["Previous_State"]],
                  target=[idx[x] for x in flows["Current_State"]],
                  value=flows["Count"], color="rgba(31,119,180,.24)")
    )])
    fig.update_layout(title="Observed transition flows between states", font=dict(size=12, color="#182230"),
                      paper_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10), height=430)
    return fig


def hist_with_lines(df: pd.DataFrame, x: str, title: str, color: str = BLUE, nbins: int = 70, lines: dict | None = None) -> go.Figure:
    fig = px.histogram(df, x=x, nbins=nbins, marginal="box", title=title, opacity=0.86)
    fig.update_traces(marker_color=color, marker_line_color="white", marker_line_width=.6)
    if lines:
        for label, val in lines.items():
            add_vline(fig, val, label, NAVY if "Moy" in label or "BE" in label else RED)
    return style_fig(fig, height=430)


def risk_table(series: pd.Series, unfavorable_lower_tail: bool = False) -> pd.DataFrame:
    arr = np.asarray(series, dtype=float)
    q005 = np.quantile(arr, .005) if len(arr) else np.nan
    q05 = np.quantile(arr, .05) if len(arr) else np.nan
    q95 = np.quantile(arr, .95) if len(arr) else np.nan
    q995 = np.quantile(arr, .995) if len(arr) else np.nan
    if unfavorable_lower_tail:
        tvar = arr[arr <= q005].mean() if np.any(arr <= q005) else np.nan
        var = q005
    else:
        tvar = arr[arr >= q995].mean() if np.any(arr >= q995) else np.nan
        var = q995
    return pd.DataFrame({
        "Indicator": ["Mean", "Standard deviation", "Q0.5%", "Q5%", "Median", "Q95%", "Q99.5%", "Selected VaR", "Selected TVaR"],
        "Value": [np.mean(arr), np.std(arr, ddof=1), q005, q05, np.median(arr), q95, q995, var, tvar]
    })


# -----------------------------------------------------------------------------
# Header and parameters
# -----------------------------------------------------------------------------
LOGO_PATH = Path(__file__).parent / "assets" / "markovreserve_logo.svg"

hero_logo, hero_text = st.columns([0.07, 0.93], vertical_alignment="center")
with hero_logo:
    st.image(str(LOGO_PATH), width=64)
with hero_text:
    st.title("MarkovReserve")
    st.caption("Semi-Markov individual claims reserving for non-life insurance portfolios.")

with st.sidebar:
    st.image(str(LOGO_PATH), width=68)
    st.markdown("### MarkovReserve")
    st.caption("Multi-State Reserving Engine")
    st.header("Settings")
    mode = st.radio("Data source", ["Demo portfolio", "Import a file"], index=0)
    if mode == "Demo portfolio":
        n_claims = st.slider("Number of claims", 500, 30000, 5000, step=500)
        seed_data = st.number_input("Data seed", value=100, min_value=1)
    else:
        uploaded = st.file_uploader("Claims file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    rbns_cap = st.selectbox("RBNS grouping", [3, 4, 5, 6], index=2)
    threshold = st.number_input("Large first-payment threshold", value=50_000, step=5_000)
    smoothing = st.number_input("Transition smoothing", value=0.5, min_value=0.0, step=0.1)
    n_sims = st.slider("Monte Carlo simulations", 100, 10000, 1000, step=100)
    seed_sim = st.number_input("Simulation seed", value=123, min_value=1)
    st.markdown("---")
    with st.expander("Client & report metadata", expanded=False):
        client_name = st.text_input("Client name", value="Demo Insurance")
        portfolio_name = st.text_input("Portfolio name", value="Motor Liability Portfolio")
        prepared_by = st.text_input("Prepared by", value="Yvan LELE")
        report_currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "XAF", "CAD"], index=0)
        report_date = st.date_input("Report date")
    with st.expander("Economic assumptions", expanded=False):
        inflation_rate = st.number_input("Annual claims inflation", value=0.0, step=0.25, format="%.2f") / 100
        discount_rate = st.number_input("Annual discount rate", value=0.0, step=0.25, format="%.2f") / 100
        avg_payment_duration = st.number_input("Approx. average future payment duration", value=2.0, min_value=0.0, step=0.5)
    st.markdown("---")
    st.caption("START/STOP: Streamlit may rerun calculations after parameter changes. Use START to recompute and STOP to freeze the current view.")
    cstart, cstop = st.columns(2)
    start_clicked = cstart.button("▶ START", type="primary", use_container_width=True)
    stop_clicked = cstop.button("■ STOP", use_container_width=True)
    if "engine_started" not in st.session_state:
        st.session_state.engine_started = True
    if start_clicked:
        st.session_state.engine_started = True
        st.cache_data.clear()
    if stop_clicked:
        st.session_state.engine_started = False
    st.markdown("---")
    st.caption("Convention: Closed0 = closure without terminal payment; Closed+ = closure with terminal payment.")


@st.cache_data(show_spinner=False)
def load_data(mode, n_claims, seed_data, uploaded_bytes):
    if mode == "Demo portfolio":
        return generate_demo_portfolio(n_claims=n_claims, seed=int(seed_data))
    if uploaded_bytes is None:
        return None
    name = getattr(uploaded_bytes, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_bytes)
    return pd.read_csv(uploaded_bytes)

uploaded_bytes = uploaded if mode == "Import a file" and 'uploaded' in locals() else None
df_raw = load_data(mode, n_claims if mode == "Demo portfolio" else 0,
                   seed_data if mode == "Demo portfolio" else 0, uploaded_bytes)
if df_raw is None:
    st.info("Import a CSV/Excel file or use the demo portfolio.")
    st.stop()

try:
    df = standardize_columns(df_raw)
except Exception as e:
    st.error(f"Format error: {e}")
    st.stop()

info = validate_portfolio(df)
max_dev = info['pay_columns']
valuation_year = st.sidebar.slider("Valuation year", int(info['ay_min']), int(info['ay_max'] + max_dev - 1), int(info['ay_max']))

if not st.session_state.get("engine_started", True):
    st.warning("Engine stopped. Adjust the parameters and click START in the sidebar to rerun the analysis.")
    st.stop()


@st.cache_data(show_spinner=True)
def build_all(df, rbns_cap, smoothing, threshold, valuation_year, n_sims, seed_sim):
    panel, aug = build_state_panel(df, rbns_cap=rbns_cap)
    tm = TransitionModel.fit(aug, smoothing=smoothing)
    pm = PaymentModel.fit(df, panel, threshold=threshold)
    sims, snap, ibnr = simulate_reserves(df, panel, valuation_year, tm, pm, n_sims=n_sims, seed=seed_sim)
    tri = cumulative_triangle(panel, valuation_year)
    cl, factors = chain_ladder_reserve(tri)
    bench_summary, bench_by_ay = benchmark_reserves(tri, panel, valuation_year)
    p1_events, lr_events = compute_payment_events(panel, threshold)
    return panel, aug, tm.table(), pm.summary_table(), pm, sims, snap, ibnr, tri, cl, factors, bench_summary, bench_by_ay, p1_events, lr_events

with st.spinner("Building states, calibrating models, generating charts and simulations..."):
    panel, aug, trans_table, pay_summary, pm, sims, snap, ibnr, tri, cl, factors, bench_summary, bench_by_ay, p1_events, lr_events = build_all(
        df, rbns_cap, smoothing, threshold, valuation_year, n_sims, int(seed_sim)
    )

total_paid_obs = panel.loc[panel['CalendarYear'] <= valuation_year, 'Pay'].sum()
reserve_mean = sims['TotalReserve'].mean()
known_mean = sims['KnownReserve'].mean()
ibnr_mean = sims['IBNRReserve'].mean()
cl_reserve = cl['ReserveCL'].sum()
cdr_mean = sims['CDR'].mean()
economic_factor = ((1 + inflation_rate) / max(1e-9, (1 + discount_rate))) ** avg_payment_duration
economic_reserve_mean = reserve_mean * economic_factor

currency_symbols = {"EUR": "€", "USD": "$", "GBP": "£", "XAF": "XAF", "CAD": "C$"}
currency_symbol = currency_symbols.get(report_currency, report_currency)

def eur_full(x):
    value = f"{x:,.0f}".replace(',', ' ')
    return f"{value} {currency_symbol}" if currency_symbol.isalpha() or currency_symbol == 'XAF' else f"{value} {currency_symbol}"

def num_full(x):
    return f"{x:,.0f}".replace(',', ' ')

kpis = [
    ("Claims", num_full(len(df)), "Claim count"),
    ("Observed paid", eur_full(total_paid_obs), "Observed cumulative cash-flows"),
    ("Individual reserve", eur_full(reserve_mean), "Best estimate Monte Carlo"),
    ("of which IBNR", eur_full(ibnr_mean), "Incurred but not reported claims"),
    ("Chain-Ladder", eur_full(cl_reserve), f"Difference vs individual: {eur_full(cl_reserve-reserve_mean)}"),
    ("Mean CDR", eur_full(cdr_mean), "One-year claims development result"),
    ("Economic BE", eur_full(economic_reserve_mean), "Inflation/discount adjusted approximation"),
]
st.markdown("<div class='kpi-grid'>" + "".join([
    f"<div class='kpi-card' title='{label}: {value}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>"
    for label, value, sub in kpis
]) + "</div>", unsafe_allow_html=True)

quality_score, quality_findings = data_quality_score(df, info)
if info['issues']:
    st.warning(" | ".join(info['issues']))
st.markdown(f"<span class='score-badge {score_class(quality_score)}'>Data readiness score: {quality_score}/100</span>", unsafe_allow_html=True)

with st.expander("📌 Excel/CSV import guide and client interpretation", expanded=False):
    st.markdown("""
    **Minimum expected format** to import a portfolio :

    | Column | Required | Description | Example |
    |---|---:|---|---|
    | `ID` or `ClNr` | Yes | Unique claim identifier | `100245` |
    | `AY` | Yes | Accident / occurrence year | `2021` |
    | `RepDel` | Recommended | Reporting delay in development years (`0` = reported in the accident year) | `1` |
    | `Pay0`, `Pay1`, ..., `PayK` | Yes | Payments by development year | `1250.50` |
    | `Open0`, `Open1`, ..., `OpenK` | Optional | Open/closed indicator at the end of development (`1` open, `0` closed) | `1` |
    | `LoB`, `cc`, `age`, `inj_part`, `AQ` | Optional | Descriptive variables used for segmentation/diagnostics | `Motor`, `12`, `35` |

    **Embedded business convention:** `Closed0` = closure without terminal payment; `Closed+` = closure with terminal payment.

    **Client message :** the application does not only produce a reserve figure. It explains how the reserve is built: claim trajectories, settlement speeds, payment behaviour, Monte Carlo uncertainty, one-year risk and Chain-Ladder benchmarking.
    """)

# Automatic executive interpretation
reserve_gap = cl_reserve - reserve_mean
gap_pct = reserve_gap / reserve_mean if reserve_mean else np.nan
if np.isfinite(gap_pct):
    if gap_pct > 0.10:
        msg = f"The Chain-Ladder method produces a reserve that is {gap_pct:.1%} above the individual model. This suggests that triangle aggregation may be conservative or sensitive to historical development patterns."
    elif gap_pct < -0.10:
        msg = f"The Chain-Ladder method produces a reserve that is {abs(gap_pct):.1%} below the individual model. A review of long-tail claims and payment assumptions is recommended."
    else:
        msg = f"The Chain-Ladder and individual reserves are relatively close ({gap_pct:.1%}  gap), which supports the overall consistency of the estimates."
    st.info("**Executive reading:** " + msg)

exec_insights = executive_insights(reserve_mean, ibnr_mean, cl_reserve, cdr_mean, snap, sims)

# -----------------------------------------------------------------------------
# Tabs enrichis
# -----------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "0. Theory & methodology", "1. Data & quality", "2. States & transitions", "3. Payments & diagnostics",
    "4. Reserves", "5. CDR & risk", "6. Benchmarks & export", "7. Sensitivity analysis",
    "8. Stress & backtesting", "9. Documentation & deployment"
])

with tab0:
    st.subheader("Executive overview")
    st.markdown("""
    <div class='info-card'>
    <b>Purpose.</b> MarkovReserve estimates non-life claim reserves at individual claim level using a semi-Markov multi-state framework. It follows each claim through IBNR, RBNP, RBNS and closed states, then compares the result with recognised collective reserving benchmarks.
    </div>
    """, unsafe_allow_html=True)
    meta_cols = st.columns(4)
    meta_cols[0].metric("Client", client_name)
    meta_cols[1].metric("Portfolio", portfolio_name)
    meta_cols[2].metric("Currency", report_currency)
    meta_cols[3].metric("Data readiness", f"{quality_score}/100")
    st.markdown("### Automatic executive summary")
    for insight in exec_insights:
        st.markdown(f"- {insight}")
    if quality_findings:
        st.markdown("### Data quality notes")
        for q in quality_findings[:6]:
            st.markdown(f"- {q}")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        **1. Data ingestion**
        - CSV / Excel import
        - quality controls
        - claim-level payment history
        - optional covariates
        """)
    with h2:
        st.markdown("""
        **2. Individual model**
        - IBNR / RBNP / RBNS states
        - duration-dependent transitions
        - first payment and link ratios
        - Monte Carlo reserve distribution
        """)
    with h3:
        st.markdown("""
        **3. Client deliverables**
        - Best Estimate, VaR, TVaR
        - one-year CDR
        - Chain-Ladder, Cape Cod, BF benchmarks
        - Excel and PDF exports
        """)
    st.markdown("### Recommended client workflow")
    st.markdown("""
    1. Load the portfolio or use the demo dataset.  
    2. Review data quality and claim-state evolution.  
    3. Validate transition and payment diagnostics.  
    4. Review reserves, CDR and benchmark comparison.  
    5. Run sensitivity scenarios.  
    6. Export the PDF report and Excel workpapers.
    """)
    st.markdown("### Deployment note")
    st.info("For client sharing, the recommended route is to deploy this Streamlit application on Streamlit Community Cloud for demos, or on Render/Azure/AWS with authentication for confidential client data.")

    st.markdown("### Theory behind the individual model")
    st.markdown("""
    <div class='info-card'>
    <b>Plain-English intuition.</b> Traditional reserving often looks at claims in aggregate triangles. MarkovReserve instead follows each claim file individually. A claim starts as <b>IBNR</b>, may become <b>RBNP</b>, then moves through one or several <b>RBNS</b> payment states, and finally becomes <b>Closed0</b> or <b>Closed+</b>. The reserve is the expected value of the future payments still to be made.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("A. Claim states and business meaning", expanded=True):
        st.markdown("""
        | State | Meaning | Business interpretation |
        |---|---|---|
        | `IBNR` | Incurred But Not Reported | The loss event has occurred but is not yet reported to the insurer. |
        | `RBNP` | Reported But Not Paid | The claim is reported but no payment has been made yet. |
        | `RBNSj` | Reported But Not Settled | The claim has already received payments and remains open. |
        | `Closed0` | Closed without terminal payment | The claim closes without an additional final payment. |
        | `Closed+` | Closed with terminal payment | The claim closes with a final payment. |

        `Closed0` and `Closed+` are absorbing states: once a claim is closed, the model assumes it does not reopen.
        """)

    with st.expander("B. Semi-Markov transition model", expanded=False):
        st.markdown(r"""
        A simple Markov chain assumes that the next state depends only on the current state. In claims reserving, this is often too restrictive because the probability of moving out of a state depends on how long the claim has already stayed there.

        We therefore define:

        $$S_t = 	ext{claim state at time } t$$

        $$D_t = 	ext{duration already spent in the current state}$$

        The augmented process is:

        $$X_t = (S_t, D_t)$$

        and it satisfies the Markov property:

        $$\mathbb{P}(X_{t+1}=b \mid X_0,\ldots,X_t)=\mathbb{P}(X_{t+1}=b \mid X_t).$$

        In matrix form, the transition probabilities are stored in a matrix:

        $$M_{ab}=\mathbb{P}(X_{t+1}=b \mid X_t=a).$$

        The distribution of the claim state after $k$ periods is obtained through matrix powers:

        $$\pi_{t+k}=\pi_t M^k.$$
        """)

    with st.expander("C. IBNR count model", expanded=False):
        st.markdown(r"""
        For claims that have occurred but are not yet reported, we model the expected number of future reports by accident year and reporting delay:

        $$\mathbb{E}[N_{ij}] = \alpha_i \beta_j,$$

        where:

        - $\alpha_i$ is the expected number of claims for accident year $i$;
        - $\beta_j$ is the probability that a claim is reported with delay $j$;
        - $\sum_j \beta_j = 1$.

        This gives a transparent way to estimate how many IBNR claims remain to be reported for each accident year.
        """)

    with st.expander("D. Payment model: first payment and link ratios", expanded=False):
        st.markdown(r"""
        Let $P_j$ be the payment made at the $j$-th payment step. The cumulative paid amount is:

        $$C_j = \sum_{k=1}^{j} P_k.$$

        Link ratios describe how the cumulative payment develops:

        $$\Lambda_j = \frac{C_{j+1}}{C_j}.$$

        Hence future cumulative payments can be represented as:

        $$C_j = P_1 \prod_{k=1}^{j-1} \Lambda_k.$$

        MarkovReserve models the first payment $P_1$ and the link ratios $\Lambda_j$. This is useful because a very large first payment is usually followed by smaller future ratios, while a small first payment can still develop significantly.
        """)

    with st.expander("E. Reserve and one-year CDR", expanded=False):
        st.markdown(r"""
        The reserve is the conditional expected value of future payments:

        $$R_t = \mathbb{E}\left[\sum_{u>t} P_u \mid \mathcal{F}_t\right].$$

        The one-year Claims Development Result is defined as:

        $$CDR_{t,t+1} = R_t - P_{t,t+1} - R_{t+1}.$$

        With this convention:

        - positive CDR = favourable development;
        - negative CDR = adverse development.

        MarkovReserve simulates many future claim paths to obtain a distribution of reserves and CDR, from which Best Estimate, VaR and TVaR are derived.
        """)

    st.markdown("### Theory behind the collective benchmarks")
    with st.expander("F. Chain-Ladder", expanded=False):
        st.markdown(r"""
        Chain-Ladder is the standard collective benchmark. It uses cumulative paid triangles by accident year and development year.

        The development factor from development year $j$ to $j+1$ is:

        $$f_j = \frac{\sum_i C_{i,j+1}}{\sum_i C_{i,j}}.$$

        The ultimate claim amount for accident year $i$ is estimated by:

        $$\widehat{U}_i = C_{i,k}\prod_{j=k}^{J-1} f_j.$$

        The reserve is:

        $$\widehat{R}_i = \widehat{U}_i - C_{i,k}.$$

        Chain-Ladder is simple and widely accepted, but it does not use individual claim information.
        """)

    with st.expander("G. Cape Cod", expanded=False):
        st.markdown(r"""
        Cape Cod introduces exposure information, such as earned premium, policy-years or claim counts. It estimates an expected ultimate loss level per exposure unit and adjusts for the maturity already observed.

        A simplified Cape Cod rate is:

        $$\widehat{q}=\frac{\sum_i C_{i,k}}{\sum_i E_i p_{i,k}},$$

        where $E_i$ is the exposure and $p_{i,k}$ is the percentage developed at the current maturity.

        The ultimate estimate is then:

        $$\widehat{U}_i = \widehat{q} E_i.$$

        In the demo version, claim count is used as a proxy exposure when no real exposure table is provided.
        """)

    with st.expander("H. Bornhuetter-Ferguson", expanded=False):
        st.markdown(r"""
        Bornhuetter-Ferguson combines observed experience with an a priori expected ultimate loss.

        If $U_i^{prior}$ is the prior ultimate and $p_{i,k}$ the percentage already developed, the reserve is:

        $$\widehat{R}_i^{BF}=U_i^{prior}(1-p_{i,k}).$$

        The estimated ultimate is:

        $$\widehat{U}_i^{BF}=C_{i,k}+U_i^{prior}(1-p_{i,k}).$$

        BF is often more stable than Chain-Ladder for immature accident years because it does not overreact to early observed payments.
        """)

with tab1:
    st.subheader("Data and quality controls")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> This section checks portfolio structure, accident-year distribution, reporting delays and early signs of heterogeneity. A client should immediately see whether the data is clean enough before any reserve estimate is produced.</div>
    """, unsafe_allow_html=True)
    st.markdown(f"<span class='score-badge {score_class(quality_score)}'>Data readiness score: {quality_score}/100</span>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, .95])
    with c1:
        st.dataframe(df.head(30), use_container_width=True)
        st.markdown("**Quality summary**")
        st.json(info)
    with c2:
        ay_counts = df.groupby('AY').size().reset_index(name='Count')
        fig = px.bar(ay_counts, x='AY', y='Count', title="Number of claims by accident year", color_discrete_sequence=[NAVY])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    st.markdown("### Portfolio descriptive statistics")
    g1, g2, g3 = st.columns(3)
    with g1:
        fig = px.histogram(df, x="RepDel", nbins=max_dev, title="Reporting delay distribution", color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)
    with g2:
        if "age" in df.columns:
            fig = px.histogram(df, x="age", nbins=25, title="Policyholder age distribution", color_discrete_sequence=[CYAN])
            st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        else:
            st.info("Column age is not available.")
    with g3:
        if "AQ" in df.columns:
            aq = df.groupby("AQ").size().reset_index(name="Count")
            fig = px.bar(aq, x="AQ", y="Count", title="Accident quarter distribution", color_discrete_sequence=[TEAL])
            st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        else:
            st.info("Column AQ is not available.")

    g4, g5 = st.columns(2)
    with g4:
        if "cc" in df.columns:
            cc = df.groupby("cc").size().reset_index(name="Count").sort_values("Count", ascending=False).head(20)
            fig = px.bar(cc, x="cc", y="Count", title="Top 20 claim causes / codes", color_discrete_sequence=[NAVY_2])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with g5:
        paid_by_dy = panel.groupby('DY')['Pay'].sum().reset_index()
        paid_by_dy["Cumulative paid"] = paid_by_dy["Pay"].cumsum()
        fig = go.Figure()
        fig.add_bar(x=paid_by_dy['DY'], y=paid_by_dy['Pay'], name="Annual payment", marker_color=BLUE)
        fig.add_trace(go.Scatter(x=paid_by_dy['DY'], y=paid_by_dy['Cumulative paid'], name="Cumulative paid", mode="lines+markers", line=dict(color=NAVY, width=3)))
        fig.update_layout(title="Payments and cumulative paid by development year", yaxis_title="Amount")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

with tab2:
    st.subheader("States, durations and transitions")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> The charts show how claims migrate from IBNR to RBNP, RBNS and then closed states. A persistent concentration in RBNS5+ indicates long-running files that may carry a material share of reserve risk.</div>
    """, unsafe_allow_html=True)
    state_counts = panel.groupby(['DY', 'State']).size().reset_index(name='Count')
    state_counts["State"] = pd.Categorical(state_counts["State"], list(STATE_COLORS.keys()), ordered=True)
    fig = px.area(state_counts.sort_values("State"), x='DY', y='Count', color='State',
                  color_discrete_map=STATE_COLORS, title="State evolution by development year")
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        top_flows = aug[aug["Previous_State"] != aug["Current_State"]].groupby(["Previous_State", "Current_State"]).size().reset_index(name="Count")
        top_flows["Transition"] = top_flows["Previous_State"] + " → " + top_flows["Current_State"]
        top_flows = top_flows.sort_values("Count", ascending=True).tail(15)
        fig = px.bar(top_flows, x="Count", y="Transition", orientation="h", title="Top observed transitions between states", color_discrete_sequence=[NAVY])
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)
    with c2:
        st.plotly_chart(transition_heatmap(trans_table), use_container_width=True)

    d1, d2 = st.columns([.8, 1.2])
    with d1:
        dur = aug.groupby("Previous_State")["Previous_Duration"].mean().reset_index(name="Observed average duration")
        fig = px.bar(dur, x="Previous_State", y="Observed average duration", title="Observed average duration by state",
                     color="Previous_State", color_discrete_map=STATE_COLORS)
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with d2:
        st.markdown("**Augmented database**")
        st.dataframe(aug.head(300), use_container_width=True)

with tab3:
    st.subheader("Payments, link ratios and diagnostics")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> This section reproduces the key diagnostics from the dissertation: P1 distribution, empirical distribution function, PP-Plot, P1-Lambda1 relationship and link-ratio histograms. The objective is to justify payment assumptions rather than present a black box.</div>
    """, unsafe_allow_html=True)
    st.dataframe(pay_summary, use_container_width=True)

    if len(p1_events):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(p1_events, x='P1', color="Class", nbins=80, marginal="box",
                               title="First payment P1 distribution", color_discrete_sequence=[BLUE, ORANGE])
            add_vline(fig, threshold, "Threshold", RED)
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        with c2:
            ecdf = empirical_cdf(p1_events["P1"])
            fig = px.line(ecdf, x="x", y="cdf", title="Empirical distribution function of P1")
            fig.update_traces(line=dict(color=NAVY, width=3))
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            pp = pp_dataframe(p1_events["P1"], pm.p1_model)
            fig = px.scatter(pp, x="Theoretical", y="Empirical", title="PP-Plot of first payment P1")
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="y=x", line=dict(color=RED, dash="dash")))
            fig.update_traces(marker=dict(color=NAVY, size=5, opacity=.55), selector=dict(mode='markers'))
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        with c4:
            if len(lr_events):
                l1 = lr_events[lr_events["LambdaIndex"] == 1]
                fig = px.scatter(l1, x="P1", y="LR", color="Class", trendline="ols" if len(l1) > 20 else None,
                                 title="First link ratio Lambda1 as a function of P1",
                                 color_discrete_sequence=[BLUE, ORANGE])
                add_vline(fig, threshold, "Threshold", RED)
                fig.update_yaxes(range=[0, min(8, max(2, np.nanquantile(l1['LR'], .98) if len(l1) else 2))])
                st.plotly_chart(style_fig(fig, 420), use_container_width=True)
            else:
                st.info("Not enough observed link ratios.")

    if len(lr_events):
        c5, c6 = st.columns(2)
        with c5:
            fig = px.histogram(lr_events, x="LR", color="LambdaIndex", nbins=80, marginal="box",
                               title="Observed link-ratio histograms Lambdaj")
            fig.update_xaxes(range=[1, min(8, np.nanquantile(lr_events['LR'], .99))])
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
        with c6:
            pp_frames = []
            for j in sorted(lr_events["LambdaIndex"].unique()):
                vals = lr_events.loc[lr_events["LambdaIndex"] == j, "LR"]
                model = pm.lr_models.get((int(min(j, 5)), 'all'))
                pp = pp_dataframe(vals, model)
                pp["Lambda"] = f"Lambda{int(j)}"
                pp_frames.append(pp)
            pp_all = pd.concat(pp_frames, ignore_index=True) if pp_frames else pd.DataFrame()
            fig = px.line(pp_all, x="Theoretical", y="Empirical", color="Lambda", title="PP-Plots of link ratios")
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="y=x", line=dict(color=RED, dash="dash")))
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

with tab4:
    st.subheader("Simulated ultimate reserves")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> The Best Estimate is the average of the simulated scenarios. VaR and TVaR measure the adverse tail of the reserve distribution and support a Solvency II-style risk view.</div>
    """, unsafe_allow_html=True)
    s = risk_summary(sims['TotalReserve'])
    a, b, c, d = st.columns(4)
    a.metric("Best estimate", f"{s['mean']:,.0f} €".replace(',', ' '))
    b.metric("Standard deviation", f"{s['std']:,.0f} €".replace(',', ' '))
    c.metric("VaR 99,5%", f"{s['p995']:,.0f} €".replace(',', ' '))
    d.metric("TVaR 99,5%", f"{s['tvar995']:,.0f} €".replace(',', ' '))
    econ = pd.DataFrame({
        'Item': ['Nominal Best Estimate', 'Annual claims inflation', 'Annual discount rate', 'Average future duration', 'Approx. economic Best Estimate'],
        'Value': [eur_full(reserve_mean), f"{inflation_rate:.2%}", f"{discount_rate:.2%}", f"{avg_payment_duration:.1f} years", eur_full(economic_reserve_mean)]
    })
    st.markdown("**Inflation / discounting approximation**")
    st.dataframe(econ, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = hist_with_lines(sims, 'TotalReserve', "Simulated total reserve distribution", BLUE,
                              lines={"BE": sims['TotalReserve'].mean(), "VaR 99.5%": np.quantile(sims['TotalReserve'], .995)})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        melted = sims[["KnownReserve", "IBNRReserve", "TotalReserve"]].melt(var_name="Component", value_name="Amount")
        fig = px.box(melted, x="Component", y="Amount", color="Component", title="Distribution of reserve components")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        by_state = snap.groupby('State').agg(N=('ID', 'count'), CumPay=('CumPay', 'sum')).reset_index()
        fig = px.bar(by_state, x="State", y="N", color="State", color_discrete_map=STATE_COLORS,
                     title="Number of claims by state at valuation date")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        st.dataframe(by_state, use_container_width=True)
    with c4:
        if len(ibnr):
            fig = px.bar(ibnr, x="AY", y="ExpectedIBNR", title="Expected IBNR claim count by accident year",
                         color_discrete_sequence=[NAVY])
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
            st.dataframe(ibnr, use_container_width=True)

with tab5:
    st.subheader("One-year CDR and development risk")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> The CDR measures one-year development result. Under the selected convention, a negative CDR indicates adverse development: payments plus the re-estimated closing reserve exceed the opening reserve.</div>
    """, unsafe_allow_html=True)
    st.markdown("Convention: `CDR = opening reserve - one-year payments - closing reserve`. A negative CDR indicates adverse development.")
    cdr_summary = risk_summary(sims['CDR'])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean CDR", f"{cdr_summary['mean']:,.0f} €".replace(',', ' '))
    c2.metric("CDR volatility", f"{cdr_summary['std']:,.0f} €".replace(',', ' '))
    c3.metric("Quantile 5%", f"{np.quantile(sims['CDR'], .05):,.0f} €".replace(',', ' '))
    c4.metric("Quantile 0,5%", f"{np.quantile(sims['CDR'], .005):,.0f} €".replace(',', ' '))

    c5, c6 = st.columns(2)
    with c5:
        fig = hist_with_lines(sims, 'CDR', "Simulated one-year CDR distribution", NAVY,
                              lines={"Mean": sims['CDR'].mean(), "Q0.5%": np.quantile(sims['CDR'], .005)})
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        ecdf = empirical_cdf(sims["CDR"])
        fig = px.line(ecdf, x="x", y="cdf", title="Empirical distribution function of CDR")
        fig.update_traces(line=dict(color=NAVY, width=3))
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        fig = px.scatter(sims, x="OneYearPayments", y="EndYearReserve", color="CDR",
                         color_continuous_scale="RdBu", title="One-year payments vs closing reserve")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    with c8:
        st.markdown("**CDR risk table**")
        st.dataframe(risk_table(sims['CDR'], unfavorable_lower_tail=True), use_container_width=True)

with tab6:
    st.subheader("Benchmark comparison and exports")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> The comparison with Chain-Ladder provides a market-recognised benchmark. The value of the individual model is to explain differences through claim-level trajectories, states, durations and payments.</div>
    """, unsafe_allow_html=True)
    comp = pd.concat([
        pd.DataFrame({'Method': ['Individual Semi-Markov'], 'Reserve': [reserve_mean], 'Key assumption': ['Individual semi-Markov trajectories + simulated payments']}),
        bench_summary
    ], ignore_index=True)
    comp['Difference vs individual'] = comp['Reserve'] - reserve_mean
    comp['Relative gap'] = comp['Difference vs individual'] / max(reserve_mean, 1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(comp, x='Method', y='Reserve', color='Method', title="Reserve comparison",
                     color_discrete_sequence=[NAVY, BLUE, ORANGE, TEAL], text_auto='.3s')
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        st.dataframe(comp, use_container_width=True)
    with c2:
        fac_df = factors.reset_index().rename(columns={'index': 'Development', 'DevelopmentFactor': 'Factor'})
        fig = px.line(fac_df, x="Development", y="Factor", markers=True, title="Chain-Ladder development factors")
        fig.update_traces(line=dict(color=NAVY, width=3), marker=dict(size=8))
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        st.dataframe(fac_df, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.imshow(tri, color_continuous_scale="Blues", aspect="auto", title="Observed cumulative triangle — heatmap")
        st.plotly_chart(style_fig(fig, 500), use_container_width=True)
    with c4:
        bench_long = bench_by_ay.melt(id_vars=['AY'], value_vars=['ReserveCL','ReserveCapeCod','ReserveBF'], var_name='Method', value_name='Reserve')
        bench_long['Method'] = bench_long['Method'].replace({'ReserveCL':'Chain-Ladder','ReserveCapeCod':'Cape Cod','ReserveBF':'Bornhuetter-Ferguson'})
        fig = px.bar(bench_long, x="AY", y="Reserve", color="Method", barmode="group", title="Collective reserves by accident year")
        st.plotly_chart(style_fig(fig, 500), use_container_width=True)

    report_payload = {
        'title': 'MarkovReserve — Client report',
        'valuation_year': valuation_year,
        'metadata': {'Client': client_name, 'Portfolio': portfolio_name, 'Prepared by': prepared_by, 'Currency': report_currency, 'Report date': str(report_date)},
        'executive_summary': pd.DataFrame({'Executive insight': exec_insights}),
        'data_quality': pd.DataFrame({'Finding': quality_findings, 'Score': [quality_score] + [''] * (len(quality_findings)-1)}),
        'economic_assumptions': pd.DataFrame({'Item': ['Claims inflation', 'Discount rate', 'Average future duration', 'Economic factor', 'Economic Best Estimate'], 'Value': [f'{inflation_rate:.2%}', f'{discount_rate:.2%}', f'{avg_payment_duration:.1f}', f'{economic_factor:.4f}', eur_full(economic_reserve_mean)]}),
        'methodology': pd.DataFrame({
            'Topic': ['Individual reserving', 'Semi-Markov states', 'IBNR count model', 'Payment model', 'Reserve and CDR', 'Chain-Ladder', 'Cape Cod', 'Bornhuetter-Ferguson'],
            'Explanation': [
                'The model follows each claim file individually instead of relying only on aggregate triangles.',
                'The future state depends on both the current claim state and the duration already spent in that state: X_t = (S_t, D_t).',
                'Future unreported claims are estimated by accident year and reporting delay: E[N_ij] = alpha_i beta_j.',
                'Payments are represented through a first payment P1 and link ratios Lambda_j = C_{j+1}/C_j.',
                'Reserve is the expected value of future payments. CDR = opening reserve - one-year payments - closing reserve.',
                'Chain-Ladder projects cumulative triangles with historical development factors.',
                'Cape Cod uses exposure and maturity to estimate expected ultimate losses.',
                'Bornhuetter-Ferguson blends observed payments with a prior expected ultimate loss.'
            ]
        }),
        'kpis': pd.DataFrame(kpis, columns=['Indicator','Value','Comment']),
        'comparison': comp,
        'risk_total_reserve': risk_table(sims['TotalReserve']),
        'risk_cdr': risk_table(sims['CDR'], unfavorable_lower_tail=True),
        'state_snapshot': snap.groupby('State').agg(N=('ID','count'), CumPay=('CumPay','sum')).reset_index(),
        'transitions': trans_table.head(40),
        'payment_models': pay_summary,
        'ibnr': ibnr,
        'cl_factors': fac_df,
        'bench_by_ay': bench_by_ay,
        'reserve_values': sims['TotalReserve'].to_numpy(),
        'cdr_values': sims['CDR'].to_numpy(),
    }
    try:
        pdf_bytes = build_pdf_report(report_payload)
        st.download_button("📄 Export PDF report", pdf_bytes, file_name="individual_reserving_client_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF export unavailable : {e}. Check that reportlab is installed via requirements.txt.")

    export = to_excel_bytes({
        'portfolio_head': df.head(1000),
        'state_panel_head': panel.head(5000),
        'augmented_head': aug.head(5000),
        'transitions': trans_table,
        'payment_models': pay_summary,
        'p1_events': p1_events.head(10000),
        'lr_events': lr_events.head(10000),
        'simulations': sims,
        'risk_total_reserve': risk_table(sims['TotalReserve']),
        'risk_cdr': risk_table(sims['CDR'], unfavorable_lower_tail=True),
        'ibnr': ibnr,
        'chain_ladder': cl,
        'benchmarks_by_ay': bench_by_ay,
        'comparison': comp,
    })
    st.download_button("⬇️ Export Excel results", export, file_name="individual_reserving_results.xlsx")


with tab7:
    st.subheader("Sensitivity analysis")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> Sensitivity analysis shows whether the reserve is robust to key modelling choices. This is essential for client discussions because it turns a single estimate into a documented range of outcomes.</div>
    """, unsafe_allow_html=True)
    st.caption("Scenarios are intentionally run with a reduced number of simulations to keep the dashboard responsive. Increase simulations for final production runs if needed.")
    sens_sims = st.slider("Sensitivity simulations per scenario", 100, max(100, min(1000, int(n_sims))), min(300, int(n_sims)), step=100)
    if st.button("Run sensitivity analysis", type="primary"):
        scenario_defs = []
        for th in sorted(set([max(1000, threshold * 0.5), threshold, threshold * 1.5])):
            scenario_defs.append((f"Threshold {th:,.0f}", rbns_cap, float(th)))
        for cap in [3, 4, 5, 6]:
            scenario_defs.append((f"RBNS{cap}+ grouping", cap, float(threshold)))
        seen = set(); scenario_defs_unique = []
        for name, cap, th in scenario_defs:
            key = (cap, round(th, 2))
            if key not in seen:
                scenario_defs_unique.append((name, cap, th)); seen.add(key)
        rows = []
        progress = st.progress(0)
        for i, (name, cap, th) in enumerate(scenario_defs_unique, start=1):
            p_s, a_s = build_state_panel(df, rbns_cap=cap)
            tm_s = TransitionModel.fit(a_s, smoothing=smoothing)
            pm_s = PaymentModel.fit(df, p_s, threshold=th)
            sims_s, _, _ = simulate_reserves(df, p_s, valuation_year, tm_s, pm_s, n_sims=int(sens_sims), seed=int(seed_sim)+i, include_ibnr=True)
            rows.append({
                'Scenario': name, 'RBNS cap': cap, 'P1 threshold': th,
                'Best Estimate': sims_s['TotalReserve'].mean(),
                'VaR 99.5%': np.quantile(sims_s['TotalReserve'], .995),
                'Mean CDR': sims_s['CDR'].mean(),
                'Gap vs central': sims_s['TotalReserve'].mean() - reserve_mean,
                'Relative gap': (sims_s['TotalReserve'].mean() - reserve_mean) / max(reserve_mean, 1)
            })
            progress.progress(i / len(scenario_defs_unique))
        sens_df = pd.DataFrame(rows)
        st.dataframe(sens_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(sens_df, x='Scenario', y='Best Estimate', color='Scenario', title='Sensitivity of Best Estimate')
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        with c2:
            fig = px.bar(sens_df, x='Scenario', y='Mean CDR', color='Scenario', title='Sensitivity of mean CDR')
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        st.download_button("⬇️ Download sensitivity CSV", sens_df.to_csv(index=False).encode('utf-8'), file_name="sensitivity_analysis.csv", mime="text/csv")
    else:
        st.info("Click 'Run sensitivity analysis' to evaluate alternative thresholds and RBNS groupings.")


with tab8:
    st.subheader("Stress testing and backtesting")
    st.markdown("""
    <div class='info-card'><b>Interpretation.</b> Stress testing translates actuarial judgement into quantified reserve impacts. Backtesting compares past estimates with subsequent run-off and helps support model governance.</div>
    """, unsafe_allow_html=True)
    stress_rows = []
    stress_rows.append({'Scenario': 'Central', 'Reserve': reserve_mean, 'Impact vs central': 0.0, 'Relative impact': 0.0})
    for label, mult in [('Severity +10%', 1.10), ('Severity +20%', 1.20), ('Severity +30%', 1.30)]:
        stress_rows.append({'Scenario': label, 'Reserve': reserve_mean * mult, 'Impact vs central': reserve_mean * (mult-1), 'Relative impact': mult-1})
    stress_rows.append({'Scenario': 'IBNR +25%', 'Reserve': known_mean + ibnr_mean * 1.25, 'Impact vs central': ibnr_mean * .25, 'Relative impact': (ibnr_mean*.25)/max(reserve_mean,1)})
    stress_rows.append({'Scenario': 'Adverse tail VaR 99.5%', 'Reserve': np.quantile(sims['TotalReserve'], .995), 'Impact vs central': np.quantile(sims['TotalReserve'], .995)-reserve_mean, 'Relative impact': np.quantile(sims['TotalReserve'], .995)/max(reserve_mean,1)-1})
    stress_rows.append({'Scenario': 'Combined severity + IBNR stress', 'Reserve': known_mean*1.20 + ibnr_mean*1.25, 'Impact vs central': known_mean*.20 + ibnr_mean*.25, 'Relative impact': (known_mean*.20 + ibnr_mean*.25)/max(reserve_mean,1)})
    stress_df = pd.DataFrame(stress_rows)
    st.dataframe(stress_df, use_container_width=True)
    fig = px.bar(stress_df, x='Scenario', y='Reserve', color='Scenario', title='Stress testing impact on reserve')
    st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    st.markdown("### Quick backtesting")
    st.caption("This module uses the full available simulated/imported run-off to compare a past valuation date with subsequent observed payments. Use reduced simulations for responsiveness.")
    possible_years = sorted([int(y) for y in panel['CalendarYear'].unique() if y < valuation_year])
    if possible_years:
        bt_years = st.multiselect("Backtesting valuation years", possible_years[-5:], default=possible_years[-3:])
        bt_sims = st.slider("Backtesting simulations per year", 50, 500, 100, step=50)
        if st.button("Run quick backtesting"):
            rows = []
            for y in bt_years:
                sims_bt, _, _ = simulate_reserves(df, panel, y, TransitionModel.fit(aug, smoothing=smoothing), pm, n_sims=int(bt_sims), seed=int(seed_sim)+int(y), include_ibnr=True)
                est = sims_bt['TotalReserve'].mean()
                actual = panel[(panel['CalendarYear'] > y) & (panel['CalendarYear'] <= valuation_year)]['Pay'].sum()
                rows.append({'Valuation year': y, 'Estimated reserve': est, 'Observed subsequent paid': actual, 'Error': est-actual, 'Relative error': (est-actual)/max(actual,1)})
            bt_df = pd.DataFrame(rows)
            st.dataframe(bt_df, use_container_width=True)
            fig = px.bar(bt_df.melt(id_vars='Valuation year', value_vars=['Estimated reserve','Observed subsequent paid'], var_name='Measure', value_name='Amount'), x='Valuation year', y='Amount', color='Measure', barmode='group', title='Backtesting: estimate vs observed subsequent paid')
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
    else:
        st.info("No earlier valuation year is available for backtesting.")

with tab9:
    st.subheader("Documentation and deployment")
    st.markdown("""
    ### Expected import format
    Required columns: `ID` or `ClNr`, `AY`, and payment columns `Pay0 ... PayK`. Recommended columns: `RepDel` and `Open0 ... OpenK`. Optional segmentation columns include `LoB`, `cc`, `age`, `inj_part`, `AQ`.

    ### Model conventions
    - `Closed0`: closure without terminal payment.  
    - `Closed+`: closure with terminal payment.  
    - Closed states are absorbing.  
    - CDR = opening reserve - one-year payments - closing reserve.  

    ### Professional deployment recommendation
    - Demo / portfolio presentation: Streamlit Community Cloud.  
    - Client confidential data: Render, Azure, AWS or private server with authentication, HTTPS and access control.  
    - Do not upload real confidential insurance data to a public demo instance.  

    ### Governance note
    MarkovReserve is a decision-support tool. Results should be reviewed by a qualified actuary, with documented assumptions, sensitivity analysis and data-quality validation.
    """)

st.markdown("---")
st.caption("MarkovReserve client-ready version — actuarial decision-support prototype. Results must be validated by a qualified actuary before regulatory use.")
