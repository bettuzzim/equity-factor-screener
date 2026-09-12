"""
app.py
-------
Streamlit dashboard for the Equity Factor Screener.

Reuses the exact same pipeline as main.py (src/data_loader.py, src/factors.py,
src/scoring.py, src/backtest.py) so the dashboard and the CLI cannot drift
out of sync — this is a presentation layer on top of the existing analysis,
not a parallel implementation.

The backtest is behind a button rather than run on load: it recomputes
factors at 43 rebalance dates for each factor variant, which takes a minute
or two and should be an explicit choice rather than a surprise.

Run:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.universe import UNIVERSE, AS_OF_DATE, SECTORS
from src.data_loader import load_prices, load_fundamentals, load_risk_free_rate
from src.backtest import run_backtest, summarize_backtest, factor_attribution
from src.viz import QUINTILE_RAMP, DIVERGING_POSITIVE, DIVERGING_NEGATIVE
from main import build_screen

st.set_page_config(page_title="Equity Factor Screener", layout="wide")

st.title("Equity Factor Screener")
st.caption(
    f"Four-factor cross-sectional model (value, quality, momentum, low volatility) "
    f"across {len(UNIVERSE)} S&P 500 constituents, universe as of {AS_OF_DATE}. "
    "Live data from Yahoo Finance and FRED, recomputed on every run."
)


@st.cache_data(ttl=3600, show_spinner="Downloading live market data...")
def get_data():
    prices, price_mode = load_prices()
    fundamentals, coverage = load_fundamentals()
    risk_free_rate, rf_mode = load_risk_free_rate()
    return prices, price_mode, fundamentals, coverage, risk_free_rate, rf_mode


@st.cache_data(ttl=3600, show_spinner="Running point-in-time backtest...")
def get_backtest(_prices, risk_free_rate):
    results = run_backtest(_prices)
    summary = summarize_backtest(results, risk_free_rate)
    attribution = factor_attribution(_prices, risk_free_rate)
    return results, summary, attribution


prices, price_mode, fundamentals, coverage, risk_free_rate, rf_mode = get_data()

if price_mode != "live":
    st.error("Live price history unavailable — cannot build the screen.")
    st.stop()

st.sidebar.header("Settings")
sector_neutral = st.sidebar.checkbox(
    "Sector-neutral scoring", value=False,
    help="Z-score within sector instead of across the whole universe. Sectors "
         "with fewer than 5 names fall back to universe-wide scoring.")
selected_sectors = st.sidebar.multiselect("Sectors", SECTORS, default=SECTORS)

sectors = pd.Series(UNIVERSE)
screen, groups, group_coverage, fallbacks = build_screen(
    prices, fundamentals, sectors, sector_neutral=sector_neutral)

filtered = screen[screen["sector"].isin(selected_sectors)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks scored", int(filtered["composite"].notna().sum()))
col2.metric("Sectors", filtered["sector"].nunique())
col3.metric("Risk-free rate", f"{risk_free_rate*100:.2f}%",
            help=f"3-month T-bill from FRED ({rf_mode})")
col4.metric("Median composite", f"{filtered['composite'].median():.2f}")

if fallbacks:
    affected = sorted({s for lst in fallbacks.values() for s in lst})
    st.info(f"Sectors too small for within-sector scoring, scored "
            f"universe-wide instead: {', '.join(affected)}")

st.subheader("1. Live screen")
display_cols = ["sector", "Value", "Quality", "Momentum", "Low Volatility",
                "composite", "quintile"]
st.dataframe(filtered[display_cols].round(2), use_container_width=True)

st.subheader("2. Factor profile of the top names")
st.caption("A high composite can come from one dominant exposure or from broad "
           "strength — the composite alone hides which, and they are different bets.")
top_names = filtered.dropna(subset=["composite"]).nlargest(15, "composite")
profile = groups.loc[top_names.index]
fig_profile = px.imshow(
    profile.values, x=list(profile.columns), y=list(profile.index),
    color_continuous_scale=[[0, DIVERGING_NEGATIVE], [0.5, "#f0efec"], [1, DIVERGING_POSITIVE]],
    zmin=-2.5, zmax=2.5, text_auto=".1f", aspect="auto",
)
fig_profile.update_layout(height=520)
st.plotly_chart(fig_profile, use_container_width=True)

st.subheader("3. Factor group correlations")
st.caption("Factors that move together are one bet wearing several names — an "
           "equal-weighted average of them concentrates rather than diversifies.")
corr = groups.corr()
fig_corr = px.imshow(
    corr.values, x=list(corr.columns), y=list(corr.index),
    color_continuous_scale=[[0, DIVERGING_NEGATIVE], [0.5, "#f0efec"], [1, DIVERGING_POSITIVE]],
    zmin=-1, zmax=1, text_auto=".2f",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("4. Point-in-time backtest")
st.caption(
    "Momentum and low volatility only. Value and quality come from a present-day "
    "fundamental snapshot; ranking past dates with them would be look-ahead bias, "
    "so they inform the live screen but are excluded from the historical test."
)

if st.button("Run backtest (about a minute)"):
    results, summary, attribution = get_backtest(prices, risk_free_rate)

    st.markdown("**Factor attribution — each factor backtested alone**")
    fig_attr = px.bar(
        attribution, x="spread_bps", y="factor", orientation="h",
        color=attribution["spread_bps"] > 0,
        color_discrete_map={True: DIVERGING_POSITIVE, False: DIVERGING_NEGATIVE},
        labels={"spread_bps": "Annualized Q1 - Q5 spread (bps)"},
    )
    fig_attr.update_layout(showlegend=False)
    fig_attr.add_vline(x=0, line_color="#0b0b0b")
    st.plotly_chart(fig_attr, use_container_width=True)
    st.caption(
        f"Equal-weight universe benchmark: "
        f"{attribution.attrs['benchmark_return']*100:.2f}% annualized, "
        f"Sharpe {attribution.attrs['benchmark_sharpe']:.2f}"
    )

    st.markdown("**Quintile performance**")
    n = results["n_quantiles"]
    quintiles = summary.iloc[:n]
    fig_q = go.Figure(go.Bar(
        x=[f"Q{i+1}" for i in range(n)],
        y=quintiles["annualized_return"] * 100,
        marker_color=QUINTILE_RAMP[:n],
    ))
    fig_q.update_layout(yaxis_title="Annualized return (%)",
                        xaxis_title="Composite quintile (Q1 = highest-scoring)")
    st.plotly_chart(fig_q, use_container_width=True)

    verdict = "monotonic" if summary.attrs["monotonic"] else "not monotonic"
    st.caption(f"Q1 - Q5 spread: {summary.attrs['spread_bps']:.0f} bps · {verdict} across quintiles.")

    st.markdown("**Cumulative performance**")
    cumulative = pd.DataFrame({
        "Q1 (highest-scoring)": (1 + results["quantile_returns"][1]).cumprod(),
        f"Q{n} (lowest-scoring)": (1 + results["quantile_returns"][n]).cumprod(),
        "Equal-weight universe": (1 + results["benchmark_returns"]).cumprod(),
    })
    fig_cum = px.line(cumulative, labels={"value": "Growth of 1 unit", "index": ""},
                      color_discrete_map={
                          "Q1 (highest-scoring)": QUINTILE_RAMP[0],
                          f"Q{n} (lowest-scoring)": QUINTILE_RAMP[-1],
                          "Equal-weight universe": "#898781"})
    st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("**Full metrics**")
    st.dataframe(summary.round(3), use_container_width=True, hide_index=True)

with st.expander("Data quality report"):
    st.caption(
        "Free fundamental data is noisy in specific, explainable ways. Values "
        "outside plausible ranges are treated as missing rather than as extreme "
        "observations — see src/data_loader.py."
    )
    rows = []
    for field, stats in coverage.items():
        rejected = ", ".join(f"{t}={v:.4g}" for t, v in stats["rejected_by_bounds"][:4]) or "—"
        rows.append({"field": field, "valid": f"{stats['valid']}/{stats['total']}",
                     "coverage": f"{stats['pct']}%", "rejected as implausible": rejected})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(
    "Not investment advice — a demonstration of factor methodology on live data. "
    "See the README for the full discussion of survivorship bias, data quality, "
    "and what this backtest can and cannot establish."
)
