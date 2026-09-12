"""
data_loader.py
----------------
Three live data sources, each fetched fresh on every run:

  1. Yahoo Finance (via yfinance) — daily price history for the universe.
     Drives momentum and low-volatility, and is the only input to the
     historical backtest (see backtest.py for why).
  2. Yahoo Finance — fundamental snapshots (P/E, P/B, EV/EBITDA, ROE,
     debt/equity). These are CURRENT values only: there is no point-in-time
     history available on the free tier, which is what confines the
     backtest to price-based factors.
  3. FRED — the 3-month Treasury bill rate, used as the risk-free rate in
     Sharpe calculations rather than a hard-coded assumption.

Free fundamental data is noisy in ways worth handling explicitly rather
than silently averaging over. Three real examples observed in this
universe:

  - JPM returns no EV/EBITDA and no debt/equity. This is correct, not a
    bug: for a bank, leverage is raw material rather than capital
    structure, so those ratios are not meaningful and the provider omits
    them. Financials therefore score on fewer inputs by construction.
  - BRK-B returns priceToBook ≈ 0.001, off by roughly three orders of
    magnitude (the true figure is near 1.6). Left unchecked, this single
    corrupt field would make Berkshire the cheapest stock in the universe
    by a wide margin.
  - AAPL returns ROE ≈ 1.49 (149%), which is genuine — years of buybacks
    have left very little book equity — but is an extreme value that would
    dominate a cross-sectional z-score.

SANITY_BOUNDS rejects the first class of problem (corrupt values become
missing data); winsorization in scoring.py handles the second (genuine
extremes are capped, not deleted).
"""

import io
import time
import urllib.request

import numpy as np
import pandas as pd

from src.universe import UNIVERSE

FRED_TBILL_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
FALLBACK_RISK_FREE = 0.02

# Plausible ranges for each raw fundamental. Values outside these bounds are
# treated as missing data, not as extreme observations — a P/B of 0.001 is a
# reporting error, whereas a P/B of 12 is simply an expensive stock.
SANITY_BOUNDS = {
    "trailing_pe":       (0.5, 500.0),
    "price_to_book":     (0.05, 100.0),
    "ev_to_ebitda":      (0.5, 200.0),
    "return_on_equity":  (-2.0, 2.0),
    "debt_to_equity":    (0.0, 1000.0),
    "profit_margin":     (-1.0, 1.0),
}

FUNDAMENTAL_FIELDS = {
    "trailing_pe":      "trailingPE",
    "price_to_book":    "priceToBook",
    "ev_to_ebitda":     "enterpriseToEbitda",
    "return_on_equity": "returnOnEquity",
    "debt_to_equity":   "debtToEquity",
    "profit_margin":    "profitMargins",
}


def _apply_sanity_bounds(field, value):
    """Returns the value if it is inside its plausible range, otherwise NaN."""
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value):
        return np.nan
    low, high = SANITY_BOUNDS[field]
    return value if low <= value <= high else np.nan


def load_prices(start="2015-01-01", end=None):
    """
    Daily adjusted close history for the whole universe, in one bulk
    request. Returns (prices_df, mode) where columns are tickers.
    """
    try:
        import yfinance as yf

        tickers = list(UNIVERSE.keys())
        # threads=False avoids a known yfinance issue where concurrent
        # downloads write to its internal SQLite cache simultaneously,
        # raising "OperationalError: database is locked".
        data = yf.download(tickers, start=start, end=end, progress=False,
                            threads=False, auto_adjust=True)["Close"]

        if data.empty:
            raise ValueError("Empty price history returned.")

        # Keep tickers with a usable history; report the rest rather than
        # silently dropping them.
        coverage = data.notna().mean()
        usable = coverage[coverage > 0.5].index.tolist()
        dropped = [t for t in tickers if t not in usable]
        if dropped:
            print(f"[data_loader] Dropped {len(dropped)} tickers with insufficient "
                  f"price history: {dropped}")

        data = data[usable]
        print(f"[data_loader] Price history: {len(usable)} tickers, "
              f"{data.index[0].date()} to {data.index[-1].date()}.")
        return data, "live"

    except Exception as e:
        print(f"[data_loader] ERROR: could not download price history "
              f"({type(e).__name__}: {e}).")
        return pd.DataFrame(), "failed"


def load_fundamentals(pause=0.15):
    """
    Current fundamental snapshot per ticker, with sanity bounds applied.
    Returns (fundamentals_df, coverage_report) where the report records how
    many usable values each field produced — data quality is part of the
    output, not a hidden detail.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        print(f"[data_loader] WARNING: yfinance unavailable ({e}).")
        return pd.DataFrame(), {}

    rows = {}
    rejected = {field: [] for field in FUNDAMENTAL_FIELDS}

    for ticker in UNIVERSE:
        try:
            info = yf.Ticker(ticker).info
            row = {}
            for field, yf_key in FUNDAMENTAL_FIELDS.items():
                raw = info.get(yf_key)
                clean = _apply_sanity_bounds(field, raw)
                if raw is not None and np.isnan(clean):
                    rejected[field].append((ticker, raw))
                row[field] = clean
            rows[ticker] = row
            time.sleep(pause)
        except Exception as e:
            print(f"[data_loader] {ticker}: fundamentals failed "
                  f"({type(e).__name__}: {e})")

    df = pd.DataFrame.from_dict(rows, orient="index")

    coverage = {}
    for field in FUNDAMENTAL_FIELDS:
        if field in df.columns:
            n_valid = int(df[field].notna().sum())
            coverage[field] = {
                "valid": n_valid,
                "total": len(df),
                "pct": round(100 * n_valid / len(df), 1) if len(df) else 0.0,
                "rejected_by_bounds": rejected[field],
            }

    print(f"[data_loader] Fundamentals: {len(df)} tickers.")
    for field, stats in coverage.items():
        note = ""
        if stats["rejected_by_bounds"]:
            examples = ", ".join(f"{t}={v:.4g}" for t, v in stats["rejected_by_bounds"][:3])
            note = f"  (rejected as implausible: {examples})"
        print(f"    {field:<18s} {stats['valid']:>3d}/{stats['total']} "
              f"({stats['pct']:>5.1f}%){note}")

    return df, coverage


def load_risk_free_rate():
    """Live 3-month T-bill rate from FRED — second independent source,
    used for Sharpe ratios instead of a static assumption."""
    try:
        with urllib.request.urlopen(FRED_TBILL_CSV, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "value"]
        df = df[df["value"] != "."]
        rate = float(df["value"].iloc[-1]) / 100
        print(f"[data_loader] Risk-free rate (FRED, 3M T-bill): {rate*100:.2f}%")
        return rate, "live"
    except Exception as e:
        print(f"[data_loader] WARNING: FRED risk-free rate unavailable "
              f"({type(e).__name__}: {e}). Using {FALLBACK_RISK_FREE*100:.1f}%.")
        return FALLBACK_RISK_FREE, "fallback"
