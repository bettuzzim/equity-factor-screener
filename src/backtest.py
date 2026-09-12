"""
backtest.py
-------------
Quarterly-rebalanced quintile portfolios, evaluated out of sample.

**Why the backtest uses only price-based factors.** Value and quality come
from a fundamental snapshot describing the universe as it is today. Ranking
2017 on today's P/E would mean forming a 2017 portfolio using information
that did not exist until years later — look-ahead bias in its purest form,
and the single most common way a factor backtest produces a Sharpe ratio
that cannot survive contact with reality. Point-in-time fundamental history
is a paid data product; it is not available here, so the honest response is
to restrict the backtest to momentum and low volatility, which are
reconstructible from price history at any past date without contamination,
and to present the four-factor model as a live screen rather than a
validated historical strategy. The alternative — running all four factors
anyway and noting the caveat in small print — would produce better-looking
numbers and worse work.

At each rebalance date the same functions that drive the live screen
recompute factors using only data up to that date, so the two paths cannot
silently disagree.

Between rebalances positions are held, not continuously re-equalized, so
weights drift with performance exactly as they would in a real book.
"""

import numpy as np
import pandas as pd

from src.factors import price_factors
from src.scoring import score_factors, group_scores, composite_score, assign_quantiles

TRADING_DAYS = 252
PRICE_GROUPS = {
    "Momentum": ["momentum_12_1", "momentum_6_1"],
    "Low Volatility": ["low_volatility"],
}


def quarterly_rebalance_dates(prices, warmup_months=12):
    """
    Quarter-start trading days, beginning only once enough history exists to
    compute a 12-month momentum factor. The warmup is why the backtest
    starts later than the price history.
    """
    warmup_end = prices.index[0] + pd.DateOffset(months=warmup_months)
    eligible = prices.index[prices.index >= warmup_end]
    if len(eligible) == 0:
        return []

    quarters = pd.Series(eligible, index=eligible).resample("QS").first().dropna()
    return list(quarters.values)


def _period_returns(prices, tickers, start, end):
    """
    Daily returns of an equal-weighted, buy-and-hold basket held from
    `start` to `end`. Weights are set at `start` and allowed to drift.
    """
    window = prices.loc[start:end, tickers].dropna(axis=1, how="any")
    if window.empty or window.shape[1] == 0 or len(window) < 2:
        return pd.Series(dtype=float)

    normalized = window / window.iloc[0]
    basket_value = normalized.mean(axis=1)
    return basket_value.pct_change().dropna()


def run_backtest(prices, n_quantiles=5, warmup_months=12, factor_groups=None):
    """
    Forms quintile portfolios on a price-based composite at each quarter
    start and holds them until the next.

    `factor_groups` defaults to the full price-based composite; passing a
    single group runs the same machinery on that factor alone, which is how
    attribution below separates factors that a composite would blend.

    Returns a dict with a daily return series per quintile, the long/short
    (Q1 - Q5) series, an equal-weighted universe benchmark, and the
    rebalance history.
    """
    factor_groups = factor_groups or PRICE_GROUPS

    rebalance_dates = quarterly_rebalance_dates(prices, warmup_months)
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough price history for a quarterly backtest.")

    quantile_returns = {q: [] for q in range(1, n_quantiles + 1)}
    benchmark_returns = []
    history = []

    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        raw = price_factors(prices, as_of=rebalance_date)
        z_scores, _ = score_factors(raw)
        groups, _ = group_scores(z_scores, groups=factor_groups)
        composite = composite_score(groups, min_groups=len(factor_groups))
        quantiles = assign_quantiles(composite, n=n_quantiles)

        selected = quantiles.dropna()
        if selected.empty:
            continue

        for q in range(1, n_quantiles + 1):
            members = selected[selected == q].index.tolist()
            if members:
                quantile_returns[q].append(
                    _period_returns(prices, members, rebalance_date, next_date))

        benchmark_returns.append(
            _period_returns(prices, list(selected.index), rebalance_date, next_date))

        history.append({
            "date": pd.Timestamp(rebalance_date).date(),
            "n_scored": int(len(selected)),
            "top_quintile": selected[selected == 1].index.tolist(),
        })

    series = {}
    for q, chunks in quantile_returns.items():
        chunks = [c for c in chunks if not c.empty]
        series[q] = pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)

    benchmark_chunks = [c for c in benchmark_returns if not c.empty]
    benchmark = pd.concat(benchmark_chunks).sort_index() if benchmark_chunks else pd.Series(dtype=float)

    long_short = (series[1] - series[n_quantiles]).dropna()

    return {
        "quantile_returns": series,
        "long_short_returns": long_short,
        "benchmark_returns": benchmark,
        "history": history,
        "rebalance_dates": rebalance_dates,
        "n_quantiles": n_quantiles,
    }


def performance_metrics(returns, risk_free_rate=0.0):
    """Annualized return, volatility, Sharpe ratio and maximum drawdown."""
    returns = returns.dropna()
    if len(returns) < 2:
        return {k: np.nan for k in
                ("annualized_return", "annualized_vol", "sharpe", "max_drawdown", "total_return")}

    cumulative = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS

    total_return = cumulative.iloc[-1] - 1
    annualized_return = cumulative.iloc[-1] ** (1 / years) - 1
    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ((annualized_return - risk_free_rate) / annualized_vol
              if annualized_vol > 0 else np.nan)

    drawdown = cumulative / cumulative.cummax() - 1

    return {
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_return": total_return,
    }


def summarize_backtest(results, risk_free_rate=0.0):
    """
    Per-quintile metrics plus the long/short spread and benchmark.

    The monotonicity of annualized return across quintiles is the real test
    of whether the composite carries information: a working factor should
    rank the quintiles in order, not merely produce a good top bucket, which
    a handful of lucky names can do on their own.
    """
    rows = []
    n = results["n_quantiles"]

    for q in range(1, n + 1):
        metrics = performance_metrics(results["quantile_returns"][q], risk_free_rate)
        rows.append({"portfolio": f"Q{q}" + (" (top)" if q == 1 else
                                             " (bottom)" if q == n else ""), **metrics})

    rows.append({"portfolio": f"Long/Short (Q1-Q{n})",
                 **performance_metrics(results["long_short_returns"], risk_free_rate)})
    rows.append({"portfolio": "Equal-weight universe",
                 **performance_metrics(results["benchmark_returns"], risk_free_rate)})

    df = pd.DataFrame(rows)

    quintile_returns = df.iloc[:n]["annualized_return"].values
    df.attrs["monotonic"] = bool(np.all(np.diff(quintile_returns) <= 0))
    df.attrs["spread_bps"] = float((quintile_returns[0] - quintile_returns[-1]) * 10000)

    return df


def factor_attribution(prices, risk_free_rate=0.0, n_quantiles=5):
    """
    Runs the same quintile backtest on each factor in isolation, then on the
    combination.

    A composite that underperforms tells you almost nothing on its own,
    because averaging two factors can hide one working and one failing —
    they cancel, and the blended result looks like noise. Splitting them is
    the difference between reporting that a model did not work and
    understanding why. Each row here is a separate point-in-time backtest,
    not a decomposition of the combined one.
    """
    rows = []

    for group_name, columns in PRICE_GROUPS.items():
        results = run_backtest(prices, n_quantiles=n_quantiles,
                               factor_groups={group_name: columns})
        summary = summarize_backtest(results, risk_free_rate)
        top = summary.iloc[0]
        bottom = summary.iloc[n_quantiles - 1]
        rows.append({
            "factor": group_name,
            "q1_return": top["annualized_return"],
            "q5_return": bottom["annualized_return"],
            "spread_bps": summary.attrs["spread_bps"],
            "q1_sharpe": top["sharpe"],
            "monotonic": summary.attrs["monotonic"],
        })

    combined = run_backtest(prices, n_quantiles=n_quantiles)
    combined_summary = summarize_backtest(combined, risk_free_rate)
    rows.append({
        "factor": "Combined (equal-weighted)",
        "q1_return": combined_summary.iloc[0]["annualized_return"],
        "q5_return": combined_summary.iloc[n_quantiles - 1]["annualized_return"],
        "spread_bps": combined_summary.attrs["spread_bps"],
        "q1_sharpe": combined_summary.iloc[0]["sharpe"],
        "monotonic": combined_summary.attrs["monotonic"],
    })

    benchmark = performance_metrics(combined["benchmark_returns"], risk_free_rate)
    attribution = pd.DataFrame(rows)
    attribution.attrs["benchmark_return"] = benchmark["annualized_return"]
    attribution.attrs["benchmark_sharpe"] = benchmark["sharpe"]

    return attribution
