"""
main.py
--------
End-to-end pipeline:

  1. Download live price history and a current fundamental snapshot for the
     universe, plus the risk-free rate from FRED
  2. Build the live four-factor screen (value, quality, momentum, low
     volatility) — cross-sectional z-scores, winsorized, combined into a
     composite and sorted into quintiles
  3. Backtest the price-based half of the model (momentum, low volatility)
     out of sample, rebalanced quarterly, with factors recomputed at each
     date from data available only up to that date
  4. Save charts and print both the screen and the backtest summary

The split between step 2 and step 3 is deliberate and is explained in
src/backtest.py: fundamentals are a present-day snapshot, so including them
in a historical backtest would be look-ahead bias.

Usage:
  python main.py
  python main.py --sector-neutral      # score within sector where samples allow
  python main.py --top-n 25
"""

import argparse
import os

import pandas as pd

from src.universe import UNIVERSE, AS_OF_DATE
from src.data_loader import load_prices, load_fundamentals, load_risk_free_rate
from src.factors import price_factors, fundamental_factors
from src.scoring import score_factors, group_scores, composite_score, assign_quantiles
from src.backtest import run_backtest, summarize_backtest, factor_attribution
from src.viz import (plot_factor_correlation, plot_quintile_performance,
                      plot_cumulative_performance, plot_composite_ranking,
                      plot_factor_profile, plot_factor_attribution)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def build_screen(prices, fundamentals, sectors, sector_neutral=False):
    """Combines price-based and fundamental factors into the live screen."""
    raw = pd.concat([price_factors(prices), fundamental_factors(fundamentals)], axis=1)
    raw = raw.loc[raw.index.intersection(prices.columns)]

    z_scores, fallbacks = score_factors(raw, sectors=sectors,
                                         sector_neutral=sector_neutral)
    groups, coverage = group_scores(z_scores)
    composite = composite_score(groups)
    quintiles = assign_quantiles(composite)

    screen = groups.copy()
    screen["composite"] = composite
    screen["quintile"] = quintiles
    screen["sector"] = sectors.reindex(screen.index)
    screen = screen.sort_values("composite", ascending=False)

    return screen, groups, coverage, fallbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector-neutral", action="store_true",
                        help="Z-score within sector where sample size allows.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="How many names to chart in the ranking.")
    args = parser.parse_args()

    print(f"\n### STEP 1 — Live data ({len(UNIVERSE)} tickers, universe as of {AS_OF_DATE}) ###")
    prices, price_mode = load_prices()
    if price_mode != "live":
        raise SystemExit("Cannot continue without price history.")

    fundamentals, coverage_report = load_fundamentals()
    risk_free_rate, rf_mode = load_risk_free_rate()

    sectors = pd.Series(UNIVERSE)

    print("\n### STEP 2 — Live four-factor screen ###")
    if args.sector_neutral:
        print("Scoring within sector where sample size allows.")
    screen, groups, coverage, fallbacks = build_screen(
        prices, fundamentals, sectors, sector_neutral=args.sector_neutral)

    if fallbacks:
        affected = sorted({s for sectors_list in fallbacks.values() for s in sectors_list})
        print(f"Sectors too small for within-sector scoring, fell back to "
              f"universe-wide: {', '.join(affected)}")

    display_cols = ["sector", "Value", "Quality", "Momentum", "Low Volatility",
                    "composite", "quintile"]
    print(screen[display_cols].head(args.top_n).round(2).to_string())

    n_unscored = int(screen["composite"].isna().sum())
    if n_unscored:
        print(f"\n{n_unscored} tickers left unscored (fewer than 3 factor groups available).")

    print("\n### STEP 3 — Backtest (price-based factors only) ###")
    print("Momentum and low volatility only — fundamentals are a present-day")
    print("snapshot and cannot be used historically without look-ahead bias.")

    results = run_backtest(prices)
    summary = summarize_backtest(results, risk_free_rate=risk_free_rate)

    first = pd.Timestamp(results["rebalance_dates"][0]).date()
    last = pd.Timestamp(results["rebalance_dates"][-1]).date()
    print(f"\n{len(results['rebalance_dates'])} quarterly rebalances, {first} to {last}, "
          f"risk-free {risk_free_rate*100:.2f}% ({rf_mode})")

    printable = summary.copy()
    for col in ("annualized_return", "annualized_vol", "max_drawdown", "total_return"):
        printable[col] = (printable[col] * 100).round(2)
    printable["sharpe"] = printable["sharpe"].round(2)
    print(printable.to_string(index=False))

    spread = summary.attrs["spread_bps"]
    monotonic = summary.attrs["monotonic"]
    print(f"\nQ1 - Q5 annualized spread: {spread:.0f} bps")
    print(f"Monotonic across quintiles: {'yes' if monotonic else 'no'}")
    if not monotonic:
        print("A non-monotonic ranking means the composite separates the extremes")
        print("but does not order the middle buckets — worth stating, not hiding.")

    print("\n### STEP 4 — Factor attribution ###")
    print("Each factor backtested alone. A blended result can hide one factor")
    print("working and another failing, which cancel into what looks like noise.")

    attribution = factor_attribution(prices, risk_free_rate=risk_free_rate)
    printable_attr = attribution.copy()
    for col in ("q1_return", "q5_return"):
        printable_attr[col] = (printable_attr[col] * 100).round(2)
    printable_attr["spread_bps"] = printable_attr["spread_bps"].round(0)
    printable_attr["q1_sharpe"] = printable_attr["q1_sharpe"].round(2)
    print(printable_attr.to_string(index=False))
    print(f"\nEqual-weight universe benchmark: "
          f"{attribution.attrs['benchmark_return']*100:.2f}% annualized, "
          f"Sharpe {attribution.attrs['benchmark_sharpe']:.2f}")

    print("\n### STEP 5 — Charts ###")
    charts = [
        ("factor_correlations.png", lambda p: plot_factor_correlation(groups, p)),
        ("factor_attribution.png", lambda p: plot_factor_attribution(attribution, p)),
        ("quintile_performance.png",
         lambda p: plot_quintile_performance(summary, results["n_quantiles"], p)),
        ("cumulative_performance.png", lambda p: plot_cumulative_performance(results, p)),
        ("composite_ranking.png",
         lambda p: plot_composite_ranking(screen, p, top_n=args.top_n)),
        ("factor_profile.png", lambda p: plot_factor_profile(screen, groups, p)),
    ]
    for filename, plot_fn in charts:
        path = os.path.join(OUTPUT_DIR, filename)
        plot_fn(path)
        print(f"Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
