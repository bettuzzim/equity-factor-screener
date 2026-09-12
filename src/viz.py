"""
viz.py
-------
Five charts, one shared style, colors assigned by the job each encoding
does rather than by preference.

  - Quintiles are an ORDINAL sequence (Q1 through Q5 is a ranking, not five
    unrelated categories), so they take a single-hue ramp running dark to
    light. Colouring them as independent categories would throw away the
    ordering the chart exists to show, and a green-to-red scheme would imply
    "Q5 is bad" when Q5 is simply the low-scoring bucket.
  - Correlations and factor z-scores are DIVERGING around a meaningful zero,
    so they take two hues with a neutral grey midpoint.
  - The benchmark line is reference chrome, not a competing series, so it is
    muted grey rather than a third identity colour.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Sequential blue ramp, dark (best) to light (worst) — quintiles are ordered.
QUINTILE_RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]

DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_NEGATIVE = "#e34948"
DIVERGING_MIDPOINT = "#f0efec"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "rv_diverging", [DIVERGING_NEGATIVE, DIVERGING_MIDPOINT, DIVERGING_POSITIVE])


def _apply_style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.grid(color=GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def plot_factor_correlation(group_scores_df, save_path):
    """
    Correlation between the factor groups themselves. This is a
    diversification check on the composite: factors that move together are
    one bet wearing several names, and an equal-weighted average of them
    quietly concentrates rather than diversifies.
    """
    corr = group_scores_df.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(corr.values, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)

    labels = corr.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color=INK_SECONDARY)
    ax.set_yticklabels(labels, fontsize=9, color=INK_SECONDARY)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.55 else INK_PRIMARY)

    ax.set_title("Factor Group Correlations", fontsize=12, weight="bold", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_quintile_performance(summary_df, n_quantiles, save_path):
    """
    Annualized return by quintile — the monotonicity test. A factor that
    carries information ranks the buckets in order; a good top bucket on its
    own can be a handful of lucky names.
    """
    quintiles = summary_df.iloc[:n_quantiles]
    returns_pct = quintiles["annualized_return"].values * 100

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _apply_style(ax, fig)

    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    bars = ax.bar(labels, returns_pct, color=QUINTILE_RAMP[:n_quantiles], width=0.62)

    for bar, value in zip(bars, returns_pct):
        offset = 0.4 if value >= 0 else -1.2
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.1f}%",
                ha="center", fontsize=9, color=INK_SECONDARY)

    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_ylabel("Annualized return (%)")
    ax.set_xlabel("Composite score quintile  (Q1 = highest-scoring)")

    monotonic = summary_df.attrs.get("monotonic", False)
    verdict = "monotonic across quintiles" if monotonic else "not monotonic across quintiles"
    ax.set_title(f"Backtest — Return by Quintile\nPrice-based composite, quarterly rebalance · {verdict}",
                 fontsize=12, weight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cumulative_performance(results, save_path):
    """Top quintile vs bottom quintile vs the equal-weighted universe."""
    n = results["n_quantiles"]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    _apply_style(ax, fig)

    def cumulative(series):
        return (1 + series.dropna()).cumprod()

    top = cumulative(results["quantile_returns"][1])
    bottom = cumulative(results["quantile_returns"][n])
    benchmark = cumulative(results["benchmark_returns"])

    ax.plot(benchmark.index, benchmark.values, color=INK_MUTED, lw=1.6,
            ls="--", label="Equal-weight universe")
    ax.plot(bottom.index, bottom.values, color=QUINTILE_RAMP[-1], lw=1.9,
            label=f"Q{n} (lowest-scoring)")
    ax.plot(top.index, top.values, color=QUINTILE_RAMP[0], lw=2.2,
            label="Q1 (highest-scoring)")

    ax.set_ylabel("Growth of 1 unit invested")
    ax.set_title("Backtest — Cumulative Performance",
                 fontsize=12, weight="bold", loc="left")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_factor_attribution(attribution_df, save_path):
    """
    Q1 - Q5 spread for each factor run on its own, and for the combination.

    Diverging colour is the right encoding here because the sign is the
    whole point: a positive spread means the factor ranked stocks in the
    intended direction, a negative one means it ranked them backwards, and
    those are qualitatively different outcomes rather than two ends of a
    magnitude scale.
    """
    df = attribution_df.iloc[::-1]
    spreads = df["spread_bps"].values
    colors = [DIVERGING_POSITIVE if s > 0 else DIVERGING_NEGATIVE for s in spreads]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    _apply_style(ax, fig)

    bars = ax.barh(df["factor"], spreads, color=colors, height=0.6)
    ax.axvline(0, color=INK_PRIMARY, lw=1.1)

    for bar, value in zip(bars, spreads):
        offset = 90 if value >= 0 else -90
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                f"{value:+.0f}", va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9, color=INK_SECONDARY)

    ax.set_xlabel("Annualized Q1 - Q5 spread (bps)\n"
                 "positive = factor ranked stocks in the intended direction · "
                 "negative = ranked them backwards")
    ax.set_title("Factor Attribution — Each Factor Backtested Alone",
                 fontsize=12, weight="bold", loc="left")

    margin = max(abs(spreads.min()), abs(spreads.max())) * 0.25
    ax.set_xlim(spreads.min() - margin, spreads.max() + margin)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_composite_ranking(screen_df, save_path, top_n=20):
    """Current screen: the highest composite scores, coloured by quintile."""
    ranked = screen_df.dropna(subset=["composite"]).nlargest(top_n, "composite")
    ranked = ranked.iloc[::-1]

    colors = [QUINTILE_RAMP[min(int(q) - 1, len(QUINTILE_RAMP) - 1)]
              if pd.notna(q) else INK_MUTED for q in ranked["quintile"]]

    fig, ax = plt.subplots(figsize=(8.5, 8))
    _apply_style(ax, fig)

    ax.barh(ranked.index, ranked["composite"], color=colors, height=0.7)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_xlabel("Composite factor score (cross-sectional z-score, equal-weighted across groups)")
    ax.set_title(f"Live Screen — Top {top_n} by Composite Score",
                 fontsize=12, weight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_factor_profile(screen_df, group_scores_df, save_path, top_n=15):
    """
    Factor profile of the top-ranked names. A high composite can come from
    one dominant exposure or from broad strength across all four; the
    composite number alone hides which, and they are different bets.
    """
    ranked = screen_df.dropna(subset=["composite"]).nlargest(top_n, "composite")
    profile = group_scores_df.loc[ranked.index]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(profile.values, cmap=DIVERGING_CMAP, vmin=-2.5, vmax=2.5,
                   aspect="auto")

    ax.set_xticks(range(len(profile.columns)))
    ax.set_xticklabels(profile.columns, rotation=25, ha="right", fontsize=9,
                       color=INK_SECONDARY)
    ax.set_yticks(range(len(profile.index)))
    ax.set_yticklabels(profile.index, fontsize=9, color=INK_SECONDARY)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    for i in range(len(profile.index)):
        for j in range(len(profile.columns)):
            value = profile.values[i, j]
            if np.isnan(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5,
                        color=INK_MUTED)
            else:
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 1.4 else INK_PRIMARY)

    ax.set_title(f"Factor Profile — Top {top_n} Names\nz-score by factor group",
                 fontsize=12, weight="bold", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
