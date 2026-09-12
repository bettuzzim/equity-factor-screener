"""
scoring.py
------------
Cross-sectional standardization and composite ranking.

Raw factors live on incompatible scales — an earnings yield of 0.05 and a
momentum of 0.30 cannot be averaged directly — so each is converted to a
cross-sectional z-score before being combined.

**Outliers are handled in two stages, deliberately.** Raw values are first
winsorized against a robust centre and spread (median ± 3 rescaled MADs),
then z-scored, then the resulting z-scores are clipped at ±3. The first
stage matters more than it looks: an extreme observation contaminates the
mean and standard deviation used to standardize every other stock, so
clipping only after the fact leaves the damage already done. Winsorizing
first means the distribution is computed from a sane sample; the final clip
is a light backstop.

The robust bound is not interchangeable with a percentile one here — see
`_winsorize` for why a 1st/99th percentile cut silently stops working at
this universe size.

**Sector-neutral scoring is available and off by default.** Comparing every
stock against the whole universe means a value screen mostly returns banks
and energy, and a quality screen mostly returns software — the factor ends
up expressing a sector bet rather than a stock-selection view. Scoring
within sector removes that. The reason it is not the default here is sample
size: this universe carries 3-8 names per sector, and a z-score computed
across three observations is not a meaningful statistic. Sectors below
MIN_SECTOR_SIZE therefore fall back to universe-wide scoring, and the run
reports which ones did.
"""

import numpy as np
import pandas as pd

from src.factors import FACTOR_GROUPS

ZSCORE_CLIP = 3.0
MAD_CLIP = 3.0
MAD_TO_SIGMA = 1.4826  # makes MAD comparable to a standard deviation under normality
MIN_SECTOR_SIZE = 5
MIN_GROUPS_REQUIRED = 3


def _winsorize(series, n_mad=MAD_CLIP):
    """
    Clips to median ± n_mad robust deviations, where the robust deviation is
    the median absolute deviation rescaled by 1.4826 so it matches a standard
    deviation for normally distributed data.

    A percentile winsorization (the more common first instinct, and what this
    function used to do) fails on samples this size. At N=53 the 99th
    percentile is interpolated between the largest and second-largest
    observation, so the bound lands right next to the outlier it is supposed
    to contain: a value 1000x out of scale gets clipped to roughly 900x out of
    scale, and the mean and standard deviation are contaminated anyway. The
    breakdown is silent — the code runs, the numbers look plausible, and the
    stated protection simply is not there.

    Median and MAD have a 50% breakdown point, so the bound is set by the bulk
    of the distribution regardless of how extreme the tail is or how few names
    are in the universe.
    """
    valid = series.dropna()
    if len(valid) < 5:
        return series

    median = valid.median()
    mad = (valid - median).abs().median() * MAD_TO_SIGMA

    if not np.isfinite(mad) or mad == 0:
        # Degenerate spread (e.g. most values identical). Fall back to the
        # empirical range rather than collapsing every observation onto the
        # median.
        return series.clip(lower=valid.min(), upper=valid.max())

    return series.clip(lower=median - n_mad * mad, upper=median + n_mad * mad)


def zscore(series, clip=ZSCORE_CLIP):
    """Winsorize, standardize, then clip. Returns NaN where input is NaN."""
    winsorized = _winsorize(series)
    mean, std = winsorized.mean(), winsorized.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return ((winsorized - mean) / std).clip(-clip, clip)


def sector_neutral_zscore(series, sectors, min_sector_size=MIN_SECTOR_SIZE):
    """
    Z-score within each sector, falling back to universe-wide scoring for
    sectors too small to standardize meaningfully. Returns (scores,
    fallback_sectors).
    """
    scores = pd.Series(np.nan, index=series.index)
    fallback_sectors = []

    for sector in sectors.unique():
        members = sectors[sectors == sector].index
        members = [m for m in members if m in series.index]
        subset = series.loc[members]

        if subset.notna().sum() >= min_sector_size:
            scores.loc[members] = zscore(subset)
        else:
            fallback_sectors.append(sector)

    if fallback_sectors:
        fallback_members = sectors[sectors.isin(fallback_sectors)].index
        fallback_members = [m for m in fallback_members if m in series.index]
        scores.loc[fallback_members] = zscore(series).loc[fallback_members]

    return scores, fallback_sectors


def score_factors(raw_factors, sectors=None, sector_neutral=False):
    """
    Standardizes every raw factor column. Returns (z_scores, fallback_info).
    """
    z_scores = pd.DataFrame(index=raw_factors.index)
    fallback_info = {}

    for column in raw_factors.columns:
        if sector_neutral and sectors is not None:
            z_scores[column], fallbacks = sector_neutral_zscore(
                raw_factors[column], sectors)
            if fallbacks:
                fallback_info[column] = fallbacks
        else:
            z_scores[column] = zscore(raw_factors[column])

    return z_scores, fallback_info


def group_scores(z_scores, groups=None):
    """
    Averages the z-scores inside each factor group, using whichever inputs
    are available for that stock.

    Averaging over available inputs rather than requiring all of them is a
    deliberate choice: financials structurally lack EV/EBITDA and
    debt/equity, so demanding complete data would drop an entire sector
    from the screen rather than scoring it on the inputs that do apply.
    The cost is that a stock scored on one input carries more estimation
    noise than one scored on three — coverage is reported alongside the
    scores so this stays visible.
    """
    groups = groups or FACTOR_GROUPS
    out = pd.DataFrame(index=z_scores.index)
    coverage = pd.DataFrame(index=z_scores.index)

    for group_name, columns in groups.items():
        present = [c for c in columns if c in z_scores.columns]
        if not present:
            continue
        out[group_name] = z_scores[present].mean(axis=1, skipna=True)
        coverage[group_name] = z_scores[present].notna().sum(axis=1)

    return out, coverage


def composite_score(group_scores_df, weights=None,
                    min_groups=MIN_GROUPS_REQUIRED):
    """
    Weighted average of the factor groups. Equal-weighted by default:
    with no reliable way to estimate forward factor premia from this data,
    an equal weighting is the honest prior — tuned weights fitted on the
    same sample they are evaluated on would be an in-sample illusion.

    Stocks scored on fewer than `min_groups` groups are returned as NaN
    rather than ranked on thin evidence.
    """
    groups = list(group_scores_df.columns)
    if weights is None:
        weights = {g: 1.0 for g in groups}

    weight_series = pd.Series({g: weights.get(g, 0.0) for g in groups})
    values = group_scores_df[groups]
    mask = values.notna()

    weighted_sum = (values.fillna(0) * weight_series).sum(axis=1)
    weight_total = (mask * weight_series).sum(axis=1)

    composite = weighted_sum / weight_total.replace(0, np.nan)
    composite[mask.sum(axis=1) < min_groups] = np.nan
    return composite


def assign_quantiles(scores, n=5, labels=None):
    """
    Sorts scores into n buckets, 1 = highest-scoring. Returns NaN where the
    score is NaN.
    """
    valid = scores.dropna()
    if len(valid) < n:
        return pd.Series(np.nan, index=scores.index)

    ranks = valid.rank(ascending=False, method="first")
    buckets = pd.qcut(ranks, n, labels=labels or range(1, n + 1))

    out = pd.Series(np.nan, index=scores.index, dtype="object")
    out.loc[valid.index] = buckets
    return out
