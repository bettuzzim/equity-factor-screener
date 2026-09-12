"""
factors.py
------------
Raw factor construction. Every factor here is returned already oriented so
that HIGHER IS BETTER, which lets scoring.py standardize everything without
tracking per-factor sign conventions.

Two design choices worth stating explicitly:

**Ratios are inverted, not negated.** Cheapness is expressed as earnings
yield (E/P), book-to-price (B/P) and EBITDA/EV rather than as minus-P/E,
minus-P/B, minus-EV/EBITDA. Price ratios are bounded below by zero and have
a long right tail, so their cross-sectional distribution is badly skewed and
a z-score of the negated ratio is driven almost entirely by the most
expensive name in the universe. Their reciprocals are far better behaved,
which is why yield-form value factors are the practitioner standard.

**Leverage enters as 1/(1+D/E).** Same reasoning: debt-to-equity is
unbounded above, so negating it hands the factor to whichever company is
most levered. The reciprocal form is monotonically decreasing in leverage
and bounded on (0, 1].

Price-based factors take an `as_of` date and look only at data up to that
point, so the same functions serve both the live screen and the historical
backtest without a separate code path that could silently disagree.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTH = 21  # trading days


def momentum(prices, as_of=None, lookback_months=12, skip_months=1):
    """
    Standard (lookback - skip) price momentum: total return over the
    lookback window, excluding the most recent `skip_months`.

    The skip is not decoration. At the one-month horizon equities exhibit
    short-term reversal, which is a distinct and opposing effect; including
    the most recent month mixes a reversal signal into a momentum factor and
    measurably degrades it. Jegadeesh and Titman (1993) construct momentum
    this way for exactly this reason.
    """
    window = prices.loc[:as_of] if as_of is not None else prices

    end_offset = skip_months * MONTH
    start_offset = lookback_months * MONTH

    if len(window) < start_offset + 1:
        return pd.Series(np.nan, index=prices.columns)

    end_price = window.iloc[-1 - end_offset] if end_offset else window.iloc[-1]
    start_price = window.iloc[-1 - start_offset]

    return (end_price / start_price - 1).replace([np.inf, -np.inf], np.nan)


def low_volatility(prices, as_of=None, lookback_months=12):
    """
    Inverse of trailing annualized realized volatility, so that calmer
    stocks score higher. Returned as -volatility rather than 1/volatility:
    volatility has no mass near zero in an equity cross-section, so it does
    not have the skew problem that motivates inverting the price ratios, and
    the negated form keeps the units interpretable.
    """
    window = prices.loc[:as_of] if as_of is not None else prices
    window = window.iloc[-(lookback_months * MONTH):]

    if len(window) < 60:
        return pd.Series(np.nan, index=prices.columns)

    daily_returns = window.pct_change()
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    return -vol.replace([np.inf, -np.inf], np.nan)


def price_factors(prices, as_of=None):
    """
    All factors computable from price history alone, at a given point in
    time. These are the only factors the historical backtest can use — see
    backtest.py.
    """
    return pd.DataFrame({
        "momentum_12_1": momentum(prices, as_of, lookback_months=12, skip_months=1),
        "momentum_6_1": momentum(prices, as_of, lookback_months=6, skip_months=1),
        "low_volatility": low_volatility(prices, as_of, lookback_months=12),
    })


def fundamental_factors(fundamentals):
    """
    Value and quality inputs derived from the current fundamental snapshot,
    oriented so higher is better. These describe the universe as it is
    today; no point-in-time history exists for them on free data, so they
    inform the live screen only.
    """
    f = fundamentals

    return pd.DataFrame({
        # Value — yield form (see module docstring)
        "earnings_yield": 1.0 / f["trailing_pe"],
        "book_to_price": 1.0 / f["price_to_book"],
        "ebitda_to_ev": 1.0 / f["ev_to_ebitda"],
        # Quality — profitability and safety
        "return_on_equity": f["return_on_equity"],
        "profit_margin": f["profit_margin"],
        "low_leverage": 1.0 / (1.0 + f["debt_to_equity"]),
    }).replace([np.inf, -np.inf], np.nan)


# Which raw factors roll up into which reported factor group.
FACTOR_GROUPS = {
    "Value": ["earnings_yield", "book_to_price", "ebitda_to_ev"],
    "Quality": ["return_on_equity", "profit_margin", "low_leverage"],
    "Momentum": ["momentum_12_1", "momentum_6_1"],
    "Low Volatility": ["low_volatility"],
}

# The subset of groups the historical backtest can evaluate without
# look-ahead bias.
PRICE_BASED_GROUPS = ["Momentum", "Low Volatility"]
