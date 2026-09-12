"""
universe.py
-------------
A real, liquid stock universe spanning all 11 GICS sectors — large-cap S&P
500 constituents, chosen for reliable data availability rather than being
an exhaustive index replica.

AS_OF_DATE documents when this list was curated. Constituent membership
changes over time; using today's list to look at history is a form of
survivorship bias — see the README's "Methodology notes and limitations"
for the full discussion. A fully rigorous version would need point-in-time
index membership, which isn't reliably available via free data sources.
"""

AS_OF_DATE = "2026-09"

UNIVERSE = {
    # --- Information Technology ---
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "CSCO": "Information Technology", "ORCL": "Information Technology",
    "ADBE": "Information Technology", "CRM": "Information Technology",
    # --- Health Care ---
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
    # --- Financials ---
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    # --- Consumer Discretionary ---
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    # --- Consumer Staples ---
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    # --- Industrials ---
    "GE": "Industrials", "CAT": "Industrials", "RTX": "Industrials",
    "UNP": "Industrials", "HON": "Industrials",
    # --- Energy ---
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # --- Utilities ---
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # --- Materials ---
    "LIN": "Materials", "SHW": "Materials", "FCX": "Materials",
    # --- Real Estate ---
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    # --- Communication Services ---
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "VZ": "Communication Services",
}

# Fixed, colorblind-checked 8+ hue order isn't practical for 11 sectors in
# one chart — see viz.py for how sector color is handled (folded to a
# muted/highlight scheme rather than 11 distinct hues).
SECTORS = sorted(set(UNIVERSE.values()))
