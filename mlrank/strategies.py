"""Human-designed (rule) strategies.

Each turns a panel (one row per month T and
stock, features at the T-1 close) into a score;
higher = better. The backtest buys the top N.
Raw (not z-scored) inputs, as in the live code
(indian_stocks_monitor/momentum_rank.py):

  c3         mean(12-1m, 6m, 3m return) / 126-day vol
             (the live strategy)
  mom12_vol  12-month return / 126-day vol
             (simpler C3: one lookback instead of three)
  low_vol    -126-day vol (defensive: calmest stocks)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _vol(p: pd.DataFrame) -> pd.Series:
    return p["vol_126"].where(p["vol_126"] > 0)


RULES = {
    "c3": lambda p: (p["raw_mom_12_1"] + p["raw_mom_6m"] + p["raw_mom_3m"]) / 3 / _vol(p),
    "mom12_vol": lambda p: p["raw_mom_12m"] / _vol(p),
    "low_vol": lambda p: -_vol(p),
}


def eligible(panel: pd.DataFrame) -> pd.Series:
    """Same universe for every strategy, as in the
    live code: >= 253 days of history (12-1m return
    exists) and a positive 126-day volatility."""
    return panel["raw_mom_12_1"].notna() & (panel["vol_126"] > 0)


def rule_scores(panel: pd.DataFrame, names=None) -> pd.DataFrame:
    names = names or list(RULES)
    out = pd.DataFrame({n: RULES[n](panel) for n in names}, index=panel.index)
    return out.replace([np.inf, -np.inf], np.nan)
