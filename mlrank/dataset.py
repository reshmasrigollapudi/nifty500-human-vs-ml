"""Build the modeling panel: features + labels.

Label for row (T, s): return from the T open to
the next rebalance open, matching live execution.
If s stops trading before the next rebalance,
exit at its last close (delisting-aware).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data_loader import MarketData, decision_dates, rebalance_dates
from .features import FEATURES, build_feature_panel

N_QUANTILES = 5


def forward_returns(md: MarketData, rebal: pd.DatetimeIndex) -> pd.DataFrame:
    cal = md.dates
    entry = md.open.reindex(rebal)
    exit_open = md.open.reindex(rebal).shift(-1)
    last_close = md.close.ffill()
    prev_of_next = pd.DatetimeIndex(
        [cal[cal.get_loc(d) - 1] for d in rebal[1:]] + [rebal[-1]]
    )
    fallback = last_close.reindex(prev_of_next)
    fallback.index = rebal
    exit_px = exit_open.fillna(fallback)
    fwd = exit_px / entry - 1
    fwd.iloc[-1] = np.nan          # last date: unknown future
    return fwd


def add_labels(panel: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    stacked = fwd.stack(future_stack=True)
    stacked.index.names = ["date", "symbol"]
    panel = panel.copy()
    panel["fwd_ret"] = stacked.reindex(panel.index).to_numpy()
    d = panel.index.get_level_values("date")
    g = panel["fwd_ret"].groupby(d)
    panel["fwd_excess"] = panel["fwd_ret"] - g.transform("mean")
    pct = g.rank(pct=True)
    panel["label_rank"] = pct - 0.5
    q = np.floor(pct * N_QUANTILES).clip(upper=N_QUANTILES - 1)
    panel["label_q"] = q            # 0..4, NaN if unlabeled
    return panel


def build_panel(md: MarketData, cfg: Config) -> pd.DataFrame:
    rebal = rebalance_dates(md.dates, cfg.rebalance_day)[1:]   # first may have no T-1
    dec = decision_dates(md.dates, rebal)
    panel = build_feature_panel(md, rebal, dec, cfg.features)
    panel = add_labels(panel, forward_returns(md, rebal))
    return panel


def labeled(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["fwd_ret"].notna()]


def feature_matrix(panel: pd.DataFrame, cols=FEATURES, fill: float | None = None):
    X = panel[list(cols)].astype(float)
    return X.fillna(fill) if fill is not None else X
