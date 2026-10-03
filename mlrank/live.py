"""Live scoring at the latest close. Shared by
scripts/live_ranking.py and the gRPC service.

The decision uses the last close in the data;
the trade happens at the next open.
"""
from __future__ import annotations

import pandas as pd

from .backtest import equity_gate_daily
from .config import Config
from .data_loader import MarketData
from .dataset import feature_matrix, labeled
from .features import FEATURES, build_feature_panel
from .models import make
from .strategies import RULES, eligible, rule_scores

ML_STRATEGIES = ["lgbm_reg"]
STRATEGIES = list(RULES) + ML_STRATEGIES


def latest_panel(md: MarketData, cfg: Config):
    """Features for the next trade, decided at the
    latest close. Returns (decision, trade, panel)."""
    dec = md.dates[-1]
    trade = dec + pd.offsets.BDay(1)              # exchange holidays not known
    panel = build_feature_panel(md, pd.DatetimeIndex([trade]), pd.DatetimeIndex([dec]),
                                cfg.features)
    return dec, trade, panel


def gate_status(md: MarketData, cfg: Config) -> dict:
    g = equity_gate_daily(md, cfg.backtest).iloc[-1]
    return {"date": md.dates[-1], "gate_on": bool(g["gate"]), "index_ok": bool(g["idx_ok"]),
            "breadth": float(g["breadth"]), "breadth_ok": bool(g["breadth_ok"])}


def train_ml(history: pd.DataFrame, name: str = "lgbm_reg"):
    """Fit an ML strategy on every labeled month."""
    lab = labeled(history)
    m = make(name)
    fill = 0.0 if m.needs_fill else None
    m.fit(feature_matrix(lab, FEATURES, fill), lab[m.target], lab.index.get_level_values("date"))
    return m


def rank(panel: pd.DataFrame, strategy: str, model=None) -> pd.Series:
    """Score -> ranked Series (symbol -> score, best
    first) over the eligible universe."""
    el = eligible(panel).to_numpy()
    if strategy in RULES:
        s = rule_scores(panel, [strategy])[strategy]
    elif strategy in ML_STRATEGIES:
        if model is None:
            raise ValueError(f"{strategy} needs a trained model")
        fill = 0.0 if model.needs_fill else None
        s = pd.Series(model.predict(feature_matrix(panel, FEATURES, fill)), index=panel.index)
    else:
        raise KeyError(f"unknown strategy {strategy}; have {STRATEGIES}")
    return s[el].dropna().droplevel("date").sort_values(ascending=False)
