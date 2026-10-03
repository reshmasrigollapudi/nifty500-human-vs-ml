"""Walk-forward training loop producing purely
out-of-sample predictions for every model."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import SplitConfig
from .dataset import feature_matrix, labeled
from .features import FEATURES
from .metrics import ic_summary, quantile_returns, rank_ic, spread_summary
from .models import make
from .splits import assert_no_leak, walk_forward


def oos_predictions(panel: pd.DataFrame, model_names: list[str], split_cfg: SplitConfig,
                    features=FEATURES, verbose: bool = True) -> pd.DataFrame:
    lab = labeled(panel)
    dates = lab.index.get_level_values("date")
    folds = walk_forward(dates, split_cfg)
    preds = pd.DataFrame(index=lab.index, columns=model_names, dtype=float)
    for fold in folds:
        assert_no_leak(fold, dates, split_cfg)
        tr = lab[dates.isin(fold.train_dates)]
        te = lab[dates.isin(fold.test_dates)]
        if verbose:
            print(fold)
        for name in model_names:
            t0 = time.time()
            m = make(name)
            fill = 0.0 if m.needs_fill else None
            Xtr = feature_matrix(tr, features, fill)
            Xte = feature_matrix(te, features, fill)
            ytr = tr[m.target] if m.target else None
            m.fit(Xtr, ytr, tr.index.get_level_values("date"))
            preds.loc[te.index, name] = m.predict(Xte)
            if verbose:
                print(f"   {name:<16} {time.time() - t0:5.1f}s")
    return preds.dropna(how="all")


def evaluate(panel: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    lab = panel.loc[preds.index]
    d = lab.index.get_level_values("date")
    rows = {}
    for name in preds.columns:
        ic = rank_ic(d, preds[name], lab["fwd_ret"])
        q = quantile_returns(d, preds[name], lab["fwd_ret"])
        rows[name] = {**ic_summary(ic), **spread_summary(q)}
    return pd.DataFrame(rows).T


def ic_series(panel, preds) -> pd.DataFrame:
    lab = panel.loc[preds.index]
    d = lab.index.get_level_values("date")
    return pd.DataFrame({n: rank_ic(d, preds[n], lab["fwd_ret"]) for n in preds.columns})


def ic_by_regime(panel, preds, col: str = "idx_gap_sma100") -> pd.DataFrame:
    """Mean IC split by market regime at decision
    time: does the model help in bear markets?"""
    ics = ic_series(panel, preds)
    reg = panel.groupby(level="date")[col].first().reindex(ics.index)
    return ics.groupby(np.where(reg > 0, "bull", "bear")).mean().T
