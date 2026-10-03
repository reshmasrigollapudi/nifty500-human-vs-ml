"""Model explanations.

permutation_importance_ic: model-agnostic. Shuffle
  one feature WITHIN each month (keeps the
  cross-sectional distribution, breaks the link
  to returns) and measure the drop in mean IC.
shap_importance: exact TreeSHAP for LightGBM,
  overall and split by market regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import rank_ic


def _shuffle_within(values: np.ndarray, date_codes: np.ndarray, rng) -> np.ndarray:
    pos = np.argsort(date_codes, kind="stable")
    order = np.lexsort((rng.random(len(values)), date_codes))
    out = values.copy()
    out[pos] = values[order]
    return out


def permutation_importance_ic(model, X: pd.DataFrame, fwd, dates,
                              n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.asarray(dates)
    codes = pd.factorize(dates)[0]
    base = rank_ic(dates, model.predict(X), fwd).mean()
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = _shuffle_within(X[col].to_numpy(), codes, rng)
            drops.append(base - rank_ic(dates, model.predict(Xp), fwd).mean())
        rows.append((col, np.mean(drops), np.std(drops)))
    out = pd.DataFrame(rows, columns=["feature", "ic_drop", "ic_drop_std"])
    return out.sort_values("ic_drop", ascending=False).reset_index(drop=True)


def shap_importance(lgbm_model, X: pd.DataFrame, regime_col: str = "idx_gap_sma100",
                    max_rows: int = 20_000, seed: int = 0) -> pd.DataFrame:
    import shap  # optional dependency

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(lgbm_model.model_)
    sv = explainer.shap_values(X)
    sv = np.asarray(sv)
    out = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)})
    if regime_col in X:
        bull = X[regime_col].to_numpy() > 0
        out["bull"] = np.abs(sv[bull]).mean(0) if bull.any() else np.nan
        out["bear"] = np.abs(sv[~bull]).mean(0) if (~bull).any() else np.nan
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
