"""Feature drift checks for the live model.

PSI (population stability index), bins from the
training distribution's quantiles:
  PSI = sum (a - e) * ln(a / e)
Rule of thumb: <0.1 stable, 0.1-0.2 watch,
>0.2 investigate / consider retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) < bins or len(a) == 0:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ep = np.histogram(e, edges)[0] / len(e)
    ap = np.histogram(a, edges)[0] / len(a)
    ep, ap = np.clip(ep, 1e-6, None), np.clip(ap, 1e-6, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def drift_report(train: pd.DataFrame, live: pd.DataFrame, psi_alert: float = 0.2,
                 bins: int = 10, alert_on: str = "psi") -> pd.DataFrame:
    """alert_on="ks" for small samples (e.g. one
    value per month), where PSI bins are too
    sparse to trust."""
    rows = []
    for col in train.columns:
        a, b = train[col].dropna(), live[col].dropna()
        ks = ks_2samp(a, b) if len(a) and len(b) else None
        rows.append({
            "feature": col,
            "psi": psi(a, b, bins),
            "ks_stat": ks.statistic if ks else np.nan,
            "ks_p": ks.pvalue if ks else np.nan,
        })
    df = pd.DataFrame(rows)
    df["alert"] = (df["psi"] > psi_alert) if alert_on == "psi" else (df["ks_p"] < 0.05)
    return df.sort_values("psi", ascending=False).reset_index(drop=True)
