"""Ranking metrics. All computed per month,
then summarized across months."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(dates, pred, y) -> pd.Series:
    """Spearman correlation per date, vectorized:
    rank within date, then Pearson on ranks."""
    df = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred, float),
                       "y": np.asarray(y, float)}).dropna()
    g = df.groupby("d")
    pr = g["p"].rank()
    yr = g["y"].rank()
    pc = pr - pr.groupby(df["d"]).transform("mean")
    yc = yr - yr.groupby(df["d"]).transform("mean")
    cov = (pc * yc).groupby(df["d"]).sum()
    vp = (pc ** 2).groupby(df["d"]).sum()
    vy = (yc ** 2).groupby(df["d"]).sum()
    ic = cov / np.sqrt(vp * vy)
    ic.index.name = "date"
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict:
    n = len(ic)
    m, s = ic.mean(), ic.std(ddof=1)
    return {
        "mean_ic": m,
        "ic_std": s,
        "icir_ann": m / s * np.sqrt(12) if s > 0 else np.nan,
        "t_stat": m / (s / np.sqrt(n)) if s > 0 else np.nan,
        "hit_rate": (ic > 0).mean(),
        "n_months": n,
    }


def quantile_returns(dates, pred, fwd, q: int = 5) -> pd.DataFrame:
    """Mean forward return per prediction quantile
    per date. Column q-1 is the top bucket."""
    df = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred, float),
                       "r": np.asarray(fwd, float)}).dropna()
    pct = df.groupby("d")["p"].rank(pct=True)
    df["b"] = np.floor(pct * q).clip(upper=q - 1).astype(int)
    return df.groupby(["d", "b"])["r"].mean().unstack("b")


def spread_summary(qret: pd.DataFrame) -> dict:
    top, bot = qret.columns.max(), qret.columns.min()
    spread = qret[top] - qret[bot]
    return {
        "top_q_monthly": qret[top].mean(),
        "bottom_q_monthly": qret[bot].mean(),
        "spread_monthly": spread.mean(),
        "spread_t": spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread))),
        "monotonic": bool(np.all(np.diff(qret.mean().to_numpy()) > 0)),
    }


def top_n_hit_rate(dates, pred, fwd, n: int = 20) -> float:
    """Share of the model's top-n that beat the
    cross-sectional median that month."""
    df = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred, float),
                       "r": np.asarray(fwd, float)}).dropna()
    med = df.groupby("d")["r"].transform("median")
    rk = df.groupby("d")["p"].rank(ascending=False, method="first")
    top = df[rk <= n]
    return float((top["r"] > med[rk <= n]).mean())
