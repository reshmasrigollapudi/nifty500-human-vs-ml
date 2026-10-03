"""ML regime model: predict P(next-month index
return < 0) and use it instead of the SMA gate.

LogisticScratch minimizes
  mean(logloss) + (alpha / 2n) ||w||^2
with full-batch gradient descent. With alpha =
1/C this matches sklearn LogisticRegression
on standardized inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitConfig
from .data_loader import MarketData
from .splits import walk_forward

REGIME_COLS = ["gap_sma100", "gap_sma250", "mom_3m", "mom_6m",
               "vol_20", "vol_60", "vol_ratio", "breadth_200"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticScratch:
    def __init__(self, alpha: float = 1.0, lr: float = 0.5, epochs: int = 5000,
                 tol: float = 1e-9, standardize: bool = True):
        self.alpha, self.lr, self.epochs = alpha, lr, epochs
        self.tol, self.standardize = tol, standardize

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu_ = X.mean(0) if self.standardize else np.zeros(X.shape[1])
        self.sd_ = X.std(0) if self.standardize else np.ones(X.shape[1])
        self.sd_[self.sd_ == 0] = 1.0
        Z = (X - self.mu_) / self.sd_
        n, p = Z.shape
        w, b, prev = np.zeros(p), 0.0, np.inf
        for _ in range(self.epochs):
            pr = _sigmoid(Z @ w + b)
            err = pr - y
            gw = Z.T @ err / n + self.alpha * w / n
            gb = err.mean()
            w -= self.lr * gw
            b -= self.lr * gb
            eps = 1e-12
            loss = -np.mean(y * np.log(pr + eps) + (1 - y) * np.log(1 - pr + eps)) \
                + self.alpha * w @ w / (2 * n)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        self.coef_, self.intercept_ = w, b
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, float) - self.mu_) / self.sd_
        return _sigmoid(Z @ self.coef_ + self.intercept_)


def regime_dataset(md: MarketData, rebal: pd.DatetimeIndex) -> pd.DataFrame:
    ix = md.index_close.ffill()
    ir = np.log(ix / ix.shift(1))
    c = md.close.ffill(limit=5)
    sma200 = c.rolling(200, min_periods=200).mean()
    valid = sma200.notna() & md.member
    breadth = ((c > sma200) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    daily = pd.DataFrame({
        "gap_sma100": ix / ix.rolling(100).mean() - 1,
        "gap_sma250": ix / ix.rolling(250).mean() - 1,
        "mom_3m": ix / ix.shift(63) - 1,
        "mom_6m": ix / ix.shift(126) - 1,
        "vol_20": ir.rolling(20).std() * np.sqrt(252),
        "vol_60": ir.rolling(60).std() * np.sqrt(252),
        "breadth_200": breadth,
    })
    daily["vol_ratio"] = daily["vol_20"] / daily["vol_60"]
    prev = md.dates[md.dates.get_indexer(rebal) - 1]
    df = daily.reindex(prev)
    df.index = rebal
    # label: index return from T-1 close to next T-1 close
    px = ix.reindex(prev).to_numpy()
    fwd = np.r_[px[1:] / px[:-1] - 1, np.nan]
    df["fwd_idx_ret"] = fwd
    df["y_down"] = np.where(np.isnan(fwd), np.nan, (fwd < 0).astype(float))
    return df


def walk_forward_regime(df: pd.DataFrame, split_cfg: SplitConfig,
                        alpha: float = 1.0) -> pd.Series:
    data = df.dropna(subset=REGIME_COLS + ["y_down"])
    probs = pd.Series(np.nan, index=df.index)
    for fold in walk_forward(data.index, split_cfg):
        tr, te = data.loc[fold.train_dates], data.loc[fold.test_dates]
        m = LogisticScratch(alpha=alpha).fit(tr[REGIME_COLS], tr["y_down"])
        probs.loc[te.index] = m.predict_proba(te[REGIME_COLS])
    return probs


def classification_report(probs: pd.Series, y: pd.Series, thr: float = 0.5) -> dict:
    d = pd.concat([probs, y], axis=1, keys=["p", "y"]).dropna()
    pred = d["p"] >= thr
    tp = (pred & (d["y"] == 1)).sum(); fp = (pred & (d["y"] == 0)).sum()
    fn = (~pred & (d["y"] == 1)).sum()
    eps = 1e-12
    ll = -np.mean(d["y"] * np.log(d["p"] + eps) + (1 - d["y"]) * np.log(1 - d["p"] + eps))
    base = d["y"].mean()
    ll0 = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return {
        "accuracy": (pred == (d["y"] == 1)).mean(),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "logloss": ll, "logloss_base_rate": ll0,
        "brier": ((d["p"] - d["y"]) ** 2).mean(),
        "n": len(d),
    }
