"""Point-in-time feature engineering.

Rule: a feature for rebalance date T uses only
data up to the T-1 close (decision date).
tests/test_leakage.py enforces this.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data_loader import MarketData

# Cross-sectional stock features (z-scored per date)
STOCK_FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1",
    "vol_20", "vol_60", "downvol_60", "dd_126",
    "dist_52w_high", "dist_sma50", "dist_sma200",
    "volume_trend", "log_turnover",
]
# sector_rel_mom_6m is built when sectors are supplied but is not a
# model input yet: the real data has no sectors file, so it was a
# constant 0. Add it back here once sectors.csv exists.
# Market-wide features (same value for every stock on a date)
REGIME_FEATURES = ["idx_gap_sma100", "idx_gap_sma250", "idx_vol_20", "breadth_200"]
# Raw (not z-scored) copies for the rule strategies
RAW_KEEP = ["mom_12m", "mom_12_1", "mom_6m", "mom_3m", "dist_52w_high"]


def daily_stock_features(md: MarketData, cfg: FeatureConfig) -> dict[str, pd.DataFrame]:
    c = md.close.ffill(limit=cfg.ffill_limit)
    v = md.volume
    r = np.log(c / c.shift(1))
    out: dict[str, pd.DataFrame] = {}

    names = {21: "mom_1m", 63: "mom_3m", 126: "mom_6m", 252: "mom_12m"}
    for w in cfg.mom_windows:
        out[names.get(w, f"mom_{w}d")] = c / c.shift(w) - 1
    out["mom_12_1"] = c.shift(cfg.skip_days) / c.shift(252) - 1

    for w in cfg.vol_windows:
        out[f"vol_{w}"] = r.rolling(w, min_periods=w).std() * np.sqrt(252)
    # Raw, for the rule strategies: same as the live C3 code
    # (std of simple daily returns over 126 days, annualised)
    out["vol_126"] = (c / c.shift(1) - 1).rolling(126, min_periods=126).std() * np.sqrt(252)
    neg = r.clip(upper=0)
    out["downvol_60"] = np.sqrt((neg ** 2).rolling(60, min_periods=60).mean()) * np.sqrt(252)

    roll_max126 = c.rolling(126, min_periods=126).max()
    out["dd_126"] = (c / roll_max126 - 1).rolling(126, min_periods=1).min()

    out["dist_52w_high"] = c / c.rolling(252, min_periods=126).max() - 1
    out["dist_sma50"] = c / c.rolling(50, min_periods=50).mean() - 1
    out["dist_sma200"] = c / c.rolling(200, min_periods=200).mean() - 1

    v20 = v.rolling(20, min_periods=15).mean()
    v120 = v.rolling(120, min_periods=90).mean()
    out["volume_trend"] = np.log(v20 / v120)
    out["log_turnover"] = np.log((c * v).rolling(60, min_periods=40).mean())
    return out


def daily_regime_features(md: MarketData) -> pd.DataFrame:
    ix = md.index_close.ffill()
    ir = np.log(ix / ix.shift(1))
    c = md.close.ffill(limit=5)
    above = (c > c.rolling(200, min_periods=200).mean()) & md.member
    valid = c.rolling(200, min_periods=200).mean().notna() & md.member
    breadth = above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({
        "idx_gap_sma100": ix / ix.rolling(100, min_periods=100).mean() - 1,
        "idx_gap_sma250": ix / ix.rolling(250, min_periods=250).mean() - 1,
        "idx_vol_20": ir.rolling(20, min_periods=20).std() * np.sqrt(252),
        "breadth_200": breadth,
    })


def _cs_zscore(s: pd.Series, dates: pd.Series, clip: float) -> pd.Series:
    g = s.groupby(dates)
    z = (s - g.transform("mean")) / g.transform("std")
    return z.clip(-clip, clip)


def build_feature_panel(
    md: MarketData,
    rebal: pd.DatetimeIndex,
    decision: pd.DatetimeIndex,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Long panel indexed by (date=T, symbol).
    Rows: index members at the decision date
    with enough history."""
    daily = daily_stock_features(md, cfg)
    syms = md.symbols
    n_d, n_s = len(rebal), len(syms)

    cols = {}
    for name, w in daily.items():
        cols[name] = w.reindex(decision).to_numpy().ravel()
    member = md.member.reindex(decision).to_numpy().ravel()

    idx = pd.MultiIndex.from_product([rebal, syms], names=["date", "symbol"])
    panel = pd.DataFrame(cols, index=idx)
    keep = member & panel["mom_6m"].notna().to_numpy()
    panel = panel[keep].copy()

    dates = panel.index.get_level_values("date").to_series(index=panel.index)
    if md.sectors is not None:
        sec = md.sectors.reindex(panel.index.get_level_values("symbol")).to_numpy()
        panel["sector"] = sec
        med = panel.groupby([dates, panel["sector"]])["mom_6m"].transform("median")
        panel["sector_rel_mom_6m"] = panel["mom_6m"] - med
    else:
        warnings.warn("no sectors file: sector_rel_mom_6m not built", stacklevel=2)
        panel["sector"] = "NA"

    for k in RAW_KEEP:
        panel[f"raw_{k}"] = panel[k]

    for f in STOCK_FEATURES:
        panel[f] = _cs_zscore(panel[f], dates, cfg.zscore_clip)

    reg = daily_regime_features(md).reindex(decision)
    reg.index = rebal
    reg_rows = reg.reindex(panel.index.get_level_values("date"))
    for f in REGIME_FEATURES:
        panel[f] = reg_rows[f].to_numpy()
    return panel


FEATURES = STOCK_FEATURES + REGIME_FEATURES
