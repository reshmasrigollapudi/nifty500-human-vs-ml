"""Load long-format tables and pivot to wide
date x symbol frames. All downstream code uses
MarketData, so swapping data sources only
touches this file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _find(data_dir: Path, stem: str) -> Path | None:
    for ext in (".csv", ".parquet"):
        p = data_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_csv(path: Path, **kw) -> pd.DataFrame:
    # Only blank cells are NaN: a symbol or sector
    # spelled "NA"/"NULL" must stay a string.
    return pd.read_csv(path, keep_default_na=False, na_values=[""],
                       float_precision="round_trip", **kw)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return read_csv(path)


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


@dataclass
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    member: pd.DataFrame          # bool, PIT index membership
    index_close: pd.Series
    gold_open: pd.Series | None
    gold_close: pd.Series | None
    sectors: pd.Series | None     # symbol -> sector

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> pd.Index:
        return self.close.columns

    @classmethod
    def from_frames(
        cls,
        prices: pd.DataFrame,
        membership: pd.DataFrame,
        index: pd.DataFrame,
        gold: pd.DataFrame | None = None,
        sectors: pd.DataFrame | None = None,
    ) -> "MarketData":
        _require(prices, PRICE_COLS, "prices")
        _require(membership, ["symbol", "start_date", "end_date"], "membership")
        _require(index, ["date", "close"], "index")

        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.drop_duplicates(["date", "symbol"], keep="last")
        # Old bhavcopies have rows with open/high/low = 0 but a real
        # close. A zero open as entry/exit gives inf/-100% labels, so
        # treat non-positive prices as missing.
        px = ["open", "high", "low", "close"]
        prices[px] = prices[px].where(prices[px] > 0)

        idx = index.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.drop_duplicates("date").set_index("date").sort_index()

        # Trading calendar = index dates
        cal = pd.DatetimeIndex(idx.index)
        wide = {}
        for f in ["open", "high", "low", "close", "volume"]:
            w = prices.pivot(index="date", columns="symbol", values=f)
            wide[f] = w.reindex(cal).sort_index(axis=1).astype(float)

        member = _membership_mask(membership, cal, wide["close"].columns)

        g_open = g_close = None
        if gold is not None and len(gold):
            g = gold.copy()
            g["date"] = pd.to_datetime(g["date"])
            g = g.drop_duplicates("date").set_index("date").sort_index()
            g = g.reindex(cal)
            g_close = g["close"].astype(float)
            g_open = g["open"].astype(float) if "open" in g else g_close

        sec = None
        if sectors is not None and len(sectors):
            sec = sectors.drop_duplicates("symbol").set_index("symbol")["sector"]
            sec = sec.reindex(wide["close"].columns)

        return cls(
            open=wide["open"], high=wide["high"], low=wide["low"],
            close=wide["close"], volume=wide["volume"], member=member,
            index_close=idx["close"].astype(float),
            gold_open=g_open, gold_close=g_close, sectors=sec,
        )

    @classmethod
    def load(cls, cfg: DataConfig) -> "MarketData":
        d = Path(cfg.data_dir)
        need = {"prices": cfg.prices, "membership": cfg.membership,
                "index": cfg.index}
        frames = {}
        for key, stem in need.items():
            p = _find(d, stem)
            if p is None:
                raise FileNotFoundError(
                    f"{stem}.csv not found in {d}. "
                    "Run scripts/make_synthetic_data.py or add your data."
                )
            frames[key] = read_table(p)
        for key, stem in {"gold": cfg.gold, "sectors": cfg.sectors}.items():
            p = _find(d, stem)
            frames[key] = read_table(p) if p else None
        return cls.from_frames(**frames)


def _membership_mask(
    membership: pd.DataFrame, cal: pd.DatetimeIndex, symbols: pd.Index
) -> pd.DataFrame:
    """Rows are membership spells. Empty end_date
    means still a member. Multiple spells per
    symbol are allowed (exit + re-entry)."""
    m = membership.copy()
    m["start_date"] = pd.to_datetime(m["start_date"])
    m["end_date"] = pd.to_datetime(m["end_date"]).fillna(cal[-1])
    out = np.zeros((len(cal), len(symbols)), dtype=bool)
    col = {s: i for i, s in enumerate(symbols)}
    cal_np = cal.values
    for sym, s, e in m[["symbol", "start_date", "end_date"]].itertuples(index=False):
        j = col.get(sym)
        if j is None:
            continue
        lo = np.searchsorted(cal_np, np.datetime64(s), side="left")
        hi = np.searchsorted(cal_np, np.datetime64(e), side="right")
        out[lo:hi, j] = True
    return pd.DataFrame(out, index=cal, columns=symbols)


# ---------- calendar helpers ----------
def rebalance_dates(cal: pd.DatetimeIndex, day: int = 1) -> pd.DatetimeIndex:
    """First trading day of each month on or after
    `day` (live rule: day=1, i.e. decide on the last
    trading day's close, trade at the next open).
    Other days are a robustness check."""
    c = cal[cal.day >= day]
    s = pd.Series(c, index=c)
    first = s.groupby([c.year, c.month]).min()
    return pd.DatetimeIndex(first.values)


def decision_dates(cal: pd.DatetimeIndex, rebal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """T-1 trading day for each rebalance date T."""
    pos = cal.get_indexer(rebal)
    if (pos < 1).any():
        raise ValueError("rebalance date has no prior trading day")
    return cal[pos - 1]
