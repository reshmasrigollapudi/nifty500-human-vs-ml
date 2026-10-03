"""Zero prices in the raw files must not reach
the wide frames or produce inf / -100% labels."""
import numpy as np
import pandas as pd

from helpers import small_frames
from mlrank.config import CFG
from mlrank.data_loader import MarketData, rebalance_dates
from mlrank.dataset import build_panel


def _frames_with_zero_opens():
    fr = dict(small_frames())
    prices = fr["prices"].copy()
    rebal = rebalance_dates(pd.DatetimeIndex(prices["date"].unique()).sort_values())
    sym = prices["symbol"].iloc[0]
    hit = prices["symbol"].eq(sym) & prices["date"].isin(rebal[-12:])
    prices.loc[hit, ["open", "high", "low"]] = 0.0
    fr["prices"] = prices
    return fr, sym, hit.sum()


def test_nonpositive_prices_become_nan():
    fr, sym, n = _frames_with_zero_opens()
    assert n > 0
    md = MarketData.from_frames(**fr)
    for f in ["open", "high", "low", "close"]:
        assert not (getattr(md, f) <= 0).any().any(), f


def test_labels_are_finite():
    fr, _, _ = _frames_with_zero_opens()
    panel = build_panel(MarketData.from_frames(**fr), CFG)
    fwd = panel["fwd_ret"].dropna()
    assert np.isfinite(fwd).all()
    assert (fwd > -1).all()
