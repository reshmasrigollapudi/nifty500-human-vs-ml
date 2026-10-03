import numpy as np
import pandas as pd

from mlrank.metrics import quantile_returns, rank_ic, spread_summary


def _frame():
    rng = np.random.default_rng(0)
    d = np.repeat(pd.date_range("2020-01-31", periods=6, freq="ME"), 50)
    y = rng.normal(size=len(d))
    return d, y


def test_ic_perfect_and_reversed():
    d, y = _frame()
    assert np.allclose(rank_ic(d, y, y), 1.0)
    assert np.allclose(rank_ic(d, -y, y), -1.0)


def test_ic_matches_scipy():
    from scipy.stats import spearmanr
    d, y = _frame()
    p = y + np.random.default_rng(1).normal(size=len(y))
    ic = rank_ic(d, p, y)
    df = pd.DataFrame({"d": d, "p": p, "y": y})
    ref = df.groupby("d").apply(lambda g: spearmanr(g["p"], g["y"])[0])
    np.testing.assert_allclose(ic.to_numpy(), ref.to_numpy())


def test_spread_positive_for_signal():
    d, y = _frame()
    s = spread_summary(quantile_returns(d, y, y))
    assert s["spread_monthly"] > 0 and s["monotonic"]
