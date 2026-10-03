"""Synthetic market with a planted, persistent
per-stock drift so momentum carries real but
weak signal. Used only to test the pipeline
end to end. Never report results from it."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(
    n_symbols: int = 200,
    start: str = "2005-01-01",
    end: str = "2026-08-31",
    n_sectors: int = 10,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    D, N = len(dates), n_symbols

    # Market factor with slow bull/bear regimes
    regime = np.zeros(D)
    state, t = 1, 0
    while t < D:
        length = rng.integers(120, 500) if state else rng.integers(60, 200)
        regime[t:t + length] = 0.0006 if state else -0.0010
        t += length
        state = 1 - state
    mkt = regime + rng.normal(0, 0.011, D)

    sec_id = rng.integers(0, n_sectors, N)
    sec_ret = rng.normal(0, 0.006, (D, n_sectors))

    beta = rng.uniform(0.6, 1.4, N)
    idio_vol = rng.uniform(0.012, 0.030, N)

    # Persistent drift (OU): the planted signal
    phi = 0.996
    eps = rng.normal(0, 0.00005, (D, N))
    alpha = np.zeros((D, N))
    for i in range(1, D):
        alpha[i] = phi * alpha[i - 1] + eps[i]

    ret = (
        beta * mkt[:, None]
        + sec_ret[:, sec_id]
        + alpha
        + rng.normal(0, 1, (D, N)) * idio_vol
    )
    ret = np.clip(ret, -0.2, 0.2)
    close = 100 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.004, (D, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (D, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (D, N))))
    vol = np.exp(rng.normal(12, 1, N))[None, :] * np.exp(rng.normal(0, 0.4, (D, N)))

    # Listings and delistings
    list_day = np.where(rng.random(N) < 0.4, 0, rng.integers(0, int(D * 0.8), N))
    delist = np.full(N, D)
    die = rng.random(N) < 0.15
    delist[die] = np.minimum(D, list_day[die] + rng.integers(500, 2500, die.sum()))
    alive = (np.arange(D)[:, None] >= list_day) & (np.arange(D)[:, None] < delist)

    syms = [f"SYN{i:04d}" for i in range(N)]
    arrs = {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    for k in arrs:
        arrs[k] = np.where(alive, arrs[k], np.nan)

    di, si = np.nonzero(alive)
    prices = pd.DataFrame({
        "date": dates[di],
        "symbol": np.array(syms)[si],
        **{k: arrs[k][di, si] for k in arrs},
    })

    member_start = np.minimum(list_day + 60, D - 1)
    membership = pd.DataFrame({
        "symbol": syms,
        "start_date": dates[member_start],
        "end_date": [dates[d - 1] if d < D else pd.NaT for d in delist],
    })

    index = pd.DataFrame({
        "date": dates,
        "close": 10_000 * np.exp(np.cumsum(np.log1p(mkt))),
    })
    g_ret = rng.normal(0.0003, 0.009, D) - 0.15 * (mkt - mkt.mean())
    g_close = 50 * np.exp(np.cumsum(np.log1p(g_ret)))
    gold = pd.DataFrame({
        "date": dates,
        "open": np.r_[g_close[0], g_close[:-1]] * (1 + rng.normal(0, 0.002, D)),
        "close": g_close,
    })
    sectors = pd.DataFrame({"symbol": syms, "sector": [f"S{s}" for s in sec_id]})
    return {"prices": prices, "membership": membership, "index": index,
            "gold": gold, "sectors": sectors}
