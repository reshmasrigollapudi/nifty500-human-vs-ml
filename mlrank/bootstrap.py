"""Stationary block bootstrap for A vs B.

Resamples months in blocks (mean length
`block`) using the SAME indices for both
strategies, so their correlation is kept.
Returns the distribution of annualized
geometric-return differences (A - B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _block_indices(T: int, n: int, block: float, rng) -> np.ndarray:
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n)
    p = 1.0 / block
    for t in range(1, T):
        jump = rng.random(n) < p
        idx[:, t] = np.where(jump, rng.integers(0, T, n), (idx[:, t - 1] + 1) % T)
    return idx


def ann_geo(r: np.ndarray, axis=-1) -> np.ndarray:
    return np.exp(np.log1p(r).mean(axis=axis) * 12) - 1


def bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 10_000,
                   block: float = 6.0, seed: int = 0) -> dict:
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    A, B = df["a"].to_numpy(), df["b"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = _block_indices(len(df), n, block, rng)
    diff = ann_geo(A[idx]) - ann_geo(B[idx])
    return {
        "observed_diff": ann_geo(A) - ann_geo(B),
        "mean_diff": diff.mean(),
        "ci_low": np.quantile(diff, 0.025),
        "ci_high": np.quantile(diff, 0.975),
        "p_a_not_better": (diff <= 0).mean(),
        "n_months": len(df),
    }
