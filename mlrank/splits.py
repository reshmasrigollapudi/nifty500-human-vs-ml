"""Purged walk-forward splits over month dates.

Train row k's label spans months k..k+horizon.
For a test block starting at month j we keep
only train rows with  k + horizon + embargo <= j,
so no training label overlaps the test period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass(frozen=True)
class Fold:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def __repr__(self) -> str:
        t, s = self.train_dates, self.test_dates
        return (f"Fold(train {t[0].date()}..{t[-1].date()} n={len(t)}, "
                f"test {s[0].date()}..{s[-1].date()} n={len(s)})")


def walk_forward(dates, cfg: SplitConfig, max_train_months: int | None = None) -> list[Fold]:
    d = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    n = len(d)
    gap = cfg.label_horizon_months + cfg.embargo_months - 1
    folds = []
    j = cfg.min_train_months
    while j < n:
        test = np.arange(j, min(j + cfg.test_months, n))
        hi = j - gap                      # exclusive end of train
        lo = 0 if cfg.expanding else max(0, hi - (max_train_months or hi))
        if hi - lo < 12:
            j += cfg.test_months
            continue
        folds.append(Fold(d[lo:hi], d[test]))
        j += cfg.test_months
    return folds


def assert_no_leak(fold: Fold, all_dates, cfg: SplitConfig) -> None:
    d = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(all_dates))))
    last_train = d.get_loc(fold.train_dates[-1])
    first_test = d.get_loc(fold.test_dates[0])
    need = last_train + cfg.label_horizon_months + cfg.embargo_months
    assert need <= first_test, f"leak: train ends {last_train}, test {first_test}"
    assert len(fold.train_dates.intersection(fold.test_dates)) == 0
