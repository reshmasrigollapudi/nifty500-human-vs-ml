import pandas as pd

from mlrank.config import SplitConfig
from mlrank.splits import assert_no_leak, walk_forward


def test_walk_forward_no_leak():
    dates = pd.date_range("2005-01-31", periods=240, freq="ME")
    for emb in (0, 1, 2):
        for h in (1, 3):
            cfg = SplitConfig(min_train_months=60, test_months=12,
                              embargo_months=emb, label_horizon_months=h)
            folds = walk_forward(dates, cfg)
            assert folds
            for f in folds:
                assert_no_leak(f, dates, cfg)
            covered = pd.DatetimeIndex([]).append([f.test_dates for f in folds])
            assert covered.is_unique
            assert covered[0] == dates[60] and covered[-1] == dates[-1]


def test_rolling_window_length():
    dates = pd.date_range("2005-01-31", periods=200, freq="ME")
    cfg = SplitConfig(min_train_months=60, test_months=12, expanding=False)
    for f in walk_forward(dates, cfg, max_train_months=48):
        assert len(f.train_dates) <= 48
