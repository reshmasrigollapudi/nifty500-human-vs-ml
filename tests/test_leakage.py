"""Perturb every price on/after date T and check
that features for T do not change. Labels for
earlier months must not change either."""
import numpy as np

from helpers import small_md
from mlrank.config import FeatureConfig
from mlrank.data_loader import decision_dates, rebalance_dates
from mlrank.features import FEATURES, build_feature_panel


def test_features_use_only_past():
    md = small_md()
    rebal = rebalance_dates(md.dates)[1:]
    dec = decision_dates(md.dates, rebal)
    k = len(rebal) - 10
    T = rebal[k]
    cfg = FeatureConfig()
    base = build_feature_panel(md, rebal, dec, cfg).xs(T, level="date")

    rng = np.random.default_rng(0)
    future = md.dates >= T
    for f in ["open", "high", "low", "close", "volume"]:
        w = getattr(md, f).copy()
        w.loc[future] = w.loc[future] * rng.uniform(0.5, 1.5, w.loc[future].shape)
        setattr(md, f, w)
    ix = md.index_close.copy()
    ix[future] = ix[future] * 1.7
    md.index_close = ix
    pert = build_feature_panel(md, rebal, dec, cfg).xs(T, level="date")

    assert base.index.equals(pert.index)
    np.testing.assert_allclose(base[FEATURES].to_numpy(), pert[FEATURES].to_numpy(),
                               equal_nan=True)
