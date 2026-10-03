"""LightGBM models.

LGBMRegressorModel: pointwise, predicts the
  cross-sectional rank of next-month return.
LGBMRankerModel: LambdaRank, a learning-to-rank
  objective optimizing NDCG per month (each
  month is one query group). Labels are return
  quintiles 0..4 (4 = best).

If lightgbm isn't installed, the regressor falls
back to sklearn HistGradientBoostingRegressor
so the pipeline still runs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import RankModel

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    lgb = None
    HAS_LGBM = False

DEFAULT_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    verbose=-1,
)


class LGBMRegressorModel(RankModel):
    target = "label_rank"
    needs_fill = False
    name = "lgbm_reg"

    def __init__(self, seed: int = 0, **params):
        self.params = {**DEFAULT_PARAMS, **params, "random_state": seed}

    def fit(self, X, y, dates=None):
        self.features_ = list(X.columns)
        if HAS_LGBM:
            self.model_ = lgb.LGBMRegressor(**self.params)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            warnings.warn("lightgbm missing: using sklearn HistGradientBoosting")
            self.model_ = HistGradientBoostingRegressor(
                max_iter=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                max_depth=self.params["max_depth"],
                min_samples_leaf=self.params["min_child_samples"],
                l2_regularization=self.params["reg_lambda"],
                random_state=self.params["random_state"],
            )
        self.model_.fit(X, np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model_.predict(X[self.features_])


class LGBMRankerModel(RankModel):
    target = "label_q"
    needs_fill = False
    name = "lgbm_rank"

    def __init__(self, seed: int = 0, **params):
        if not HAS_LGBM:
            raise ImportError("pip install lightgbm to use LGBMRankerModel")
        base = {k: v for k, v in DEFAULT_PARAMS.items()}
        self.params = {**base, **params, "random_state": seed,
                       "objective": "lambdarank",
                       "lambdarank_truncation_level": 40}

    def fit(self, X, y, dates):
        # LightGBM needs rows sorted by query group
        order = np.argsort(np.asarray(dates), kind="stable")
        Xs = X.iloc[order]
        ys = np.asarray(y)[order].astype(int)
        ds = np.asarray(dates)[order]
        _, group = np.unique(ds, return_counts=True)
        self.features_ = list(X.columns)
        self.model_ = lgb.LGBMRanker(**self.params)
        self.model_.fit(Xs, ys, group=group)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.features_])
