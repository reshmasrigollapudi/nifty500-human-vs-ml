"""Common model interface.

Every model:
  target: which panel column it trains on
          ("label_rank", "label_q", or None)
  needs_fill: True if NaNs must be imputed
  fit(X, y, dates) / predict(X)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RankModel:
    name = "base"
    target: str | None = "label_rank"
    needs_fill = True

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> "RankModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class MomentumBaseline(RankModel):
    """The live strategy as a 'model':
    score = trailing momentum. No fitting."""
    target = None
    needs_fill = False

    def __init__(self, col: str = "mom_12m", fallback: str = "mom_6m"):
        self.col, self.fallback = col, fallback
        self.name = f"baseline_{col}"

    def fit(self, X, y, dates):
        return self

    def predict(self, X):
        # Flex scoring: young stocks use shorter window
        s = X[self.col]
        if self.fallback in X:
            s = s.fillna(X[self.fallback])
        return s.to_numpy()
