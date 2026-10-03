"""Histogram decision tree + gradient boosting
in NumPy. Same core idea as LightGBM:

1. Bin each feature into <= n_bins quantile
   buckets once (uint8 matrix B).
2. At a node, per feature, build a histogram of
   gradient sums and counts per bin (bincount).
3. Scan bins left->right with cumulative sums.
   Split gain for squared loss (G = grad sum,
   n = count, lam = L2 on leaf value):
     G_L^2/(n_L+lam) + G_R^2/(n_R+lam)
       - G^2/(n+lam)
4. Leaf value = G / (n + lam).

Boosting (squared loss): residual r = y - F,
fit a tree to r, F += lr * tree(x), repeat.
"""
from __future__ import annotations

import numpy as np

from .base import RankModel


class Binner:
    def __init__(self, n_bins: int = 32):
        assert 2 <= n_bins <= 255
        self.n_bins = n_bins

    def fit(self, X: np.ndarray) -> "Binner":
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, f], qs)) for f in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        B = np.empty(X.shape, dtype=np.uint8)
        for f, e in enumerate(self.edges_):
            B[:, f] = np.searchsorted(e, X[:, f], side="right")
        return B


class HistTree:
    """Works on pre-binned data. Nodes stored in
    flat arrays; feature == -1 marks a leaf."""

    def __init__(self, max_depth=3, min_samples_leaf=100, lam=1.0,
                 min_gain=0.0, n_bins=32):
        self.max_depth, self.min_leaf = max_depth, min_samples_leaf
        self.lam, self.min_gain, self.n_bins = lam, min_gain, n_bins

    def fit_binned(self, B: np.ndarray, g: np.ndarray, idx: np.ndarray | None = None):
        n_feat, nb = B.shape[1], self.n_bins
        feat, thr, left, right, value = [], [], [], [], []

        def new_node(v):
            feat.append(-1); thr.append(0); left.append(-1)
            right.append(-1); value.append(v)
            return len(feat) - 1

        if idx is None:
            idx = np.arange(len(g))
        root = new_node(g[idx].sum() / (len(idx) + self.lam))
        stack = [(root, idx, 0)]
        offsets = np.arange(n_feat) * nb

        while stack:
            node, rows, depth = stack.pop()
            m = len(rows)
            if depth >= self.max_depth or m < 2 * self.min_leaf:
                continue
            Bn = B[rows].astype(np.int64) + offsets
            gn = g[rows]
            flat = Bn.ravel()
            G = np.bincount(flat, np.repeat(gn, n_feat), n_feat * nb).reshape(n_feat, nb)
            C = np.bincount(flat, minlength=n_feat * nb).reshape(n_feat, nb)
            GL, CL = np.cumsum(G, 1)[:, :-1], np.cumsum(C, 1)[:, :-1]
            Gt, Ct = gn.sum(), m
            GR, CR = Gt - GL, Ct - CL
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = (GL**2 / (CL + self.lam) + GR**2 / (CR + self.lam)
                        - Gt**2 / (Ct + self.lam))
            gain[(CL < self.min_leaf) | (CR < self.min_leaf)] = -np.inf
            f, t = np.unravel_index(np.argmax(gain), gain.shape)
            if not np.isfinite(gain[f, t]) or gain[f, t] <= self.min_gain:
                continue
            go_left = B[rows, f] <= t
            lr, rr = rows[go_left], rows[~go_left]
            ln = new_node(g[lr].sum() / (len(lr) + self.lam))
            rn = new_node(g[rr].sum() / (len(rr) + self.lam))
            feat[node], thr[node], left[node], right[node] = f, t, ln, rn
            stack.append((ln, lr, depth + 1))
            stack.append((rn, rr, depth + 1))

        self.feat_ = np.array(feat); self.thr_ = np.array(thr)
        self.left_ = np.array(left); self.right_ = np.array(right)
        self.value_ = np.array(value, float)
        return self

    def predict_binned(self, B: np.ndarray) -> np.ndarray:
        node = np.zeros(len(B), dtype=np.int64)
        rows = np.arange(len(B))
        for _ in range(self.max_depth):
            f = self.feat_[node]
            internal = f >= 0
            go_left = B[rows, np.where(internal, f, 0)] <= self.thr_[node]
            nxt = np.where(go_left, self.left_[node], self.right_[node])
            node = np.where(internal, nxt, node)
        return self.value_[node]

    def feature_gain_counts(self, n_features: int) -> np.ndarray:
        used = self.feat_[self.feat_ >= 0]
        return np.bincount(used, minlength=n_features)


class RegressionTreeScratch(RankModel):
    target = "label_rank"
    needs_fill = True
    name = "tree_scratch"

    def __init__(self, max_depth=4, min_samples_leaf=200, n_bins=32):
        self.binner = Binner(n_bins)
        self.tree = HistTree(max_depth, min_samples_leaf, lam=0.0, n_bins=n_bins)

    def fit(self, X, y, dates=None):
        X = np.asarray(X, float)
        B = self.binner.fit(X).transform(X)
        self.tree.fit_binned(B, np.asarray(y, float))
        return self

    def predict(self, X):
        return self.tree.predict_binned(self.binner.transform(np.asarray(X, float)))


class GradientBoostingScratch(RankModel):
    target = "label_rank"
    needs_fill = True
    name = "gbm_scratch"

    def __init__(self, n_estimators=150, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=200, subsample=0.8, lam=1.0, n_bins=32, seed=0):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth, self.min_leaf = max_depth, min_samples_leaf
        self.subsample, self.lam, self.n_bins = subsample, lam, n_bins
        self.seed = seed

    def fit(self, X, y, dates=None):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.binner_ = Binner(self.n_bins).fit(X)
        B = self.binner_.transform(X)
        self.f0_ = y.mean()
        F = np.full(len(y), self.f0_)
        self.trees_ = []
        n_sub = int(len(y) * self.subsample)
        for _ in range(self.n_estimators):
            resid = y - F
            idx = rng.choice(len(y), n_sub, replace=False) if self.subsample < 1 else None
            t = HistTree(self.max_depth, self.min_leaf, self.lam, n_bins=self.n_bins)
            t.fit_binned(B, resid, idx)
            F += self.lr * t.predict_binned(B)
            self.trees_.append(t)
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        B = self.binner_.transform(np.asarray(X, float))
        out = np.full(len(B), self.f0_)
        for t in self.trees_:
            out += self.lr * t.predict_binned(B)
        return out

    def split_counts(self) -> np.ndarray:
        return sum(t.feature_gain_counts(self.n_features_) for t in self.trees_)
