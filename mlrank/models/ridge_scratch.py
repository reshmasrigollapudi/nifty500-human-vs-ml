"""Ridge regression in NumPy, two ways.

Objective:  ||Xc w - yc||^2 + alpha ||w||^2
(Xc, yc centered, so the intercept is not
penalized; same as sklearn.linear_model.Ridge)

closed form:  w = (Xc'Xc + alpha I)^-1 Xc'yc
gradient:     g = 2 (Xc'(Xc w - yc) + alpha w)
Both reach the same minimum (convex).
"""
from __future__ import annotations

import numpy as np

from .base import RankModel


class RidgeScratch(RankModel):
    target = "label_rank"
    needs_fill = True

    def __init__(self, alpha: float = 10.0, method: str = "closed",
                 lr: float = 0.1, epochs: int = 2000, tol: float = 1e-10):
        self.alpha, self.method = alpha, method
        self.lr, self.epochs, self.tol = lr, epochs, tol
        self.name = f"ridge_scratch_{method}"

    def fit(self, X, y, dates=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.x_mean_ = X.mean(0)
        self.y_mean_ = y.mean()
        Xc, yc = X - self.x_mean_, y - self.y_mean_
        n, p = Xc.shape
        if self.method == "closed":
            A = Xc.T @ Xc + self.alpha * np.eye(p)
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        else:
            w = np.zeros(p)
            prev = np.inf
            for _ in range(self.epochs):
                resid = Xc @ w - yc
                grad = 2 * (Xc.T @ resid + self.alpha * w) / n
                w -= self.lr * grad
                loss = (resid @ resid + self.alpha * w @ w) / n
                if abs(prev - loss) < self.tol:
                    break
                prev = loss
            self.coef_ = w
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_
