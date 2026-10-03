import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from mlrank.models.ridge_scratch import RidgeScratch
from mlrank.models.tree_scratch import GradientBoostingScratch, RegressionTreeScratch
from mlrank.regime import LogisticScratch


def _reg_data(n=4000, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.5 * rng.normal(size=n) + 3.0
    return X, y


def test_ridge_closed_matches_sklearn():
    X, y = _reg_data()
    m = RidgeScratch(alpha=5.0).fit(X, y)
    ref = Ridge(alpha=5.0).fit(X, y)
    np.testing.assert_allclose(m.coef_, ref.coef_, rtol=1e-8)
    np.testing.assert_allclose(m.intercept_, ref.intercept_, rtol=1e-8)


def test_ridge_gd_matches_closed():
    X, y = _reg_data()
    a = RidgeScratch(alpha=5.0).fit(X, y)
    b = RidgeScratch(alpha=5.0, method="gd", lr=0.2, epochs=5000, tol=1e-14).fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-5)


def test_logistic_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 4))
    y = (X @ [1.0, -2.0, 0.5, 0.0] + rng.normal(size=3000) > 0).astype(float)
    X = (X - X.mean(0)) / X.std(0)
    m = LogisticScratch(alpha=2.0, lr=1.0, epochs=20000, tol=1e-14).fit(X, y)
    ref = LogisticRegression(C=0.5, tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(m.predict_proba(X), ref.predict_proba(X)[:, 1], atol=1e-4)


def test_tree_and_gbm_learn_nonlinear():
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, size=(6000, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=6000)
    Xt = rng.uniform(-3, 3, size=(2000, 3))
    yt = np.sin(Xt[:, 0]) + 0.5 * Xt[:, 1] ** 2

    def r2(p):
        return 1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()

    tree = RegressionTreeScratch(max_depth=6, min_samples_leaf=20).fit(X, y)
    gbm = GradientBoostingScratch(n_estimators=300, learning_rate=0.1, max_depth=3,
                                  min_samples_leaf=20).fit(X, y)
    assert r2(tree.predict(Xt)) > 0.85
    assert r2(gbm.predict(Xt)) > 0.95
    assert gbm.split_counts()[2] < gbm.split_counts()[:2].min()   # noise feature
