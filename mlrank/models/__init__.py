"""Model registry: name -> factory."""
from .base import MomentumBaseline, RankModel
from .lgbm_models import HAS_LGBM, LGBMRankerModel, LGBMRegressorModel
from .ridge_scratch import RidgeScratch
from .tree_scratch import GradientBoostingScratch, RegressionTreeScratch

REGISTRY = {
    "baseline_mom12": lambda: MomentumBaseline("mom_12m", "mom_6m"),
    "ridge_scratch": lambda: RidgeScratch(alpha=100.0),
    "tree_scratch": lambda: RegressionTreeScratch(max_depth=4, min_samples_leaf=500),
    "gbm_scratch": lambda: GradientBoostingScratch(n_estimators=150),
    "lgbm_reg": lambda: LGBMRegressorModel(),
}
if HAS_LGBM:
    REGISTRY["lgbm_rank"] = lambda: LGBMRankerModel()

# The ML strategy compared with C3. The other models
# (incl. the from-scratch GBM) stay available via --models
DEFAULT_MODELS = ["lgbm_reg"]


def make(name: str) -> RankModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name}; have {sorted(REGISTRY)}")
    m = REGISTRY[name]()
    m.name = name
    return m
