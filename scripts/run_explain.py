"""Feature importance on the most recent fold:
within-month permutation importance (any model)
and SHAP (LightGBM only)."""
import _path  # noqa: F401
import pandas as pd

from mlrank.dataset import feature_matrix, labeled
from mlrank.explain import permutation_importance_ic, shap_importance
from mlrank.features import FEATURES
from mlrank.io import base_args, load_all, out_dir
from mlrank.models import make
from mlrank.splits import walk_forward

p = base_args(__doc__)
p.add_argument("--model", default="lgbm_reg")
args = p.parse_args()

cfg, md, panel = load_all(args)
o = out_dir(args)
lab = labeled(panel)
d = lab.index.get_level_values("date")
folds = walk_forward(d, cfg.splits)
full = [f for f in folds if len(f.test_dates) == cfg.splits.test_months]
fold = (full or folds)[-1]            # last fold with a full test year
tr, te = lab[d.isin(fold.train_dates)], lab[d.isin(fold.test_dates)]
m = make(args.model)
fill = 0.0 if m.needs_fill else None
m.fit(feature_matrix(tr, FEATURES, fill), tr[m.target], tr.index.get_level_values("date"))
Xte = feature_matrix(te, FEATURES, fill)

perm = permutation_importance_ic(m, Xte, te["fwd_ret"], te.index.get_level_values("date"))
perm.to_csv(o / f"perm_importance_{args.model}.csv", index=False)
print(f"Permutation importance ({args.model}, test {fold.test_dates[0].date()}..)")
print(perm.round(5).to_string())

if args.model.startswith("lgbm"):
    try:
        sh = shap_importance(m, Xte)
        sh.to_csv(o / f"shap_{args.model}.csv", index=False)
        print("\nSHAP mean |value| (overall / bull / bear)")
        print(sh.round(5).to_string())
    except ImportError:
        print("\npip install shap for SHAP values")
