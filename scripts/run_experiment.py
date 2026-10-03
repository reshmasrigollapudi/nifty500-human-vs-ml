"""Walk-forward OOS predictions for all models,
then the IC / quantile-spread comparison table."""
import _path  # noqa: F401
import pandas as pd

from mlrank.experiment import evaluate, ic_by_regime, ic_series, oos_predictions
from mlrank.io import base_args, load_all, out_dir, save_frame
from mlrank.models import DEFAULT_MODELS

p = base_args(__doc__)
p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
args = p.parse_args()

cfg, md, panel = load_all(args)
preds = oos_predictions(panel, args.models, cfg.splits)
o = out_dir(args)
save_frame(preds, o / "oos_predictions.csv")

table = evaluate(panel, preds)
table.to_csv(o / "model_comparison.csv")
ic_series(panel, preds).to_csv(o / "ic_by_month.csv")
pd.set_option("display.width", 160)
cols = ["mean_ic", "icir_ann", "t_stat", "hit_rate", "spread_monthly",
        "spread_t", "monotonic", "n_months"]
print("\nOut-of-sample ranking quality")
print(table[cols].astype(float, errors="ignore").round(4).to_string())
print("\nMean IC by regime (index vs SMA100)")
print(ic_by_regime(panel, preds).round(4).to_string())
print(f"\nsaved -> {o}")
