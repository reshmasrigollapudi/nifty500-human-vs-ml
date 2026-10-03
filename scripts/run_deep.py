"""Walk-forward 1D-CNN on raw return sequences,
compared by IC against the saved OOS table.
Needs torch."""
import _path  # noqa: F401
import numpy as np
import pandas as pd

from mlrank.dataset import labeled
from mlrank.deep.cnn1d import CNN1DRanker, build_sequences
from mlrank.metrics import ic_summary, rank_ic
from mlrank.io import base_args, load_all, load_frame, out_dir, save_frame
from mlrank.splits import walk_forward

p = base_args(__doc__)
p.add_argument("--epochs", type=int, default=8)
args = p.parse_args()

cfg, md, panel = load_all(args)
o = out_dir(args)
lab = labeled(panel)
print("building sequences ...")
X = build_sequences(md, lab)
d = lab.index.get_level_values("date")
pred = pd.Series(np.nan, index=lab.index)
for fold in walk_forward(d, cfg.splits):
    print(fold)
    tr = np.asarray(d.isin(fold.train_dates))
    te = np.asarray(d.isin(fold.test_dates))
    m = CNN1DRanker(epochs=args.epochs).fit(X[tr], lab["label_rank"].to_numpy()[tr])
    pred[te] = m.predict(X[te])
pred = pred.dropna()
ic = rank_ic(pred.index.get_level_values("date"), pred, lab.loc[pred.index, "fwd_ret"])
print(pd.Series(ic_summary(ic)).round(4).to_string())
preds = load_frame(o / "oos_predictions.csv") if (o / "oos_predictions.csv").exists() else None
if preds is not None:
    preds["cnn1d"] = pred.reindex(preds.index)
    save_frame(preds, o / "oos_predictions.csv")
    print("added cnn1d to oos_predictions.csv (rerun run_backtest.py)")
