"""Logistic regime classifier vs the SMA gate,
both evaluated by classification metrics and by
the baseline momentum portfolio's results."""
import _path  # noqa: F401
import pandas as pd

from mlrank.backtest import (equity_gate, falling_knife_filter,  ## CHANGED
                             run_backtest, sma_regime)
from mlrank.io import base_args, load_all, out_dir
from mlrank.models import make
from mlrank.regime import classification_report, regime_dataset, walk_forward_regime

p = base_args(__doc__)
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--alpha", type=float, default=1.0)
args = p.parse_args()

cfg, md, panel = load_all(args)
o = out_dir(args)
rebal = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
rd = regime_dataset(md, rebal)
probs = walk_forward_regime(rd, cfg.splits, alpha=args.alpha)
test = probs.dropna().index

sma = sma_regime(md, rebal, cfg.backtest.regime_sma)
gate = equity_gate(md, rebal, cfg.backtest)        ## NEW: SMA | breadth
sma_prob = (~sma).astype(float)            # SMA says "down" when below
gate_prob = (~gate).astype(float)                  ## NEW
print("Classification (OOS months)")
print(pd.DataFrame({
    "logistic": classification_report(probs.loc[test], rd["y_down"].loc[test], args.threshold),
    "sma_gate": classification_report(sma_prob.loc[test], rd["y_down"].loc[test], 0.5),
    "sma_or_breadth": classification_report(                       ## NEW
        gate_prob.loc[test], rd["y_down"].loc[test], 0.5),
}).round(4).to_string())

base = make("baseline_mom12")
scores = pd.Series(base.predict(panel), index=panel.index)
scores = scores[panel.index.get_level_values("date").isin(test)]
elig = falling_knife_filter(panel, cfg.backtest.max_below_52w_high)
gates = {
    "no_gate": None,
    "sma_gate": sma,
    "sma_or_breadth": gate,                         ## NEW: live gate
    "ml_gate": probs < args.threshold,
    "ml_or_gate": (probs < args.threshold) | gate,  ## CHANGED: OR, live gate
}
rows = {}
for k, g in gates.items():
    rows[k] = run_backtest(md, scores, cfg.backtest, regime_on=g, eligible=elig).stats
res = pd.DataFrame(rows).T
res.to_csv(o / "regime_comparison.csv")
probs.to_frame("p_down").to_csv(o / "regime_probs.csv")
print("\nBaseline momentum portfolio under each gate")
print(res[["cagr", "sharpe", "max_dd", "calmar"]].round(3).to_string())
