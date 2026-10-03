"""Monthly production job. Run after the T-1
close with data through T-1; T is taken as the
next business day.
1. train the chosen model on all labeled months
2. save a versioned artifact + metadata
3. score the latest (unlabeled) month
4. apply live rules -> target portfolio
5. append predictions to the prediction log"""
import datetime as dt
import json
import pickle

import _path  # noqa: F401
import pandas as pd

from mlrank.backtest import equity_gate_daily, falling_knife_filter  ## CHANGED
from mlrank.config import MODEL_DIR
from mlrank.dataset import feature_matrix, labeled
from mlrank.features import FEATURES, build_feature_panel
from mlrank.io import base_args, load_all, out_dir
from mlrank.models import make

p = base_args(__doc__)
p.add_argument("--model", default="lgbm_rank")
args = p.parse_args()
args.rebuild = True

cfg, md, panel = load_all(args)
o = out_dir(args)
lab = labeled(panel)
dec = md.dates[-1]                                   # T-1 close
latest = dec + pd.offsets.BDay(1)                    # T
live = build_feature_panel(md, pd.DatetimeIndex([latest]),
                           pd.DatetimeIndex([dec]), cfg.features)

try:
    m = make(args.model)
except (KeyError, ImportError) as e:
    print(f"{e}; falling back to gbm_scratch")
    m = make("gbm_scratch")
fill = 0.0 if m.needs_fill else None
m.fit(feature_matrix(lab, FEATURES, fill), lab[m.target], lab.index.get_level_values("date"))

stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR.mkdir(exist_ok=True)
art = MODEL_DIR / f"{m.name}_{stamp}.pkl"
with open(art, "wb") as f:
    pickle.dump(m, f)
d = lab.index.get_level_values("date")
meta = {
    "model": m.name, "artifact": art.name, "features": FEATURES,
    "train_start": str(d.min().date()), "train_end": str(d.max().date()),
    "train_rows": len(lab), "scored_date": str(latest.date()),
    "config": {"top_n": cfg.backtest.top_n, "regime_sma": cfg.backtest.regime_sma,
               "breadth_sma": cfg.backtest.breadth_sma,               ## NEW
               "breadth_threshold": cfg.backtest.breadth_threshold},  ## NEW
}
(MODEL_DIR / f"{m.name}_{stamp}.json").write_text(json.dumps(meta, indent=2))

scores = pd.Series(m.predict(feature_matrix(live, FEATURES, fill)), index=live.index)
elig = falling_knife_filter(live, cfg.backtest.max_below_52w_high)
g = equity_gate_daily(md, cfg.backtest).iloc[-1]   ## CHANGED: SMA | breadth
regime_on = bool(g["gate"])
ranked = scores[elig.to_numpy()].sort_values(ascending=False)
top = ranked.head(cfg.backtest.top_n).reset_index()
top.columns = ["date", "symbol", "score"]

live_dir = o / "live"
live_dir.mkdir(exist_ok=True)
top.to_csv(live_dir / f"target_{latest.date()}.csv", index=False)
log = scores.rename("score").reset_index()
log["model"], log["artifact"] = m.name, art.name
log_path = o / "prediction_log.csv"
log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

print(f"model     {m.name}  trained {meta['train_start']}..{meta['train_end']}")
print(f"artifact  {art}")
print(f"index leg   NIFTY50 > SMA{cfg.backtest.regime_sma}: {bool(g['idx_ok'])}")  ## NEW
print(f"breadth leg {g['breadth']:.1%} above SMA{cfg.backtest.breadth_sma} "         ## NEW
      f"(need {cfg.backtest.breadth_threshold:.0%}): {bool(g['breadth_ok'])}")
print(f"regime    {'ON  -> hold stocks' if regime_on else 'OFF -> 100% gold'}")
print(f"\nTop {cfg.backtest.top_n} for {latest.date()}:")
print(top.to_string(index=False))
