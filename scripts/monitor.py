"""Live model health:
- realized IC of each logged month vs the
  backtest's OOS IC
- drift of raw and regime features: last 12
  months vs the training window
- fallback rule: trailing 3-month live IC < 0
  -> switch back to the momentum baseline

Note: z-scored features barely drift by design
(each month is re-standardized), so drift is
checked on raw and market-level columns."""
import _path  # noqa: F401
import pandas as pd

from mlrank.drift import drift_report
from mlrank.features import REGIME_FEATURES
from mlrank.io import base_args, load_all, out_dir
from mlrank.metrics import ic_summary, rank_ic

p = base_args(__doc__)
p.add_argument("--window", type=int, default=3)
args = p.parse_args()
args.rebuild = True

cfg, md, panel = load_all(args)
o = out_dir(args)

log_path = o / "prediction_log.csv"
if log_path.exists():
    log = pd.read_csv(log_path, parse_dates=["date"])
    log = log.drop_duplicates(["date", "symbol"], keep="last").set_index(["date", "symbol"])
    real = panel["fwd_ret"].reindex(log.index)
    ok = real.notna()
    if ok.any():
        ic = rank_ic(log.index.get_level_values("date")[ok], log["score"][ok], real[ok])
        print("Live IC by month")
        print(ic.round(4).to_string())
        bt = o / "model_comparison.csv"
        if bt.exists():
            ref = pd.read_csv(bt, index_col=0)
            name = log["model"].iloc[-1]
            if name in ref.index:
                print(f"backtest OOS mean IC ({name}): {ref.loc[name, 'mean_ic']:.4f}")
        print(pd.Series(ic_summary(ic)).round(4).to_string())
        if len(ic) >= args.window and ic.tail(args.window).mean() < 0:
            print(f"\nALERT: trailing {args.window}m IC < 0 -> fall back to baseline")
    else:
        print("No realized labels yet for logged predictions.")
else:
    print("No prediction_log.csv yet (run train_live.py).")

dates = panel.index.get_level_values("date").unique().sort_values()
recent = dates[-12:]
train = dates[:-12]
d = panel.index.get_level_values("date")
stock_cols = ["raw_mom_12m", "raw_dist_52w_high"]
rep_s = drift_report(panel.loc[d.isin(train), stock_cols],
                     panel.loc[d.isin(recent), stock_cols])
# one value per month for market-level features
reg = panel.groupby(level="date")[REGIME_FEATURES].first()
rep_r = drift_report(reg.loc[train], reg.loc[recent], bins=4, alert_on="ks")
rep = pd.concat([rep_s, rep_r], ignore_index=True)
rep.to_csv(o / "drift_report.csv", index=False)
print("\nFeature drift (last 12m vs history)")
print(rep.round(4).to_string())
