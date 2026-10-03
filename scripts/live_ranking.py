"""This month's live output: rank NIFTY 500 by a
strategy (default C3) at the latest close and
print the top N to hold, a watchlist, and the
gate (stocks or GOLDBEES).

Run after the last trading day's close with data
through that day; trade at the next open.
Writes outputs/real/live/<strategy>_<date>.csv.
"""
import _path  # noqa: F401
import pandas as pd

from mlrank.backtest import equity_gate_daily
from mlrank.data_loader import MarketData
from mlrank.io import base_args, config_for, out_dir
from mlrank.live import latest_panel, rank
from mlrank.strategies import RULES

p = base_args(__doc__)
p.add_argument("--strategy", default="c3", choices=list(RULES))
p.add_argument("--watch", type=int, default=10, help="extra ranks to show")
args = p.parse_args()

cfg = config_for(args)
md = MarketData.load(cfg.data)
bt = cfg.backtest
dec, trade, panel = latest_panel(md, cfg)
ranked = rank(panel, args.strategy)

n = bt.top_n
out = pd.DataFrame({
    "rank": range(1, len(ranked) + 1),
    "symbol": ranked.index,
    "score": ranked.to_numpy(),
    "mom_12_1": panel.xs(trade, level="date").loc[ranked.index, "raw_mom_12_1"].to_numpy(),
    "mom_6m": panel.xs(trade, level="date").loc[ranked.index, "raw_mom_6m"].to_numpy(),
    "mom_3m": panel.xs(trade, level="date").loc[ranked.index, "raw_mom_3m"].to_numpy(),
    "vol_126": panel.xs(trade, level="date").loc[ranked.index, "vol_126"].to_numpy(),
})
g = equity_gate_daily(md, bt).loc[dec]
top = "HOLD" if g["gate"] else "ref"          # gate off: ranking is reference only
out["action"] = [top if r <= n else "watch" for r in out["rank"]]
live_dir = out_dir(args) / "live"
live_dir.mkdir(exist_ok=True)
path = live_dir / f"{args.strategy}_{dec.date()}.csv"
out.to_csv(path, index=False)

pd.set_option("display.width", 160)
print(f"Strategy {args.strategy}  ·  decision close {dec.date()}  ·  trade at the next open")
print(f"Gate: NIFTY 50 > SMA{bt.regime_sma}: {bool(g['idx_ok'])}  |  "
      f"breadth {g['breadth']:.1%} above SMA{bt.breadth_sma} (need {bt.breadth_threshold:.0%}): "
      f"{bool(g['breadth_ok'])}")
if g["gate"]:
    print(f"-> STOCKS: hold the top {n}; sell anything ranked below {n}\n")
else:
    print("-> GOLD: sell all stocks, hold 100% GOLDBEES (ranking shown for reference)\n")
show = out.head(n + args.watch).copy()
for c in ["mom_12_1", "mom_6m", "mom_3m", "vol_126"]:
    show[c] = (show[c] * 100).round(1)
print(show.round({"score": 3}).to_string(index=False))
print(f"\nranked {len(out)} eligible stocks  ·  saved -> {path}")
