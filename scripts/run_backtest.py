"""Human vs ML under the live rules.

Human: the rule strategies in mlrank/strategies.py
(C3, mom12_vol, low_vol). ML: LightGBM's
walk-forward predictions (run_experiment.py).
All get the same universe, top N, gate, gold,
costs and timing.

Test period: from the first ML prediction (after
the 7-year training window, Nov 2009) to the end.
  - CAGR / max drawdown / Sharpe / turnover
  - calendar-year returns and wins
  - average CAGR over random 3-15 year windows
  - bootstrap of each CAGR difference vs C3
Context only: the rule strategies over the full
history (ML can't be tested before 2009).
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd

from mlrank.backtest import buy_and_hold, equity_gate, run_backtest
from mlrank.bootstrap import bootstrap_diff
from mlrank.io import base_args, load_all, load_frame, out_dir
from mlrank.metrics import ic_summary, rank_ic
from mlrank.strategies import RULES, eligible, rule_scores

p = base_args(__doc__)
p.add_argument("--windows", type=int, default=1000)
p.add_argument("--n-boot", type=int, default=10_000)
p.add_argument("--reference", default="c3")
args = p.parse_args()

cfg, md, panel = load_all(args)
o = out_dir(args)
bt = cfg.backtest
el = eligible(panel)
scores = rule_scores(panel)
ml_path = o / "oos_predictions.csv"
if ml_path.exists():
    ml = load_frame(ml_path)
    scores = scores.join(ml, how="left")
    start = ml.index.get_level_values("date").min()
else:
    print("no oos_predictions.csv: rule strategies only")
    start = scores.dropna().index.get_level_values("date").min()
scores = scores[el.to_numpy()]
dates = scores.index.get_level_values("date")
rebal = pd.DatetimeIndex(sorted(dates.unique()))
gate = equity_gate(md, rebal, bt)

results, equity, monthly = {}, {}, {}
for name in scores.columns:
    r = run_backtest(md, scores[name], bt, regime_on=gate, start=start)
    test = dates >= start
    ic = ic_summary(rank_ic(dates[test], scores.loc[test, name],
                            panel.loc[scores.index[test], "fwd_ret"]))
    results[name] = {
        **{k: r.stats[k] for k in ["cagr", "max_dd", "sharpe", "vol", "turnover_ann"]},
        "rank_ic": ic["mean_ic"],
    }
    equity[name], monthly[name] = r.equity, r.monthly
    r.equity.to_frame("equity").to_csv(o / f"equity_{name}.csv")
results["nifty50_buy_hold"] = {k: buy_and_hold(md.index_close, start)[k]
                               for k in ["cagr", "max_dd", "sharpe", "vol"]}
table = pd.DataFrame(results).T
table.to_csv(o / "strategy_comparison.csv")

# Context: rule strategies over all the history they have
full = {}
for name in RULES:
    r = run_backtest(md, scores[name], bt, regime_on=gate)
    full[name] = {"from": r.equity.index[0].date(), **{k: r.stats[k] for k in ["cagr", "max_dd"]}}
full = pd.DataFrame(full).T
full.to_csv(o / "rules_full_history.csv")

# Calendar-year returns (first and last years are partial)
E = pd.DataFrame(equity)
yearly = E.groupby(E.index.year).last().pct_change()
yearly.iloc[0] = E.groupby(E.index.year).last().iloc[0] / E.iloc[0] - 1
yearly.to_csv(o / "strategy_yearly.csv")
wins = yearly.idxmax(axis=1).value_counts().reindex(E.columns, fill_value=0)

# Random windows: same windows for every strategy
Em = E.reindex(rebal[rebal >= start], method="ffill")
rng = np.random.default_rng(cfg.seed)
rows = []
for _ in range(args.windows):
    L = 12 * int(rng.integers(3, 16))
    if L >= len(Em):
        continue
    i = int(rng.integers(0, len(Em) - L))
    a, b = Em.iloc[i], Em.iloc[i + L]
    rows.append((b / a) ** (12 / L) - 1)
W = pd.DataFrame(rows)
W.to_csv(o / "strategy_windows.csv", index=False)
win_summary = pd.DataFrame({
    "mean_cagr": W.mean(),
    "worst5pct_cagr": W.quantile(0.05),
    "pct_windows_best": W.idxmax(axis=1).value_counts(normalize=True).reindex(W.columns, fill_value=0),
    "years_best": wins,
})
win_summary.to_csv(o / "strategy_windows_summary.csv")

ref = args.reference
boot = pd.DataFrame({n: bootstrap_diff(monthly[n], monthly[ref], n=args.n_boot)
                     for n in monthly if n != ref}).T
boot.to_csv(o / f"bootstrap_vs_{ref}.csv")

pd.set_option("display.width", 180)
pct = lambda df: (df * 100).round(1)
print(f"\nLive rules: top {bt.top_n}, costs {bt.buy_cost:.2%} buy / {bt.sell_cost:.2%} sell, "
      f"gate on {gate[gate.index >= start].mean():.0%} of months")
print(f"Test period {start.date()} .. {E.index[-1].date()}\n")
print(pct(table[["cagr", "max_dd", "vol"]])
      .join(table[["sharpe", "turnover_ann", "rank_ic"]].astype(float).round(2))
      .sort_values("cagr", ascending=False).to_string())
print("\nCalendar-year returns (%)")
print(pct(yearly).to_string())
print("\nRandom 3-15 year windows (%), and calendar years won")
print(pct(win_summary[["mean_cagr", "worst5pct_cagr", "pct_windows_best"]])
      .join(win_summary["years_best"]).sort_values("mean_cagr", ascending=False).to_string())
print(f"\nBootstrap: CAGR difference vs {ref} (annualised, {args.n_boot:,} runs)")
print(boot[["observed_diff", "ci_low", "ci_high", "p_a_not_better"]].round(3).to_string())
print("\nContext: rule strategies over their full history (no ML before 2009)")
print(full.assign(cagr=pct(full["cagr"].astype(float)), max_dd=pct(full["max_dd"].astype(float))).to_string())
print(f"\nsaved -> {o}")
