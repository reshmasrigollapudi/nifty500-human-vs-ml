"""Rebalance-day check: CAGR of every strategy when
trading on the 1st, 10th or 20th of the month.
Run the pipeline with --rebalance-day 10 / 20 first
(make days)."""
import _path  # noqa: F401
import pandas as pd

from mlrank.config import OUT_DIR

DAYS = {1: "real", 10: "real_day10", 20: "real_day20"}
cols = {}
for day, folder in DAYS.items():
    f = OUT_DIR / folder / "strategy_comparison.csv"
    if f.exists():
        cols[f"day {day}"] = pd.read_csv(f, index_col=0)["cagr"]
t = pd.DataFrame(cols)
t["average"] = t.mean(axis=1)
t["best - worst day"] = t.drop(columns="average").max(axis=1) - t.drop(columns="average").min(axis=1)
t.to_csv(OUT_DIR / "real" / "rebalance_days.csv")
print("CAGR by rebalance day (%), test period Nov 2009 - Sep 2026")
print((t * 100).round(1).sort_values("average", ascending=False).to_string())
