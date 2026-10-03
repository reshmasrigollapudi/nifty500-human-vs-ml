"""Write a synthetic market to data/synthetic/
so the whole pipeline runs without real data."""
import argparse

import _path  # noqa: F401
from mlrank.config import DATA_DIR
from mlrank.synthetic import generate

p = argparse.ArgumentParser()
p.add_argument("--symbols", type=int, default=200)
p.add_argument("--start", default="2005-01-01")
p.add_argument("--end", default="2026-08-31")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

out = DATA_DIR / "synthetic"
out.mkdir(parents=True, exist_ok=True)
for name, df in generate(a.symbols, a.start, a.end, seed=a.seed).items():
    df.to_csv(out / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    print(f"{name:<11} {len(df):>9,} rows -> {out / (name + '.csv')}")
