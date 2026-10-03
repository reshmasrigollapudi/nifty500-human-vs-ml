"""Convert the indian_stocks_monitor NSE dataset
(survivorship-free, 2000 ->) into data/*.csv.

Open/close come straight from the monitor's
total-return adjusted pickles. High/low come from
the raw bhavcopies times the same adjustment
factor (adj_close / raw close). Volume is set to
rupee turnover / adj close, so close * volume is
the true traded value through splits."""
import argparse
import glob
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from mlrank.config import DATA_DIR

p = argparse.ArgumentParser()
p.add_argument("--src", default=str(Path.home() / "Documents/indian_stocks_monitor"))
a = p.parse_args()

src = Path(a.src)
sys.path.insert(0, str(src / "backtest"))
from build_nse_dataset import SERIES, apply_renames, load_renames  # noqa: E402

D = src / "data"


def read_bhav(path):
    z = zipfile.ZipFile(path)
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    df.columns = [c.strip() for c in df.columns]
    if "TckrSymb" in df.columns:                  # UDiFF (Jul 2024 ->)
        df = df[df.FinInstrmTp.eq("STK")]
        df = df.rename(columns={
            "TckrSymb": "sym", "SctySrs": "series", "TradDt": "date",
            "HghPric": "high", "LwPric": "low", "ClsPric": "close",
            "TtlTradgVol": "qty", "TtlTrfVal": "val"})
    else:
        df = df.rename(columns={
            "SYMBOL": "sym", "SERIES": "series", "TIMESTAMP": "date",
            "HIGH": "high", "LOW": "low", "CLOSE": "close",
            "TOTTRDQTY": "qty", "TOTTRDVAL": "val"})
    df["sym"] = df.sym.astype(str).str.strip()
    df["series"] = df.series.astype(str).str.strip()
    df = df[df.series.isin(SERIES)]
    out = df[["date", "sym", "series", "high", "low", "close", "qty", "val"]].copy()
    out["date"] = pd.to_datetime(out.date, format="mixed", dayfirst=False)
    return out


files = sorted(glob.glob(str(D / "nse_bhav" / "*.zip")))
print(f"reading {len(files)} bhavcopies ...", flush=True)
raw = pd.concat([read_bhav(f) for f in files], ignore_index=True)
sc = load_renames()
raw["symbol"] = apply_renames(raw.sym.values.astype(object), raw.date.values, sc)
raw["pri"] = raw.series.map(SERIES)
raw = (raw.sort_values(["symbol", "date", "pri"])
          .drop_duplicates(["symbol", "date"], keep="first"))

C = pd.read_pickle(D / "nse_close_adj.pkl")
O = pd.read_pickle(D / "nse_open_adj.pkl")
adj = pd.DataFrame({"close": C.stack(), "open": O.stack()})
adj.index.names = ["date", "symbol"]
prices = raw.merge(adj.reset_index(), on=["date", "symbol"],
                   how="inner", suffixes=("_raw", ""))
F = prices.close / prices.close_raw
prices["high"] = prices.high * F
prices["low"] = prices.low * F
# old bhavcopies have zero/blank TOTTRDVAL on some rows
val = prices.val.where(prices.val > 0, prices.qty * prices.close_raw)
prices["volume"] = val / prices.close
prices = prices[["date", "symbol", "open", "high", "low", "close", "volume"]]
prices = prices.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])

# Daily bool membership -> spells (end_date inclusive)
M = pd.read_pickle(D / "nse_membership_daily.pkl")
spells = []
for sym in M.columns[M.any()]:
    m = M[sym].to_numpy()
    d = np.diff(np.r_[False, m, False].astype(int))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        end = pd.NaT if e == len(m) - 1 else M.index[e]
        spells.append((sym, M.index[s], end))
membership = pd.DataFrame(spells, columns=["symbol", "start_date", "end_date"])

# Calendar = index dates, so keep only days with stock prices
n50 = pd.read_pickle(D / "nifty50_nse.pkl")
n50 = n50[n50.index.isin(C.index)]
index = n50.rename("close").rename_axis("date").reset_index()

g = pd.read_pickle(D / "gold_long.pkl")
gold = g.rename("close").rename_axis("date").reset_index()
if "GOLDBEES" in O.columns:                       # real open where it exists
    gold["open"] = gold.date.map(O["GOLDBEES"]).fillna(gold.close)

tables = {"prices": prices, "membership": membership,
          "index": index, "gold": gold}
for name, df in tables.items():
    df.to_csv(DATA_DIR / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    print(f"{name:<11} {len(df):>10,} rows -> {DATA_DIR / (name + '.csv')}")
print(f"prices {prices.date.min().date()} .. {prices.date.max().date()}, "
      f"{prices.symbol.nunique():,} symbols")
