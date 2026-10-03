"""Fill the gaps in the NIFTY 500 membership history.

Constituent lists exist only at snapshot dates. Where
two lists are more than --min-gap days apart,
membership.csv carries the older list forward
unchanged. Inside each gap this script:
  1. applies the logged index changes
     (nifty500_changes.csv) on their effective dates
  2. drops a stock after its last trading day
  3. compares the result (plus changes logged for the
     end date) with the list at the end of the gap,
     before any reconciliation

Writes two files next to membership.csv:
  membership_bridged.csv      changes the log missed
                              switch at the end of the
                              gap (no lookahead)
  membership_bridged_mid.csv  they switch at the gap
                              midpoint (sensitivity; only
                              where the log covers the gap)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from mlrank.config import DATA_DIR
from mlrank.data_loader import _membership_mask, read_csv

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default=str(Path.home() / "Documents/indian_stocks_monitor"))
p.add_argument("--min-gap", type=int, default=180, help="days between lists")
a = p.parse_args()

src = Path(a.src)
sys.path.insert(0, str(src / "backtest"))
from build_nse_dataset import apply_renames, load_renames  # noqa: E402

cal = pd.DatetimeIndex(pd.to_datetime(read_csv(DATA_DIR / "index.csv")["date"]))
px = read_csv(DATA_DIR / "prices.csv")[["date", "symbol", "close"]]
px["date"] = pd.to_datetime(px["date"])
px = px[px["close"] > 0]
first_trade = px.groupby("symbol")["date"].min()
last_trade = px.groupby("symbol")["date"].max()
syms = pd.Index(sorted(px["symbol"].unique()))
del px

spells = read_csv(DATA_DIR / "membership.csv")
M = _membership_mask(spells, cal, syms)

# Change log -> latest NSE symbol, same mapping as the price data
sc = load_renames()
ch = read_csv(src / "data" / "nifty500_changes.csv")
ch["effective_date"] = pd.to_datetime(ch["effective_date"])
ch = ch.sort_values("effective_date", kind="stable")
raw = ch["symbol"].str.replace("DELISTED_", "", regex=False).to_numpy(dtype=object)
ch["key"] = apply_renames(raw, ch["effective_date"].to_numpy(), sc)

# Gaps: list changes more than min_gap days apart
changed = M.ne(M.shift()).any(axis=1)
changed.iloc[0] = False
chg = M.index[changed.to_numpy()]
gaps = [(s, e) for s, e in zip(chg[:-1], chg[1:]) if (e - s).days > a.min_gap]


def bridge(start, end, mid_fix=None):
    """Daily member sets for trading days in [start, end)."""
    days = cal[(cal >= start) & (cal < end)]
    ev = ch[(ch["effective_date"] > start) & (ch["effective_date"] < end)]
    ev_day = days[np.minimum(days.searchsorted(ev["effective_date"]), len(days) - 1)]
    cur = set(syms[M.loc[start].to_numpy()])
    out, j = {}, 0
    mid = start + (end - start) / 2
    for d in days:
        while j < len(ev) and ev_day[j] <= d:
            k, act = ev["key"].iloc[j], ev["action"].iloc[j]
            if k in last_trade.index:
                (cur.add if act == "ADD" else cur.discard)(k)
            j += 1
        if mid_fix is not None and d >= mid:
            extra, missing = mid_fix
            cur = (cur - extra) | {s for s in missing if first_trade[s] <= d}
            mid_fix = None
        cur = {s for s in cur if last_trade[s] >= d}
        out[d] = set(cur)
    return out


def to_spells(mask: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym in mask.columns[mask.any().to_numpy()]:
        m = mask[sym].to_numpy()
        dif = np.diff(np.r_[False, m, False].astype(int))
        for s, e in zip(np.flatnonzero(dif == 1), np.flatnonzero(dif == -1) - 1):
            end = pd.NaT if e == len(m) - 1 else mask.index[e]
            rows.append((sym, mask.index[s], end))
    return pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"])


M_end, M_mid = M.copy(), M.copy()
report = []
for s, e in gaps:
    A = set(syms[M.loc[s].to_numpy()])
    B = set(syms[M.loc[e].to_numpy()])
    est = bridge(s, e)
    last = est[max(est)]
    # changes logged for the end date itself are explained; only
    # the rest is unknown timing, and only where the log reaches
    on_end = set(ch.loc[ch["effective_date"].between(max(est), e) & ch["action"].eq("ADD"), "key"])
    off_end = set(ch.loc[ch["effective_date"].between(max(est), e) & ch["action"].eq("REMOVE"), "key"])
    at_end = (last | on_end) - off_end
    extra, missing = at_end - B, B - at_end
    covered = e <= ch["effective_date"].max()
    est_mid = bridge(s, e, mid_fix=(extra, missing)) if covered else est
    for d, members in est.items():
        M_end.loc[d] = syms.isin(list(members))
    for d, members in est_mid.items():
        M_mid.loc[d] = syms.isin(list(members))
    ev = ch[(ch["effective_date"] > s) & (ch["effective_date"] < e)]
    report.append({
        "gap": f"{s.date()} -> {e.date()}",
        "years": round((e - s).days / 365.25, 1),
        "log_events": len(ev),
        "unmatched": int((~ev["key"].isin(last_trade.index)).sum()),
        "carry_match": len(A & B) / len(B),
        "bridged_match": len(at_end & B) / len(B),
        "extra": len(extra), "missing": len(missing),
        "mid_fix": covered,
        "mean_size": np.mean([len(v) for v in est.values()]),
    })

pd.set_option("display.width", 160)
print(pd.DataFrame(report).round(3).to_string(index=False))
for name, mask in {"membership_bridged": M_end, "membership_bridged_mid": M_mid}.items():
    out = to_spells(mask)
    out.to_csv(DATA_DIR / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    print(f"{name:<24} {len(out):>5} spells, members/day mean "
          f"{mask.sum(axis=1)[mask.any(axis=1)].mean():.0f}")
