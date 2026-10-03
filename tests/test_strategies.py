"""Rule strategies and live timing match the live
code (indian_stocks_monitor/momentum_rank.py)."""
import numpy as np
import pandas as pd

from helpers import small_md
from mlrank.config import CFG
from mlrank.data_loader import decision_dates, rebalance_dates
from mlrank.dataset import build_panel
from mlrank.strategies import eligible, rule_scores


def test_rebalance_on_first_trading_day():
    md = small_md()
    rebal = rebalance_dates(md.dates)
    dec = decision_dates(md.dates, rebal[1:])
    s = pd.Series(md.dates, index=md.dates)
    assert (rebal == s.groupby([md.dates.year, md.dates.month]).min().values).all()
    # decision = last trading day of the previous month
    last = s.groupby([md.dates.year, md.dates.month]).max().values[:-1]
    assert (dec == last).all()


def test_c3_matches_live_formula():
    md = small_md()
    panel = build_panel(md, CFG)
    sc = rule_scores(panel)["c3"].dropna()
    T = sc.index.get_level_values("date")[-1]
    dec = md.dates[md.dates.get_loc(T) - 1]
    checked = 0
    for sym in sc.xs(T, level="date").index[:10]:
        p = md.close[sym].loc[:dec].dropna()
        if p.isna().any() or len(p) < 253:
            continue
        A = p.iloc[-22] / p.iloc[-253] - 1
        B = p.iloc[-1] / p.iloc[-127] - 1
        C = p.iloc[-1] / p.iloc[-64] - 1
        vol = p.pct_change().iloc[-126:].std() * np.sqrt(252)
        assert np.isclose(sc.loc[(T, sym)], ((A + B + C) / 3) / vol, rtol=1e-9)
        checked += 1
    assert checked >= 5


def test_eligible_needs_one_year():
    md = small_md()
    panel = build_panel(md, CFG)
    el = eligible(panel)
    assert el.any() and not el.all()
    assert panel.loc[el, "raw_mom_12_1"].notna().all()


def test_rebalance_day_shifts_trade_date():
    md = small_md()
    r10 = rebalance_dates(md.dates, day=10)
    assert (r10.day >= 10).all()
    for T in r10:
        month = md.dates[(md.dates.year == T.year) & (md.dates.month == T.month)]
        assert T == month[month.day >= 10].min()
