import numpy as np
import pandas as pd

from helpers import small_md
from mlrank.backtest import run_backtest
from mlrank.config import BacktestConfig, Config
from mlrank.dataset import build_panel


def test_costs_only_with_flat_prices():
    md = small_md()
    for f in ["open", "high", "low", "close"]:
        w = getattr(md, f)
        setattr(md, f, w.where(w.isna(), 100.0))
    md.gold_open = md.gold_close = pd.Series(50.0, index=md.dates)
    panel = build_panel(md, Config())
    rng = np.random.default_rng(0)
    score = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    cfg = BacktestConfig(buy_cost=0.001, sell_cost=0.001, top_n=10)
    r = run_backtest(md, score, cfg)
    assert r.equity.iloc[-1] <= cfg.capital
    n_side = r.trades["value"].sum()
    assert np.isclose(cfg.capital - r.equity.iloc[-1], 0.001 * n_side, rtol=0.05)


def test_regime_off_goes_to_gold():
    md = small_md()
    panel = build_panel(md, Config())
    rebal = pd.DatetimeIndex(panel.index.get_level_values("date").unique())
    off = pd.Series(False, index=rebal)
    r = run_backtest(md, panel["mom_6m"], BacktestConfig(), regime_on=off)
    assert all(h == ["GOLD"] for h in r.holdings.values())
    assert set(r.trades["symbol"]) == {"GOLD"}


def test_winners_run_keeps_survivors():
    md = small_md()
    panel = build_panel(md, Config())
    r = run_backtest(md, panel["mom_6m"], BacktestConfig(top_n=10))
    ds = sorted(r.holdings)
    for a, b in zip(ds[:-1], ds[1:]):
        assert len(r.holdings[b]) <= 10
    buys = r.trades[r.trades.side == "BUY"]
    sells = r.trades[r.trades.side == "SELL"]
    # a name is never sold and re-bought on the same date
    both = pd.merge(buys, sells, on=["date", "symbol"])
    assert both.empty


## NEW: gate is bitwise OR of both legs
def test_equity_gate_is_or():
    from mlrank.backtest import equity_gate_daily
    md = small_md()
    g = equity_gate_daily(md, BacktestConfig())
    assert (g["gate"] == (g["idx_ok"] | g["breadth_ok"])).all()
    # breadth leg alone can keep us in equity
    assert (g["gate"] & ~g["idx_ok"]).any() or not g["breadth_ok"].any()
    # index leg alone can keep us in equity
    assert (g["gate"] & ~g["breadth_ok"]).any()
    assert ((g["breadth"] >= 0.6) == g["breadth_ok"]).loc[g["breadth"].notna()].all()
