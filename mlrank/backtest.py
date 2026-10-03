"""Monthly top-N backtest matching the live rules:

- T = first trading day of the month; decision
  on the T-1 close (last trading day), execution
  at the T open
- winners_run: sell names that left the top N,
  split freed cash equally across new entrants,
  never trim survivors
- regime OFF: everything into gold (or cash)
  gate = NIFTY50>SMA100 | breadth>=60%
- costs on traded value: 0.20% buy, 0.30% sell
- held stock that stops trading is sold at its
  last close on the next rebalance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data_loader import MarketData


@dataclass
class BacktestResult:
    equity: pd.Series            # daily close marks
    monthly: pd.Series           # rebalance-to-rebalance returns
    trades: pd.DataFrame
    holdings: dict               # T -> list of symbols (or ["GOLD"])
    stats: dict


def _stats(equity: pd.Series, traded: float, rf: float) -> dict:
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": (r.mean() * 252 - rf) / vol if vol > 0 else np.nan,
        "max_dd": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "turnover_ann": traded / equity.mean() / years,
        "years": years,
    }


def run_backtest(
    md: MarketData,
    scores: pd.Series,
    cfg: BacktestConfig,
    regime_on: pd.Series | None = None,
    eligible: pd.Series | None = None,
    start: pd.Timestamp | None = None,
) -> BacktestResult:
    """scores: MultiIndex (date=T, symbol), higher = better.
    regime_on: bool per T. eligible: bool per (T, symbol)."""
    scores = scores.dropna()
    if eligible is not None:
        el = eligible.reindex(scores.index).fillna(False).astype(bool)
        scores = scores[el.to_numpy()]
    rebal = pd.DatetimeIndex(sorted(scores.index.get_level_values("date").unique()))
    if start is not None:
        rebal = rebal[rebal >= start]
    cal = md.dates
    close_ff = md.close.ffill()
    op = md.open
    g_close = md.gold_close.ffill() if md.gold_close is not None else None
    g_open = md.gold_open if md.gold_open is not None else None

    cash = cfg.capital
    shares: dict[str, float] = {}
    gold = 0.0
    trades, holdings, traded = [], {}, 0.0
    eq_parts = []

    def px(sym, T):
        p = op.at[T, sym]
        if np.isnan(p):
            prev = cal[cal.get_loc(T) - 1]
            p = close_ff.at[prev, sym]
        return p

    def gpx(T):
        p = g_open.at[T] if g_open is not None else np.nan
        if np.isnan(p):
            prev = cal[cal.get_loc(T) - 1]
            p = g_close.at[prev]
        return p

    for k, T in enumerate(rebal):
        on = True if regime_on is None else bool(regime_on.get(T, True))
        s = scores.xs(T, level="date")
        tradable = op.loc[T, s.index].notna().to_numpy()
        target = list(s[tradable].sort_values(ascending=False).index[: cfg.top_n])

        def sell(sym):
            nonlocal cash, traded
            p = px(sym, T)
            val = shares.pop(sym) * p
            cash += val * (1 - cfg.sell_cost)
            traded += val
            trades.append((T, sym, "SELL", val))

        def buy(sym, amount):
            nonlocal cash, traded
            p = px(sym, T)
            shares[sym] = shares.get(sym, 0.0) + amount * (1 - cfg.buy_cost) / p
            cash -= amount
            traded += amount
            trades.append((T, sym, "BUY", amount))

        if not on:
            for sym in list(shares):
                sell(sym)
            if g_close is not None and gold == 0 and cash > 0:
                p = gpx(T)
                gold = cash * (1 - cfg.gold_cost) / p
                traded += cash
                trades.append((T, "GOLD", "BUY", cash))
                cash = 0.0
            holdings[T] = ["GOLD"] if gold > 0 else ["CASH"]
        else:
            if gold > 0:
                val = gold * gpx(T)
                cash += val * (1 - cfg.gold_cost)
                traded += val
                trades.append((T, "GOLD", "SELL", val))
                gold = 0.0
            if cfg.mode == "full":
                for sym in list(shares):
                    sell(sym)
            for sym in [x for x in shares if x not in target]:
                sell(sym)
            entrants = [x for x in target if x not in shares]
            entrants = entrants[: max(0, cfg.top_n - len(shares))]
            if entrants and cash > 0:
                each = cash / len(entrants)
                for sym in entrants:
                    buy(sym, each)
            holdings[T] = sorted(shares)

        # mark to market at daily closes until next rebalance
        end = rebal[k + 1] if k + 1 < len(rebal) else cal[-1]
        seg = cal[(cal >= T) & (cal < end)] if k + 1 < len(rebal) else cal[cal >= T]
        if shares:
            syms = list(shares)
            val = close_ff.loc[seg, syms].to_numpy() @ np.array([shares[x] for x in syms])
        else:
            val = np.zeros(len(seg))
        if gold > 0:
            val = val + gold * g_close.loc[seg].to_numpy()
        eq_parts.append(pd.Series(val + cash, index=seg))

    equity = pd.concat(eq_parts)
    monthly = equity.reindex(rebal).pct_change().dropna()
    tr = pd.DataFrame(trades, columns=["date", "symbol", "side", "value"])
    return BacktestResult(equity, monthly, tr, holdings,
                          _stats(equity, traded, cfg.rf_annual))


def buy_and_hold(series: pd.Series, start, rf: float = 0.0) -> dict:
    s = series.ffill()
    s = s[s.index >= start]
    return _stats(s / s.iloc[0], 0.0, rf)


def sma_regime(md: MarketData, rebal: pd.DatetimeIndex, window: int) -> pd.Series:
    """Index above its SMA at the T-1 close.
    Index leg only; kept for comparison."""
    ix = md.index_close.ffill()
    ok = ix > ix.rolling(window, min_periods=window).mean()
    prev = md.dates[md.dates.get_indexer(rebal) - 1]
    return pd.Series(ok.reindex(prev).to_numpy(), index=rebal)


## NEW: gate = index leg | breadth leg
def equity_gate_daily(md: MarketData, cfg: BacktestConfig) -> pd.DataFrame:
    """Stay in equity when
         NIFTY 50 > SMA100
       | share of NIFTY 500 members above
         their own SMA200 >= 60%
    Bitwise OR on boolean Series. A leg with
    too little history (NaN) counts as False.
    Breadth denominator: PIT members that have
    a full SMA200 (>= 200 trading days)."""
    ix = md.index_close.ffill()
    ix_sma = ix.rolling(cfg.regime_sma, min_periods=cfg.regime_sma).mean()
    idx_ok = (ix > ix_sma).fillna(False).astype(bool)

    c = md.close.ffill(limit=5)
    s200 = c.rolling(cfg.breadth_sma, min_periods=cfg.breadth_sma).mean()
    valid = s200.notna() & md.member
    above = (c > s200) & valid
    breadth = above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    breadth_ok = (breadth >= cfg.breadth_threshold).fillna(False).astype(bool)

    return pd.DataFrame({
        "idx_ok": idx_ok,
        "breadth": breadth,
        "breadth_ok": breadth_ok,
        "gate": idx_ok | breadth_ok,
    })


## NEW: sample gate at T-1 close
def equity_gate(md: MarketData, rebal: pd.DatetimeIndex,
                cfg: BacktestConfig) -> pd.Series:
    daily = equity_gate_daily(md, cfg)
    prev = md.dates[md.dates.get_indexer(rebal) - 1]
    return pd.Series(daily["gate"].reindex(prev).to_numpy(), index=rebal)


def falling_knife_filter(panel: pd.DataFrame, max_below: float) -> pd.Series:
    return panel["raw_dist_52w_high"] >= -max_below
