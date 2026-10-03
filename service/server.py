"""gRPC server for the strategy service.

  python service/server.py               # real data, port 50051
  python service/server.py --synthetic   # small fake data

Loads the market data and panel once at start-up.
The ML model is trained on first use and cached.
ML backtests use the walk-forward predictions from
scripts/run_experiment.py (no future data).
"""
import sys
import threading
from concurrent import futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlrank.backtest import _stats, equity_gate, run_backtest  # noqa: E402
from mlrank.live import ML_STRATEGIES, STRATEGIES, gate_status, latest_panel, rank, train_ml  # noqa: E402
from mlrank.strategies import RULES, eligible, rule_scores  # noqa: E402
from service import strategy_pb2 as pb  # noqa: E402
from service import strategy_pb2_grpc as rpc  # noqa: E402


def _day(ts) -> str:
    return str(pd.Timestamp(ts).date())


class StrategyServicer(rpc.StrategyServiceServicer):
    def __init__(self, md, cfg, panel, oos: pd.DataFrame | None = None):
        self.md, self.cfg, self.panel, self.oos = md, cfg, panel, oos
        self.dec, self.trade, self.live = latest_panel(md, cfg)
        self.gate = gate_status(md, cfg)
        self._models, self._lock = {}, threading.Lock()

    def _model(self, name):
        with self._lock:                       # train once, even with parallel calls
            if name not in self._models:
                self._models[name] = train_ml(self.panel, name)
            return self._models[name]

    def _check(self, strategy, context):
        if strategy not in STRATEGIES:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"unknown strategy '{strategy}'; have {STRATEGIES}")

    def GetGate(self, request, context):
        g = self.gate
        return pb.GateResponse(date=_day(g["date"]), gate_on=g["gate_on"], index_ok=g["index_ok"],
                               breadth=g["breadth"], breadth_ok=g["breadth_ok"])

    def GetRanking(self, request, context):
        s = request.strategy or "c3"
        self._check(s, context)
        model = self._model(s) if s in ML_STRATEGIES else None
        n = request.top_n or self.cfg.backtest.top_n
        top = rank(self.live, s, model).head(n)
        return pb.RankingResponse(
            strategy=s, decision_date=_day(self.dec), trade_date=_day(self.trade),
            gate_on=self.gate["gate_on"],
            stocks=[pb.RankedStock(rank=i + 1, symbol=sym, score=float(v))
                    for i, (sym, v) in enumerate(top.items())])

    def _scores(self, s, context) -> pd.Series:
        if s in RULES:
            sc = rule_scores(self.panel, [s])[s]
        elif self.oos is None or s not in self.oos:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"no walk-forward predictions for {s}: run scripts/run_experiment.py")
        else:
            sc = self.oos[s].reindex(self.panel.index)
        return sc[eligible(self.panel).to_numpy()].dropna()

    def RunBacktest(self, request, context):
        s = request.strategy or "c3"
        self._check(s, context)
        sc = self._scores(s, context)
        d = sc.index.get_level_values("date")
        end = pd.Timestamp(request.end) if request.end else None
        keep = np.ones(len(sc), dtype=bool)
        if request.start:
            keep &= d >= pd.Timestamp(request.start)
        if end is not None:
            keep &= d <= end
        sc = sc[keep]
        if sc.empty:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no scores in that period")
        rebal = pd.DatetimeIndex(sorted(sc.index.get_level_values("date").unique()))
        bt = self.cfg.backtest
        r = run_backtest(self.md, sc, bt, regime_on=equity_gate(self.md, rebal, bt))
        rupees = r.equity if end is None else r.equity[:end]
        eq = rupees / rupees.iloc[0]               # growth of 1.0, for the stream
        for T in rebal:
            if not context.is_active():            # client went away
                return
            h = r.holdings.get(T, [])
            gold = h in (["GOLD"], ["CASH"])
            yield pb.BacktestUpdate(month=pb.MonthPoint(
                date=_day(T), equity=float(eq.asof(T)), in_gold=gold, holdings=0 if gold else len(h)))
        traded = r.trades.loc[r.trades["date"] <= eq.index[-1], "value"].sum()
        st = _stats(rupees, traded, bt.rf_annual)
        yield pb.BacktestUpdate(summary=pb.BacktestSummary(
            cagr=st["cagr"], max_drawdown=st["max_dd"], sharpe=st["sharpe"],
            turnover=st["turnover_ann"], start=_day(eq.index[0]), end=_day(eq.index[-1])))


def serve(servicer, port: int = 50051, workers: int = 4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    rpc.add_StrategyServiceServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server, bound


if __name__ == "__main__":
    from mlrank.io import base_args, load_all, load_frame, out_dir

    p = base_args(__doc__)
    p.add_argument("--port", type=int, default=50051)
    args = p.parse_args()
    cfg, md, panel = load_all(args)
    f = out_dir(args) / "oos_predictions.csv"
    oos = load_frame(f) if f.exists() else None
    print("building live features ...", flush=True)
    server, port = serve(StrategyServicer(md, cfg, panel, oos), args.port)
    print(f"StrategyService ready on port {port} (Ctrl+C to stop)", flush=True)
    server.wait_for_termination()
