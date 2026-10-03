"""Command-line client for the strategy service.

  python service/client.py gate
  python service/client.py ranking --strategy c3 --top 20
  python service/client.py backtest --strategy lgbm_reg --start 2009-11-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc  # noqa: E402

from service import strategy_pb2 as pb  # noqa: E402
from service import strategy_pb2_grpc as rpc  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("command", choices=["gate", "ranking", "backtest"])
p.add_argument("--host", default="localhost:50051")
p.add_argument("--strategy", default="c3")
p.add_argument("--top", type=int, default=0)
p.add_argument("--start", default="")
p.add_argument("--end", default="")
a = p.parse_args()

stub = rpc.StrategyServiceStub(grpc.insecure_channel(a.host))
try:
    if a.command == "gate":
        g = stub.GetGate(pb.GateRequest())
        print(f"{g.date}: {'STOCKS' if g.gate_on else 'GOLD'}  "
              f"(NIFTY>SMA100 {g.index_ok}, breadth {g.breadth:.1%} ok={g.breadth_ok})")
    elif a.command == "ranking":
        r = stub.GetRanking(pb.RankingRequest(strategy=a.strategy, top_n=a.top))
        note = "" if r.gate_on else "  [gate off: hold GOLDBEES, ranking is reference only]"
        print(f"{r.strategy}: decided {r.decision_date}, trade {r.trade_date}{note}")
        for s in r.stocks:
            print(f"{s.rank:>3}  {s.symbol:<12} {s.score:8.3f}")
    else:
        req = pb.BacktestRequest(strategy=a.strategy, start=a.start, end=a.end)
        for u in stub.RunBacktest(req):            # streamed: one message per month
            if u.HasField("month"):
                m = u.month
                if m.date[5:7] == "01":            # print once a year
                    print(f"{m.date}  equity x{m.equity:7.2f}  {'GOLD' if m.in_gold else f'{m.holdings} stocks'}")
            else:
                s = u.summary
                print(f"\n{a.strategy} {s.start}..{s.end}: CAGR {s.cagr:.1%}, "
                      f"max drawdown {s.max_drawdown:.1%}, Sharpe {s.sharpe:.2f}, turnover {s.turnover:.1f}x/yr")
except grpc.RpcError as e:
    sys.exit(f"{e.code().name}: {e.details()}")
