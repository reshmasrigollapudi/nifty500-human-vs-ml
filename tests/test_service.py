"""gRPC service end to end on synthetic data: a
real server on a free port, called through the
generated client stub."""
import grpc
import pandas as pd

from helpers import small_md
from mlrank.config import CFG
from mlrank.dataset import build_panel
from mlrank.live import rank
from service import strategy_pb2 as pb
from service import strategy_pb2_grpc as rpc
from service.server import StrategyServicer, serve

_ctx = {}


def _stub():
    if not _ctx:
        md = small_md()
        panel = build_panel(md, CFG)
        svc = StrategyServicer(md, CFG, panel)
        server, port = serve(svc, port=0)
        _ctx.update(svc=svc, server=server,
                    stub=rpc.StrategyServiceStub(grpc.insecure_channel(f"localhost:{port}")))
    return _ctx["stub"], _ctx["svc"]


def test_gate_matches_live_module():
    stub, svc = _stub()
    g = stub.GetGate(pb.GateRequest())
    assert g.gate_on == svc.gate["gate_on"]
    assert g.gate_on == (g.index_ok or g.breadth_ok)


def test_ranking_rule_and_ml():
    stub, svc = _stub()
    r = stub.GetRanking(pb.RankingRequest(strategy="c3", top_n=5))
    expect = rank(svc.live, "c3").head(5)
    assert [s.symbol for s in r.stocks] == list(expect.index)
    assert [s.rank for s in r.stocks] == [1, 2, 3, 4, 5]
    ml = stub.GetRanking(pb.RankingRequest(strategy="lgbm_reg"))
    assert len(ml.stocks) == CFG.backtest.top_n


def test_backtest_streams_months_then_summary():
    stub, _ = _stub()
    ups = list(stub.RunBacktest(pb.BacktestRequest(strategy="c3", start="2014-01-01")))
    months = [u.month for u in ups if u.HasField("month")]
    assert ups[-1].HasField("summary") and len(months) > 12
    assert abs(months[0].equity - 1.0) < 0.01
    assert pd.Timestamp(months[0].date) >= pd.Timestamp("2014-01-01")
    assert -1 < ups[-1].summary.max_drawdown <= 0


def test_bad_requests_get_clear_errors():
    stub, _ = _stub()
    for call, req, code in [
        (stub.GetRanking, pb.RankingRequest(strategy="nope"), grpc.StatusCode.INVALID_ARGUMENT),
        (lambda q: list(stub.RunBacktest(q)), pb.BacktestRequest(strategy="lgbm_reg"),
         grpc.StatusCode.FAILED_PRECONDITION),           # no walk-forward predictions loaded
    ]:
        try:
            call(req)
            raise AssertionError("expected an error")
        except grpc.RpcError as e:
            assert e.code() == code, e.code()
