PY ?= python3
FLAG ?= --synthetic      # set FLAG= for real data

synthetic:
	$(PY) scripts/make_synthetic_data.py

membership:                   # needs ../indian_stocks_monitor data
	$(PY) scripts/build_membership.py

test:
	$(PY) tests/run_all.py

all:                          # panel -> ML walk-forward -> comparison -> SHAP
	$(PY) scripts/build_panel.py $(FLAG)
	$(PY) scripts/run_experiment.py $(FLAG)
	$(PY) scripts/run_backtest.py $(FLAG)
	$(PY) scripts/run_explain.py $(FLAG) --model lgbm_reg

live:                         # this month's C3 top 20 + gate
	$(PY) scripts/live_ranking.py

proto:                        # regenerate gRPC code from service/strategy.proto
	$(PY) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/strategy.proto

serve:                        # gRPC strategy service on port 50051
	$(PY) service/server.py

# Same pipeline on the other membership versions;
# outputs/real_<version>/
robustness:
	for m in membership membership_bridged_mid; do \
	  $(PY) scripts/build_panel.py --membership $$m && \
	  $(PY) scripts/run_experiment.py --membership $$m && \
	  $(PY) scripts/run_backtest.py --membership $$m || exit 1; \
	done

# Rebalance-day check: same pipeline trading on the 10th / 20th
days:
	for d in 10 20; do \
	  $(PY) scripts/build_panel.py --rebalance-day $$d && \
	  $(PY) scripts/run_experiment.py --rebalance-day $$d && \
	  $(PY) scripts/run_backtest.py --rebalance-day $$d || exit 1; \
	done
	$(PY) scripts/compare_days.py
