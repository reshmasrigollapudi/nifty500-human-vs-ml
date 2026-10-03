# ML Stock Ranking: NIFTY 500

Learning-to-rank on point-in-time NIFTY 500
data, benchmarked against a production
momentum strategy under identical live rules.

**Question:** can ML rank next-month returns
better than 12-month momentum, after costs,
out of sample?

## The ML problem
- **Rows:** one per (month-end T, stock) for
  index members at T-1.
- **Features (18):** 14 cross-sectional
  (momentum 1/3/6/12/12-1m, volatility,
  downside vol, drawdown, distance to 52w
  high / SMA50 / SMA200, volume trend,
  turnover), z-scored within each month;
  plus 4 market-regime features.
  Sector-relative momentum is built only
  when `data/sectors.csv` exists and is
  not yet a model input.
- **Label:** return from the T open to the
  next rebalance open, as a within-month
  rank (regression) or quintile (LambdaRank).
  Delisted stocks exit at their last close.
- **Data hygiene:** non-positive prices
  (old bhavcopy rows with open = 0) are
  set to NaN on load
  (`tests/test_data.py`).
- **Membership gaps:** historical NIFTY 500
  lists are snapshots with gaps of up to 4
  years (2006-2016). `build_membership.py`
  fills them with the logged index changes
  and drops stocks after their last trade:
  96-97% match with the list at each gap's
  end vs 78-82% for carry-forward. Default
  is `membership_bridged`; `make robustness`
  reruns on carry-forward and midpoint
  versions.
- **No lookahead:** features use the T-1
  close only. `tests/test_leakage.py`
  perturbs all prices from T onward and
  asserts features for T don't change.

## Validation
- Purged walk-forward: 7-year minimum train,
  yearly retrain, 1-month embargo
  (`mlrank/splits.py`).
- Ranking metrics: rank IC, ICIR, t-stat,
  hit rate, quintile spread, monotonicity.
- Portfolio test: OOS scores go through the
  live rules (top 20, winners_run, 52w-high
  guard, costs). Equity gate:
  NIFTY 50 > SMA100 OR >= 60% of NIFTY 500
  members above their SMA200; else gold.
- Significance: 10,000-run stationary block
  bootstrap of CAGR difference vs baseline.
- Noise floor: edges under ~2.9pp CAGR
  (rebalance-day variance) don't count.

## Models
| name | what | from scratch |
|---|---|---|
| baseline_mom12 | live strategy score | - |
| ridge_scratch | ridge, closed form + GD | yes |
| tree_scratch | histogram regression tree | yes |
| gbm_scratch | gradient boosting on hist trees | yes |
| lgbm_reg | LightGBM pointwise | - |
| lgbm_rank | LightGBM LambdaRank | - |
| cnn1d | 1D-CNN on raw returns | - |

Regime model: logistic regression from
scratch predicting P(next-month NIFTY < 0),
compared with the SMA100 gate.

Scratch models are tested against sklearn
(`tests/test_models.py`).

## Run
```bash
pip install -r requirements.txt
make synthetic && make test   # sanity check
make all                      # synthetic run
make all FLAG=                # real data in data/
make live FLAG=               # monthly job
```

Outputs land in `outputs/{real|synthetic}/`.

## Layout
```
mlrank/
  config.py        every parameter
  data_loader.py   files -> MarketData
  features.py      PIT features
  dataset.py       labels + panel
  splits.py        purged walk-forward
  metrics.py       IC, quantile spread
  models/          baseline, scratch, LightGBM
  experiment.py    OOS prediction loop
  backtest.py      live-rule portfolio sim
  bootstrap.py     block bootstrap A vs B
  regime.py        logistic regime model
  explain.py       permutation + SHAP
  drift.py         PSI / KS monitoring
  deep/cnn1d.py    sequence model (torch)
scripts/           one entry point per step
tests/             leakage, splits, models,
                   metrics, backtest
```

## Production loop
`train_live.py` trains on all labeled months,
saves a versioned artifact + JSON metadata,
scores today, applies the live rules, and logs
predictions. `monitor.py` computes realized
live IC, checks feature drift, and flags a
fallback to the baseline if trailing 3-month
IC turns negative.

## Results
Fill in from `outputs/real/` after running
on real data. Report the table even if the
baseline wins.

Synthetic data is only for pipeline testing.
Never quote numbers from it.
