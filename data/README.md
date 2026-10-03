# Data contract

Drop these files here as .csv (.parquet also works).
Only `mlrank/data_loader.py` reads them.

## prices (required)
| col | notes |
|---|---|
| date | trading date |
| symbol | NSE ticker |
| open, high, low, close | split/bonus ADJUSTED |
| volume | shares |

One row per (date, symbol). Include delisted
stocks: survivorship-free is the point.

## membership (required)
| col | notes |
|---|---|
| symbol | |
| start_date | joined NIFTY 500 |
| end_date | left (blank = still in) |

One row per membership spell; a stock can have
several (exit and re-entry).

## index (required)
`date, close` for NIFTY 50. Its dates define
the trading calendar.

## gold (optional)
`date, open, close` for GOLDBEES. Without it,
regime-OFF months sit in cash.

## sectors (optional)
`symbol, sector`. Enables sector-relative
momentum.
