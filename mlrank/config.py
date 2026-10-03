"""Central config. Every tunable lives here."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models_store"


@dataclass(frozen=True)
class DataConfig:
    # Stems only: loader tries .csv, then .parquet
    data_dir: Path = DATA_DIR
    prices: str = "prices"          # date,symbol,open,high,low,close,volume
    # symbol,start_date,end_date. "membership_bridged" fills the
    # snapshot gaps with the change log (scripts/build_membership.py);
    # "membership" is the raw carry-forward version.
    membership: str = "membership_bridged"
    index: str = "index"            # date,close   (NIFTY 50)
    gold: str = "gold"              # date,open,close (GOLDBEES)
    sectors: str = "sectors"        # symbol,sector  (optional)


@dataclass(frozen=True)
class FeatureConfig:
    mom_windows: tuple = (21, 63, 126, 252)   # 1m,3m,6m,12m
    skip_days: int = 21                        # for 12-1 momentum
    vol_windows: tuple = (20, 60)
    min_history_days: int = 126                # need mom_6m to enter panel
    zscore_clip: float = 3.0
    ffill_limit: int = 5                       # suspension gaps


@dataclass(frozen=True)
class SplitConfig:
    min_train_months: int = 84     # 7 years before first test
    test_months: int = 12          # retrain yearly
    embargo_months: int = 1
    label_horizon_months: int = 1
    expanding: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 1_000_000.0
    top_n: int = 20
    mode: str = "winners_run"      # or "full"
    buy_cost: float = 0.0020       # live: 0.20% buy
    sell_cost: float = 0.0030      # live: 0.30% sell
    gold_cost: float = 0.0005
    regime_sma: int = 100
    breadth_sma: int = 200             ## NEW: breadth leg SMA
    breadth_threshold: float = 0.60    ## NEW: >= 60% above SMA200
    max_below_52w_high: float = 0.20   # optional guard; not a live rule
    rf_annual: float = 0.0


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    rebalance_day: int = 1         # live: 1st trading day of the month
    seed: int = 42


CFG = Config()
