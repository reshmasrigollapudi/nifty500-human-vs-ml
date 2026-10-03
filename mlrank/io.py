"""Shared script helpers: load data, cache the
panel, write outputs."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from .config import CFG, DATA_DIR, OUT_DIR, Config
from .data_loader import MarketData, read_csv
from .dataset import build_panel


def base_args(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-dir", default=str(DATA_DIR),
                   help="folder with prices/membership/index[/gold/sectors]")
    p.add_argument("--synthetic", action="store_true",
                   help="use data/synthetic (see make_synthetic_data.py)")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--rebuild", action="store_true", help="ignore cached panel")
    p.add_argument("--membership", default=CFG.data.membership,
                   help="membership file stem in the data dir")
    p.add_argument("--rebalance-day", type=int, default=CFG.rebalance_day,
                   help="trade on the first trading day on/after this day of month")
    return p


def config_for(args) -> Config:
    d, mem = Path(args.data_dir), args.membership
    if args.synthetic:
        d, mem = DATA_DIR / "synthetic", "membership"   # no snapshot gaps
    return dataclasses.replace(CFG, rebalance_day=args.rebalance_day,
                               data=dataclasses.replace(CFG.data, data_dir=d, membership=mem))


def out_dir(args) -> Path:
    name = "synthetic" if args.synthetic else "real"
    if not args.synthetic and args.membership != CFG.data.membership:
        name += "_" + args.membership          # robustness runs
    if args.rebalance_day != CFG.rebalance_day:
        name += f"_day{args.rebalance_day}"
    o = Path(args.out) / name
    o.mkdir(parents=True, exist_ok=True)
    return o


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """(date, symbol)-indexed frame -> CSV."""
    df.to_csv(path, date_format="%Y-%m-%d")


def load_frame(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "symbol"])


def load_all(args):
    cfg = config_for(args)
    md = MarketData.load(cfg.data)
    cache = out_dir(args) / "panel.csv"
    if cache.exists() and not args.rebuild:
        panel = load_frame(cache)
    else:
        panel = build_panel(md, cfg)
        save_frame(panel, cache)
    return cfg, md, panel
