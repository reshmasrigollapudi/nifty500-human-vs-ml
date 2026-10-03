"""README charts (light + dark PNGs) in docs/:
  equity_{mode}.png          growth of 1 rupee, test period
  rebalance_days_{mode}.png  CAGR by rebalance day
Run after run_backtest.py and compare_days.py."""
import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from mlrank.config import DATA_DIR, OUT_DIR, ROOT  # noqa: E402

# Validated palette (dataviz skill): categorical slots 1-4, ordinal blue ramp
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
                  grid="#e1e0d9", ref="#898781",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
                  days=["#86b6ef", "#2a78d6", "#104281"]),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", ref="#898781",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"],
                 days=["#184f95", "#2a78d6", "#86b6ef"]),
}
NAMES = {"c3": "C3 (live)", "lgbm_reg": "LightGBM (ML)", "mom12_vol": "mom12_vol",
         "low_vol": "low_vol"}
ORDER = ["c3", "lgbm_reg", "mom12_vol", "low_vol"]          # color follows the entity
o = OUT_DIR / "real"
docs = ROOT / "docs"
docs.mkdir(exist_ok=True)

eq = pd.DataFrame({n: pd.read_csv(o / f"equity_{n}.csv", index_col=0, parse_dates=True)["equity"]
                   for n in ORDER})
eq = eq / eq.iloc[0]
ix = pd.read_csv(DATA_DIR / "index.csv", parse_dates=["date"]).set_index("date")["close"]
ix = ix[ix.index >= eq.index[0]].reindex(eq.index).ffill()
eq["nifty"] = ix / ix.iloc[0]
cagr = pd.read_csv(o / "strategy_comparison.csv", index_col=0)["cagr"]
days = pd.read_csv(o / "rebalance_days.csv", index_col=0)


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def equity_chart(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=t["surface"])
    style(ax, t)
    ax.plot(eq.index, eq["nifty"], color=t["ref"], lw=1.5, ls="--", label="NIFTY 50")
    for n, c in zip(ORDER, t["series"]):
        ax.plot(eq.index, eq[n], color=c, lw=2, label=NAMES[n])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    # direct labels at the line ends, nudged apart in log space
    ends = sorted(((np.log10(eq[n].iloc[-1]), n) for n in ORDER + ["nifty"]), reverse=True)
    placed = []
    for y, n in ends:
        if placed and placed[-1] - y < 0.07:
            y = placed[-1] - 0.07
        placed.append(y)
        c = t["ref"] if n == "nifty" else t["series"][ORDER.index(n)]
        name = "NIFTY 50" if n == "nifty" else NAMES[n]
        val = cagr["nifty50_buy_hold" if n == "nifty" else n]
        ax.plot([eq.index[-1] + pd.Timedelta(days=60), eq.index[-1] + pd.Timedelta(days=200)],
                [10 ** y] * 2, color=c, lw=2, solid_capstyle="round")
        ax.text(eq.index[-1] + pd.Timedelta(days=260), 10 ** y, f"{name}  {val:.1%}",
                color=t["text"], fontsize=9, va="center")
    ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=1700))
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2010, eq.index[-1].year + 1, 2)]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.spines["bottom"].set_bounds(*matplotlib.dates.date2num([eq.index[0], eq.index[-1]]))  # axis ends with the data
    ax.set_title("Growth of ₹1, Nov 2009 – Sep 2026 (live rules, CAGR at right)",
                 color=t["text"], fontsize=11, loc="left", pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    fig.tight_layout()
    fig.savefig(docs / f"equity_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def days_chart(mode):
    t = THEMES[mode]
    cols = ["day 1", "day 10", "day 20"]
    d = days.loc[ORDER, cols] * 100
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, facecolor=t["surface"])
    style(ax, t)
    w, x = 0.26, np.arange(len(ORDER))
    for k, (col, c) in enumerate(zip(cols, t["days"])):
        ax.bar(x + (k - 1) * w, d[col], width=w - 0.02, color=c, label=f"rebalance on {col}",
               edgecolor=t["surface"], linewidth=1)
    ax.set_xticks(x, [NAMES[n] for n in ORDER], color=t["text2"], fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("CAGR by rebalance day (test period Nov 2009 – Sep 2026)",
                 color=t["text"], fontsize=11, loc="left", pad=12)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    ax.set_ylim(0, d.values.max() * 1.18)
    fig.tight_layout()
    fig.savefig(docs / f"rebalance_days_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


for mode in THEMES:
    equity_chart(mode)
    days_chart(mode)
print(f"charts -> {docs}")
