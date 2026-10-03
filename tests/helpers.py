import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlrank.data_loader import MarketData  # noqa: E402
from mlrank.synthetic import generate  # noqa: E402


@lru_cache(maxsize=1)
def small_frames():
    return generate(n_symbols=60, start="2012-01-01", end="2016-12-31", seed=3)


def small_md() -> MarketData:
    return MarketData.from_frames(**small_frames())
