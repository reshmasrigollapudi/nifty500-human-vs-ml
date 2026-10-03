"""Build and cache the (date, symbol) panel,
then print a data-quality summary."""
import _path  # noqa: F401
from mlrank.features import FEATURES
from mlrank.io import base_args, load_all, out_dir

args = base_args(__doc__).parse_args()
args.rebuild = True
cfg, md, panel = load_all(args)
d = panel.index.get_level_values("date")
print(f"rows            {len(panel):,}")
print(f"months          {d.nunique()}  ({d.min().date()} .. {d.max().date()})")
print(f"symbols/month   {panel.groupby(level='date').size().median():.0f} median")
print(f"labeled rows    {panel['fwd_ret'].notna().sum():,}")
print("\nNaN share per feature:")
print(panel[FEATURES].isna().mean().round(3).to_string())
print(f"\ncached -> {out_dir(args) / 'panel.csv'}")
