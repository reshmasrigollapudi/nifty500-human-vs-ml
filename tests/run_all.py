"""Run all tests without pytest:
python tests/run_all.py"""
import importlib
import sys
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))
failed = 0
for f in sorted(here.glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"PASS {f.stem}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {f.stem}.{name}")
                traceback.print_exc()
print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
