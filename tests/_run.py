"""Minimal standalone runner so test modules work without pytest installed. Each
test module ends with `if __name__ == "__main__": run(globals())`, which invokes
every top-level `test_*` callable and reports pass/fail. Pytest, if later added,
discovers the same functions.
"""

from __future__ import annotations

import sys
import traceback


def approx(value: float, target: float, tol: float) -> bool:
    """Absolute-tolerance float comparison, so tests need no pytest."""
    return abs(value - target) <= tol


def run(namespace: dict) -> None:
    tests = sorted(
        (name, fn)
        for name, fn in namespace.items()
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception:
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    total = len(tests)
    print(f"{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
