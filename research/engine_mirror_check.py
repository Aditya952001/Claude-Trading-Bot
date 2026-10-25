"""
research/engine_mirror_check.py - Prove strategy_engine.evaluate() and the
vectorised mirror in rf_backtest.build_signals() agree bar-by-bar.

The backtest is only trustworthy if its mirror matches the code that actually
trades. This samples real bars across symbols, runs the REAL engine on the
trailing window ending at each bar, and compares the direction with the
vectorised signal for that same bar.

Usage: .venv/bin/python research/engine_mirror_check.py [--samples 300]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from config import StrategyConfig  # noqa: E402
from strategy_engine import StrategyEngine, StrategyParams  # noqa: E402

UNIVERSE = ["R_10", "R_25", "R_50", "R_75", "R_100",
            "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
            "JD10", "JD25", "JD50", "JD75", "JD100",
            "stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--variant", default="v10", choices=["v10", "v11"])
    args = ap.parse_args()

    cfg = StrategyConfig()
    eng = StrategyEngine(cfg)
    params = StrategyParams()
    rng = random.Random(7)

    agree = disagree = skipped = 0
    mismatches = []
    per_symbol = max(1, args.samples // len(UNIVERSE))

    for sym in UNIVERSE:
        p = rb.CANDLE_DIR / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = rb.load(sym)
        sig = rb.build_signals(sym, df, variant=args.variant)
        sig_dir = {int(r.bar): r.direction for r in sig.itertuples()}
        n = len(df)
        bars = list(range(210, n - 1))
        rng.shuffle(bars)
        for i in bars[:per_symbol]:
            window = df.iloc[:i + 1]
            out = eng.evaluate(sym, "synthetic", window, params)
            live = out.direction if out else None
            mirror = sig_dir.get(i)
            if live is None and mirror is None:
                agree += 1
            elif live == mirror:
                agree += 1
            else:
                disagree += 1
                if len(mismatches) < 8:
                    mismatches.append((sym, str(df.index[i]), live, mirror))

    total = agree + disagree
    print(f"engine vs mirror ({args.variant}): {agree}/{total} agree "
          f"({agree / total * 100:.1f}%)")
    for sym, ts, live, mirror in mismatches:
        print(f"  mismatch {sym} {ts}: engine={live} mirror={mirror}")
    if disagree:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
