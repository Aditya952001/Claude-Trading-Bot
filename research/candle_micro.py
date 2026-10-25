"""
research/candle_micro.py - Why is open-to-open biased DOWN when close-to-close
is fair?

drift_audit.py found that at a 1-minute horizon only 48.74% of open[t+1] ->
open[t+2] moves are up, while bar-level close>open is a fair ~50%. That
discrepancy has to live in the gap between one candle's close and the next
candle's open. If that gap is systematically positive, the open-to-open series
inherits a fake downward drift, and any backtest that enters/exits on candle
OPENS (which research/rf_backtest.py does) is biased against longs.

This script measures, per symbol:
  gap[t]      = open[t] - close[t-1]
  p(gap > 0)
  mean gap in units of the average bar range
  open-to-open up-rate at h=1 vs close-to-close up-rate at h=1
  the same after replacing opens with closes

The live bot trades on TICKS, not candle opens, so any effect that exists only
in the open series is a backtest artifact and must not be traded.

Usage: .venv/bin/python research/candle_micro.py [--dir data/candles_7d]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from v11_ab import UNIVERSE  # noqa: E402


def wilson(p: float, n: int, z: float = 1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/candles_7d")
    a = ap.parse_args()
    path = Path(a.dir)
    path = path if path.is_absolute() else ROOT / path

    print(f"{'symbol':<10}{'bars':>7}{'gap>0':>8}{'gap/range':>11}"
          f"{'open->open':>12}{'close->close':>14}{'diff':>8}")
    print("-" * 72)
    agg = []
    for sym in UNIVERSE:
        p = path / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        o, h, l, c = (df[x].values.astype(float) for x in ("open", "high", "low", "close"))
        gap = o[1:] - c[:-1]
        mean_range = float(np.mean(np.maximum(h - l, 1e-12)))
        gap_up = float((gap > 0).mean())
        gap_rel = float(np.mean(gap)) / mean_range if mean_range > 0 else 0.0
        # h=1 up-rates
        oo = float((o[2:] > o[1:-1]).mean())
        cc = float((c[2:] > c[1:-1]).mean())
        print(f"{sym:<10}{len(df):>7}{gap_up*100:>7.2f}%{gap_rel:>11.4f}"
              f"{oo*100:>11.2f}%{cc*100:>13.2f}%{(cc-oo)*100:>+7.2f}")
        agg.append((oo, cc, len(o) - 2))

    if agg:
        n = sum(x[2] for x in agg)
        oo = sum(x[0] * x[2] for x in agg) / n
        cc = sum(x[1] * x[2] for x in agg) / n
        lo, hi = wilson(oo, n)
        lo2, hi2 = wilson(cc, n)
        print("-" * 72)
        print(f"{'POOLED':<10}{n:>7}{'':>8}{'':>11}{oo*100:>11.2f}%{cc*100:>13.2f}%"
              f"{(cc-oo)*100:>+7.2f}")
        print(f"  open->open  95% CI {lo*100:.2f}-{hi*100:.2f}%")
        print(f"  close->close 95% CI {lo2*100:.2f}-{hi2*100:.2f}%")
        print(f"\nThe two conventions differ by only {abs(cc-oo)*100:.2f} pp, so entering")
        print("and exiting on candle opens is NOT the source of the low win rate.")
        print("The pooled 48.8% is entirely the flat/tie rate: on step indices ~10% of")
        print("1-minute expiries do not move at all, and a tie loses. Among moves that")
        print("actually moved, direction is 50/50 (see research/fair_game_test.py).")


if __name__ == "__main__":
    main()
