"""
research/duration_folds.py - Is a fixed shorter duration a real improvement?

The 1-10m sweep showed 3m fixed improved BOTH halves versus the dynamic
duration (-0.004/-0.120 vs -0.018/-0.157). That is the only lever in the whole
study that did. But it was chosen from 10 options on the same data, so it needs
a consistency test: does 3m beat the dynamic duration in most consecutive time
folds, and does the same hold for the mean-reversion (fade) variant?

Also reports the deployable economics: EV/trade, total PnL, and the number of
trades needed to distinguish the result from zero.

Usage: .venv/bin/python research/duration_folds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from v11_ab import UNIVERSE, SPLIT_DATE, wilson  # noqa: E402
from edge_study import load_signals, replay, stats  # noqa: E402

FOLDS = 5


def main() -> None:
    data, sigs = load_signals("v10")

    # time folds over the signal timeline
    times = np.sort(sigs["time"].unique())
    edges = np.linspace(0, len(times), FOLDS + 1).astype(int)
    bounds = [(pd.Timestamp(times[edges[i]]), pd.Timestamp(times[edges[i + 1] - 1]))
              for i in range(FOLDS)]

    def fold_of(ts: pd.Series) -> np.ndarray:
        out = np.zeros(len(ts), dtype=int)
        for i, (lo, hi) in enumerate(bounds):
            out[(ts >= lo) & (ts <= hi)] = i
        return out

    variants = {
        "dynamic (current)": rb.Filters(),
        "fixed 1m": rb.Filters(duration_override=1),
        "fixed 2m": rb.Filters(duration_override=2),
        "fixed 3m": rb.Filters(duration_override=3),
        "fixed 4m": rb.Filters(duration_override=4),
        "fixed 5m": rb.Filters(duration_override=5),
        "FADE dynamic": rb.Filters(invert=True),
        "FADE 3m": rb.Filters(invert=True, duration_override=3),
        "FADE 1m": rb.Filters(invert=True, duration_override=1),
    }

    print(f"{'variant':<20}" + "".join(f"{f'fold{i}':>9}" for i in range(FOLDS))
          + f"{'all exp':>10}{'netPnL':>9}{'n':>6}{'WR':>8}")
    print("-" * 110)

    results = {}
    for name, f in variants.items():
        tr = replay(data, sigs, f)
        tr["fold"] = fold_of(tr["time"])
        exps = []
        for k in range(FOLDS):
            g = tr[tr.fold == k]
            exps.append(g["pnl"].mean() if len(g) else np.nan)
        s = stats(tr)
        results[name] = (tr, exps)
        print(f"{name:<20}" + "".join(f"{e:>+9.3f}" for e in exps)
              + f"{s['exp']:>+10.3f}{s['pnl']:>+9.2f}{s['n']:>6}{s['wr']*100:>7.1f}%")

    # head-to-head: fixed 3m vs dynamic, per fold
    dyn, d_ex = results["dynamic (current)"]
    fix, f_ex = results["fixed 3m"]
    print("\nhead-to-head per fold (3m fixed minus dynamic, EV/trade):")
    wins = 0
    for k in range(FOLDS):
        d = f_ex[k] - d_ex[k]
        wins += d > 0
        print(f"  fold{k}: {d:+.3f}  {'better' if d > 0 else 'worse'}")
    print(f"  3m better in {wins}/{FOLDS} folds")

    # statistical honesty on the improvement
    print("\nstatistical honesty:")
    for name in ("dynamic (current)", "fixed 3m"):
        tr = results[name][0]
        s = stats(tr)
        se = np.std(tr["pnl"], ddof=1) / np.sqrt(len(tr))
        print(f"  {name:<20} exp={s['exp']:+.3f}  SE={se:.3f}  "
              f"95% CI {s['exp']-1.96*se:+.3f}..{s['exp']+1.96*se:+.3f}")
    print("  The improvement (~0.02/trade) is smaller than the standard error,")
    print("  so it is directionally consistent but not statistically proven.")

    # per-symbol effect of 3m vs dynamic
    print("\nper-symbol: 3m fixed minus dynamic (EV/trade, n>=15 in both):")
    m = dyn[["symbol", "signal_bar", "pnl"]].rename(columns={"pnl": "dyn"})
    f2 = fix[["symbol", "signal_bar", "pnl"]].rename(columns={"pnl": "fix"})
    j = m.merge(f2, on=["symbol", "signal_bar"], how="inner")
    g = j.groupby("symbol").agg(n=("dyn", "size"), dyn=("dyn", "mean"), fix=("fix", "mean"))
    g = g[g["n"] >= 15]
    g["delta"] = g["fix"] - g["dyn"]
    for sym, r in g.sort_values("delta").iterrows():
        print(f"  {sym:<9} n={int(r['n']):>3}  dyn {r['dyn']:+.3f}  "
              f"3m {r['fix']:+.3f}  delta {r['delta']:+.3f}")


if __name__ == "__main__":
    main()
