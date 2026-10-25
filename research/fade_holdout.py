"""
research/fade_holdout.py - Test the fade-search survivors on a true holdout.

fade_search.py inspected 822 conditions and 9 cleared breakeven on both halves
and in >=4/5 folds. But those 9 were *selected* using the validation half, so
that half is contaminated. The only honest test left is data that was never
touched by the search.

data/candles_recent/ covers 2026-10-24 15:42 .. 2026-10-25 05:01 UTC and does
not overlap data/candles/ (which ends 2026-10-24 05:25). It is a clean holdout.

This script replays each surviving rule there, with the real payout table, and
also runs two controls:
  * fade every bar (no condition)
  * the bot's own momentum premise

Usage: .venv/bin/python research/fade_holdout.py
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
from v11_ab import UNIVERSE, wilson  # noqa: E402
from fade_search import family  # noqa: E402

RECENT = ROOT / "data" / "candles_recent"
PAY = {"stpRNG": 0.95, "JD": 0.91, "R_": 0.88, "1HZ": 0.88}

# (N, condition) - the 9 survivors from fade_search.py
SURVIVORS = [
    (10, {"rsi": (80, 101)}),
    (8, {"rsi": (80, 101)}),
    (5, {"run": (8, 9), "adx": (30, 101)}),
    (10, {"run": (7, 8), "adx": (30, 101)}),
    (1, {"run": (7, 8), "pctb": (0.7, 1.01)}),
    (1, {"run": (7, 8), "adx": (30, 101)}),
    (1, {"run": (6, 7), "rsi": (60, 101)}),
    (10, {"run": (5, 6), "adx": (0, 22)}),
    (3, {"run": (7, 8), "adx": (30, 101)}),
]


def load_recent() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        p = RECENT / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df[["open", "high", "low", "close"]].astype(float)
        if len(df) < 200:
            continue
        d = rb.compute_indicators(df, rb.CFG)
        o, c = d["open"].values, d["close"].values
        n = len(o)
        sign = np.sign(c - o)
        run = np.ones(n, dtype=int)
        for i in range(1, n):
            run[i] = run[i - 1] + 1 if (sign[i] != 0 and sign[i] == sign[i - 1]) else 1
        base = pd.DataFrame({
            "symbol": sym, "fam": family(sym), "time": d.index,
            "rsi": d["rsi"].values, "pctb": d["bb_pctb"].values,
            "adx": d["adx"].values, "atr_pctile": d["atr_pctile"].values,
            "run": run, "up_bar": sign > 0,
            "entry": np.r_[o[1:], np.nan],
        })
        for N in (1, 3, 5, 8, 10):
            exit_ = np.r_[o[1 + N:], np.full(1 + N, np.nan)]
            up_move = exit_ > base["entry"].values
            b = base.copy()
            b["N"] = N
            b["fade_win"] = base["up_bar"].values ^ up_move
            frames.append(b.dropna(subset=["entry", "fade_win"]))
    x = pd.concat(frames, ignore_index=True)
    return x[(x["adx"].notna()) & (x["rsi"].notna()) & (x["pctb"].notna())]


def ev(wr: float, payout: float) -> float:
    return wr * payout - (1 - wr)


def report(name: str, s: pd.DataFrame, N: int) -> None:
    if s.empty:
        print(f"{name:<46}{'n=0':>10}")
        return
    n = len(s)
    w = int(s["fade_win"].sum())
    wr = w / n
    lo, hi = wilson(wr, n)
    # payout differs by symbol family: use the actual mix
    pay = s["fam"].map(PAY).fillna(0.88)
    pnl = np.where(s["fade_win"], pay, -1.0)
    be = 1 / (1 + float(pay.mean()))
    print(f"{name:<46}{n:>10}{wr*100:>8.2f}%{lo*100:>6.1f}-{hi*100:<6.1f}"
          f"{be*100:>7.2f}%{pnl.mean():>+10.3f}{pnl.sum():>+9.2f}"
          f"{'YES' if wr > be else 'no':>6}")


def main() -> None:
    x = load_recent()
    print(f"holdout bars: {len(x):,} rows, {x['symbol'].nunique()} symbols, "
          f"{x['time'].min():%Y-%m-%d %H:%M} .. {x['time'].max():%Y-%m-%d %H:%M} UTC")
    print(f"duration covered: {(x['time'].max()-x['time'].min()).total_seconds()/3600:.1f} h\n")

    print(f"{'rule':<46}{'n':>10}{'WR':>9}{'95% CI':>14}{'breakeven':>9}"
          f"{'EV/trade':>10}{'netPnL':>9}{'beat?':>6}")
    print("-" * 113)

    # controls
    for N in (1, 5, 10):
        report(f"CONTROL fade every bar, {N}m", x[x["N"] == N], N)
    print()

    for N, cond in SURVIVORS:
        s = x[x["N"] == N]
        for col, (lo, hi) in cond.items():
            s = s[(s[col] >= lo) & (s[col] < hi)]
        label = f"{N}m " + " & ".join(
            f"{c}[{a:g},{b:g})" for c, (a, b) in cond.items())
        report(label, s, N)

    print("\nper-family breakdown of the strongest rule (run 7-8 & ADX>=30, 10m):")
    s = x[(x["N"] == 10) & (x["run"] >= 7) & (x["run"] < 8) & (x["adx"] >= 30)]
    for fam, g in s.groupby("fam"):
        report(f"  {fam}", g, 10)

    print("\nper-symbol breakdown of that rule:")
    for sym, g in s.groupby("symbol"):
        if len(g) < 3:
            continue
        report(f"  {sym}", g, 10)

    # ---- non-overlapping replay of the strongest rule ----
    print("\nnon-overlapping replay (one trade per symbol per 10 bars) of "
          "fade run7-8 & ADX>=30, 10m:")
    rows = []
    for sym, g in x[(x["N"] == 10) & (x["run"] >= 7) & (x["run"] < 8) & (x["adx"] >= 30)].groupby("symbol"):
        g = g.sort_values("time")
        last = -10 ** 9
        idx = 0
        for _, r in g.iterrows():
            pos = x.index.get_loc(r.name)
            if pos <= last:
                continue
            last = pos + 10
            rows.append(r)
    if rows:
        s2 = pd.DataFrame(rows)
        report("  non-overlapping", s2, 10)
    else:
        print("  no trades")


if __name__ == "__main__":
    main()
