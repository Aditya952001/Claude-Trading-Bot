"""
research/exploration_monitor.py - Sequential evidence check for the Rise/Fall
exploration sample.

Why: collecting demo trades and eyeballing the win rate is exactly how people
fool themselves. Looking at the data after every trade ("optional stopping")
inflates the false-positive rate. This applies two pre-registered rules:

1. SPRT (Wald sequential probability ratio test)
       H0: p = payout breakeven      (no edge)
       H1: p = breakeven + edge      (useful edge)
   Boundaries A = log((1-beta)/alpha), B = log(beta/(1-alpha)).
   Cross A -> declare edge, cross B -> declare no edge. Valid at any stopping
   point, so it is safe to check after every trade.

2. Futility bound (practical, not significance-based)
       Stop when the 95% Wilson UPPER bound is below breakeven + margin,
       i.e. even a lucky sample can no longer hide a useful edge.

Usage
    python research/exploration_monitor.py                # synthetic RF sample
    python research/exploration_monitor.py --since EPOCH
    python research/exploration_monitor.py --edge 0.03 --margin 0.02

Read-only.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "bot_memory.sqlite3"


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def load(since: float | None) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    sql = ("SELECT id, exit_time, symbol, direction, stake, pnl "
           "FROM trades WHERE contract_type='rise_fall' "
           "AND status IN ('closed','sold') AND pnl IS NOT NULL "
           "AND asset_class='synthetic' ")
    args: list = []
    if since is not None:
        sql += "AND exit_time >= ? "
        args.append(since)
    sql += "ORDER BY exit_time ASC"
    df = pd.read_sql_query(sql, con, params=args)
    con.close()
    return df


def sprt_path(wins: np.ndarray, p0: float, p1: float) -> np.ndarray:
    llr = np.cumsum(np.where(wins, math.log(p1 / p0), math.log((1 - p1) / (1 - p0))))
    return llr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=None,
                    help="only trades closed after this epoch")
    ap.add_argument("--edge", type=float, default=0.03,
                    help="edge size H1 must beat (default 3pp)")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="futility margin over breakeven (default 2pp)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.20)
    a = ap.parse_args()

    df = load(a.since)
    if df.empty:
        print("No closed synthetic Rise/Fall trades in scope.")
        return

    payout = float((df.loc[df.pnl > 0, "pnl"] / df.loc[df.pnl > 0, "stake"]).median())
    be = 1.0 / (1.0 + payout)
    wins = (df.pnl.values > 0)
    n, k = len(wins), int(wins.sum())
    p_hat = k / n
    lo, hi = wilson(p_hat, n)

    p0, p1 = be, be + a.edge
    A = math.log((1 - a.beta) / a.alpha)
    B = math.log(a.beta / (1 - a.alpha))
    llr = sprt_path(wins, p0, p1)
    cur = float(llr[-1]) if len(llr) else 0.0

    print(f"\n=== Exploration monitor ({n} synthetic Rise/Fall trades) ===")
    print("Caveat: the journal spans several configs (learner tweaks, guard "
          "changes), so it is observational, not a controlled experiment.")
    print(f"realised payout {payout:.3f} -> breakeven {be*100:.1f}%")
    print(f"observed {k}W/{n-k}L = {p_hat*100:.1f}%  "
          f"(95% CI {lo*100:.1f}-{hi*100:.1f})")
    print(f"distance to breakeven: {(p_hat-be)*100:+.1f}pp, "
          f"to futility margin ({100*(be+a.margin):.1f}%): {(p_hat-be-a.margin)*100:+.1f}pp")

    print("\nSPRT  H0 p={:.3f}  H1 p={:.3f}  alpha={:.2f} beta={:.2f}".format(p0, p1, a.alpha, a.beta))
    print(f"  LLR = {cur:+.3f}   boundaries: B={B:+.3f} (no edge)   A={A:+.3f} (edge)")
    if cur >= A:
        print("  => EDGE: stop and promote (then confirm on fresh data)")
    elif cur <= B:
        print("  => NO EDGE: stop exploring this hypothesis")
    else:
        print("  => CONTINUE: inconclusive")

    # how many trades until each boundary at the *current* win rate?
    def n_to(bound: float) -> int | None:
        if n == 0:
            return None
        per = p_hat * math.log(p1 / p0) + (1 - p_hat) * math.log((1 - p1) / (1 - p0))
        if per * np.sign(bound) <= 0:
            return None
        need = abs(bound - cur) / abs(per)
        return int(math.ceil(need)) if np.isfinite(need) else None

    n_edge, n_fail = n_to(A), n_to(B)
    print("  at the current win rate: "
          + (f"~{n_edge} more trades to prove an edge" if n_edge else
             "never proves an edge at this win rate")
          + ", "
          + (f"~{n_fail} more to prove there is none" if n_fail else
             "futility boundary is unreachable (win rate is above breakeven)"))

    # futility: upper Wilson bound below breakeven+margin
    print("\nFutility rule: stop when 95% upper bound < {:.1f}%".format(100 * (be + a.margin)))
    if hi < be + a.margin:
        print(f"  TRIGGERED (upper {hi*100:.1f}% < {100*(be+a.margin):.1f}%) "
              f"-> no edge of practical size")
    else:
        # smallest n at the current win rate where the upper bound would clear
        need = None
        for nn in range(max(n, 10), 20000, 10):
            if wilson(p_hat, nn)[1] < be + a.margin:
                need = nn - n
                break
        if need:
            print(f"  not triggered (upper {hi*100:.1f}%); ~{need} more trades at "
                  f"this win rate would clear it")
        else:
            print(f"  not triggered (upper {hi*100:.1f}%); the upper bound converges "
                  f"to the observed {p_hat*100:.1f}%, which is above the "
                  f"{100*(be+a.margin):.1f}% margin, so futility can only trigger "
                  f"if the win rate falls")

    # cumulative trajectory, one row per 5 trades
    print("\nTrajectory (every 5th trade):")
    print(f"  {'n':>4} {'WR':>7} {'exp':>7} {'95% CI':>16} {'LLR':>7}")
    for i in range(4, n, 5):
        w = wins[: i + 1]
        p = w.mean()
        l, h = wilson(p, len(w))
        print(f"  {len(w):>4} {p*100:>6.1f}% {df.pnl.values[:i+1].mean():>+7.3f} "
              f"  {l*100:>5.1f}-{h*100:>5.1f}% {llr[i]:>+7.2f}")

    # per-day drift: is today behaving like the prior sample?
    df["day"] = pd.to_datetime(df.exit_time, unit="s").dt.strftime("%Y-%m-%d")
    print("\nBy day:")
    for day, g in df.groupby("day"):
        w = (g.pnl.values > 0)
        l, h = wilson(w.mean(), len(w))
        print(f"  {day}  n={len(w):>3}  WR={w.mean()*100:>5.1f}%  "
              f"exp={g.pnl.mean():>+6.3f}  PnL={g.pnl.sum():>+7.2f}  "
              f"CI {l*100:.0f}-{h*100:.0f}%")


if __name__ == "__main__":
    main()
