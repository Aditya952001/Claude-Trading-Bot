"""
research/analyze_session.py - Score a live demo Rise/Fall session against the
offline baseline.

Usage
    python research/analyze_session.py                 # today's RF trades
    python research/analyze_session.py --since 1792... # explicit epoch
    python research/analyze_session.py --last 20       # last N RF trades

Reports per-trade detail, aggregate expectancy with a 95% Wilson interval, a
one-sided binomial check against the payout breakeven, and the same breakdowns
the backtest uses (symbol / direction / expiry / regime / score). Read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "bot_memory.sqlite3"

BASELINE_WR = 0.480          # offline replay, 862 trades
BASELINE_EXP = -0.091        # per $1 staked


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def binom_upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


def load(since: float | None, last: int | None) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    sql = ("SELECT * FROM trades WHERE contract_type='rise_fall' "
           "AND status IN ('closed','sold') AND pnl IS NOT NULL ")
    args: list = []
    if since is not None:
        sql += "AND exit_time >= ? "
        args.append(since)
    sql += "ORDER BY exit_time DESC"
    if last:
        sql += f" LIMIT {int(last)}"
    df = pd.read_sql_query(sql, con, params=args)
    con.close()
    return df.sort_values("exit_time").reset_index(drop=True)


def stats(df: pd.DataFrame) -> dict:
    pnl = df["pnl"].values.astype(float)
    n = len(pnl)
    if n == 0:
        return dict(n=0, wr=0.0, pnl=0.0, exp=0.0, pf=0.0, dd=0.0, lo=0.0, hi=1.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], eq]))
                - np.concatenate([[0.0], eq])).max())
    wr = float((pnl > 0).mean())
    lo, hi = wilson(wr, n)
    return dict(n=n, wr=wr, pnl=float(pnl.sum()), exp=float(pnl.mean()),
                pf=float(wins.sum() / abs(losses.sum())) if len(losses) else 99.0,
                dd=dd, lo=lo, hi=hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=None,
                    help="epoch seconds; default = trades closed today (UTC)")
    ap.add_argument("--last", type=int, default=None)
    a = ap.parse_args()

    since = a.since
    scope = "today"
    if since is None:
        since = float(pd.Timestamp.now("UTC").normalize().timestamp())
    if a.last:
        scope = f"last {a.last}"

    df = load(since if not a.last else None, a.last)
    if df.empty:
        print(f"No closed Rise/Fall trades ({scope}).")
        return

    payout = (df.loc[df.pnl > 0, "pnl"] / df.loc[df.pnl > 0, "stake"]).median()
    payout = float(payout) if np.isfinite(payout) else 0.88
    breakeven = 1.0 / (1.0 + payout)

    pd.set_option("display.width", 200)
    print(f"\n=== Rise/Fall session report ({scope}) ===")
    print(f"{len(df)} closed trades | realised payout {payout:.3f} "
          f"-> breakeven {breakeven*100:.1f}% | newsest "
          f"{pd.to_datetime(df.exit_time.max(), unit='s'):%Y-%m-%d %H:%M}Z\n")

    m = stats(df)
    tail = binom_upper_tail(int((df.pnl > 0).sum()), m["n"], breakeven)
    print(f"Win rate        : {m['wr']*100:5.1f}%  (95% CI {m['lo']*100:.1f}-{m['hi']*100:.1f})"
          f"   [backtest 48.0%]")
    print(f"Net PnL         : {m['pnl']:+8.2f}   expectancy {m['exp']:+.3f}/trade"
          f"   [backtest {BASELINE_EXP:+.3f}]")
    print(f"Profit factor   : {m['pf']:5.2f}       max drawdown {m['dd']:.2f}")
    print(f"P(WR >= observed if truly at breakeven) = {tail:.3f}"
          f"   (small => hint of edge, but n={m['n']} is tiny)\n")

    cols = ["exit_time", "symbol", "direction", "stake", "pnl", "duration",
            "score", "confidence", "regime", "entry_price", "exit_price"]
    show = df.copy()
    show["exit_time"] = pd.to_datetime(show["exit_time"], unit="s").dt.strftime("%m-%d %H:%M")
    show["duration"] = ((df["exit_time"] - df["entry_time"]) / 60).round(1)
    for c in ("score", "confidence"):
        if c in show:
            show[c] = show[c].round(2)
    print(show[[c for c in cols if c in show]].to_string(index=False))
    print()

    def breakdown(col):
        if col not in df:
            return
        print(f"-- by {col}")
        for key, g in df.groupby(col):
            s = stats(g)
            print(f"   {str(key):<12} n={s['n']:>3}  WR={s['wr']*100:5.1f}%  "
                  f"exp={s['exp']:+.3f}  PnL={s['pnl']:+.2f}")
        print()

    df["dur_min"] = ((df["exit_time"] - df["entry_time"]) / 60).round(0)
    for c in ("symbol", "direction", "dur_min", "regime"):
        breakdown(c)

    # explicit score buckets (confluence strength)
    if "score" in df:
        df["score_bucket"] = df["score"].abs().round(0).astype(int)
        breakdown("score_bucket")

    out = ROOT / "research" / "session_report.json"
    out.write_text(json.dumps({
        "scope": scope,
        "payout": payout,
        "breakeven": breakeven,
        "aggregate": m,
        "binom_tail_vs_breakeven": tail,
        "trades": json.loads(df.to_json(orient="records", date_format="iso")),
    }, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
