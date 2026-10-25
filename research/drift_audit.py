"""
research/drift_audit.py - Is there ANY exploitable directional bias in the
Rise/Fall universe?

NOTE: this was the first cut. It counts a FLAT expiry as "not up", which
dilutes the up-rate on step indices (10% of their 1-minute moves are flat).
research/fair_game_test.py is the corrected version: it measures direction
among non-flat moves only and prices the tie cost separately. Use that one for
the verdict; this file is kept for the raw counts.

Everything else in this repo (score gates, ADX, RSI, filters, fades) is an
attempt to predict direction. All of it can only work if the underlying price
process is not a fair martingale. This script tests that directly and with
proper multiple-testing control, so the answer is not "we found a filter".

For each symbol and each fixed expiry d (1..10 minutes) we measure the raw
probability that the price is UP at expiry, using the exact trade convention of
the bot:

    entry  = open[signal_bar + 1]
    exit   = open[signal_bar + 1 + d]
    up     = exit > entry

If the process is a fair martingale that probability is 0.50 for every symbol
and every d. The contract pays 0.88-0.95 per $1, so you need 51.3%-53.2% to
break even. We therefore test:

  H0: p_up = 0.50            (fair game -> nothing can be profitable)
  H1: p_up != 0.50           (a real directional bias exists)

with a Wilson 95% CI, a two-sided binomial test, and Bonferroni correction
across every (symbol, horizon) pair tested. A result must survive the
correction AND hold in both the train and validation halves to be believable.

Also reported:
  * bar-level up fraction (close > open) - detects intrabar drift/bias,
  * per-symbol tick/step structure for the step indices,
  * the breakeven win rate implied by the real payout for each symbol.

Usage: .venv/bin/python research/drift_audit.py [--dir data/candles_7d]
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

import rf_backtest as rb  # noqa: E402
from v11_ab import UNIVERSE  # noqa: E402

HORIZONS = list(range(1, 11))
TRAIN_FRAC = 0.60


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def binom_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p-value (no scipy dependency)."""
    if n == 0:
        return 1.0
    # sum of probabilities of outcomes no more likely than the observed one
    def logpmf(i: int) -> float:
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * math.log(p0) + (n - i) * math.log1p(-p0))
    obs = logpmf(k)
    tot = 0.0
    for i in range(n + 1):
        lp = logpmf(i)
        if lp <= obs + 1e-9:
            tot += math.exp(lp)
    return min(1.0, tot)


def up_series(df: pd.DataFrame, d: int) -> np.ndarray:
    """Boolean 'up at expiry' for every possible signal bar."""
    o = df["open"].values
    n = len(o)
    # signal bar s -> entry s+1, exit s+1+d
    last = n - 1 - d
    if last <= 0:
        return np.array([], dtype=bool)
    entry = o[1:last + 1]
    exit_ = o[1 + d:last + 1 + d]
    return exit_ > entry


def breakeven(pay: float) -> float:
    return 1.0 / (1.0 + pay)


def load_dir(path: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for sym in UNIVERSE:
        p = path / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        data[sym] = df[["open", "high", "low", "close"]].astype(float)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/candles_7d")
    a = ap.parse_args()
    d = Path(a.dir)
    path = d if d.is_absolute() else ROOT / d

    data = load_dir(path)
    if not data:
        raise SystemExit(f"no candles in {path}")
    span = (min(x.index[0] for x in data.values()), max(x.index[-1] for x in data.values()))
    print(f"candles: {len(data)} symbols, {span[0]:%Y-%m-%d %H:%M} -> {span[1]:%Y-%m-%d %H:%M} UTC")
    print(f"horizons: {HORIZONS} minutes, train={TRAIN_FRAC:.0%} / validation={1-TRAIN_FRAC:.0%}\n")

    pay = rb.payouts()
    tests = len(data) * len(HORIZONS)
    alpha = 0.05
    bonf = alpha / tests
    print(f"multiple-testing: {tests} (symbol x horizon) tests, "
          f"Bonferroni threshold p < {bonf:.2e}\n")

    rows = []
    for sym, df in data.items():
        n = len(df)
        split = int(n * TRAIN_FRAC)
        be = breakeven(pay.get(sym, 0.88))
        # bar-level bias: close > open
        bar_up = float((df["close"].values > df["open"].values).mean())
        bar_n = n
        blo, bhi = wilson(bar_up, bar_n)
        for h in HORIZONS:
            u = up_series(df, h)
            if len(u) == 0:
                continue
            k, m = int(u.sum()), len(u)
            p = k / m
            lo, hi = wilson(p, m)
            pv = binom_two_sided(k, m)
            # train / validation halves
            ktr, ntr = int(u[:split - h].sum()), max(1, split - h)
            kva, nva = int(u[split - h:].sum()), max(1, len(u) - (split - h))
            rows.append(dict(symbol=sym, h=h, n=m, k=k, p=p, lo=lo, hi=hi, pval=pv,
                             be=be, bar_up=bar_up, bar_n=bar_n, bar_lo=blo, bar_hi=bhi,
                             p_tr=ktr / ntr, n_tr=ntr, p_va=kva / nva, n_va=nva))
    r = pd.DataFrame(rows)

    # ---- headline: does ANY symbol/horizon beat its own breakeven? -------------
    print("=" * 100)
    print("A. BEST RAW BIAS PER SYMBOL (horizon with highest p_up), vs breakeven")
    print("=" * 100)
    print(f"{'symbol':<10}{'best h':>7}{'n':>7}{'p_up':>8}{'95% CI':>15}"
          f"{'breakeven':>11}{'edge':>8}{'p-value':>11}{'survives':>10}")
    print("-" * 100)
    survivors = []
    for sym, g in r.groupby("symbol"):
        b = g.loc[g["p"].idxmax()]
        edge = b["p"] - b["be"]
        surv = (b["pval"] < bonf) and (edge > 0)
        if surv:
            survivors.append(b)
        print(f"{sym:<10}{int(b['h']):>7}{int(b['n']):>7}{b['p']*100:>7.2f}%"
              f"{b['lo']*100:>7.1f}-{b['hi']*100:<7.1f}{b['be']*100:>10.2f}%"
              f"{edge*100:>+7.2f}{b['pval']:>11.3g}{'YES' if surv else 'no':>10}")
    print(f"\nsymbols whose best horizon beats breakeven after correction: "
          f"{len(survivors)}/{r['symbol'].nunique()}")

    # ---- B. pooled: the universe as a whole --------------------------------
    print("\n" + "=" * 100)
    print("B. POOLED UP-PROBABILITY BY HORIZON (all 23 symbols, every bar)")
    print("=" * 100)
    print(f"{'h (min)':>8}{'n':>10}{'p_up':>9}{'95% CI':>16}{'vs 50%':>9}{'p-value':>12}")
    print("-" * 60)
    for h, g in r.groupby("h"):
        k, m = int(g["k"].sum()), int(g["n"].sum())
        p = k / m
        lo, hi = wilson(p, m)
        pv = binom_two_sided(k, m)
        print(f"{int(h):>8}{m:>10}{p*100:>8.2f}%{lo*100:>8.2f}-{hi*100:<7.2f}"
              f"{(p-0.5)*100:>+8.2f}{pv:>12.3g}")

    # ---- C. bar-level bias ------------------------------------------------
    print("\n" + "=" * 100)
    print("C. BAR-LEVEL BIAS (close > open) - is the generator itself tilted?")
    print("=" * 100)
    print(f"{'symbol':<10}{'bars':>9}{'up bars':>10}{'95% CI':>16}{'p-value':>12}")
    print("-" * 60)
    for sym, g in r.groupby("symbol"):
        b = g.iloc[0]
        pv = binom_two_sided(int(round(b["bar_up"] * b["bar_n"])), int(b["bar_n"]))
        print(f"{sym:<10}{int(b['bar_n']):>9}{b['bar_up']*100:>9.2f}%"
              f"{b['bar_lo']*100:>8.2f}-{b['bar_hi']*100:<7.2f}{pv:>12.3g}")

    # ---- D. stability: does any signal hold in both halves? ---------------
    print("\n" + "=" * 100)
    print("D. TRAIN vs VALIDATION (best-bias horizons only; a real edge must hold in both)")
    print("=" * 100)
    print(f"{'symbol':<10}{'h':>4}{'train p':>10}{'train n':>9}{'valid p':>10}"
          f"{'valid n':>9}{'delta':>9}{'both>BE':>9}")
    print("-" * 75)
    for sym, g in r.groupby("symbol"):
        b = g.loc[g["p"].idxmax()]
        both = (b["p_tr"] > b["be"]) and (b["p_va"] > b["be"])
        print(f"{sym:<10}{int(b['h']):>4}{b['p_tr']*100:>9.2f}%{int(b['n_tr']):>9}"
              f"{b['p_va']*100:>9.2f}%{int(b['n_va']):>9}"
              f"{(b['p_va']-b['p_tr'])*100:>+8.2f}{'YES' if both else 'no':>9}")

    # ---- E. conclusion ---------------------------------------------------
    print("\n" + "=" * 100)
    print("E. WHAT THIS MEANS FOR THE STRATEGY")
    print("=" * 100)
    pooled = {h: (g["k"].sum() / g["n"].sum(), g["n"].sum()) for h, g in r.groupby("h")}
    worst_be = min(breakeven(v) for v in pay.values())
    print(f"best pooled up-probability across all horizons: "
          f"{max(p for p, _ in pooled.values())*100:.2f}%")
    print(f"lowest breakeven in the universe: {worst_be*100:.2f}% "
          f"(highest payout {max(pay.values()):.2f})")
    if not survivors:
        print("\nNo symbol/horizon has a directional bias that survives multiple-testing")
        print("correction. The prices are consistent with a fair martingale, so the")
        print("only thing a filter can change is HOW OFTEN you trade - never the sign")
        print("of the expected value. Every extra entry at a 0.88-0.95 payout is a")
        print("2.5%-6% expected loss on stake.")
    else:
        print(f"\n{len(survivors)} candidate(s) survived; see section D for stability")
        print("before believing any of them.")


if __name__ == "__main__":
    main()
