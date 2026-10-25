"""
research/fair_game_test.py - The definitive test: is direction predictable at all?

drift_audit.py counted "up at expiry" and found 48.7% pooled. That number is
misleading: on step indices 10.3% of 1-minute moves are exactly FLAT (the price
sits on a 0.1 grid and does not move), and a flat expiry is a loss for both
directions. Flat moves dilute the up-rate without saying anything about skill.

This script therefore separates the two questions:

  1. DIRECTION: among moves that actually moved, is the up-fraction 50%?
     This is the only thing a signal can predict. If it is 50% for every
     symbol and expiry, no entry rule can ever have positive expectancy.

  2. TIES: how often is the expiry exactly flat? A tie is a structural cost
     that no signal can avoid, so it makes the required win rate worse.

For each symbol and expiry d in 1..10 minutes:

    n_nonflat, p_up = up / (up + down), Wilson CI, exact binomial p-value
    tie_rate
    structural EV of a *perfect coin-flip* direction call at the real payout
    EV of a perfect oracle (upper bound: what 100% accuracy would pay)

Bonferroni correction across every (symbol, expiry) pair. A directional bias
must survive the correction AND hold in both train and validation halves.

Usage: .venv/bin/python research/fair_game_test.py [--dir data/candles_7d]
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
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def binom_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    if n == 0:
        return 1.0

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


def move_table(df: pd.DataFrame, d: int) -> tuple[np.ndarray, np.ndarray]:
    """entry=open[s+1], exit=open[s+1+d] -> (delta, is_up) for every bar."""
    o = df["open"].values
    n = len(o)
    last = n - 1 - d
    if last <= 0:
        return np.array([]), np.array([], dtype=bool)
    entry = o[1:last + 1]
    exit_ = o[1 + d:last + 1 + d]
    delta = exit_ - entry
    return delta, delta > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/candles_7d")
    a = ap.parse_args()
    path = Path(a.dir)
    path = path if path.is_absolute() else ROOT / path

    pay = rb.payouts()
    data = {}
    for sym in UNIVERSE:
        p = path / f"{sym}_60.csv"
        if p.exists():
            data[sym] = pd.read_csv(p, parse_dates=["time"], index_col="time")[
                ["open", "high", "low", "close"]].astype(float)
    if not data:
        raise SystemExit(f"no candles in {path}")

    tests = len(data) * len(HORIZONS)
    bonf = 0.05 / tests
    print(f"candles: {len(data)} symbols | horizons {HORIZONS}m | "
          f"{tests} tests | Bonferroni p < {bonf:.2e}\n")

    rows = []
    for sym, df in data.items():
        n = len(df)
        split = int(n * TRAIN_FRAC)
        for d in HORIZONS:
            delta, up = move_table(df, d)
            if len(delta) == 0:
                continue
            nz = delta != 0
            m = int(nz.sum())
            k = int((delta[nz] > 0).sum())
            p = k / m if m else 0.5
            lo, hi = wilson(p, m)
            # train / validation on non-flat moves
            tr = nz[:max(0, split - d)]
            va = nz[max(0, split - d):]
            dtr = delta[:max(0, split - d)]
            dva = delta[max(0, split - d):]
            mtr = int(tr.sum()); ktr = int((dtr[tr] > 0).sum()) if mtr else 0
            mva = int(va.sum()); kva = int((dva[va] > 0).sum()) if mva else 0
            rows.append(dict(
                symbol=sym, h=d, n_all=len(delta), n_nonflat=m, up=k, p_up=p,
                lo=lo, hi=hi, pval=binom_two_sided(k, m),
                tie=1.0 - m / len(delta),
                pay=pay.get(sym, 0.88),
                p_tr=(ktr / mtr if mtr else 0.5), n_tr=mtr,
                p_va=(kva / mva if mva else 0.5), n_va=mva,
            ))
    r = pd.DataFrame(rows)
    r["be"] = 1.0 / (1.0 + r["pay"])

    # ------------------------------------------------------------------ 1
    print("=" * 104)
    print("1. DIRECTION AMONG NON-FLAT MOVES - pooled by expiry")
    print("=" * 104)
    print(f"{'expiry':>7}{'moves':>10}{'up':>9}{'p_up':>8}{'95% CI':>16}"
          f"{'tie rate':>10}{'p-value':>12}{'coin-flip EV':>14}")
    print("-" * 104)
    for d, g in r.groupby("h"):
        m = int(g["n_nonflat"].sum()); k = int(g["up"].sum())
        p = k / m
        lo, hi = wilson(p, m)
        tie = 1 - m / int(g["n_all"].sum())
        # EV of a coin-flip call, tie = loss, averaged over the universe payouts
        ev = float(np.mean(0.5 * (1 - g["tie"]) * g["pay"]
                           - (1 - 0.5 * (1 - g["tie"]))))
        print(f"{int(d):>6}m{m:>10}{k:>9}{p*100:>7.2f}%"
              f"{lo*100:>7.2f}-{hi*100:<7.2f}{tie*100:>9.2f}%"
              f"{binom_two_sided(k, m):>12.3g}{ev:>+13.3f}")

    # ------------------------------------------------------------------ 2
    print("\n" + "=" * 104)
    print("2. DOES ANY SYMBOL/EXPIRY SHOW A REAL DIRECTIONAL BIAS?")
    print("=" * 104)
    print(f"{'symbol':<10}{'best d':>7}{'moves':>9}{'p_up':>8}{'95% CI':>16}"
          f"{'edge vs 50':>12}{'p-value':>11}{'survives':>10}")
    print("-" * 104)
    survivors = []
    for sym, g in r.groupby("symbol"):
        b = g.loc[(g["p_up"] - 0.5).abs().idxmax()]
        edge = (b["p_up"] - 0.5) * 100
        surv = b["pval"] < bonf
        if surv:
            survivors.append((sym, int(b["h"]), b["p_up"], b["pval"]))
        print(f"{sym:<10}{int(b['h']):>7}{int(b['n_nonflat']):>9}"
              f"{b['p_up']*100:>7.2f}%{b['lo']*100:>7.2f}-{b['hi']*100:<7.2f}"
              f"{edge:>+11.2f}{b['pval']:>11.3g}{'YES' if surv else 'no':>10}")
    print(f"\nresults surviving Bonferroni: {len(survivors)}/{r['symbol'].nunique()} symbols")

    # ------------------------------------------------------------------ 3
    print("\n" + "=" * 104)
    print("3. STRUCTURAL COST OF TIES (a tie is a loss for both directions)")
    print("=" * 104)
    print(f"{'symbol':<10}{'tie 1m':>9}{'tie 3m':>9}{'tie 6m':>9}{'tie 10m':>10}"
          f"{'payout':>8}{'breakeven':>11}{'best coin-flip WR':>19}")
    print("-" * 104)
    for sym, g in r.groupby("symbol"):
        def t(h):
            x = g[g["h"] == h]
            return float(x["tie"].iloc[0]) if len(x) else float("nan")
        pay_s = float(g["pay"].iloc[0]); be = 1 / (1 + pay_s)
        best = max(0.5 * (1 - t(h)) for h in (1, 3, 6, 10))
        print(f"{sym:<10}{t(1)*100:>8.2f}%{t(3)*100:>8.2f}%{t(6)*100:>8.2f}%"
              f"{t(10)*100:>9.2f}%{pay_s:>8.2f}{be*100:>10.2f}%{best*100:>18.2f}%")

    # ------------------------------------------------------------------ 3b
    print("\n" + "=" * 104)
    print("3b. LOWEST-STRUCTURAL-LOSS CONFIGURATIONS (coin-flip call, ties lose)")
    print("    This is the best a strategy with no edge can do: minimise the house")
    print("    edge + tie cost. Every line is still negative.")
    print("=" * 104)
    print(f"{'rank':>5}{'symbol':<10}{'expiry':>8}{'tie':>8}{'payout':>8}"
          f"{'EV per $1':>11}{'WR needed':>11}{'max WR (no skill)':>19}")
    print("-" * 104)
    ev_rows = []
    for sym, g in r.groupby("symbol"):
        for _, x in g.iterrows():
            ev = 0.5 * (1 - x["tie"]) * x["pay"] - (1 - 0.5 * (1 - x["tie"]))
            ev_rows.append((ev, sym, int(x["h"]), x["tie"], x["pay"], x["be"]))
    ev_rows.sort(reverse=True)
    for i, (ev, sym, h, tie, pay_s, be) in enumerate(ev_rows[:8], 1):
        print(f"{i:>5}{sym:<10}{h:>7}m{tie*100:>7.2f}%{pay_s:>8.2f}"
              f"{ev*100:>+10.2f}%{be*100:>10.2f}%{0.5*(1-tie)*100:>18.2f}%")
    worst = ev_rows[-1]
    print(f"\nworst configuration: {worst[1]} {worst[2]}m at {worst[0]*100:+.2f}%/trade")
    print(f"best  configuration: {ev_rows[0][1]} {ev_rows[0][2]}m at "
          f"{ev_rows[0][0]*100:+.2f}%/trade")
    print(f"spread between best and worst: "
          f"{(ev_rows[0][0]-worst[0])*100:.2f} pp per trade")

    # ------------------------------------------------------------------ 4
    print("\n" + "=" * 104)
    print("4. STABILITY - best-bias expiry, train vs validation")
    print("=" * 104)
    print(f"{'symbol':<10}{'d':>4}{'train p_up':>12}{'train n':>9}"
          f"{'valid p_up':>12}{'valid n':>9}{'delta':>9}{'same side':>11}")
    print("-" * 80)
    flips = 0
    for sym, g in r.groupby("symbol"):
        b = g.loc[(g["p_up"] - 0.5).abs().idxmax()]
        same = (b["p_tr"] - 0.5) * (b["p_va"] - 0.5) > 0
        flips += 0 if same else 1
        print(f"{sym:<10}{int(b['h']):>4}{b['p_tr']*100:>11.2f}%{int(b['n_tr']):>9}"
              f"{b['p_va']*100:>11.2f}%{int(b['n_va']):>9}"
              f"{(b['p_va']-b['p_tr'])*100:>+8.2f}{'yes' if same else 'FLIP':>11}")
    print(f"\nbest-bias expiry keeps its sign in both halves for "
          f"{r['symbol'].nunique()-flips}/{r['symbol'].nunique()} symbols "
          f"(a pure-noise search would flip about half)")

    # ------------------------------------------------------------------ 5
    print("\n" + "=" * 104)
    print("5. VERDICT - does any measured bias clear the payout?")
    print("=" * 104)
    m = int(r["n_nonflat"].sum()); k = int(r["up"].sum())
    p = k / m
    lo, hi = wilson(p, m)
    print(f"All symbols, all expiries pooled: {p*100:.3f}% up among {m:,} non-flat moves")
    print(f"  95% CI {lo*100:.3f}-{hi*100:.3f}%  "
          f"({'50% inside' if lo <= 0.5 <= hi else '50% OUTSIDE'})")

    r["edge"] = (r["p_up"] - 0.5).abs()
    # Actual win probability of betting the measured bias, including ties (a tie
    # loses). p_up is measured among non-flat moves, so multiply by (1 - tie).
    best_side = np.where(r["p_up"] >= 0.5, r["p_up"], 1 - r["p_up"])
    r["win_prob"] = best_side * (1 - r["tie"])
    r["edge_signed"] = r["win_prob"] - r["be"]
    clears = r[r["edge_signed"] > 0]
    print(f"\nDirectional bias surviving Bonferroni: {len(survivors)}/23")
    print(f"Symbol/expiry pairs whose bias ALSO clears their own breakeven "
          f"(after tie losses): {len(clears)}/{len(r)}")
    best = r.loc[r["edge"].idxmax()]
    print(f"\nLargest measurable bias anywhere: {best['symbol']} at {int(best['h'])}m, "
          f"{(best['p_up']-0.5)*100:+.2f} pp (95% CI "
          f"{(best['lo']-0.5)*100:+.2f} to {(best['hi']-0.5)*100:+.2f})")
    print(f"Betting that bias gives a {best['win_prob']*100:.2f}% win rate against a "
          f"{best['be']*100:.2f}% breakeven -> {(best['edge_signed'])*100:+.2f} pp/trade.")
    print(f"\nRequired edge to break even: {(min(r['be'])-0.5)*100:.2f} to "
          f"{(max(r['be'])-0.5)*100:.2f} pp above a coin flip.")
    print(f"Measured edge, best case:   {(best['p_up']-0.5)*100:+.2f} pp.")
    print(f"\nBest coin-flip win rate available (zero-skill, best config): "
          f"{max(0.5*(1-x['tie']) for _, x in r.iterrows())*100:.2f}% "
          f"vs a breakeven of {min(r['be'])*100:.2f}%.")
    if len(clears) == 0:
        print("\nCONCLUSION: the price process is a fair martingale. No symbol, no")
        print("expiry and no measured bias clears the payout once tie losses are")
        print("counted. Every entry rule can only change WHEN you trade, never the")
        print("sign of expectancy. There is nothing left to filter: the instrument")
        print("is priced so that no directional strategy can be profitable.")
    else:
        print(f"\n{len(clears)} pair(s) clear breakeven on this sample. They are")
        print("selected maxima; they must be confirmed on data not used here")
        print("before any capital is risked.")


if __name__ == "__main__":
    main()
