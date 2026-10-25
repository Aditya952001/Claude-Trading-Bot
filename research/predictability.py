"""
research/predictability.py - Is there ANY exploitable structure in the 1m candles?

The entry-rule study found no bucket that beats breakeven on both halves. That
could mean (a) the rules are bad, or (b) the instrument is a martingale and no
rule can work. This script tests (b) directly, independently of the strategy:

  1. autocorrelation of 1m log returns, lags 1..10, per symbol
  2. hit rate of "next N minutes continue the last bar" (momentum)
  3. hit rate of "next N minutes reverse the last bar" (mean reversion)
  4. conditional hit rate after k consecutive same-direction bars
  5. conditional hit rate at RSI / %B extremes
  6. the same, split into 5 consecutive time folds, to see consistency

For every test the null is 50% (a driftless martingale). A real edge must be
stable across folds and economically larger than the payout breakeven
(51.3% at 0.95, 52.4% at 0.91, 53.2% at 0.88).

Usage: .venv/bin/python research/predictability.py
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

HORIZONS = [1, 2, 3, 4, 5, 6, 8, 10]


def folds(index: pd.DatetimeIndex, k: int = 5):
    """Split a time index into k contiguous folds."""
    edges = np.linspace(0, len(index), k + 1).astype(int)
    return [(index[edges[i]], index[edges[i + 1] - 1]) for i in range(k)]


def fold_of(ts: pd.Series, edges) -> np.ndarray:
    out = np.zeros(len(ts), dtype=int)
    for i, (lo, hi) in enumerate(edges):
        out[(ts >= lo) & (ts <= hi)] = i
    return out


def section(t: str) -> None:
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def hit(wins: np.ndarray, n: int) -> str:
    if n == 0:
        return "n=0"
    p = wins / n
    lo, hi = wilson(p, n)
    return f"n={n:>6}  hit={p*100:>5.2f}%  CI {lo*100:>5.2f}-{hi*100:<5.2f}"


def main() -> None:
    data = {s: rb.load(s) for s in UNIVERSE if (rb.CANDLE_DIR / f"{s}_60.csv").exists()}
    print(f"symbols: {len(data)}")

    # ------------------------------------------------------------------
    section("1. AUTOCORRELATION OF 1m LOG RETURNS (lag 1..10)")
    print("A driftless martingale has ~0 autocorrelation. Values here are tiny,")
    print("and the 95% band for n~20000 is about +/-0.014.")
    print(f"\n{'symbol':<9}" + "".join(f"{f'lag{l}':>8}" for l in range(1, 11)))
    print("-" * 89)
    all_ac = []
    for sym, df in data.items():
        r = np.log(df["close"]).diff().dropna().values
        r = r[np.isfinite(r)]
        acs = []
        for lag in range(1, 11):
            if len(r) > lag + 10:
                acs.append(np.corrcoef(r[:-lag], r[lag:])[0, 1])
            else:
                acs.append(np.nan)
        all_ac.append(acs)
        print(f"{sym:<9}" + "".join(f"{a:>8.4f}" for a in acs))
    med = np.nanmedian(np.array(all_ac), axis=0)
    print(f"{'MEDIAN':<9}" + "".join(f"{a:>8.4f}" for a in med))

    # ------------------------------------------------------------------
    section("2. MOMENTUM vs MEAN-REVERSION (all symbols pooled)")
    print("Entry at next bar open, exit at open + N bars. 'mom' = trade in the")
    print("direction of the signal bar; 'rev' = fade it. 50% = coin flip.\n")
    print(f"{'N':>3}  {'momentum':<44}{'reversal':<44}")
    print("-" * 92)
    for N in HORIZONS:
        mom_w = mom_n = rev_w = rev_n = 0
        for sym, df in data.items():
            o = df["open"].values
            c = df["close"].values
            n = len(o)
            if n < N + 3:
                continue
            # signal bar i closes up/down; enter at open[i+1], exit at open[i+1+N]
            i = np.arange(1, n - N - 1)
            up = c[i] > o[i]
            move = o[i + 1 + N] - o[i + 1]
            mom = np.where(up, move > 0, move < 0)
            rev = np.where(up, move < 0, move > 0)
            mom_w += int(mom.sum()); mom_n += len(mom)
            rev_w += int(rev.sum()); rev_n += len(rev)
        print(f"{N:>3}  {hit(mom_w, mom_n):<44}{hit(rev_w, rev_n):<44}")

    # ------------------------------------------------------------------
    section("3. AFTER k CONSECUTIVE SAME-DIRECTION BARS (pooled, N=5)")
    print("Continuation vs reversal of a run. A trend-following edge would show")
    print("continuation > 53%; a mean-reversion edge reversal > 53%.\n")
    N = 5
    print(f"{'run':>4}  {'continuation':<44}{'reversal':<44}")
    print("-" * 92)
    for k in (2, 3, 4, 5, 6):
        cont_w = cont_n = rev_w = rev_n = 0
        for sym, df in data.items():
            o, c = df["open"].values, df["close"].values
            n = len(o)
            sign = np.sign(c - o)
            for i in range(k, n - N - 2):
                run = sign[i - k + 1:i + 1]
                if not np.all(run == run[0]) or run[0] == 0:
                    continue
                move = o[i + 1 + N] - o[i + 1]
                cont = (move > 0) if run[0] > 0 else (move < 0)
                cont_w += int(cont); cont_n += 1
                rev_w += int(not cont); rev_n += 1
        print(f"{k:>4}  {hit(cont_w, cont_n):<44}{hit(rev_w, rev_n):<44}")

    # ------------------------------------------------------------------
    section("4. RSI / BOLLINGER EXTREMES -> next 5m direction (pooled)")
    print("Classic mean-reversion hypothesis: buy oversold, sell overbought.\n")
    feats = []
    for sym, df in data.items():
        d = rb.compute_indicators(df, rb.CFG)
        feats.append(pd.DataFrame({
            "symbol": sym,
            "time": d.index,
            "rsi": d["rsi"].values,
            "pctb": d["bb_pctb"].values,
            "close": d["close"].values,
            "open": d["open"].values,
            "fwd": pd.Series(d["open"].values).shift(-(N + 1)).values,
            "entry": pd.Series(d["open"].values).shift(-1).values,
        }))
    f = pd.concat(feats, ignore_index=True).dropna(subset=["rsi", "pctb", "fwd", "entry"])
    f["move"] = f["fwd"] - f["entry"]

    def cond(name, mask):
        x = f[mask]
        if x.empty:
            print(f"{name:<28}n=0")
            return
        up = (x["move"] > 0).values
        print(f"{name:<28}{hit(int(up.sum()), len(up))}")

    cond("RSI < 30 (buy)", f["rsi"] < 30)
    cond("RSI < 25 (buy)", f["rsi"] < 25)
    cond("RSI 30-40 (buy)", (f["rsi"] >= 30) & (f["rsi"] < 40))
    cond("RSI 40-60 (none)", (f["rsi"] >= 40) & (f["rsi"] < 60))
    cond("RSI 60-70 (sell)", (f["rsi"] >= 60) & (f["rsi"] < 70))
    cond("RSI > 70 (sell)", f["rsi"] > 70)
    cond("RSI > 75 (sell)", f["rsi"] > 75)
    cond("%B < 0.05 (buy)", f["pctb"] < 0.05)
    cond("%B < 0.15 (buy)", f["pctb"] < 0.15)
    cond("%B > 0.85 (sell)", f["pctb"] > 0.85)
    cond("%B > 0.95 (sell)", f["pctb"] > 0.95)
    print("\n(For sell rows, 'hit' is the up-move rate; a sell edge needs it LOW.)")

    # ------------------------------------------------------------------
    section("5. STABILITY ACROSS 5 TIME FOLDS (baseline rule vs coin flip)")
    print("If the instrument is a martingale, every fold should sit near 50%.")
    print("A rule that only works in one fold is noise.\n")
    data_sigs = {}
    frames = []
    for sym, df in data.items():
        s = rb.build_signals(sym, df, variant="v10")
        if s.empty:
            continue
        s = s.copy()
        s["symbol"] = sym
        s["time"] = df.index[s["bar"].values]
        frames.append(s)
    sigs = pd.concat(frames, ignore_index=True)
    all_edges = folds(pd.DatetimeIndex(sorted(sigs["time"].unique())), 5)
    sigs["fold"] = fold_of(sigs["time"], all_edges)

    tr = rb.run_all(data, sigs)
    tr = tr.merge(sigs[["symbol", "bar", "time", "fold"]].rename(columns={"bar": "signal_bar"}),
                  on=["symbol", "signal_bar"], how="left")
    print(f"{'fold':<6}{'period':<28}{'n':>6}{'WR':>8}{'exp/trade':>11}{'netPnL':>9}")
    print("-" * 70)
    for k, g in tr.groupby("fold"):
        lo, hi = all_edges[k]
        n = len(g); w = int(g["win"].sum())
        print(f"{k:<6}{f'{lo:%m-%d} .. {hi:%m-%d}':<28}{n:>6}{w/n*100:>7.1f}%"
              f"{g['pnl'].mean():>+11.3f}{g['pnl'].sum():>+9.2f}")
    print("\nCoin-flip control (same folds, random direction, same durations):")
    rng = np.random.default_rng(0)
    print(f"{'fold':<6}{'period':<28}{'n':>6}{'WR':>8}{'exp/trade':>11}{'netPnL':>9}")
    print("-" * 70)
    for k, g in tr.groupby("fold"):
        lo, hi = all_edges[k]
        win = rng.random(len(g)) < 0.5
        pnl = np.where(win, g["pnl"].where(g["win"], 0.9).fillna(0.9), -1.0)
        # payout is per-trade; approximate with realised mean win payout
        pay = g.loc[g["win"], "pnl"].mean() if g["win"].any() else 0.9
        pnl = np.where(win, pay, -1.0)
        print(f"{k:<6}{f'{lo:%m-%d} .. {hi:%m-%d}':<28}{len(g):>6}"
              f"{win.mean()*100:>7.1f}%{pnl.mean():>+11.3f}{pnl.sum():>+9.2f}")

    # ------------------------------------------------------------------
    section("6. DATA COVERAGE")
    for sym, df in data.items():
        gaps = df.index.to_series().diff().dt.total_seconds().div(60)
        print(f"  {sym:<9} {len(df):>6} bars  {df.index.min():%m-%d %H:%M} .. "
              f"{df.index.max():%m-%d %H:%M}  maxgap={gaps.max():.0f}m")


if __name__ == "__main__":
    main()
