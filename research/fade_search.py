"""
research/fade_search.py - Last honest attempt at finding a real edge.

predictability.py showed the bot's momentum premise is slightly wrong: fading a
run of same-direction bars hits ~51.2% at N=6, versus 49.1-49.5% for momentum.
That is a genuine mean-reversion effect, but 51.2% is still under breakeven
(51.28% at 0.95 payout, 52.4% at 0.91, 53.2% at 0.88).

This script searches the full bar universe (440k+ bars, not just strategy
signals) for a *conditional* mean-reversion edge strong enough to clear
breakeven, and validates it the only way that counts: on a held-out time half
and across 5 consecutive folds.

Conditions searched
  * run length k = 2..8
  * RSI bands, Bollinger %B bands
  * ADX bands, ATR-percentile bands
  * body size (|close-open| / ATR)
  * hour of day
  * symbol family (R_ / 1HZ / JD / stpRNG)
  * pairwise combinations of the above (run x RSI, run x %B, run x family...)

Every condition is evaluated with:
  train  = first 70% of time
  valid  = last 30% of time
  folds  = 5 contiguous time blocks
A candidate must clear breakeven on BOTH halves and in at least 4/5 folds.

Usage: .venv/bin/python research/fade_search.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from v11_ab import UNIVERSE, wilson  # noqa: E402

HORIZONS = [1, 3, 5, 6, 8, 10]
TRAIN_FRAC = 0.70
MIN_N_TRAIN = 300
MIN_N_VALID = 150

BREAKEVEN = {  # win rate needed to break even, by payout
    "stpRNG": 1 / 1.95,
    "JD": 1 / 1.91,
    "R_": 1 / 0.88,
    "1HZ": 1 / 0.88,
}


def family(sym: str) -> str:
    if sym.startswith("stp"):
        return "stpRNG"
    if sym.startswith("JD"):
        return "JD"
    if sym.startswith("1HZ"):
        return "1HZ"
    return "R_"


def build_bars() -> pd.DataFrame:
    """One row per (symbol, signal bar, horizon) with fade outcome + features."""
    frames = []
    for sym in UNIVERSE:
        p = rb.CANDLE_DIR / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = rb.load(sym)
        d = rb.compute_indicators(df, rb.CFG)
        o = d["open"].values
        c = d["close"].values
        n = len(o)
        if n < 200:
            continue
        sign = np.sign(c - o)
        # run length of consecutive same-direction bars ending at i
        run = np.zeros(n, dtype=int)
        for i in range(1, n):
            if sign[i] != 0 and sign[i] == sign[i - 1]:
                run[i] = run[i - 1] + 1
            else:
                run[i] = 0
        run = run + 1  # 1 = first bar of a run
        base = pd.DataFrame({
            "symbol": sym,
            "fam": family(sym),
            "time": d.index,
            "rsi": d["rsi"].values,
            "pctb": d["bb_pctb"].values,
            "adx": d["adx"].values,
            "atr_pctile": d["atr_pctile"].values,
            "body_atr": np.abs(c - o) / np.where(d["atr"].values > 0, d["atr"].values, np.nan),
            "hour": d.index.hour,
            "run": run,
            "up_bar": (sign > 0),
            "entry": np.r_[o[1:], np.nan],
        })
        for N in HORIZONS:
            exit_ = np.r_[o[1 + N:], np.full(1 + N, np.nan)]
            up_move = exit_ > base["entry"].values
            # fade: short an up bar, long a down bar -> wins when move opposes bar
            fade_win = base["up_bar"].values ^ up_move
            b = base.copy()
            b["N"] = N
            b["fade_win"] = fade_win
            frames.append(b.dropna(subset=["entry", "fade_win"]))
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["adx"].notna()) & (out["rsi"].notna()) & (out["pctb"].notna())]
    return out


def evaluate(x: pd.DataFrame, N: int, cond: dict, split_time) -> dict | None:
    s = x[(x["N"] == N)]
    for col, (lo, hi) in cond.items():
        s = s[(s[col] >= lo) & (s[col] < hi)]
    if s.empty:
        return None
    tr = s[s["time"] < split_time]
    va = s[s["time"] >= split_time]
    if len(tr) < MIN_N_TRAIN or len(va) < MIN_N_VALID:
        return None
    out = {"N": N, "cond": cond, "n_tr": len(tr), "n_va": len(va),
           "wr_tr": float(tr["fade_win"].mean()),
           "wr_va": float(va["fade_win"].mean())}
    # per-fold consistency
    q = pd.qcut(s["time"].rank(method="first"), 5, labels=False)
    wr_folds = s.groupby(q)["fade_win"].mean().values
    out["folds"] = wr_folds
    return out


def fmt_cond(cond: dict) -> str:
    parts = []
    for col, (lo, hi) in cond.items():
        if col == "fam":
            parts.append(f"fam={lo}")
        elif col == "hour":
            parts.append(f"h{lo:.0f}-{hi:.0f}")
        else:
            parts.append(f"{col}[{lo:g},{hi:g})")
    return " & ".join(parts)


def main() -> None:
    print("building bar universe ...")
    x = build_bars()
    print(f"{len(x):,} rows across {x['symbol'].nunique()} symbols, "
          f"{x['time'].min():%m-%d} .. {x['time'].max():%m-%d}")

    all_times = np.sort(x["time"].unique())
    split_time = pd.Timestamp(all_times[int(len(all_times) * TRAIN_FRAC)])
    if split_time.tzinfo is None:
        split_time = split_time.tz_localize("UTC")
    print(f"train < {split_time:%Y-%m-%d %H:%M} UTC, valid >= it")

    # ---- single conditions ----
    SINGLES = [
        ("run", [(k, k + 1) for k in range(1, 9)]),
        ("rsi", [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 101)]),
        ("pctb", [(0, .05), (.05, .15), (.15, .3), (.3, .45), (.45, .6),
                  (.6, .75), (.75, .85), (.85, .95), (.95, 1.01)]),
        ("adx", [(0, 20), (20, 22), (22, 25), (25, 30), (30, 40), (40, 101)]),
        ("atr_pctile", [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]),
        ("body_atr", [(0, .25), (.25, .5), (.5, 1.0), (1.0, 2.0), (2.0, 1e9)]),
        ("hour", [(h, h + 1) for h in range(24)]),
        ("fam", [("R_", "R_"), ("1HZ", "1HZ"), ("JD", "JD"), ("stpRNG", "stpRNG")]),
    ]

    print("\n" + "=" * 100)
    print("A. SINGLE-CONDITION FADE EDGE (best by validation win rate)")
    print("=" * 100)
    print(f"{'N':>3} {'condition':<34}{'n_tr':>7}{'wr_tr':>8}{'n_va':>7}{'wr_va':>8}"
          f"{'need':>7}{'both?':>7}{'folds>need':>11}")
    print("-" * 100)
    singles = []
    for N in HORIZONS:
        for col, bins in SINGLES:
            for lo, hi in bins:
                r = evaluate(x, N, {col: (lo, hi)}, split_time)
                if r:
                    singles.append(r)
    singles.sort(key=lambda r: -r["wr_va"])
    shown = 0
    for r in singles:
        if shown >= 30:
            break
        need = BREAKEVEN["stpRNG"]  # most forgiving payout
        both = "YES" if (r["wr_tr"] >= need and r["wr_va"] >= need) else "no"
        nf = int((r["folds"] >= need).sum())
        print(f"{r['N']:>3} {fmt_cond(r['cond']):<34}{r['n_tr']:>7}{r['wr_tr']*100:>7.2f}%"
              f"{r['n_va']:>7}{r['wr_va']*100:>7.2f}%{need*100:>6.2f}%{both:>7}{nf:>9}/5")
        shown += 1

    # ---- pairwise conditions ----
    print("\n" + "=" * 100)
    print("B. PAIRWISE CONDITIONS (run x X), ranked by validation win rate")
    print("=" * 100)
    print(f"{'N':>3} {'condition':<46}{'n_tr':>7}{'wr_tr':>8}{'n_va':>7}{'wr_va':>8}"
          f"{'need':>7}{'both?':>7}{'folds':>7}")
    print("-" * 100)
    pairs = []
    run_bins = [(k, k + 1) for k in range(2, 9)]
    others = [
        ("rsi", [(0, 40), (40, 60), (60, 101)]),
        ("pctb", [(0, .3), (.3, .7), (.7, 1.01)]),
        ("adx", [(0, 22), (22, 30), (30, 101)]),
        ("atr_pctile", [(0, .5), (.5, 1.01)]),
        ("fam", [("R_", "R_"), ("1HZ", "1HZ"), ("JD", "JD"), ("stpRNG", "stpRNG")]),
    ]
    for N in HORIZONS:
        for rb_ in run_bins:
            for col, bins in others:
                for lo, hi in bins:
                    r = evaluate(x, N, {"run": rb_, col: (lo, hi)}, split_time)
                    if r:
                        pairs.append(r)
    pairs.sort(key=lambda r: -r["wr_va"])
    shown = 0
    for r in pairs:
        if shown >= 40:
            break
        need = BREAKEVEN["stpRNG"]
        both = "YES" if (r["wr_tr"] >= need and r["wr_va"] >= need) else "no"
        nf = int((r["folds"] >= need).sum())
        print(f"{r['N']:>3} {fmt_cond(r['cond']):<46}{r['n_tr']:>7}{r['wr_tr']*100:>7.2f}%"
              f"{r['n_va']:>7}{r['wr_va']*100:>7.2f}%{need*100:>6.2f}%{both:>7}{nf:>5}/5")
        shown += 1

    # ---- summary ----
    print("\n" + "=" * 100)
    print("C. SUMMARY")
    print("=" * 100)
    need = BREAKEVEN["stpRNG"]
    survivors = [r for r in singles + pairs
                 if r["wr_tr"] >= need and r["wr_va"] >= need
                 and int((r["folds"] >= need).sum()) >= 4]
    print(f"conditions with n>=300 train / n>=150 valid tested: {len(singles) + len(pairs)}")
    print(f"cleared breakeven ({need*100:.2f}%) on both halves AND in >=4/5 folds: "
          f"{len(survivors)}")
    for r in survivors[:15]:
        print(f"  N={r['N']}  {fmt_cond(r['cond'])}  "
              f"train {r['wr_tr']*100:.2f}%  valid {r['wr_va']*100:.2f}%  "
              f"folds {[round(v*100,1) for v in r['folds']]}")
    if not survivors:
        print("\n  No condition clears breakeven out of sample and across folds.")
        print("  The best raw edge (fade a long run) is real but too small:")
        best = max(singles, key=lambda r: r["wr_tr"] + r["wr_va"])
        print(f"    best single: N={best['N']} {fmt_cond(best['cond'])} "
              f"train {best['wr_tr']*100:.2f}% valid {best['wr_va']*100:.2f}%")
        print(f"    breakeven at best available payout (stpRNG 0.95): {need*100:.2f}%")


if __name__ == "__main__":
    main()
