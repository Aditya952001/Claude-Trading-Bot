"""
research/htf_sweep.py - Follow-up to htf_test.py.

htf_test showed the boolean "price must be on the correct side of
ema_higher_tf" gate does NOT hold up (train improves, validation doesn't).
Live loss #319 was extreme, though: price sat 7 ATR BELOW the 200-EMA.
So sweep a signed distance threshold instead of a boolean:

    long  requires close_vs_higher >= -tol
    short requires close_vs_higher <=  tol

tol = 1e9 is the baseline. Report train and validation separately so an
overfit shows up immediately (train improves, validation does not).
"""
from __future__ import annotations


import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_backtest as rb


def wl(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return dict(n=0, wr=0.0, exp=0.0, pnl=0.0)
    p = tr["pnl"].values
    return dict(n=len(p), wr=float(tr["win"].mean()), exp=float(p.mean()),
                pnl=float(p.sum()))


def main() -> None:
    data, sigs = rb.load_all()
    pay = rb.payouts()
    synth = [s for s in data if not s.startswith("frx")]
    # Use the SAME split point as htf_test.py (computed on all signals) so the
    # two reports are directly comparable.
    cut = int(sigs["epoch"].quantile(0.65))
    sigs = sigs[sigs.symbol.isin(synth)]
    tr_s = sigs[sigs.epoch < cut]
    va_s = sigs[sigs.epoch >= cut]

    def subset(df: pd.DataFrame, tol: float) -> pd.DataFrame:
        if tol > 1e8:
            return df
        long = (df["direction"] == "long").values
        v = df["close_vs_higher"].values
        # long may sit at most `tol` ATR below the higher-tf EMA; short at most
        # `tol` ATR above it.
        keep = ((v >= -tol) & long) | ((v <= tol) & ~long)
        return df[keep]

    print("synthetic only | tol = max distance (in ATR) allowed on the WRONG side "
          "of ema_higher_tf\n")
    print(f"{'tol':>6} | {'train n':>7} {'WR':>6} {'exp':>7} | "
          f"{'val n':>6} {'WR':>6} {'exp':>7} | {'dExp val':>8}")
    print("-" * 78)
    base_val = None
    for tol in (1e9, 6.0, 4.0, 2.0, 1.0, 0.5, 0.0):
        tr = rb.run_all(data, subset(tr_s, tol), rb.Filters(symbols=synth))
        va = rb.run_all(data, subset(va_s, tol), rb.Filters(symbols=synth))
        mt, mv = wl(tr), wl(va)
        if base_val is None:
            base_val = mv["exp"]
        label = "base" if tol > 1e8 else f"{tol:.1f}"
        print(f"{label:>6} | {mt['n']:>7} {mt['wr']*100:>5.1f}% {mt['exp']:>+7.3f} | "
              f"{mv['n']:>6} {mv['wr']*100:>5.1f}% {mv['exp']:>+7.3f} | "
              f"{mv['exp'] - base_val:>+8.3f}")

    # distribution of close_vs_higher for winners vs losers on validation
    va = rb.run_all(data, va_s, rb.Filters(symbols=synth))
    print("\nvalidation baseline trades, close_vs_higher deciles:")
    va = va.copy()
    va["bucket"] = pd.qcut(va["close_vs_higher"], 5, duplicates="drop")
    for b, g in va.groupby("bucket", observed=True):
        m = wl(g)
        print(f"  {str(b):<24} n={m['n']:>4}  WR={m['wr']*100:>5.1f}%  exp={m['exp']:>+6.3f}")


if __name__ == "__main__":
    main()
