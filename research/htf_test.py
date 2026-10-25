"""
research/htf_test.py - Does requiring higher-timeframe agreement improve the
Rise/Fall engine out-of-sample?

Hypothesis taken from live trade #319 (1HZ100V long, lost):
    price was BELOW ema_higher_tf (1053 vs 1068) yet the engine went long,
    because `adx >= 25` bypasses the higher-timeframe check.

Test
    1. baseline (engine as shipped), train vs validation
    2. + require_htf_agreement (long needs close > ema_higher_tf, short <)
    3. segment baseline trades into "agrees" vs "bypass (disagrees)" so we can
       see whether the ADX bypass really is the losing population
    4. same, restricted to synthetics (production's rf_allowed_assets)

Read-only research. Nothing here touches production code.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_backtest as rb

DAY = 14.0


def wl(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return dict(n=0, wr=0.0, pnl=0.0, exp=0.0, pf=0.0)
    p = tr["pnl"].values
    wins, losses = p[p > 0], p[p < 0]
    return dict(
        n=len(p), wr=float(tr["win"].mean()), pnl=float(p.sum()),
        exp=float(p.mean()),
        pf=float(wins.sum() / abs(losses.sum())) if len(losses) else 99.0,
    )


def fmt(name: str, m: dict, payout: float) -> str:
    be = 1.0 / (1.0 + payout)
    se = math.sqrt(max(be * (1 - be) / max(1, m["n"]), 1e-12))
    z = (m["wr"] - be) / se if m["n"] else 0.0
    return (f"  {name:<24} n={m['n']:>5}  WR={m['wr']*100:>5.1f}%  "
            f"exp={m['exp']:>+7.3f}  PnL={m['pnl']:>+8.2f}  "
            f"PF={m['pf']:>4.2f}  z={z:>+5.2f}")


def agree(tr: pd.DataFrame) -> pd.Series:
    long = (tr["direction"] == "long").values
    v = tr["close_vs_higher"].values
    return np.where(long, v > 0.0, v < 0.0)


def avg_pay(tr: pd.DataFrame, pay: dict) -> float:
    if tr.empty:
        return 0.88
    return float(np.mean([pay.get(s, 0.88) for s in tr.symbol]))


def report(tag: str, tr: pd.DataFrame, va: pd.DataFrame, pay: dict) -> None:
    p_tr, p_va = avg_pay(tr, pay), avg_pay(va, pay)
    print(f"\n{tag}")
    print(fmt("train (65%)", wl(tr), p_tr))
    print(fmt("validation (35%)", wl(va), p_va))


def main() -> None:
    data, sigs = rb.load_all()
    pay = rb.payouts()
    synthetics = [s for s in data if not s.startswith("frx")]

    cut = int(sigs["epoch"].quantile(0.65))
    print(f"Split {pd.to_datetime(cut, unit='s')} UTC | "
          f"train signals {int((sigs.epoch < cut).sum())} / "
          f"val {int((sigs.epoch >= cut).sum())}")
    print(f"symbols: {len(data)} total, {len(synthetics)} synthetic")

    for universe_name, universe in (("ALL SYMBOLS", list(data)),
                                    ("SYNTHETIC ONLY (production policy)", synthetics)):
        sub = sigs[sigs["symbol"].isin(universe)]
        tr_s = sub[sub.epoch < cut]
        va_s = sub[sub.epoch >= cut]

        base = rb.Filters(symbols=universe)
        htf = rb.Filters(symbols=universe, require_htf_agreement=True)

        tr_b = rb.run_all(data, tr_s, base)
        va_b = rb.run_all(data, va_s, base)
        tr_h = rb.run_all(data, tr_s, htf)
        va_h = rb.run_all(data, va_s, htf)

        print("\n" + "=" * 108)
        print(universe_name)
        print("=" * 108)
        report("BASELINE (shipped engine)", tr_b, va_b, pay)
        report("+ HTF agreement required", tr_h, va_h, pay)

        print("\n  population split of BASELINE trades")
        for label, frame in (("train", tr_b), ("validation", va_b)):
            if frame.empty:
                continue
            a = agree(frame)
            print(f"   {label}:")
            print(fmt("   htf agrees", wl(frame[a]), avg_pay(frame[a], pay)))
            print(fmt("   htf DISAGREES", wl(frame[~a]), avg_pay(frame[~a], pay)))

        # how many baseline trades actually came from the ADX bypass?
        if not tr_b.empty:
            bypass = (~agree(tr_b)) & (tr_b["adx"] >= 25.0)
            print(f"\n   bypass trades (adx>=25 & htf disagreement): "
                  f"train {int(bypass.sum())}/{len(tr_b)} "
                  f"({bypass.mean()*100:.1f}%)")

        # direction breakdown of the filter's effect (validation only)
        if not va_b.empty and not va_h.empty:
            print("\n  validation by direction")
            for d in ("long", "short"):
                b = va_b[va_b.direction == d]
                h = va_h[va_h.direction == d]
                print(fmt(f"   baseline {d}", wl(b), avg_pay(b, pay)))
                print(fmt(f"   htf      {d}", wl(h), avg_pay(h, pay)))

    # per-symbol stability of the filter on validation
    if not sigs.empty:
        va_s = sigs[(sigs.epoch >= cut) & sigs.symbol.isin(synthetics)]
        va_b = rb.run_all(data, va_s, rb.Filters(symbols=synthetics))
        va_h = rb.run_all(data, va_s,
                          rb.Filters(symbols=synthetics, require_htf_agreement=True))
        print("\n" + "=" * 108)
        print("VALIDATION per symbol (synthetic, n>=15 in baseline) "
              "delta exp = htf - baseline")
        print("=" * 108)
        rows = []
        for sym, g in va_b.groupby("symbol"):
            if len(g) < 15:
                continue
            h = va_h[va_h.symbol == sym]
            mb, mh = wl(g), wl(h)
            rows.append((sym, mb["n"], mb["wr"], mb["exp"],
                         mh["n"], mh["wr"], mh["exp"], mh["exp"] - mb["exp"]))
        rows.sort(key=lambda r: -r[-1])
        for r in rows:
            print(f"  {r[0]:<9} base n={r[1]:>3} WR={r[2]*100:>5.1f}% exp={r[3]:>+6.3f}"
                  f"   | htf n={r[4]:>3} WR={r[5]*100:>5.1f}% exp={r[6]:>+6.3f}"
                  f"   | dExp={r[7]:>+6.3f}")

    # walk-forward: does the filter help in each month independently?
    print("\n" + "=" * 108)
    print("WALK-FORWARD by month (synthetic only): baseline exp -> htf exp")
    print("=" * 108)
    syn_sigs = sigs[sigs.symbol.isin(synthetics)].copy()
    syn_sigs["month"] = pd.to_datetime(syn_sigs["epoch"], unit="s").dt.strftime("%Y-%m")
    for month, chunk in syn_sigs.groupby("month"):
        if len(chunk) < 100:
            continue
        b = rb.run_all(data, chunk, rb.Filters(symbols=synthetics))
        h = rb.run_all(data, chunk,
                       rb.Filters(symbols=synthetics, require_htf_agreement=True))
        mb, mh = wl(b), wl(h)
        if mb["n"] == 0:
            continue
        print(f"  {month}  base n={mb['n']:>4} WR={mb['wr']*100:>5.1f}% exp={mb['exp']:>+6.3f}"
              f"   ->  htf n={mh['n']:>4} WR={mh['wr']*100:>5.1f}% exp={mh['exp']:>+6.3f}"
              f"   (dExp {mh['exp']-mb['exp']:>+6.3f})")


if __name__ == "__main__":
    main()
