"""
research/policy_test.py - Compare trade-selection policies on train vs validation.

Answers: which (symbol family, expiry, direction) policy loses least / makes most
out-of-sample, and is any measured edge statistically distinguishable from zero?
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_backtest as rb

DAY = 14.0


def wl(tr):
    p = tr["pnl"].values
    n = len(p)
    if not n:
        return dict(n=0, wr=0, pnl=0, exp=0, pf=0, dd=0, tpd=0)
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    wins, losses = p[p > 0], p[p < 0]
    return dict(n=n, wr=float(tr["win"].mean()), pnl=float(p.sum()),
                exp=float(p.mean()), dd=dd, tpd=n / DAY,
                pf=float(wins.sum() / abs(losses.sum())) if len(losses) else 99.0)


def z_vs_breakeven(m, payout):
    """One-sided z-score of measured win rate vs the payout breakeven."""
    be = 1.0 / (1.0 + payout)
    se = math.sqrt(max(be * (1 - be) / max(1, m["n"]), 1e-12))
    return (m["wr"] - be) / se, be


def fmt(name, m, payout):
    z, be = z_vs_breakeven(m, payout)
    return (f"{name:<34} n={m['n']:>4} ({m['tpd']:>4.1f}/d)  WR={m['wr']*100:>5.1f}% "
            f"(be {be*100:.1f}%)  exp={m['exp']:>+6.3f}  PnL={m['pnl']:>+7.2f}  "
            f"PF={m['pf']:>4.2f}  z={z:>+5.2f}")


FAMILIES = {
    "R_": lambda s: s.startswith("R_"),
    "1HZ": lambda s: s.startswith("1HZ"),
    "JD": lambda s: s.startswith("JD"),
    "stp": lambda s: s.startswith("stp"),
    "frx": lambda s: s.startswith("frx"),
}
SYNTH_ALL = None  # filled in main()
SYNTH_LOW_EDGE = ["stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5",
                  "JD10", "JD25", "JD50", "JD75", "JD100"]


def main():
    global SYNTH_ALL
    data, sigs = rb.load_all()
    SYNTH_ALL = [s for s in data if not s.startswith("frx")]
    cut = int(sigs.epoch.quantile(0.65))
    tr_s, va_s = sigs[sigs.epoch < cut], sigs[sigs.epoch >= cut]
    pay = rb.payouts()

    def run(sub, f):
        return rb.run_all(data, sub, f)

    def avg_pay(tr):
        return float(np.mean([pay.get(s, 0.88) for s in tr.symbol])) if len(tr) else 0.88

    all_tr = run(tr_s, rb.Filters())
    all_va = run(va_s, rb.Filters())
    p_all = avg_pay(all_tr)

    def fam_set(names):
        return sorted({s for s in data if any(FAMILIES[n](s) for n in names)})

    policies = {
        "all": rb.Filters(),
        "synthetics only (no forex)": rb.Filters(symbols=[s for s in data if not s.startswith("frx")]),
        "stp only": rb.Filters(symbols=fam_set(["stp"])),
        "JD only": rb.Filters(symbols=fam_set(["JD"])),
        "stp+JD": rb.Filters(symbols=SYNTH_LOW_EDGE),
        "stp+JD, 4-8m": rb.Filters(symbols=SYNTH_LOW_EDGE, durations=[4, 5, 6, 7, 8]),
        "stp+JD, 4-6m": rb.Filters(symbols=SYNTH_LOW_EDGE, durations=[4, 5, 6]),
        "R_ only": rb.Filters(symbols=fam_set(["R_"])),
        "1HZ only": rb.Filters(symbols=fam_set(["1HZ"])),
        "forex only": rb.Filters(symbols=fam_set(["frx"])),
        "long only": rb.Filters(directions=["long"]),
        "short only": rb.Filters(directions=["short"]),
        "4m only": rb.Filters(durations=[4]),
        "5m only": rb.Filters(durations=[5]),
        "6m only": rb.Filters(durations=[6]),
        "3m only": rb.Filters(durations=[3]),
        # --- contrarian / horizon probes (synthetics only) ---
        "fade ALL signals": rb.Filters(symbols=SYNTH_ALL, invert=True),
        "fade long signals (engine dur)": rb.Filters(symbols=SYNTH_ALL, invert=True, directions=["long"]),
        "fade long signals @1m": rb.Filters(symbols=SYNTH_ALL, invert=True, directions=["long"], duration_override=1),
        "fade long signals @2m": rb.Filters(symbols=SYNTH_ALL, invert=True, directions=["long"], duration_override=2),
        "fade short signals @1m": rb.Filters(symbols=SYNTH_ALL, invert=True, directions=["short"], duration_override=1),
        "fade short signals @2m": rb.Filters(symbols=SYNTH_ALL, invert=True, directions=["short"], duration_override=2),
        "follow short signals @2m": rb.Filters(symbols=SYNTH_ALL, directions=["short"], duration_override=2),
        "all signals @1m": rb.Filters(symbols=SYNTH_ALL, duration_override=1),
        "all signals @2m": rb.Filters(symbols=SYNTH_ALL, duration_override=2),
    }

    print(f"Split {pd.to_datetime(cut, unit='s')} UTC | train signals {len(tr_s)} / val {len(va_s)}")
    print(f"Average payout of baseline trades: {p_all:.3f}  -> breakeven "
          f"{100/(1+p_all):.1f}%\n")

    print("=" * 130)
    print("POLICY COMPARISON  (train 65%  |  validation 35%)")
    print("=" * 130)
    rows = []
    for name, f in policies.items():
        tr = run(tr_s, f)
        va = run(va_s, f)
        if tr.empty or va.empty:
            continue
        p = avg_pay(tr)
        mtr, mva = wl(tr), wl(va)
        print(f"\n{name}   (avg payout {p:.3f}, breakeven {100/(1+p):.1f}%)")
        print("  " + fmt("train", mtr, p))
        print("  " + fmt("validation", mva, p))
        rows.append({"policy": name, "payout": p, "train": mtr, "val": mva})

    # per-family aggregate
    print("\n" + "=" * 130)
    print("PER FAMILY (all data, baseline signals)")
    print("=" * 130)
    fam_rows = []
    for name, fn in FAMILIES.items():
        tr = all_tr[all_tr.symbol.map(fn)]
        if tr.empty:
            continue
        m = wl(tr)
        p = avg_pay(tr)
        print(fmt(f"{name} ({p:.2f} payout)", m, p))
        fam_rows.append({"family": name, "payout": p, **m})

    # per symbol, train-ranked whitelist -> OOS
    print("\n" + "=" * 130)
    print("WHITELIST TEST: pick the best symbols on TRAIN, then measure on VALIDATION")
    print("=" * 130)
    tr_b = run(tr_s, rb.Filters())
    va_b = run(va_s, rb.Filters())
    ranked = (tr_b.groupby("symbol")
              .apply(lambda g: pd.Series(wl(g)), include_groups=False)
              .sort_values("exp", ascending=False))
    for k in (3, 5, 8, 12):
        top = ranked.head(k).index.tolist()
        tr_k = tr_b[tr_b.symbol.isin(top)]
        va_k = va_b[va_b.symbol.isin(top)]
        if len(tr_k) < 20 or len(va_k) < 10:
            continue
        print(f"\ntop {k} on train: {top}")
        print("  " + fmt("train", wl(tr_k), avg_pay(tr_k)))
        print("  " + fmt("validation", wl(va_k), avg_pay(va_k)))

    print("\n" + "=" * 130)
    print("SAMPLE SIZE: trades needed to prove a +2pp edge at 80% power")
    be = 1 / (1 + p_all)
    n_need = (1.96 + 0.84) ** 2 * be * (1 - be) / (0.02 ** 2)
    print(f"  ~{n_need:.0f} trades. At the current ~{wl(all_va)['tpd']:.0f} trades/day "
          f"(validation), that is ~{n_need / max(0.1, wl(all_va)['tpd']):.0f} days of "
          f"uninterrupted trading for ONE configuration.")
    print("=" * 130)

    (Path(__file__).parent / "policy.json").write_text(
        json.dumps({"policies": rows, "families": fam_rows,
                    "n_needed_2pp": n_need}, indent=2, default=float))


if __name__ == "__main__":
    main()
