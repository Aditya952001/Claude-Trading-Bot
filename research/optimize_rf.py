"""
research/optimize_rf.py - Iterative research loop for the Rise/Fall bot.

Rounds
  1 BASELINE      current engine, ~14 days of 1m candles, all symbols
  2 DIAGNOSE      win rate / expectancy by symbol, direction, expiry, feature bucket
  3 PROPOSE       candidate filter rules generated from the TRAIN slice only
  4 SELECT        greedy forward selection on TRAIN, scored by lower confidence bound
  5 VALIDATE      selected rule on the untouched VALIDATION slice
  6 WALK-FORWARD  refit on the past, score the next slice (stability check)

Read-only: no orders are placed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import rf_backtest as rb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research"
DAYS = 14.0


# ---------------------------------------------------------------------------
def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def stats(tr: pd.DataFrame, label: str = "") -> dict:
    m = rb.metrics(tr, label)
    if m["n"]:
        se = float(tr["pnl"].std(ddof=1) / math.sqrt(m["n"]))
        lo, hi = wilson(m["wr"], m["n"])
        m.update(se=se, wr_lo=lo, wr_hi=hi, trades_per_day=m["n"] / DAYS,
                 lcbs=m["exp"] - se)
    else:
        m.update(se=0.0, wr_lo=0.0, wr_hi=0.0, trades_per_day=0.0, lcbs=-9.9)
    return m


def fmt(m: dict) -> str:
    return (f"{m['label']:<30} n={m['n']:>5} ({m['trades_per_day']:>4.1f}/d)  "
            f"WR={m['wr']*100:>5.1f}% [{m['wr_lo']*100:>4.1f}-{m['wr_hi']*100:<4.1f}]  "
            f"PnL={m['pnl']:>8.2f}  exp={m['exp']:>+6.3f}  lcb={m['lcbs']:>+6.3f}  "
            f"PF={m['pf']:>4.2f}")


def combine(a: rb.Filters, b: dict) -> rb.Filters:
    f = rb.Filters(**{k: getattr(a, k) for k in a.__dataclass_fields__})
    for k, v in b.items():
        if k in ("symbols", "directions", "durations"):
            cur = getattr(f, k)
            setattr(f, k, sorted(set(v) & set(cur)) if cur is not None else sorted(v))
        elif k in ("adx_min", "body_long_min"):
            setattr(f, k, max(getattr(f, k), v))
        elif k in ("adx_max", "body_short_max", "max_score", "min_score"):
            setattr(f, k, min(getattr(f, k), v))
        elif k == "hours":
            f.hours = v
        else:
            old = getattr(f, k)
            setattr(f, k, (max(old[0], v[0]), min(old[1], v[1])))
    return f


def spec_text(spec: dict) -> str:
    parts = []
    for k, v in spec.items():
        if k in ("symbols", "directions", "durations"):
            parts.append(f"{k}=[{','.join(map(str, v))}]")
        elif isinstance(v, tuple):
            parts.append(f"{k}=[{v[0]:.3g},{v[1]:.3g}]")
        else:
            parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return " & ".join(parts)


# ---------------------------------------------------------------------------
def quant(trades: pd.DataFrame, col: str, q: float, default: float) -> float:
    v = trades[col]
    return float(v.quantile(q)) if len(v) else default


def build_candidates(train: pd.DataFrame) -> list[tuple[str, dict]]:
    """Rules proposed from the TRAIN slice only (no peeking at validation)."""
    c: list[tuple[str, dict]] = []
    if train.empty:
        return c

    def add(name, spec):
        if name not in {n for n, _ in c}:
            c.append((name, spec))

    fam = {"R_": lambda s: s.startswith("R_"), "1HZ": lambda s: s.startswith("1HZ"),
           "JD": lambda s: s.startswith("JD"), "stp": lambda s: s.startswith("stp"),
           "frx": lambda s: s.startswith("frx")}
    for name, fn in fam.items():
        syms = sorted({s for s in train["symbol"].unique() if fn(s)})
        if syms:
            add(f"family:{name}", {"symbols": syms})
    for sym, g in train.groupby("symbol"):
        if len(g) >= 60:
            add(f"symbol:{sym}", {"symbols": [sym]})

    for d in ("long", "short"):
        if (train["direction"] == d).sum() >= 60:
            add(f"direction:{d}", {"directions": [d]})

    for durs in ([3], [4], [5], [3, 4], [3, 4, 5], [3, 4, 5, 6], [5, 6], [6, 7, 8]):
        if train["duration"].isin(durs).sum() >= 60:
            add(f"dur:{durs}", {"durations": durs})

    for d in ("long", "short"):
        g = train[train["direction"] == d]
        if len(g) < 60:
            continue
        is_short = d == "short"

        def q(col, lo, hi):
            return quant(g, col, lo, -1e9), quant(g, col, hi, 1e9)

        for lo, hi in [(25, 1e9), (28, 1e9), (22, 28), (25, 35), (25, 45),
                       (28, 50), (20, 25), (30, 1e9)]:
            if ((g.adx >= lo) & (g.adx <= hi)).sum() >= 40:
                add(f"{d}|adx:{lo}-{hi}", {"adx_min": lo, "adx_max": hi})

        rsi_specs = ([(42, 65), (45, 65), (48, 65), (50, 65), (35, 55), (40, 60),
                      q("rsi", .2, .8), q("rsi", .4, .9)] if is_short
                     else [(33, 50), (35, 55), (38, 55), (35, 45), (40, 60), (45, 62),
                           q("rsi", .2, .8), q("rsi", .1, .6)])
        for lo, hi in rsi_specs:
            if lo < hi:
                add(f"{d}|rsi:{lo:.1f}-{hi:.1f}",
                    {"rsi_short" if is_short else "rsi_long": (lo, hi)})

        bb_specs = ([(25, 65), (30, 65), (35, 70), (40, 80), (45, 85), q("bb_pctb", .2, .8)]
                    if is_short else
                    [(10, 45), (15, 55), (20, 60), (25, 65), (30, 70), q("bb_pctb", .2, .8)])
        for lo, hi in bb_specs:
            if lo < hi:
                add(f"{d}|bb:{lo:.2f}-{hi:.2f}",
                    {"bb_short" if is_short else "bb_long": (lo, hi)})

        for lo, hi in [(0, .4), (0, .5), (0, .6), (.2, .6), (.2, .8), (.4, 1.0), (.6, 1.0)]:
            if ((g.atr_pctile >= lo) & (g.atr_pctile <= hi)).sum() >= 40:
                add(f"{d}|atr:{lo}-{hi}", {"atr_pctile": (lo, hi)})

        macd_specs = ([(-0.06, -0.003), (-0.03, -0.003), (-0.10, -0.01), (-1, -0.02)]
                      if is_short else
                      [(0.003, 0.03), (0.003, 0.06), (0.01, 0.06), (0.003, 0.10)])
        for lo, hi in macd_specs:
            add(f"{d}|macd:{lo}-{hi}",
                {"macd_rel_short" if is_short else "macd_rel_long": (lo, hi)})

        gap_specs = ([(-0.30, -0.02), (-0.15, -0.02), (-0.10, -0.015), (-1, -0.05)]
                     if is_short else
                     [(0.02, 0.15), (0.02, 0.30), (0.05, 0.30), (0.015, 0.10)])
        for lo, hi in gap_specs:
            add(f"{d}|gap:{lo}-{hi}",
                {"ema_gap_short" if is_short else "ema_gap_long": (lo, hi)})

        for thr in (0.0, 0.1, 0.2, 0.3):
            add(f"{d}|body>{thr}",
                {"body_short_max": -thr} if is_short else {"body_long_min": thr})

    for a, b in [(0, 6), (6, 12), (12, 18), (18, 24)]:
        if train.hour.between(a, b - 1).sum() >= 100:
            add(f"hours:{a}-{b}", {"hours": (a, b - 1)})

    return c


def greedy(fit_sigs, data, base, max_rules=3, min_n=100, verbose=True):
    """Forward selection on the fit slice. Returns (selected rules, filter, trades)."""
    def run_on(f):
        return rb.run_all(data, fit_sigs, f)

    cur = base
    cur_tr = run_on(cur)
    cur_m = stats(cur_tr, "base")
    selected = []
    cands = build_candidates(cur_tr)
    for _ in range(max_rules):
        best = None
        for name, spec in cands:
            f = combine(cur, spec)
            tr = run_on(f)
            if len(tr) < min_n:
                continue
            m = stats(tr)
            if m["trades_per_day"] < 0.5:
                continue
            if best is None or m["lcbs"] > best[0]:
                best = (m["lcbs"], name, spec, m)
        if best is None or best[0] <= cur_m["lcbs"] + 0.005:
            break
        _, name, spec, m = best
        selected.append({"name": name, "spec": spec, "train_n": m["n"],
                         "train_exp": m["exp"], "train_wr": m["wr"],
                         "train_lcbs": best[0]})
        if verbose:
            print(f"  + {name:<34} n={m['n']:>4} WR={m['wr']*100:5.1f}% "
                  f"exp={m['exp']:+.3f} lcb={best[0]:+.3f}")
        cur = combine(cur, spec)
        cur_tr = run_on(cur)
        cur_m = stats(cur_tr, "base")
    return selected, cur, cur_tr


def horizon_scan(data, sigs, horizons) -> pd.DataFrame:
    """Directional hit rate of raw signals at each forward horizon (payout-aware)."""
    pay = rb.payouts()
    out = []
    for sym, bars in data.items():
        s = sigs[sigs.symbol == sym]
        if s.empty:
            continue
        o = bars["open"].values
        n = len(o)
        for h in horizons:
            for d in ("long", "short"):
                b = s.loc[s.direction == d, "bar"].values
                b = b[(b + 1 + h) < n]
                if not len(b):
                    continue
                entry, ex = o[b + 1], o[b + 1 + h]
                move = (ex - entry) if d == "long" else (entry - ex)
                hit = move > 0
                pnl = np.where(hit, pay.get(sym, 0.88), -1.0)
                out.append({"symbol": sym, "direction": d, "horizon": h,
                            "n": len(b), "hit": int(hit.sum()), "pnl": pnl.sum()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def main():
    print("Loading candles + building signals ...", flush=True)
    data, sigs = rb.load_all()
    print(f"  {len(data)} symbols, {len(sigs)} raw signals\n", flush=True)

    cutoff = int(sigs["epoch"].quantile(0.65))
    train_sigs = sigs[sigs.epoch < cutoff]
    val_sigs = sigs[sigs.epoch >= cutoff]
    print(f"Split at {pd.to_datetime(cutoff, unit='s')} UTC "
          f"(train {len(train_sigs)} signals / val {len(val_sigs)})\n")

    def run(f, part):
        return rb.run_all(data, train_sigs if part == "train" else val_sigs, f)

    base = rb.Filters()
    tr_train, tr_val = run(base, "train"), run(base, "val")
    tr_all = pd.concat([tr_train, tr_val])

    print("=" * 120)
    print("ROUND 1  BASELINE (current engine, no extra filters)")
    print(fmt(stats(tr_all, "ALL 14 days")))
    print(fmt(stats(tr_train, "train 65%")))
    print(fmt(stats(tr_val, "validation 35%")))
    print(fmt(stats(run(rb.Filters(invert=True), "train"), "FADE baseline / train")))
    print(fmt(stats(run(rb.Filters(invert=True), "val"), "FADE baseline / validation")))
    print("=" * 120)

    print("\nROUND 2  DIAGNOSE (baseline trades, all data)\n")
    print("-- by symbol")
    for sym, g in tr_all.groupby("symbol"):
        m = stats(g)
        print(f"   {sym:<11} n={m['n']:>4}  WR={m['wr']*100:>5.1f}%  "
              f"exp={m['exp']:>+6.3f}  PnL={m['pnl']:>+7.2f}")
    print("\n-- by direction / expiry")
    for col in ("direction", "duration"):
        for key, g in tr_all.groupby(col):
            m = stats(g)
            print(f"   {col[:4]}={str(key):<7} n={m['n']:>4}  WR={m['wr']*100:>5.1f}%  "
                  f"exp={m['exp']:>+6.3f}  PnL={m['pnl']:>+7.2f}")
    print()

    print("\n-- HORIZON SCAN (raw signals, no filters)")
    hs = horizon_scan(data, sigs, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])
    print("   horizon |    n    | hit long | hit short | EV/trade @ avg payout")
    for h in sorted(hs["horizon"].unique()):
        g = hs[hs.horizon == h]
        n = int(g.n.sum())
        hl = g[g.direction == "long"].hit.sum() / max(1, g[g.direction == "long"].n.sum())
        sh = g[g.direction == "short"].hit.sum() / max(1, g[g.direction == "short"].n.sum())
        evl = g[g.direction == "long"].pnl.sum() / max(1, g[g.direction == "long"].n.sum())
        evs = g[g.direction == "short"].pnl.sum() / max(1, g[g.direction == "short"].n.sum())
        print(f"   {h:>3}m    | {n:>7} |  {hl*100:>5.1f}%  |  {sh*100:>5.1f}%   | "
              f"long {evl:>+6.3f} / short {evs:>+6.3f}")
    print()

    for col, bins in [
        ("adx", [20, 24, 26, 28, 30, 35, 40, 100]),
        ("rsi", [0, 35, 40, 45, 50, 55, 58, 62, 100]),
        ("bb_pctb", [0, .2, .3, .4, .5, .6, .7, .8, 1.01]),
        ("atr_pctile", [0, .2, .4, .6, .8, 1.01]),
        ("macd_rel", [-1, -.05, -.02, -.01, 0, .01, .02, .05, 1]),
        ("ema_gap", [-1, -.3, -.1, -.03, 0, .03, .1, .3, 1]),
    ]:
        b = pd.cut(tr_all[col], bins)
        print(f"-- {col}")
        for key, g in tr_all.groupby(b, observed=True):
            if len(g) >= 10:
                m = stats(g)
                print(f"   {str(key):<16} n={m['n']:>4}  WR={m['wr']*100:>5.1f}%  "
                      f"exp={m['exp']:>+6.3f}  PnL={m['pnl']:>+7.2f}")
        print()

    print("=" * 120)
    print("ROUND 3/4  PROPOSE + GREEDY SELECT (TRAIN only)")
    selected, fitted, fit_tr = greedy(train_sigs, data, base, max_rules=3, min_n=100)
    print(f"\nSELECTED: {spec_text({k: v for s in selected for k, v in s['spec'].items()}) or 'NONE'}")

    print("\nROUND 5  VALIDATION (untouched 35%)\n")
    vt = run(fitted, "val")
    print(fmt(stats(fit_tr, "fitted / train")))
    print(fmt(stats(vt, "fitted / VALIDATION")))
    print(fmt(stats(pd.concat([fit_tr, vt]), "fitted / all")))

    print("\nROUND 6  WALK-FORWARD (refit on past, test next 20%)\n")
    qs = np.quantile(sigs["epoch"].values, [0.2, 0.4, 0.6, 0.8, 1.0]).astype("int64")
    wf = []
    for i, cut in enumerate(qs[:-1]):
        fit_s = sigs[sigs.epoch < cut]
        test_s = sigs[(sigs.epoch >= cut) & (sigs.epoch < qs[i + 1])]
        if len(fit_s) < 300 or len(test_s) < 100:
            continue
        sel_i, f_i, _ = greedy(fit_s, data, base, max_rules=2, min_n=80, verbose=False)
        test_tr = rb.run_all(data, test_s, f_i)
        m = stats(test_tr, f"WF {i+1} ({pd.to_datetime(cut, unit='s'):%m-%d})")
        wf.append(m)
    total = sum(m["n"] for m in wf)
    for m in wf:
        print(fmt(m))
    if total:
        pnl = sum(m["pnl"] for m in wf)
        wr = sum(m["wr"] * m["n"] for m in wf) / total
        print(f"\n  Walk-forward aggregate: n={total}  WR={wr*100:.1f}%  "
              f"PnL={pnl:+.2f}  exp={pnl/total:+.3f}")

    report = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "symbols": list(data.keys()),
        "baseline_all": stats(tr_all, "baseline"),
        "baseline_train": stats(tr_train, "baseline train"),
        "baseline_val": stats(tr_val, "baseline val"),
        "selected_rules": selected,
        "fitted_train": stats(fit_tr, "fitted train"),
        "fitted_val": stats(vt, "fitted val"),
        "walk_forward": wf,
        "per_symbol_baseline": {s: stats(g) for s, g in tr_all.groupby("symbol")},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nSaved research/report.json")


if __name__ == "__main__":
    main()
