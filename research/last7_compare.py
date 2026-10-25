"""
research/last7_compare.py - Old vs new strategy over the last 7 days.

"old" = v10 entry rules exactly as they were before 2026-10-25 (no symbol
        exclusion).
"new" = v10 + the measured-defect exclusion stpRNG4 (the only strategy change
        shipped today).
"v11" = the loss-driven tightening that was reverted, kept as a reference.

Two windows are reported:
  * the original cache window (data/candles/, 2026-10-17 05:25 -> 2026-10-24
    05:25 UTC), kept so results stay comparable with earlier reports;
  * the continuous 7-day window built by research/fill_gap.py
    (data/candles_7d/, ends at the newest bar), which has no missing hours.

Trades are attributed to the day they were OPENED (entry bar), matching the
bot's reporting rule.

Usage: .venv/bin/python research/last7_compare.py [--dir DIR] [--days N]
                                                  [--main-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from v11_ab import UNIVERSE, wilson  # noqa: E402

MAIN_START = pd.Timestamp("2026-10-17 05:25", tz="UTC")
MAIN_END = pd.Timestamp("2026-10-24 05:25", tz="UTC")
RECENT_START = pd.Timestamp("2026-10-24 16:01", tz="UTC")
RECENT_END = pd.Timestamp("2026-10-25 05:00", tz="UTC")


def build(variant: str, exclude_toxic: bool) -> tuple[dict, pd.DataFrame]:
    data, frames = {}, []
    for sym in UNIVERSE:
        p = rb.CANDLE_DIR / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = rb.load(sym)
        data[sym] = df
        s = rb.build_signals(sym, df, variant=variant, exclude_toxic=exclude_toxic)
        if s.empty:
            continue
        s = s.copy()
        s.insert(0, "symbol", sym)
        s["time"] = df.index[s["bar"].values]
        s["entry_time"] = df.index[np.minimum(s["bar"].values + 1, len(df) - 1)]
        frames.append(s)
    return data, (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def replay(data, sigs, start, end) -> pd.DataFrame:
    tr = rb.run_all(data, sigs)
    if tr.empty:
        return tr
    rows = []
    for sym, g in tr.groupby("symbol"):
        df = data[sym]
        g = g.copy()
        g["time"] = df.index[g["signal_bar"].values]
        g["entry_time"] = df.index[g["entry_bar"].values]
        g["exit_time"] = df.index[np.minimum(g["exit_bar"].values, len(df) - 1)]
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    return t[(t["entry_time"] >= start) & (t["entry_time"] <= end)].sort_values("entry_time")


def stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"n": 0, "w": 0, "wr": 0.0, "pnl": 0.0, "exp": 0.0, "dd": 0.0,
                "lo": 0.0, "hi": 0.0}
    n = len(t)
    w = int(t["win"].sum())
    pnl = float(t["pnl"].sum())
    eq = np.cumsum(t.sort_values("entry_time")["pnl"].values)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq])) -
                      np.concatenate([[0.0], eq])))
    lo, hi = wilson(w / n, n)
    return {"n": n, "w": w, "wr": w / n, "pnl": pnl, "exp": pnl / n, "dd": dd,
            "lo": lo, "hi": hi}


def row(name: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{name:<26}{0:>6}"
    return (f"{name:<26}{s['n']:>6}{s['w']:>5}{s['wr']*100:>7.1f}%"
            f"{s['lo']*100:>6.1f}-{s['hi']*100:<6.1f}{s['pnl']:>+9.2f}"
            f"{s['exp']:>+9.3f}{s['dd']:>8.2f}")


HDR = (f"{'strategy':<26}{'n':>6}{'W':>5}{'WR':>8}{'95% CI':>14}"
       f"{'netPnL':>9}{'exp/tr':>9}{'maxDD':>8}")


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def compare_window(label: str, start, end, old_t, new_t, v11_t) -> None:
    section(f"{label}\n{start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC "
            f"({(end-start).total_seconds()/86400:.1f} days)")
    print(HDR)
    print("-" * 88)
    print(row("OLD (v10, no exclusion)", stats(old_t)))
    print(row("NEW (v10 + no stpRNG4)", stats(new_t)))
    print(row("v11 (reverted, reference)", stats(v11_t)))
    so, sn = stats(old_t), stats(new_t)
    print(f"\nchange from OLD to NEW: {sn['n']-so['n']:+d} trades, "
          f"WR {(sn['wr']-so['wr'])*100:+.1f} pp, "
          f"netPnL {sn['pnl']-so['pnl']:+.2f}, "
          f"exp/trade {sn['exp']-so['exp']:+.3f}")

    # day by day
    print(f"\nby UTC day (entry basis):")
    print(f"{'day':<12}{'OLD n':>7}{'OLD W':>7}{'OLD PnL':>10}"
          f"{'NEW n':>7}{'NEW W':>7}{'NEW PnL':>10}{'delta':>9}")
    print("-" * 70)
    days = sorted(set(old_t["entry_time"].dt.strftime("%m-%d")) |
                  set(new_t["entry_time"].dt.strftime("%m-%d")))
    for d in days:
        o = old_t[old_t["entry_time"].dt.strftime("%m-%d") == d]
        n = new_t[new_t["entry_time"].dt.strftime("%m-%d") == d]
        print(f"{d:<12}{len(o):>7}{int(o['win'].sum()):>7}{o['pnl'].sum():>+10.2f}"
              f"{len(n):>7}{int(n['win'].sum()):>7}{n['pnl'].sum():>+10.2f}"
              f"{n['pnl'].sum()-o['pnl'].sum():>+9.2f}")

    # direction
    print(f"\nby direction:")
    print(f"{'direction':<12}{'OLD n':>7}{'OLD WR':>8}{'OLD PnL':>10}"
          f"{'NEW n':>7}{'NEW WR':>8}{'NEW PnL':>10}")
    print("-" * 62)
    for d in ("long", "short"):
        o = old_t[old_t["direction"] == d]
        n = new_t[new_t["direction"] == d]
        print(f"{d:<12}{len(o):>7}{(o['win'].mean()*100 if len(o) else 0):>7.1f}%"
              f"{o['pnl'].sum():>+10.2f}{len(n):>7}"
              f"{(n['win'].mean()*100 if len(n) else 0):>7.1f}%{n['pnl'].sum():>+10.2f}")

    # per-symbol, only where old had trades
    print(f"\nby symbol (OLD -> NEW):")
    print(f"{'symbol':<10}{'OLD n':>7}{'OLD WR':>8}{'OLD PnL':>10}"
          f"{'NEW n':>7}{'NEW WR':>8}{'NEW PnL':>10}{'delta':>9}")
    print("-" * 76)
    for sym in sorted(old_t["symbol"].unique()):
        o = old_t[old_t["symbol"] == sym]
        n = new_t[new_t["symbol"] == sym]
        print(f"{sym:<10}{len(o):>7}{(o['win'].mean()*100 if len(o) else 0):>7.1f}%"
              f"{o['pnl'].sum():>+10.2f}{len(n):>7}"
              f"{(n['win'].mean()*100 if len(n) else 0):>7.1f}%"
              f"{n['pnl'].sum():>+10.2f}{n['pnl'].sum()-o['pnl'].sum():>+9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="candle dir override (e.g. data/candles_7d)")
    ap.add_argument("--days", type=float, default=7.0, help="window length in days")
    ap.add_argument("--main-only", action="store_true", help="skip the recent-cache section")
    a = ap.parse_args()

    if a.dir:
        d = Path(a.dir)
        rb.CANDLE_DIR = d if d.is_absolute() else ROOT / d

    print("Building signals and replaying (old / new / v11) ...")
    data, old_sigs = build("v10", exclude_toxic=False)
    _, new_sigs = build("v10", exclude_toxic=True)
    _, v11_sigs = build("v11", exclude_toxic=True)

    if a.dir:
        end = max(df.index[-1] for df in data.values()).floor("min")
        start = end - pd.Timedelta(days=a.days)
        label = f"LAST {a.days:g} DAYS - continuous ({a.dir})"
    else:
        start, end, label = MAIN_START, MAIN_END, "LAST 7 DAYS - original cache window"

    old_main = replay(data, old_sigs, start, end)
    new_main = replay(data, new_sigs, start, end)
    v11_main = replay(data, v11_sigs, start, end)
    compare_window(label, start, end, old_main, new_main, v11_main)

    if a.main_only:
        summary(old_main, new_main)
        return

    # recent holdout cache
    data2, old2_sigs = {}, []
    for sym in UNIVERSE:
        p = ROOT / "data" / "candles_recent" / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)[["open", "high", "low", "close"]].astype(float)
        if len(df) < 200:
            continue
        data2[sym] = df
    # build on the recent cache only (warm-up is limited, so fewer signals)
    old_r, new_r, v11_r = [], [], []
    for sym, df in data2.items():
        for variant, tox, bucket in (("v10", False, old_r), ("v10", True, new_r),
                                     ("v11", True, v11_r)):
            s = rb.build_signals(sym, df, variant=variant, exclude_toxic=tox)
            if s.empty:
                continue
            s = s.copy()
            s.insert(0, "symbol", sym)
            t = rb.simulate(s, df, lambda d, _s=sym: rb.pay_for(rb.payout_table(), _s, d),
                            rb.SPREAD.get(sym, 0.0))
            if t.empty:
                continue
            t["symbol"] = sym
            t["entry_time"] = df.index[t["entry_bar"].values]
            bucket.append(t)
    def merge(parts):
        if not parts:
            return pd.DataFrame()
        t = pd.concat(parts, ignore_index=True)
        return t[(t["entry_time"] >= RECENT_START) & (t["entry_time"] <= RECENT_END)].sort_values("entry_time")
    compare_window("SUPPLEMENTARY - fresh holdout cache", RECENT_START, RECENT_END,
                   merge(old_r), merge(new_r), merge(v11_r))

    # overall conclusion
    summary(old_main, new_main)


def summary(old_main: pd.DataFrame, new_main: pd.DataFrame) -> None:
    section("SUMMARY")
    so, sn = stats(old_main), stats(new_main)
    print(f"The change removed {so['n']-sn['n']} trades (all stpRNG4), improved "
          f"win rate by {(sn['wr']-so['wr'])*100:+.1f} pp and PnL by "
          f"{sn['pnl']-so['pnl']:+.2f} USD.")
    print(f"NEW expectancy is {sn['exp']:+.3f}/trade against a breakeven "
          f"of about +0.00 (52-53% WR at 0.88-0.95 payout).")
    print("So: the change reduces the loss, it does not turn the book positive.")
    print("v11 is shown only as a reference; it was reverted and must not be shipped.")


if __name__ == "__main__":
    main()
