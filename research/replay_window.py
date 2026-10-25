"""
research/replay_window.py - "What would the strategy have done between X and Y?"

Fetches fresh 1m candles for the Rise/Fall universe, replays the real entry
rules over a UTC window, and reports trades / wins / losses / PnL with the
same payout table and cooldown logic as the 862-trade baseline.

Read-only: only `ticks_history` and `active_symbols` are requested.

Usage:
  .venv/bin/python research/replay_window.py --start 2026-10-24T18:30 --end 2026-10-25T03:30
  .venv/bin/python research/replay_window.py --start ... --end ... --variant v11
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from deriv_data import CandleClient  # noqa: E402

UNIVERSE = ["R_10", "R_25", "R_50", "R_75", "R_100",
            "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
            "JD10", "JD25", "JD50", "JD75", "JD100",
            "stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"]

CACHE = ROOT / "data" / "candles_recent"


async def fetch_all(bars: int) -> dict[str, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    client = await CandleClient.connect("demo")
    out = {}
    try:
        for sym in UNIVERSE:
            try:
                df = await client.candles(sym, bars)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {sym}: {exc}")
                continue
            if df.empty:
                continue
            df.to_csv(CACHE / f"{sym}_60.csv")
            out[sym] = df
    finally:
        await client.close()
    return out


def load_cached() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in UNIVERSE:
        p = CACHE / f"{sym}_60.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if len(df):
                out[sym] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC, e.g. 2026-10-24T18:30")
    ap.add_argument("--end", required=True, help="UTC, e.g. 2026-10-25T03:30")
    ap.add_argument("--variant", default="v10", choices=["v10", "v11"])
    ap.add_argument("--bars", type=int, default=800)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    data = asyncio.run(fetch_all(args.bars)) if args.refresh else load_cached()
    if not data:
        print("no candles; pass --refresh")
        return

    cover_lo = min(d.index.min() for d in data.values())
    cover_hi = max(d.index.max() for d in data.values())
    print(f"candles: {len(data)} symbols, {cover_lo:%Y-%m-%d %H:%M} .. {cover_hi:%Y-%m-%d %H:%M} UTC")
    print(f"window : {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC  (rules: {args.variant})\n")

    table = rb.payout_table()
    trades = []
    sig_count = 0
    for sym, df in data.items():
        s = rb.build_signals(sym, df, variant=args.variant)
        if s.empty:
            continue
        s = s.copy()
        s["time"] = df.index[s["bar"].values]
        s = s[(s["time"] >= start) & (s["time"] < end)]
        sig_count += len(s)
        if s.empty:
            continue
        tr = rb.simulate(s, df, lambda d, _s=sym: rb.pay_for(table, _s, d),
                         rb.SPREAD.get(sym, 0.0))
        if tr.empty:
            continue
        tr["symbol"] = sym
        tr["time"] = df.index[tr["signal_bar"].values]
        tr = tr[(tr["time"] >= start) & (tr["time"] < end)]
        trades.append(tr)

    if not trades:
        print(f"signals: {sig_count}\ntrades : 0")
        return
    t = pd.concat(trades, ignore_index=True).sort_values("time")
    n = len(t)
    w = int(t["win"].sum())
    l = n - w
    pnl = float(t["pnl"].sum())
    payout = float(t["pnl"][t["win"]].mean()) if w else 0.0
    be = 1.0 / (1.0 + payout) * 100 if payout else 0.0

    print(f"signals generated : {sig_count}")
    print(f"trades executed   : {n}")
    print(f"wins / losses     : {w}W / {l}L  ({w / n * 100:.1f}%)")
    print(f"net PnL           : {pnl:+.2f} USD  (avg {pnl / n:+.3f}/trade)")
    print(f"avg payout        : {payout:.2f}  -> breakeven {be:.1f}%")

    print("\nper hour (UTC):")
    t["hour"] = t["time"].dt.strftime("%m-%d %H")
    for h, g in t.groupby("hour"):
        print(f"  {h}  {len(g):>2} trades  {int(g['win'].sum())}W/{len(g)-int(g['win'].sum())}L"
              f"  {g['pnl'].sum():+.2f}")

    print("\nper symbol:")
    for sym, g in t.groupby("symbol"):
        print(f"  {sym:<9}{len(g):>3} trades  {int(g['win'].sum())}W/"
              f"{len(g)-int(g['win'].sum())}L  {g['pnl'].sum():+.2f}")

    print("\ntrade list:")
    for r in t.itertuples():
        print(f"  {r.time:%m-%d %H:%M}  {r.symbol:<9}{r.direction:<6}"
              f"{r.duration:>2}m  {'WIN ' if r.win else 'LOSS'}  {r.pnl:+.2f}")


if __name__ == "__main__":
    main()
