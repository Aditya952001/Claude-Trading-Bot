"""
research/fill_gap.py - Build a continuous 7-day candle set.

The main cache (data/candles/*_60.csv) ends around 2026-10-24 05:2x UTC; the
fresh cache (data/candles_recent/) starts around 2026-10-24 15:4x UTC, leaving
a ~10h hole. This script fetches the missing recent bars from Deriv (read-only
`ticks_history`), merges them with the main cache, and writes the result to
data/candles_7d/ so nothing that already uses data/candles/ is disturbed.

Usage:
  .venv/bin/python research/fill_gap.py --bars 2000
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

from deriv_data import CandleClient  # noqa: E402
from v11_ab import UNIVERSE  # noqa: E402

SRC = ROOT / "data" / "candles"
DST = ROOT / "data" / "candles_7d"


def merge(sym: str, fresh: pd.DataFrame) -> pd.DataFrame:
    p = SRC / f"{sym}_60.csv"
    if not p.exists():
        return fresh
    old = pd.read_csv(p, parse_dates=["time"], index_col="time")[["open", "high", "low", "close"]].astype(float)
    both = pd.concat([old, fresh])
    both = both[~both.index.duplicated(keep="last")].sort_index()
    return both


async def run(symbols: list[str], bars: int) -> None:
    DST.mkdir(parents=True, exist_ok=True)
    client = await CandleClient.connect()
    ok = 0
    try:
        for i, sym in enumerate(symbols, 1):
            try:
                fresh = await client.candles(sym, bars)
                if fresh.empty:
                    print(f"  {sym:<10} EMPTY", flush=True)
                    continue
                out = merge(sym, fresh)
                out.to_csv(DST / f"{sym}_60.csv", index_label="time")
                gaps = (out.index.to_series().diff().dt.total_seconds().fillna(60) > 60).sum()
                print(f"  [{i}/{len(symbols)}] {sym:<10} fresh={len(fresh):>5} "
                      f"total={len(out):>6} {out.index[0]:%m-%d %H:%M} -> "
                      f"{out.index[-1]:%m-%d %H:%M} gaps={gaps}", flush=True)
                ok += 1
            except Exception as exc:
                print(f"  {sym:<10} FAIL {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(0.25)
    finally:
        await client.close()
    print(f"\nwrote {ok}/{len(symbols)} symbols to {DST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()] or UNIVERSE
    asyncio.run(run(syms, a.bars))
