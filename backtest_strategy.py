#!/usr/bin/env python3
"""
backtest_strategy.py - Walk-forward test of strategy_engine.py on candle data.

Uses the SAME code path as live (StrategyEngine.scan == evaluate on each closed bar),
takes at most one trade per symbol at a time with a cooldown (like the live risk
manager), and judges outcomes honestly:

  rise_fall  : win only if the exit price is strictly beyond the entry (ties lose);
               profit = +payout on a win, -1 stake on a loss. Breakeven win rate is
               1/(1+payout), e.g. 54.1% at 0.85, 52.6% at 0.90.
  multiplier : stop = sl_mult*ATR, target = rr*stop, simulated bar by bar. If a bar
               touches both, the stop is assumed first (conservative). Results in R.

Examples
  python backtest_strategy.py --fetch frxEURUSD --asset forex --granularity 300 --count 5000 --save data/EURUSD_5m.csv
  python backtest_strategy.py --csv data/EURUSD_5m.csv --symbol frxEURUSD --asset forex --payout 0.88
  python backtest_strategy.py --csv data/R_75_5m.csv --symbol R_75 --asset synthetic --profile strict
  python backtest_strategy.py --csv data/XAU_5m.csv --symbol frxXAUUSD --asset commodities --mode multiplier
  python backtest_strategy.py ... --tune max_ext_fast_atr=1.0 min_clv_rise_fall=0.6

CSV columns: epoch (or time/timestamp/datetime), open, high, low, close.
Get enough data: 5000 candles of 5m is only ~17 days. A win rate means little below ~100
trades; the report prints how many trades you need to confirm the edge it measured.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses as dc
import math
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from strategy_engine import PROFILES, EntryTuning, StrategyEngine, StrategyParams

try:                                   # only needed for --fetch
    from config import Config, StrategyConfig
except ImportError:                    # pragma: no cover
    Config = StrategyConfig = None


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: str) -> None:
    """Write epoch SECONDS. Unit-independent: pandas 3 stores datetimes in seconds, older versions in ns."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    epoch = (df.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    out = df[["open", "high", "low", "close"]].copy()
    out.insert(0, "epoch", np.asarray(epoch, dtype="int64"))
    out.to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ("epoch", "time", "timestamp", "datetime", "date") if c in df.columns)
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        v = float(t.iloc[0])
        unit = "ns" if v > 1e17 else "us" if v > 1e14 else "ms" if v > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = idx
    if idx.nunique() < max(10, len(df) // 2):
        sys.exit(f"{path}: timestamps are collapsed/duplicated (saved by the old --save bug?). Re-fetch the data.")
    return out[~out.index.duplicated()].sort_index()


async def _fetch(symbol: str, granularity: int, count: int) -> pd.DataFrame:
    from deriv_client import DerivClient
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()
    try:
        all_candles: List[dict] = []
        end_val: Union[str, int] = "latest"
        remaining = count
        while remaining > 0:
            batch_size = min(remaining, 1000)   # Deriv limits candles to 1,000 per request
            raw = await client.get_candles(symbol, granularity, batch_size, end=end_val)
            if not raw:
                break
            all_candles = raw + all_candles
            remaining -= len(raw)
            earliest_dt = pd.to_datetime(raw[0]["epoch"], unit="s", utc=True)
            print(f"  [fetch] received {len(raw)} bars (accumulated {len(all_candles)}/{count}), earliest {earliest_dt}")
            if len(raw) < batch_size:
                break
            end_val = int(raw[0]["epoch"]) - 1
            if remaining > 0:
                await asyncio.sleep(0.2)
    finally:
        await client.close()
    df = pd.DataFrame(all_candles)
    if "epoch" in df.columns:
        df.index = pd.to_datetime(df.pop("epoch").astype(int), unit="s", utc=True)
    df = df[["open", "high", "low", "close"]].astype(float)
    return df[~df.index.duplicated()].sort_index()


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(engine: StrategyEngine, df: pd.DataFrame, symbol: str, asset: str, mode: str,
             params: Optional[StrategyParams] = None, payout: float = 0.85, sl_mult: float = 1.5,
             rr: float = 2.0, max_hold: int = 24, cooldown_bars: int = 3) -> pd.DataFrame:
    params = params or StrategyParams()
    bar = float(pd.Series(df.index).diff().dt.total_seconds().median())
    c, hi, lo = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    n, rows, next_free = len(df), [], 0
    for pos, sig in engine.scan(symbol, asset, df, params, contract_type=mode):
        if pos < next_free:
            continue                                   # a position would still be open / cooling down
        d = 1 if sig.direction == "long" else -1
        entry = c[pos]
        if mode == "rise_fall":
            h = max(1, math.ceil(sig.duration * 60 / bar))
            if pos + h >= n:
                continue
            move = d * (c[pos + h] - entry)
            win, pnl, hold = move > 0, (payout if move > 0 else -1.0), h
        else:
            stop = sl_mult * sig.atr
            sl, tp = entry - d * stop, entry + d * rr * stop
            r, hold = None, max_hold
            for k in range(1, max_hold + 1):
                j = pos + k
                if j >= n:
                    break
                hit_sl = lo[j] <= sl if d > 0 else hi[j] >= sl
                hit_tp = hi[j] >= tp if d > 0 else lo[j] <= tp
                if hit_sl or hit_tp:
                    if hit_sl:
                        # Account for gap/spike slippage beyond the stop
                        overshoot = (hi[j] - entry) / stop if d < 0 else (entry - lo[j]) / stop
                        r = -max(1.0, float(overshoot))
                    else:
                        r = rr
                    hold = k   # stop first if both touched
                    break
            if r is None:
                if pos + max_hold >= n:
                    continue
                r = d * (c[pos + max_hold] - entry) / stop
            win, pnl = r > 0, r
        rows.append({"pos": pos, "time": df.index[pos], "direction": sig.direction, "regime": sig.regime,
                     "duration": sig.duration, "score": sig.score, "win": bool(win), "pnl": float(pnl)})
        next_free = pos + hold + cooldown_bars
    return pd.DataFrame(rows)


def sweep_horizons(df: pd.DataFrame, trades: pd.DataFrame, horizons: List[int]) -> Dict[int, float]:
    """Win rate of the SAME entries if held for a fixed number of bars (picks the best expiry)."""
    c, out = df["close"].to_numpy(), {}
    for h in horizons:
        ok = trades[trades.pos + h < len(c)]
        mv = [(1 if r.direction == "long" else -1) * (c[r.pos + h] - c[r.pos]) for r in ok.itertuples()]
        out[h] = float(np.mean(np.array(mv) > 0)) if mv else float("nan")
    return out


# ----------------------------------------------------------------------------
# Statistics / report
# ----------------------------------------------------------------------------
def wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def trades_needed(wr: float, breakeven: float, z: float = 1.645) -> float:
    gap = wr - breakeven
    return float("inf") if gap <= 0 else (z ** 2) * breakeven * (1 - breakeven) / gap ** 2


def summarize(tr: pd.DataFrame, mode: str, payout: float) -> dict:
    n = len(tr)
    if n == 0:
        return {"n": 0}
    w = int(tr.win.sum())
    wr = w / n
    lo, hi = wilson(w, n)
    eq = tr.pnl.cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if n else 0.0
    gains, losses = tr.pnl[tr.pnl > 0].sum(), -tr.pnl[tr.pnl < 0].sum()
    be = 1 / (1 + payout) if mode == "rise_fall" else None
    return {"n": n, "wins": w, "wr": wr, "ci": (lo, hi), "expectancy": float(tr.pnl.mean()),
            "total": float(tr.pnl.sum()), "max_dd": dd, "pf": float(gains / losses) if losses > 0 else float("inf"),
            "breakeven": be, "z": ((wr - be) / math.sqrt(be * (1 - be) / n)) if be else None}


def _line(label: str, s: dict, mode: str) -> str:
    if s["n"] == 0:
        return f"  {label:<16} no trades"
    unit = "stakes" if mode == "rise_fall" else "R"
    base = (f"  {label:<16} n={s['n']:<4} win={s['wr']:.1%} [{s['ci'][0]:.1%}-{s['ci'][1]:.1%}]  "
            f"exp/trade={s['expectancy']:+.3f}{unit[0]}  total={s['total']:+.1f} {unit}  maxDD={s['max_dd']:.1f}  PF={s['pf']:.2f}")
    if s["breakeven"]:
        base += f"  (breakeven {s['breakeven']:.1%}, z={s['z']:+.2f})"
    return base


def report(engine: StrategyEngine, df: pd.DataFrame, tr: pd.DataFrame, mode: str, payout: float,
           split: float, horizons: List[int]) -> None:
    print(_line("ALL", summarize(tr, mode, payout), mode))
    if len(tr):
        cut = df.index[int(len(df) * split)]
        print(_line(f"in-sample", summarize(tr[tr.time < cut], mode, payout), mode))
        print(_line(f"out-of-sample", summarize(tr[tr.time >= cut], mode, payout), mode))
        for col in ("direction", "regime"):
            for k, g in tr.groupby(col):
                print(_line(f"{col}={k}", summarize(g, mode, payout), mode))
        s = summarize(tr, mode, payout)
        if mode == "rise_fall":
            need = trades_needed(s["wr"], s["breakeven"])
            print(f"\n  Trades needed to confirm this win rate beats breakeven at 95% confidence: "
                  f"{'never (it is below breakeven)' if math.isinf(need) else f'~{need:,.0f}'}  (you have {s['n']})")
            if horizons:
                sw = sweep_horizons(df, tr, horizons)
                print("  Win rate if every entry were held N bars: " + "  ".join(f"{h}b={v:.1%}" for h, v in sw.items()))
    funnel = engine.rejects.most_common(10)
    print("\n  Why bars were rejected (funnel): " + ", ".join(f"{k}={v}" for k, v in funnel))


# ----------------------------------------------------------------------------
def _parse_tune(items: List[str]) -> Dict[str, object]:
    fields = {f.name: f for f in dc.fields(EntryTuning)}
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in fields:
            sys.exit(f"unknown tuning field '{k}'. Valid: {', '.join(fields)}")
        t = type(getattr(EntryTuning(), k))
        out[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--fetch", metavar="SYMBOL", help="download candles with your Deriv token (.env)")
    ap.add_argument("--symbol")
    ap.add_argument("--asset", default="synthetic", choices=["synthetic", "forex", "commodities", "crypto"])
    ap.add_argument("--mode", default="rise_fall", choices=["rise_fall", "multiplier"])
    ap.add_argument("--granularity", type=int, default=300)
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--save")
    ap.add_argument("--payout", type=float, default=0.85, help="net payout on a win, e.g. 0.85")
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    ap.add_argument("--tune", nargs="*", help="override EntryTuning fields, e.g. max_ext_fast_atr=1.0")
    ap.add_argument("--split", type=float, default=0.6)
    ap.add_argument("--sl-mult", type=float, default=1.5)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--cooldown", type=int, default=3)
    ap.add_argument("--horizons", default="1,2,3,6,12")
    a = ap.parse_args()

    if a.fetch:
        df, symbol = asyncio.run(_fetch(a.fetch, a.granularity, a.count)), a.fetch
        if a.save:
            save_csv(df, a.save)
    else:
        df, symbol = load_csv(a.csv), a.symbol or a.csv
    if a.symbol:
        symbol = a.symbol

    tuning = dc.replace(PROFILES[a.profile], **_parse_tune(a.tune))
    scfg = StrategyConfig() if StrategyConfig else None
    engine = StrategyEngine(scfg, tuning)
    print(f"{symbol} | {a.asset} | {a.mode} | profile={a.profile} | {len(df)} bars "
          f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    tr = simulate(engine, df, symbol, a.asset, a.mode, payout=a.payout, sl_mult=a.sl_mult, rr=a.rr,
                  max_hold=a.max_hold, cooldown_bars=a.cooldown)
    report(engine, df, tr, a.mode, a.payout, a.split, [int(x) for x in a.horizons.split(",") if x])


if __name__ == "__main__":
    main()
