"""
research/live_scan.py - Read-only "why is the bot silent?" diagnostic.

Fetches fresh 1m candles for every symbol the Rise/Fall bot actually streams,
runs the real StrategyEngine on them, and prints the per-symbol indicator state
plus the FIRST gate that blocks an entry. Nothing here can place an order: it
only sends `ticks_history` / `active_symbols` requests to the demo account.

The gate list below mirrors StrategyEngine.evaluate() (v10). To catch drift the
script also runs the real evaluate() and flags any disagreement, so a stale
diagnostic can never quietly tell you the wrong reason.

Usage:
    .venv/bin/python research/live_scan.py
    .venv/bin/python research/live_scan.py --bars 300 --symbols R_75,1HZ100V
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from config import AssetClass, Config                      # noqa: E402
from strategy_engine import (StrategyEngine, StrategyParams,  # noqa: E402
                             compute_indicators, classify_regime)
from deriv_data import CandleClient                        # noqa: E402


def load_params(db: str = "data/bot_memory.sqlite3") -> StrategyParams:
    """Read the live strategy_params blob the running bot is using."""
    import sqlite3
    con = sqlite3.connect(ROOT / db)
    try:
        row = con.execute("SELECT value FROM kv WHERE key='strategy_params'").fetchone()
    finally:
        con.close()
    if not row or not row[0]:
        return StrategyParams()
    raw = json.loads(row[0])
    return StrategyParams(**{k: v for k, v in raw.items()
                             if k in StrategyParams.__dataclass_fields__})


def _checks(symbol: str, last: pd.Series, prev: pd.Series, df: pd.DataFrame,
            regime: str, params: StrategyParams, score: float, adx: float,
            atr: float, macd_rel: float):
    """Return (long_failures, short_failures, context) mirroring evaluate()."""
    req = max(2.0, min(4.5, float(params.min_score)))

    if "1HZ" in symbol.upper():
        max_bb_long = 0.74 if adx >= 28.0 else 0.68
        max_rsi_long = 60.0 if adx >= 28.0 else 57.0
    elif adx >= 30.0:
        max_bb_long, max_rsi_long = 0.78, 63.0
    elif adx >= 25.0:
        max_bb_long, max_rsi_long = 0.74, 62.0
    else:
        max_bb_long, max_rsi_long = 0.70, 60.0
    max_rsi_long = min(max_rsi_long, float(params.rsi_long_max))

    if adx >= 30.0:
        min_bb_short, min_rsi_short = 0.18, 28.0
    elif adx >= 25.0:
        min_bb_short, min_rsi_short = 0.32, 33.0
    else:
        min_bb_short, min_rsi_short = 0.38, 35.0
    min_rsi_short = max(min_rsi_short, float(params.rsi_short_min))

    c = float(last["close"])
    long_fail = []
    if score < req:
        long_fail.append(f"score {score:+.1f} < {req:.1f}")
    if not last["ema_fast"] > last["ema_slow"]:
        long_fail.append("ema9<=ema21")
    if not (last["ema_fast"] - last["ema_slow"]) >= 0.015 * atr:
        long_fail.append("ema spread <0.015 ATR")
    if not c >= last["ema_fast"]:
        long_fail.append("close<ema9")
    if not (c - last["ema_fast"]) <= 0.80 * atr:
        long_fail.append("close >0.80 ATR above ema9")
    if not c >= last["ema_trend"]:
        long_fail.append("close<ema50")
    if not (c - last["ema_trend"]) <= 2.20 * atr:
        long_fail.append("close >2.20 ATR above ema50")
    if not (adx >= 25.0 or c >= last["ema_higher_tf"]):
        long_fail.append("close<ema200 (adx<25)")
    if not last["rsi"] <= max_rsi_long:
        long_fail.append(f"rsi {last['rsi']:.1f} > {max_rsi_long:.0f}")
    if not 33.0 <= last["rsi"]:
        long_fail.append(f"rsi {last['rsi']:.1f} < 33")
    if not 0.12 <= last["bb_pctb"] <= max_bb_long:
        long_fail.append(f"bb {last['bb_pctb']:.2f} outside [0.12,{max_bb_long:.2f}]")
    if last["bb_pctb"] > 0.70 and last["rsi"] > 60.0:
        long_fail.append("bb>0.70 & rsi>60 (extended)")
    if not (last["macd_hist"] > 0 and macd_rel >= 0.003):
        long_fail.append(f"macd_rel {macd_rel:+.4f} < 0.003")
    if not c > last["open"]:
        long_fail.append("bearish candle")

    short_fail = []
    if score > -req:
        short_fail.append(f"score {score:+.1f} > -{req:.1f}")
    if not last["ema_fast"] < last["ema_slow"]:
        short_fail.append("ema9>=ema21")
    if not (last["ema_slow"] - last["ema_fast"]) >= 0.015 * atr:
        short_fail.append("ema spread <0.015 ATR")
    if not c <= last["ema_fast"]:
        short_fail.append("close>ema9")
    if not (last["ema_fast"] - c) <= 0.80 * atr:
        short_fail.append("close >0.80 ATR below ema9")
    if not c <= last["ema_trend"]:
        short_fail.append("close>ema50")
    if not (last["ema_trend"] - c) <= 2.20 * atr:
        short_fail.append("close >2.20 ATR below ema50")
    if not (adx >= 25.0 or c <= last["ema_higher_tf"]):
        short_fail.append("close>ema200 (adx<25)")
    if not min_rsi_short <= last["rsi"] <= 65.0:
        short_fail.append(f"rsi {last['rsi']:.1f} outside [{min_rsi_short:.0f},65]")
    if not min_bb_short <= last["bb_pctb"] <= 0.85:
        short_fail.append(f"bb {last['bb_pctb']:.2f} outside [{min_bb_short:.2f},0.85]")
    if last["bb_pctb"] < (min_bb_short + 0.05) and last["rsi"] < 42.0:
        short_fail.append("floor-bounce trap")
    if not (last["macd_hist"] < 0 and macd_rel <= -0.003):
        short_fail.append(f"macd_rel {macd_rel:+.4f} > -0.003")
    if not c < last["open"]:
        short_fail.append("bullish candle")

    ctx = {"req_score": req, "max_bb_long": max_bb_long, "max_rsi_long": max_rsi_long,
           "min_bb_short": min_bb_short, "min_rsi_short": min_rsi_short}
    return long_fail, short_fail, ctx


def diagnose(symbol: str, asset_class: str, candles: pd.DataFrame,
             cfg: Config, params: StrategyParams) -> dict:
    """Run the real engine plus a gate-by-gate explanation of a rejection."""
    eng = StrategyEngine(cfg.strategy)
    sig = eng.evaluate(symbol, asset_class, candles, params)

    out = {"symbol": symbol, "signal": sig.direction if sig else None, "reason": "",
           "score": None, "adx": None, "rsi": None, "bb": None, "macd_rel": None,
           "regime": None, "atr_pctile": None, "long_fail": [], "short_fail": [],
           "mismatch": False}

    if len(candles) < cfg.strategy.min_candles:
        out["reason"] = f"only {len(candles)} candles (<{cfg.strategy.min_candles})"
        return out

    df = compute_indicators(candles, cfg.strategy)
    last, prev = df.iloc[-1], df.iloc[-2]
    needed = ["ema_fast", "ema_slow", "ema_trend", "ema_higher_tf", "rsi", "atr",
              "bb_pctb", "adx", "macd_hist", "atr_pctile"]
    if last[needed].isna().any() or last["atr"] <= 0:
        out["reason"] = "indicators not ready (NaN/ATR=0)"
        return out

    regime = classify_regime(last, cfg.strategy)
    adx = float(last["adx"])
    atr = max(1e-9, float(last["atr"]))
    macd_rel = float(last["macd_hist"]) / atr
    votes = eng._votes(df, last, prev, regime, params)
    w = params.component_weights
    score = sum(w.get(k, 1.0) * v for k, v in votes.items())

    out.update(score=score, adx=adx, rsi=float(last["rsi"]),
               bb=float(last["bb_pctb"]), macd_rel=macd_rel, regime=regime,
               atr_pctile=float(last["atr_pctile"]),
               close=float(last["close"]), open=float(last["open"]),
               ema_fast=float(last["ema_fast"]), ema_slow=float(last["ema_slow"]),
               ema_trend=float(last["ema_trend"]),
               ema_higher=float(last["ema_higher_tf"]), votes=votes)

    # Pre-checks, in evaluate()'s order.
    if regime == "high_vol" and not cfg.strategy.trade_high_vol:
        out["reason"] = "high_vol regime blocked"
        return out
    if params.is_blocked(asset_class, regime):
        out["reason"] = f"learner blocked {asset_class}|{regime}"
        return out
    if adx < 20.0:
        out["reason"] = f"adx {adx:.1f} < 20"
        return out
    if "JD" in symbol.upper() and adx < 22.0:
        out["reason"] = f"adx {adx:.1f} < 22 (JD)"
        return out
    if "1HZ" in symbol.upper() and adx < 22.0:
        out["reason"] = f"adx {adx:.1f} < 22 (1HZ)"
        return out

    long_fail, short_fail, ctx = _checks(symbol, last, prev, df, regime, params,
                                         score, adx, atr, macd_rel)
    out["long_fail"], out["short_fail"] = long_fail, short_fail
    out["ctx"] = ctx

    if sig is not None:
        out["reason"] = "SIGNAL"
    else:
        # Closest side = fewest failed conditions.
        if len(long_fail) <= len(short_fail):
            out["reason"] = f"long blocked by: {', '.join(long_fail) or '?'}"
        else:
            out["reason"] = f"short blocked by: {', '.join(short_fail) or '?'}"

    # Drift check: the mirror must agree with the real engine.
    mirror_dir = None
    if not long_fail:
        mirror_dir = "long"
    elif not short_fail:
        mirror_dir = "short"
    if mirror_dir != out["signal"]:
        out["mismatch"] = True

    return out


async def fetch(client: CandleClient, symbol: str, bars: int) -> pd.DataFrame:
    try:
        return await client.candles(symbol, bars)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {symbol}: fetch failed: {exc}")
        return pd.DataFrame()


async def rf_universe(client: CandleClient, cfg: Config) -> list[str]:
    """The exact symbols main._validate_symbols() keeps for RISE_FALL mode.

    BOOM/CRASH/forex look available in `active_symbols` but have no CALL/PUT
    contracts, so the bot never streams them. Scanning them would report gates
    on instruments that cannot produce a trade.
    """
    open_now, suspended = set(), set()
    try:
        res = await client.request({"active_symbols": "brief"})
        for s in res.get("active_symbols", []):
            sym = s.get("underlying_symbol", s.get("symbol"))
            if s.get("exchange_is_open") == 0 or s.get("is_trading_suspended") == 1:
                suspended.add(sym)
            else:
                open_now.add(sym)
    except Exception as exc:  # noqa: BLE001
        print(f"(active_symbols unavailable: {exc})")

    kept = []
    for sc in cfg.symbols:
        if not (sc.enabled and sc.supports_rise_fall):
            continue
        if sc.asset_class.value not in cfg.rf_allowed_assets:
            continue
        if open_now and sc.symbol not in open_now:
            continue
        try:
            await asyncio.sleep(0.15)
            info = await client.request({"contracts_for": sc.symbol})
            avail = (info.get("contracts_for") or {}).get("available", [])
            if not any(c.get("contract_category") == "callput"
                       and c.get("contract_type") in ("CALL", "PUT") for c in avail):
                continue
        except Exception:  # noqa: BLE001
            pass  # keep it, same fallback the bot uses
        kept.append(sc.symbol)
    return kept


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=260)
    ap.add_argument("--symbols", default="", help="comma list; default = bot's RF universe")
    ap.add_argument("--db", default="data/bot_memory.sqlite3")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cfg = Config.from_env()
    params = load_params(args.db)

    if args.symbols:
        wanted = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        wanted = None

    client = await CandleClient.connect(cfg.deriv.environment.value)
    try:
        if wanted is None:
            wanted = await rf_universe(client, cfg)

        print(f"params v{params.version}: min_score={params.min_score} "
              f"rsi_long_max={params.rsi_long_max} rsi_short_min={params.rsi_short_min}")
        print(f"scanning {len(wanted)} symbols x {args.bars} 1m candles "
              f"({cfg.deriv.environment.value})")
        print("universe: " + ", ".join(wanted) + "\n")

        # CandleClient is a single-socket request/response loop: one request at a
        # time, otherwise the recv() calls collide.
        frames = {}
        for sym in wanted:
            frames[sym] = await fetch(client, sym, args.bars)
    finally:
        await client.close()

    rows = []
    for sc in cfg.symbols:
        if sc.symbol not in frames:
            continue
        df = frames[sc.symbol]
        if df.empty:
            continue
        rows.append(diagnose(sc.symbol, sc.asset_class.value, df, cfg, params))

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    rows.sort(key=lambda r: (r["signal"] is None, -(r["adx"] or 0)))
    hdr = (f"{'symbol':<10}{'adx':>6}{'rsi':>6}{'bb':>6}{'macd':>9}{'score':>7}"
           f"  {'regime':<9} verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["adx"] is None:
            print(f"{r['symbol']:<10}{'-':>6}{'-':>6}{'-':>6}{'-':>9}{'-':>7}  {'-':<9} {r['reason']}")
            continue
        verdict = "SIGNAL " + r["signal"].upper() if r["signal"] else r["reason"]
        print(f"{r['symbol']:<10}{r['adx']:>6.1f}{r['rsi']:>6.1f}{r['bb']:>6.2f}"
              f"{r['macd_rel']:>+9.4f}{r['score']:>+7.1f}  {r['regime']:<9} {verdict}")

    # Aggregate the blocking gates so the dominant reason is obvious.
    from collections import Counter
    gates = Counter()
    for r in rows:
        if r["signal"]:
            gates["SIGNAL"] += 1
            continue
        fails = r["long_fail"] if len(r["long_fail"]) <= len(r["short_fail"]) else r["short_fail"]
        if not fails:
            gates[r["reason"]] += 1
            continue
        # Count only the first failing condition (the binding gate).
        gates[fails[0].split(" ")[0]] += 1

    print(f"\n{len(rows)} symbols | {sum(1 for r in rows if r['signal'])} signals now")
    print("binding gate (first failure, closest side):")
    for gate, n in gates.most_common(8):
        print(f"  {n:>3}x  {gate}")

    mism = [r["symbol"] for r in rows if r["mismatch"]]
    if mism:
        print(f"\n!! diagnostic drift vs real engine on: {', '.join(mism)}")
        return 1
    print("\nmirror check: agrees with StrategyEngine.evaluate() on all symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
