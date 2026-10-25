#!/usr/bin/env python3
"""
scratch/fast_replay.py

Fast, vectorized precomputed replay for:
October 25, 2026: 12:00 AM IST (00:00) to 09:00 AM IST (09:00).
Replays the exact High-Conviction Strategy rules across all active instruments.
"""
import os
import sys
import pickle
import datetime
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from strategy_engine import (
    compute_indicators, classify_regime, calculate_dynamic_duration,
    COMPONENTS, StrategyParams, Signal
)
from memory_and_learning import MemoryStore

IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
TZ_IST = datetime.timezone(IST_OFFSET)

START_TIME_IST = datetime.datetime(2026, 10, 25, 0, 0, 0, tzinfo=TZ_IST)
END_TIME_IST = datetime.datetime(2026, 10, 25, 9, 0, 0, tzinfo=TZ_IST)

START_EPOCH = int(START_TIME_IST.timestamp())
END_EPOCH = int(END_TIME_IST.timestamp())

WIN_PAYOUT = 0.88   # Standard Rise/Fall payout for 88% return on $1.00 stake
LOSS_COST = 1.00    # $1.00 stake lost if out of the money

def run_fast_replay():
    print("=" * 75)
    print("DERIV RISE/FALL STRATEGY AUDIT & REPLAY (HIGH-SPEED VECTORIZED)")
    print(f"Time Range : {START_TIME_IST.strftime('%Y-%m-%d %I:%M %p IST')} to {END_TIME_IST.strftime('%Y-%m-%d %I:%M %p IST')}")
    print(f"Epoch Window: {START_EPOCH} to {END_EPOCH} (540 1-minute bars)")
    print("=" * 75)

    cfg = Config.from_env()
    store = MemoryStore(cfg.learning.db_path)
    raw_params = store.get_kv("strategy_params")
    params = StrategyParams.from_json(raw_params) if raw_params else StrategyParams()

    cache_file = "scratch/cached_replay_candles.pkl"
    if not os.path.exists(cache_file):
        print(f"Error: {cache_file} not found!")
        return

    with open(cache_file, "rb") as f:
        symbol_candles = pickle.load(f)

    print(f"Loaded {len(symbol_candles)} symbols from cache.")
    print("Pre-computing technical indicators for all symbols...")

    computed_dfs = {}
    for sym, raw_df in symbol_candles.items():
        try:
            df = compute_indicators(raw_df, cfg.strategy)
            computed_dfs[sym] = df
        except Exception as e:
            print(f"Error computing indicators for {sym}: {e}")

    print("Indicators ready! Running minute-by-minute evaluation across all instruments...\n")

    all_trades = []
    symbol_open_until = {s: 0 for s in symbol_candles}
    bar_evaluations = 0

    rejections = {
        "ADX < 22 (Chop / Sideways)": 0,
        "Overhead 200 EMA Resistance (Longs)": 0,
        "Bollinger / RSI Overbought Top (Ceiling)": 0,
        "Floor Bounce / Oversold Trap (Shorts)": 0,
        "Low Momentum / Score < Threshold": 0,
        "MACD Rel Momentum Insufficient": 0,
    }

    # Iterate minute by minute through the 9 hours
    for epoch in range(START_EPOCH, END_EPOCH + 1, 60):
        dt_ist = datetime.datetime.fromtimestamp(epoch, TZ_IST)

        for sym, df in computed_dfs.items():
            if epoch < symbol_open_until[sym]:
                continue  # already in a trade on this symbol

            idx_matches = df.index[df['epoch'] == epoch]
            if len(idx_matches) == 0:
                continue
            i = idx_matches[0]

            if i < cfg.strategy.min_candles:
                continue

            bar_evaluations += 1
            last = df.iloc[i]
            prev = df.iloc[i - 1]

            needed = ["ema_fast", "ema_slow", "ema_trend", "ema_higher_tf", "rsi", "atr", "bb_pctb",
                      "adx", "macd_hist", "atr_pctile"]
            if last[needed].isna().any() or last["atr"] <= 0:
                continue

            regime = classify_regime(last, cfg.strategy)
            if regime == "high_vol" and not cfg.strategy.trade_high_vol:
                continue
            if params.is_blocked("synthetic", regime):
                continue

            # Votes
            v = {k: 0 for k in COMPONENTS}
            close, rsi, pctb = last["close"], last["rsi"], last["bb_pctb"]
            prev_rsi = prev["rsi"]

            if close > last["ema_higher_tf"] and last["ema_fast"] > last["ema_slow"]:
                v["trend"] = 1
            elif close < last["ema_higher_tf"] and last["ema_fast"] < last["ema_slow"]:
                v["trend"] = -1

            diff = (df["ema_fast"] - df["ema_slow"]).iloc[max(0, i - 4):i + 1]
            if len(diff) >= 2:
                if diff.iloc[-1] > 0 and (diff.iloc[-1] >= diff.iloc[-2] or (diff.iloc[:-1].tail(3) <= 0).any()):
                    v["cross"] = 1
                elif diff.iloc[-1] < 0 and (diff.iloc[-1] <= diff.iloc[-2] or (diff.iloc[:-1].tail(3) >= 0).any()):
                    v["cross"] = -1

            if 38.0 <= rsi <= 64.0 and rsi >= prev_rsi:
                v["rsi"] = 1
            elif 36.0 <= rsi <= 62.0 and rsi <= prev_rsi:
                v["rsi"] = -1

            if 0.15 <= pctb <= 0.70 and last["close"] > last["open"]:
                v["bbands"] = 1
            elif 0.30 <= pctb <= 0.85 and last["close"] < last["open"]:
                v["bbands"] = -1

            h = last["macd_hist"]
            if h > 0:
                v["macd"] = 1
            elif h < 0:
                v["macd"] = -1

            w = params.component_weights
            score = sum(w.get(k, 1.0) * val for k, val in v.items())
            denom = sum(w.get(k, 1.0) for k in v) or 1.0

            adx_val = float(last["adx"])
            if adx_val < 22.0:
                rejections["ADX < 22 (Chop / Sideways)"] += 1
                continue

            # Toxic RF filter
            if any(tok in sym.upper() for tok in ("1HZ15V", "FRXUSDJPY", "FRXUSDCHF", "FRXXAUUSD", "FRXEURUSD")):
                continue

            atr_val = max(1e-9, float(last["atr"]))
            macd_rel = float(last["macd_hist"]) / atr_val
            req_score = max(2.0, min(4.5, float(params.min_score)))

            # Dynamic BB & RSI bounds
            if "1HZ" in sym.upper():
                max_bb_long = 0.74 if adx_val >= 28.0 else 0.68
                max_rsi_long = 60.0 if adx_val >= 28.0 else 57.0
            elif adx_val >= 30.0:
                max_bb_long = 0.78
                max_rsi_long = 62.0
            elif adx_val >= 25.0:
                max_bb_long = 0.74
                max_rsi_long = 62.0
            else:
                max_bb_long = 0.70
                max_rsi_long = 60.0

            if adx_val >= 30.0:
                min_bb_short = 0.18
                min_rsi_short = 28.0
            elif adx_val >= 25.0:
                min_bb_short = 0.32
                min_rsi_short = 33.0
            else:
                min_bb_short = 0.38
                min_rsi_short = 35.0

            max_rsi_long = min(max_rsi_long, float(params.rsi_long_max))
            min_rsi_short = max(min_rsi_short, float(params.rsi_short_min))

            direction = None

            # LONG Check
            is_long = (
                score >= req_score
                and last["ema_fast"] > last["ema_slow"]
                and (last["ema_fast"] - last["ema_slow"]) >= 0.015 * atr_val
                and last["close"] >= last["ema_fast"]
                and (last["close"] - last["ema_fast"]) <= 0.80 * atr_val
                and last["close"] >= last["ema_trend"]
                and (last["close"] - last["ema_trend"]) <= 2.20 * atr_val
                and last["close"] >= last["ema_higher_tf"]
                and last["rsi"] <= max_rsi_long
                and 33.0 <= last["rsi"]
                and 0.12 <= last["bb_pctb"] <= max_bb_long
                and not (last["bb_pctb"] > 0.72 and last["rsi"] > 59.0)
                and not (last["bb_pctb"] > 0.70 and macd_rel < 0.015)
                and (last["macd_hist"] > 0 and macd_rel >= 0.008)
                and last["close"] > last["open"]
            )
            if is_long:
                direction = "long"

            # SHORT Check
            is_short = (
                score <= -req_score
                and last["ema_fast"] < last["ema_slow"]
                and (last["ema_slow"] - last["ema_fast"]) >= 0.015 * atr_val
                and last["close"] <= last["ema_fast"]
                and (last["ema_fast"] - last["close"]) <= 0.80 * atr_val
                and last["close"] <= last["ema_trend"]
                and (last["ema_trend"] - last["close"]) <= 2.20 * atr_val
                and (adx_val >= 30.0 or last["close"] <= last["ema_higher_tf"])
                and min_rsi_short <= last["rsi"] <= 65.0
                and min_bb_short <= last["bb_pctb"] <= 0.85
                and not (last["bb_pctb"] < (min_bb_short + 0.05) and last["rsi"] < 42.0)
                and not (last["bb_pctb"] < 0.28 and last["rsi"] < 40.0)
                and (last["macd_hist"] < 0 and macd_rel <= -0.008)
                and last["close"] < last["open"]
            )
            if is_short:
                direction = "short"

            if direction is None:
                # Track why it was rejected
                if score >= req_score and last["close"] < last["ema_higher_tf"]:
                    rejections["Overhead 200 EMA Resistance (Longs)"] += 1
                elif score >= req_score and (last["bb_pctb"] > max_bb_long or last["rsi"] > max_rsi_long):
                    rejections["Bollinger / RSI Overbought Top (Ceiling)"] += 1
                elif score <= -req_score and (last["bb_pctb"] < min_bb_short or last["rsi"] < min_rsi_short):
                    rejections["Floor Bounce / Oversold Trap (Shorts)"] += 1
                elif abs(score) < req_score:
                    rejections["Low Momentum / Score < Threshold"] += 1
                elif abs(macd_rel) < 0.008:
                    rejections["MACD Rel Momentum Insufficient"] += 1
                continue

            # Signal triggered!
            dyn_dur, dyn_reason = calculate_dynamic_duration(sym, "synthetic", last, regime)
            entry_price = float(last["close"])
            exit_idx = i + dyn_dur

            if exit_idx >= len(df):
                continue

            exit_price = float(df.iloc[exit_idx]["close"])
            exit_epoch = int(df.iloc[exit_idx]["epoch"])
            exit_time_ist = datetime.datetime.fromtimestamp(exit_epoch, TZ_IST)

            if direction == "long":
                won = exit_price > entry_price
            else:
                won = exit_price < entry_price

            tie = (exit_price == entry_price)

            if tie:
                pnl = 0.0
                outcome = "TIE"
            elif won:
                pnl = WIN_PAYOUT
                outcome = "WON"
            else:
                pnl = -LOSS_COST
                outcome = "LOST"

            all_trades.append({
                "time_ist": dt_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time_ist": exit_time_ist.strftime("%H:%M:%S"),
                "symbol": sym,
                "direction": "RISE (CALL)" if direction == "long" else "FALL (PUT)",
                "duration_min": dyn_dur,
                "score": f"{score:+.1f}",
                "adx": f"{adx_val:.1f}",
                "rsi": f"{float(last['rsi']):.1f}",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "outcome": outcome,
                "pnl": pnl,
                "reason": dyn_reason
            })
            symbol_open_until[sym] = exit_epoch

    print("=" * 75)
    print("                    FINAL AUDIT RESULTS")
    print("=" * 75)
    print(f"Total Bar Checks Evaluated: {bar_evaluations} across {len(computed_dfs)} symbols")

    print("\n--- Market Filter Rejection Breakdown ---")
    for reason, count in sorted(rejections.items(), key=lambda x: -x[1]):
        pct = (count / bar_evaluations * 100) if bar_evaluations > 0 else 0
        print(f"  * {reason:<45}: {count:>5} bars ({pct:>5.1f}%)")

    if not all_trades:
        print("\nSUMMARY: 0 Trades executed during this window (12:00 AM to 9:00 AM IST).")
        print("Total PnL: $0.00 USD")
        return

    df_res = pd.DataFrame(all_trades)
    wins = len(df_res[df_res["outcome"] == "WON"])
    losses = len(df_res[df_res["outcome"] == "LOST"])
    ties = len(df_res[df_res["outcome"] == "TIE"])
    tot = len(df_res)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
    tot_pnl = df_res["pnl"].sum()

    print(f"\nTotal Trades Triggered : {tot}")
    print(f"Wins                   : {wins}")
    print(f"Losses                 : {losses}")
    print(f"Ties                   : {ties}")
    print(f"Win Rate               : {win_rate:.1f}%")
    print(f"Total Profit / Loss    : {'+' if tot_pnl >= 0 else ''}${tot_pnl:.2f} USD")

    print("\n--- Individual Executed Trades ---")
    cols = ["time_ist", "symbol", "direction", "duration_min", "entry_price", "exit_price", "outcome", "pnl", "adx", "rsi"]
    print(df_res[cols].to_string(index=False))

    print("\n--- Performance by Symbol ---")
    by_sym = df_res.groupby("symbol").agg(
        trades=("outcome", "count"),
        wins=("outcome", lambda x: (x == "WON").sum()),
        losses=("outcome", lambda x: (x == "LOST").sum()),
        pnl=("pnl", "sum")
    )
    by_sym["win_rate"] = (by_sym["wins"] / by_sym["trades"] * 100).round(1).astype(str) + "%"
    print(by_sym.to_string())

if __name__ == "__main__":
    run_fast_replay()
