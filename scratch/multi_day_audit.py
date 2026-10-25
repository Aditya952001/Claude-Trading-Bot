#!/usr/bin/env python3
"""
scratch/multi_day_audit.py

Comprehensive 3-Day Historical Strategy Audit (Oct 22, 23, 24, 25, 2026).
Evaluates bar-by-bar across 22 synthetic instruments.
Compares:
  A) Current Live Strategy (Dynamic duration)
  B) Calibrated High-Conviction (4-min expiry, ADX >= 25)
  C) Balanced Active (3-min expiry, ADX >= 22)
Reports Day-by-Day performance and Stake scaling ($1, $5, $10).
"""
import os
import sys
import pickle
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from strategy_engine import (
    compute_indicators, classify_regime, calculate_dynamic_duration,
    COMPONENTS, StrategyParams
)

IST_OFFSET = datetime.timedelta(hours=5, minutes=30)
TZ_IST = datetime.timezone(IST_OFFSET)

# 3-Day Window in IST:
# Start: 2026-10-22 00:00:00 IST
# End:   2026-10-25 09:00:00 IST
START_DT = datetime.datetime(2026, 10, 22, 0, 0, 0, tzinfo=TZ_IST)
END_DT = datetime.datetime(2026, 10, 25, 9, 0, 0, tzinfo=TZ_IST)

START_EPOCH = int(START_DT.timestamp())
END_EPOCH = int(END_DT.timestamp())

WIN_PAYOUT_PCT = 0.88

def run_multi_day_audit():
    print("=" * 80)
    print("      3-DAY DERIV HISTORICAL AUDIT (OCTOBER 22 - 25, 2026)")
    print(f"Time Window: {START_DT.strftime('%Y-%m-%d %H:%M IST')} -> {END_DT.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"Duration   : 77 Hours = 4,620 minutes")
    print("=" * 80)

    cfg = Config.from_env()

    with open("scratch/three_days_candles.pkl", "rb") as f:
        candles_dict = pickle.load(f)

    print(f"Loaded {len(candles_dict)} symbols. Computing technical indicators across 3+ days...")
    dfs = {}
    for sym, raw_df in candles_dict.items():
        try:
            # Sort and deduplicate
            df = raw_df.drop_duplicates(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
            df = compute_indicators(df, cfg.strategy)
            dfs[sym] = df
        except Exception as e:
            print(f"  Error on {sym}: {e}")

    print("Indicators ready! Running multi-configuration simulation...\n")

    configs = [
        {"name": "Current Engine (Dynamic Expiry, ADX >= 22)", "dur_mode": "dynamic", "min_adx": 22.0},
        {"name": "Calibrated Engine (4-Min Expiry, ADX >= 25)", "dur_mode": 4, "min_adx": 25.0},
        {"name": "Active Engine (3-Min Expiry, ADX >= 22)", "dur_mode": 3, "min_adx": 22.0},
    ]

    results_by_config = {}

    for c in configs:
        cname = c["name"]
        dur_mode = c["dur_mode"]
        min_adx = c["min_adx"]

        trades = []
        sym_open = {s: 0 for s in dfs}

        for epoch in range(START_EPOCH, END_EPOCH + 1, 60):
            dt_ist = datetime.datetime.fromtimestamp(epoch, TZ_IST)
            date_str = dt_ist.strftime("%Y-%m-%d")

            for sym, df in dfs.items():
                if epoch < sym_open[sym]:
                    continue

                matches = df.index[df['epoch'] == epoch]
                if len(matches) == 0:
                    continue
                i = matches[0]
                if i < 200:
                    continue

                last, prev = df.iloc[i], df.iloc[i - 1]
                adx_val = float(last["adx"])
                if adx_val < min_adx:
                    continue

                if any(tok in sym.upper() for tok in ("1HZ15V", "FRXUSDJPY", "FRXUSDCHF", "FRXXAUUSD", "FRXEURUSD")):
                    continue

                atr_val = max(1e-9, float(last["atr"]))
                macd_rel = float(last["macd_hist"]) / atr_val

                v = {k: 0 for k in COMPONENTS}
                if last["close"] > last["ema_higher_tf"] and last["ema_fast"] > last["ema_slow"]: v["trend"] = 1
                elif last["close"] < last["ema_higher_tf"] and last["ema_fast"] < last["ema_slow"]: v["trend"] = -1

                diff = (df["ema_fast"] - df["ema_slow"]).iloc[max(0, i - 4):i + 1]
                if len(diff) >= 2:
                    if diff.iloc[-1] > 0 and (diff.iloc[-1] >= diff.iloc[-2] or (diff.iloc[:-1].tail(3) <= 0).any()): v["cross"] = 1
                    elif diff.iloc[-1] < 0 and (diff.iloc[-1] <= diff.iloc[-2] or (diff.iloc[:-1].tail(3) >= 0).any()): v["cross"] = -1

                if 38.0 <= last["rsi"] <= 64.0 and last["rsi"] >= prev["rsi"]: v["rsi"] = 1
                elif 36.0 <= last["rsi"] <= 62.0 and last["rsi"] <= prev["rsi"]: v["rsi"] = -1

                if 0.15 <= last["bb_pctb"] <= 0.70 and last["close"] > last["open"]: v["bbands"] = 1
                elif 0.30 <= last["bb_pctb"] <= 0.85 and last["close"] < last["open"]: v["bbands"] = -1

                if last["macd_hist"] > 0: v["macd"] = 1
                elif last["macd_hist"] < 0: v["macd"] = -1

                score = sum(v.values())

                if "1HZ" in sym.upper():
                    max_bb_long = 0.74 if adx_val >= 28.0 else 0.68
                    max_rsi_long = 60.0 if adx_val >= 28.0 else 57.0
                elif adx_val >= 30.0: max_bb_long, max_rsi_long = 0.78, 62.0
                elif adx_val >= 25.0: max_bb_long, max_rsi_long = 0.74, 62.0
                else: max_bb_long, max_rsi_long = 0.70, 60.0

                if adx_val >= 30.0: min_bb_short, min_rsi_short = 0.18, 28.0
                elif adx_val >= 25.0: min_bb_short, min_rsi_short = 0.32, 33.0
                else: min_bb_short, min_rsi_short = 0.38, 35.0

                direction = None
                if (score >= 2.0 and last["ema_fast"] > last["ema_slow"]
                    and (last["ema_fast"] - last["ema_slow"]) >= 0.015 * atr_val
                    and last["close"] >= last["ema_fast"] and (last["close"] - last["ema_fast"]) <= 0.80 * atr_val
                    and last["close"] >= last["ema_trend"] and (last["close"] - last["ema_trend"]) <= 2.20 * atr_val
                    and last["close"] >= last["ema_higher_tf"]
                    and 33.0 <= last["rsi"] <= max_rsi_long
                    and 0.12 <= last["bb_pctb"] <= max_bb_long
                    and not (last["bb_pctb"] > 0.72 and last["rsi"] > 59.0)
                    and not (last["bb_pctb"] > 0.70 and macd_rel < 0.015)
                    and (last["macd_hist"] > 0 and macd_rel >= 0.008)
                    and last["close"] > last["open"]):
                    direction = "long"
                elif (score <= -2.0 and last["ema_fast"] < last["ema_slow"]
                    and (last["ema_slow"] - last["ema_fast"]) >= 0.015 * atr_val
                    and last["close"] <= last["ema_fast"] and (last["ema_fast"] - last["close"]) <= 0.80 * atr_val
                    and last["close"] <= last["ema_trend"] and (last["ema_trend"] - last["close"]) <= 2.20 * atr_val
                    and (adx_val >= 30.0 or last["close"] <= last["ema_higher_tf"])
                    and min_rsi_short <= last["rsi"] <= 65.0
                    and min_bb_short <= last["bb_pctb"] <= 0.85
                    and not (last["bb_pctb"] < (min_bb_short + 0.05) and last["rsi"] < 42.0)
                    and not (last["bb_pctb"] < 0.28 and last["rsi"] < 40.0)
                    and (last["macd_hist"] < 0 and macd_rel <= -0.008)
                    and last["close"] < last["open"]):
                    direction = "short"

                if not direction:
                    continue

                if dur_mode == "dynamic":
                    regime = classify_regime(last, cfg.strategy)
                    d, _ = calculate_dynamic_duration(sym, "synthetic", last, regime)
                else:
                    d = int(dur_mode)

                if i + d >= len(df):
                    continue

                exit_p = float(df.iloc[i + d]["close"])
                entry_p = float(last["close"])
                won = (exit_p > entry_p) if direction == "long" else (exit_p < entry_p)
                tie = (exit_p == entry_p)

                pnl_1 = 0.88 if won else (0.0 if tie else -1.0)
                pnl_5 = 4.40 if won else (0.0 if tie else -5.0)
                pnl_10 = 8.80 if won else (0.0 if tie else -10.0)

                sym_open[sym] = int(df.iloc[i + d]["epoch"])

                trades.append({
                    "date": date_str,
                    "epoch": epoch,
                    "symbol": sym,
                    "direction": direction,
                    "duration": d,
                    "won": won,
                    "pnl_1": pnl_1,
                    "pnl_5": pnl_5,
                    "pnl_10": pnl_10
                })

        results_by_config[cname] = pd.DataFrame(trades)

    # Print Comparative Summary
    print("=" * 80)
    print("               3-DAY OVERALL PERFORMANCE COMPARISON")
    print("=" * 80)
    for name, df_t in results_by_config.items():
        if df_t.empty:
            print(f"{name}: 0 trades")
            continue
        tot = len(df_t)
        wins = df_t["won"].sum()
        losses = tot - wins
        wr = (wins / tot * 100)
        pnl1 = df_t["pnl_1"].sum()
        pnl5 = df_t["pnl_5"].sum()
        pnl10 = df_t["pnl_10"].sum()

        print(f"\n▶ CONFIG: {name}")
        print(f"  Total Trades : {tot} (avg {tot/3.3:.1f} trades/day)")
        print(f"  Win / Loss   : {wins} Wins / {losses} Losses")
        print(f"  Win Rate     : {wr:.1f}%")
        print(f"  Net PnL ($1 Stake) : {'+' if pnl1>=0 else ''}${pnl1:.2f} USD")
        print(f"  Net PnL ($5 Stake) : {'+' if pnl5>=0 else ''}${pnl5:.2f} USD")
        print(f"  Net PnL ($10 Stake): {'+' if pnl10>=0 else ''}${pnl10:.2f} USD")

    # Print Day-by-Day Breakdown for Calibrated Engine
    print("\n" + "=" * 80)
    print("    DAY-BY-DAY BREAKDOWN FOR CALIBRATED ENGINE (4-Min Expiry, ADX >= 25)")
    print("=" * 80)
    df_cal = results_by_config["Calibrated Engine (4-Min Expiry, ADX >= 25)"]
    if not df_cal.empty:
        by_date = df_cal.groupby("date").agg(
            trades=("won", "count"),
            wins=("won", "sum"),
            losses=("won", lambda x: (~x).sum()),
            pnl_1=("pnl_1", "sum"),
            pnl_5=("pnl_5", "sum"),
            pnl_10=("pnl_10", "sum"),
        )
        by_date["win_rate"] = (by_date["wins"] / by_date["trades"] * 100).round(1).astype(str) + "%"
        cols = ["trades", "wins", "losses", "win_rate", "pnl_1", "pnl_5", "pnl_10"]
        print(by_date[cols].to_string())

if __name__ == "__main__":
    run_multi_day_audit()
