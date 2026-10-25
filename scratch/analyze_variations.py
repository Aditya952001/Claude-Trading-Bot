import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle, pandas as pd
from config import Config
from strategy_engine import compute_indicators, classify_regime, calculate_dynamic_duration, COMPONENTS, StrategyParams

with open('scratch/cached_replay_candles.pkl', 'rb') as f:
    candles = pickle.load(f)

cfg = Config.from_env()
dfs = {sym: compute_indicators(df, cfg.strategy) for sym, df in candles.items()}

from scratch.fast_replay import START_EPOCH, END_EPOCH

print(f"{'Mode':<15} | {'Min ADX':<8} | {'Trades':<7} | {'Wins':<5} | {'Losses':<7} | {'Win Rate':<9} | {'Net PnL':<10}")
print("-" * 75)

for dur_mode in [2, 3, 4, 5, 'dynamic']:
    for min_adx in [22.0, 25.0, 27.0, 30.0]:
        trades = []
        sym_open = {s: 0 for s in dfs}
        for epoch in range(START_EPOCH, END_EPOCH + 1, 60):
            for sym, df in dfs.items():
                if epoch < sym_open[sym]:
                    continue
                matches = df.index[df['epoch'] == epoch]
                if len(matches) == 0:
                    continue
                i = matches[0]
                if i < 200:
                    continue
                last = df.iloc[i]
                prev = df.iloc[i-1]
                adx_val = float(last["adx"])
                if adx_val < min_adx:
                    continue
                if any(tok in sym.upper() for tok in ("1HZ15V", "FRXUSDJPY", "FRXUSDCHF", "FRXXAUUSD", "FRXEURUSD")):
                    continue

                # check long/short using same logic
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

                if dur_mode == 'dynamic':
                    regime = classify_regime(last, cfg.strategy)
                    d, _ = calculate_dynamic_duration(sym, "synthetic", last, regime)
                else:
                    d = int(dur_mode)

                if i + d >= len(df):
                    continue
                exit_p = float(df.iloc[i + d]["close"])
                entry_p = float(last["close"])
                won = (exit_p > entry_p) if direction == "long" else (exit_p < entry_p)
                pnl = 0.88 if won else -1.0
                trades.append(pnl)
                sym_open[sym] = int(df.iloc[i + d]["epoch"])

        w = sum(1 for p in trades if p > 0)
        l = sum(1 for p in trades if p < 0)
        tot = len(trades)
        wr = (w / tot * 100) if tot > 0 else 0.0
        net = sum(trades)
        print(f"{str(dur_mode):<15} | {min_adx:<8.1f} | {tot:<7} | {w:<5} | {l:<7} | {wr:<8.1f}% | ${net:+.2f}")
