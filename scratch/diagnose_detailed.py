import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import pandas as pd
from config import Config
from deriv_client import DerivClient
from strategy_engine import StrategyEngine, StrategyParams, compute_indicators

async def check():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()
    strat = StrategyEngine(cfg.strategy)
    params = StrategyParams()
    
    symbols = [
        'R_10', 'R_25', 'R_50', 'R_75', 'R_100',
        '1HZ10V', '1HZ25V', '1HZ50V', '1HZ75V', '1HZ100V',
        'JD10', 'JD25', 'JD50', 'JD75', 'JD100',
        'stpRNG', 'stpRNG2', 'stpRNG3', 'stpRNG4', 'stpRNG5',
        'frxEURUSD', 'frxGBPUSD', 'frxUSDJPY', 'frxAUDUSD', 'frxUSDCAD', 'frxXAUUSD', 'frxXAGUSD'
    ]
    
    print(f"{'SYMBOL':<10} | {'SCORE':<5} | {'ADX':<5} | {'RSI':<5} | {'%B':<5} | {'C>EMA200':<8} | {'C>EMA50':<7} | {'FAILS'}")
    print("-" * 95)
    
    for sym in symbols:
        try:
            candles = await client.get_candles(sym, 60, 249)
            if not candles or len(candles) < 50:
                continue
            df = compute_indicators(pd.DataFrame(candles), cfg.strategy)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            adx = float(last["adx"])
            rsi = float(last["rsi"])
            pctb = float(last["bb_pctb"])
            close = float(last["close"])
            ema_200 = float(last["ema_higher_tf"])
            ema_50 = float(last["ema_trend"])
            atr = max(1e-9, float(last["atr"]))
            macd_rel = float(last["macd_hist"]) / atr
            
            regime = "trending" if adx > 25 else "ranging"
            votes = strat._votes(df, last, prev, regime, params)
            score = sum(votes.values())
            
            sig = strat.evaluate(sym, "forex" if "frx" in sym else "synthetic", pd.DataFrame(candles), params)
            
            fails = []
            if adx < 18.0:
                fails.append(f"ADX<18({adx:.1f})")
            if "JD" in sym and adx < 21.0:
                fails.append("JD ADX<21")
            if "1HZ" in sym and adx < 22.0:
                fails.append("1HZ ADX<22")
            if abs(score) < 2.0:
                fails.append(f"Score={score:.1f}<2.0")
            
            # Check Long fails
            if score >= 1.0:
                if close < ema_200:
                    fails.append("Close<EMA200")
                if close < ema_50:
                    fails.append("Close<EMA50")
                if last["close"] <= last["open"]:
                    fails.append("Red Candle")
                if pctb > 0.82:
                    fails.append(f"%B>{0.82:.2f}")
                if (close - float(last["ema_fast"])) > 0.55 * atr:
                    fails.append("Fast EMA Stretch")
            elif score <= -1.0:
                if close > ema_200:
                    fails.append("Close>EMA200")
                if close > ema_50:
                    fails.append("Close>EMA50")
                if last["close"] >= last["open"]:
                    fails.append("Green Candle")
                if pctb < 0.18:
                    fails.append(f"%B<{0.18:.2f}")
                if (float(last["ema_fast"]) - close) > 0.55 * atr:
                    fails.append("Fast EMA Stretch")
            
            status = f"*** SIGNAL: {sig.direction} ***" if sig else ", ".join(fails[:3])
            c_200 = "YES" if close > ema_200 else "NO"
            c_50 = "YES" if close > ema_50 else "NO"
            print(f"{sym:<10} | {score:<5.1f} | {adx:<5.1f} | {rsi:<5.1f} | {pctb:<5.2f} | {c_200:<8} | {c_50:<7} | {status}")
            await asyncio.sleep(0.15)
        except Exception as e:
            pass

if __name__ == "__main__":
    asyncio.run(check())
