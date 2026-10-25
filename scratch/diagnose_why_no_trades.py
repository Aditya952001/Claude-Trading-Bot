import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import pandas as pd
from config import Config
from deriv_client import DerivClient
from strategy_engine import StrategyEngine, StrategyParams, compute_indicators

async def diagnose():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()
    
    symbols = [
        'R_10', 'R_25', 'R_50', 'R_100',
        '1HZ10V', '1HZ15V', '1HZ25V', '1HZ30V', '1HZ50V', '1HZ75V', '1HZ90V', '1HZ100V',
        'JD10', 'JD25', 'JD50', 'JD75', 'JD100',
        'frxEURUSD', 'frxGBPUSD', 'frxUSDJPY', 'frxAUDUSD', 'frxUSDCAD', 'frxUSDCHF', 'frxEURJPY', 'frxXAUUSD', 'frxXAGUSD'
    ]
    
    strat = StrategyEngine(cfg.strategy)
    params = StrategyParams(rsi_long_max=58.0, rsi_short_min=42.0, min_score=2.0)
    
    print(f"{'SYMBOL':<10} | {'ADX':<5} | {'RSI':<5} | {'EMA9>21':<7} | {'%B':<5} | {'SCORE':<5} | {'DIAGNOSTIC STATUS'}")
    print("-" * 80)
    
    for sym in symbols:
        try:
            candles = await client.get_candles(sym, granularity=60, count=249)
            if not candles or len(candles) < 50:
                print(f"{sym:<10} | Insufficient candles")
                continue
            df = compute_indicators(pd.DataFrame(candles), cfg.strategy)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            adx = float(last['adx'])
            rsi = float(last['rsi'])
            b_pct = float(last['bb_pctb'])
            ema_fast = float(last['ema_fast'])
            ema_slow = float(last['ema_slow'])
            ema_up = ema_fast > ema_slow
            regime = "trending" if adx > 25 else "ranging"
            votes = strat._votes(df, last, prev, regime, params)
            score = sum(votes.values())
            
            # Check reasons
            reason = "Passes to Confluence"
            if adx < 22.0:
                reason = f"Filtered: ADX {adx:.1f} < 22.0 (Low trend strength)"
            elif "JD" in sym and adx < 25.0:
                reason = f"Filtered: JD ADX {adx:.1f} < 25.0 (Stagnation trap)"
            elif "1HZ" in sym and adx < 25.0:
                reason = f"Filtered: 1HZ ADX {adx:.1f} < 25.0 (Micro-chop)"
            else:
                asset_class = "forex" if "frx" in sym else "synthetic"
                signal = strat.evaluate(sym, asset_class, pd.DataFrame(candles), params)
                if signal:
                    reason = f"*** SIGNAL DETECTED: {signal.direction.upper()} ({signal.dynamic_duration}m) ***"
                else:
                    reason = f"ADX OK ({adx:.1f}), Score={score:.1f}/4.0, Waiting for entry trigger"
            
            print(f"{sym:<10} | {adx:<5.1f} | {rsi:<5.1f} | {str(ema_up):<7} | {b_pct:<5.2f} | {score:<5.1f} | {reason}")
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"{sym:<10} | Error: {e}")
            await asyncio.sleep(1.0)
            
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(diagnose())
