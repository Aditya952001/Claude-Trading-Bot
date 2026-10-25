import asyncio
import pandas as pd
import numpy as np
from config import Config
from deriv_client import DerivClient, CandleFeed
from strategy_engine import StrategyEngine, compute_indicators
from memory_and_learning import StrategyParams

async def check():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()
    engine = StrategyEngine(cfg.strategy)
    params = StrategyParams(min_score=3.0)

    try:
        symbols_to_check = [s.symbol for s in cfg.symbols if s.enabled][:12] # check sample 12 symbols
        print(f"Testing candle history & signals on {len(symbols_to_check)} symbols...")
        total_bars = 0
        total_signals = 0

        for sym in symbols_to_check:
            cf = await client.request({
                "ticks_history": sym,
                "style": "candles",
                "granularity": 60,
                "count": 100,
                "end": "latest",
            })
            candles_data = cf.get("candles", [])
            if not candles_data:
                continue
            df = pd.DataFrame(candles_data)
            df = df.rename(columns={"epoch": "time"})
            # Evaluate across last 20 bars
            sym_signals = 0
            for i in range(80, len(df)):
                sub_df = df.iloc[:i+1]
                sig = engine.evaluate(sym, "synthetic", sub_df, params)
                if sig:
                    sym_signals += 1
                    total_signals += 1
            print(f"  {sym:12} : {sym_signals} signals in {len(df)-80} bars")
            total_bars += (len(df) - 80)
            await asyncio.sleep(0.3)

        print(f"\nTotal bars evaluated: {total_bars} | Total signals: {total_signals} ({total_signals/total_bars*100:.1f}%)")
    finally:
        await client.close()

asyncio.run(check())
