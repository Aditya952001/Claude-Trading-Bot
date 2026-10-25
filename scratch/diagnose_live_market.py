import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deriv_client import DerivClient, CandleFeed
from config import Config
from strategy_engine import StrategyEngine, compute_indicators, classify_regime
from memory_and_learning import MemoryStore, SelfLearner

async def inspect():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()
    engine = StrategyEngine(cfg.strategy)
    store = MemoryStore(cfg.learning.db_path)
    learner = SelfLearner(store, cfg.learning)
    params = learner.params

    for sc in cfg.symbols:
        if not sc.enabled:
            continue
        sym = sc.symbol
        feed = CandleFeed(sym, sc.granularity)
        sub = await client.subscribe_candles(sym, sc.granularity, 250, feed.handle)
        await asyncio.sleep(1.2)
        df = feed.frame()
        if len(df) < 150:
            print(f"{sym}: only {len(df)} candles")
            await client.unsubscribe(sub)
            continue
        ind_df = compute_indicators(df, cfg.strategy)
        last, prev = ind_df.iloc[-1], ind_df.iloc[-2]
        regime = classify_regime(last, cfg.strategy)
        votes = engine._votes(ind_df, last, prev, regime, params)
        score = sum(params.component_weights.get(k, 1.0) * v for k, v in votes.items())
        
        c = float(last["close"])
        p = float(prev["close"])
        o = float(last["open"])
        adx = float(last["adx"])
        rsi = float(last["rsi"])
        prsi = float(prev["rsi"])
        macdh = float(last["macd_hist"])
        pmacdh = float(prev["macd_hist"])
        pctb = float(last["bb_pctb"])
        ema9 = float(last["ema_fast"])
        ema21 = float(last["ema_slow"])
        ema50 = float(last["ema_trend"])
        ema200 = float(last["ema_higher_tf"])

        sig = engine.evaluate(sym, sc.asset_class.value, df, params)

        print("="*65)
        print(f"SYMBOL: {sym} | Price: {c:.4f} | Regime: {regime} | Score: {score:.1f} | Signal: {sig.direction if sig else 'NONE'}")
        print(f"  Votes: {votes}")
        print(f"  ADX: {adx:.1f} (>=20: {adx>=20.0})")
        print(f"  RSI: {rsi:.1f} (prev: {prsi:.1f})")
        print(f"  MACD_Hist: {macdh:.4f} (prev: {pmacdh:.4f})")
        print(f"  BB%B: {pctb:.2f}")
        print(f"  Candle: Close vs Open: {c > o} | Close vs Prev: {c > p}")
        print(f"  EMA Alignment: Close > 200EMA: {c > ema200} | Close > 50EMA: {c > ema50} | Fast > Slow: {ema9 > ema21}")
        
        # Why no signal?
        reasons = []
        if adx < 20.0:
            reasons.append(f"ADX {adx:.1f} < 20.0 (Chop)")
        if abs(score) < params.min_score:
            reasons.append(f"|Score| {abs(score):.1f} < min_score {params.min_score:.1f}")
        # Long check
        long_fails = []
        if score < params.min_score: long_fails.append(f"score {score:.1f} < {params.min_score}")
        if not (42.0 <= rsi <= 60.0): long_fails.append(f"RSI {rsi:.1f} not in [42,60]")
        if rsi < prsi: long_fails.append("RSI declining")
        if macdh <= 0: long_fails.append(f"MACDh {macdh:.4f} <= 0")
        if not (0.20 <= pctb <= 0.68): long_fails.append(f"BB%B {pctb:.2f} not in [0.20,0.68]")
        if not (c > o): long_fails.append("Bearish candle (close <= open)")
        if not (c > p): long_fails.append("Close <= prev close")
        if not (c > ema200): long_fails.append(f"Below 200 EMA ({c:.2f} <= {ema200:.2f})")
        if not (c > ema50): long_fails.append(f"Below 50 EMA ({c:.2f} <= {ema50:.2f})")

        # Short check
        short_fails = []
        if score > -params.min_score: short_fails.append(f"score {score:.1f} > -{params.min_score}")
        if not (40.0 <= rsi <= 58.0): short_fails.append(f"RSI {rsi:.1f} not in [40,58]")
        if rsi > prsi: short_fails.append("RSI rising")
        if macdh >= 0: short_fails.append(f"MACDh {macdh:.4f} >= 0")
        if not (0.32 <= pctb <= 0.80): short_fails.append(f"BB%B {pctb:.2f} not in [0.32,0.80]")
        if not (c < o): short_fails.append("Bullish candle (close >= open)")
        if not (c < p): short_fails.append("Close >= prev close")
        if not (c < ema200): short_fails.append(f"Above 200 EMA ({c:.2f} >= {ema200:.2f})")
        if not (c < ema50): short_fails.append(f"Above 50 EMA ({c:.2f} >= {ema50:.2f})")

        print(f"  Long Blocking Factors ({len(long_fails)}): {', '.join(long_fails[:4])}")
        print(f"  Short Blocking Factors ({len(short_fails)}): {', '.join(short_fails[:4])}")

        await client.unsubscribe(sub)
    await client.close()

asyncio.run(inspect())
