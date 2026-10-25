import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deriv_client import DerivClient, CandleFeed
from config import Config
from strategy_engine import StrategyEngine, compute_indicators, classify_regime
from memory_and_learning import MemoryStore, SelfLearner

async def test_bars():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()
    engine = StrategyEngine(cfg.strategy)
    store = MemoryStore(cfg.learning.db_path)
    learner = SelfLearner(store, cfg.learning)
    params = learner.params

    total_candles = 0
    signals_count = 0
    rejections = {
        "adx_below_20": 0,
        "score_too_low": 0,
        "rsi_out_of_bounds": 0,
        "rsi_not_turning": 0,
        "macdh_wrong_sign": 0,
        "bb_pctb_out_of_bounds": 0,
        "candle_wrong_color": 0,
        "candle_not_higher_than_prev": 0,
        "close_vs_200ema": 0,
        "close_vs_50ema": 0,
    }

    for sc in cfg.symbols:
        if not sc.enabled:
            continue
        feed = CandleFeed(sc.symbol, sc.granularity)
        sub = await client.subscribe_candles(sc.symbol, sc.granularity, 250, feed.handle)
        await asyncio.sleep(1.2)
        df = feed.frame()
        if len(df) < 150:
            await client.unsubscribe(sub)
            continue
        ind_df = compute_indicators(df, cfg.strategy)
        for i in range(150, len(ind_df)):
            total_candles += 1
            slice_df = ind_df.iloc[:i+1]
            last = slice_df.iloc[-1]
            prev = slice_df.iloc[-2]
            regime = classify_regime(last, cfg.strategy)
            votes = engine._votes(slice_df, last, prev, regime, params)
            score = sum(params.component_weights.get(k, 1.0) * v for k, v in votes.items())
            
            sig = engine.evaluate(sc.symbol, sc.asset_class.value, slice_df, params)
            if sig:
                signals_count += 1
                rsi_val = sig.indicators.get('rsi', 0)
                print(f"SIGNAL on {sc.symbol} at bar {i}: {sig.direction} score={sig.score:.2f} rsi={rsi_val:.1f}")
            else:
                # Track why it failed for the best candidate direction
                if last["adx"] < 20.0:
                    rejections["adx_below_20"] += 1
                if abs(score) < params.min_score:
                    rejections["score_too_low"] += 1
                if score >= params.min_score: # Attempted long
                    if not (42.0 <= last["rsi"] <= 60.0): rejections["rsi_out_of_bounds"] += 1
                    elif last["rsi"] < prev["rsi"]: rejections["rsi_not_turning"] += 1
                    elif last["macd_hist"] <= 0: rejections["macdh_wrong_sign"] += 1
                    elif not (0.20 <= last["bb_pctb"] <= 0.68): rejections["bb_pctb_out_of_bounds"] += 1
                    elif not (last["close"] > last["open"]): rejections["candle_wrong_color"] += 1
                    elif not (last["close"] > prev["close"]): rejections["candle_not_higher_than_prev"] += 1
                    elif not (last["close"] > last["ema_higher_tf"]): rejections["close_vs_200ema"] += 1
                    elif not (last["close"] > last["ema_trend"]): rejections["close_vs_50ema"] += 1
                elif score <= -params.min_score: # Attempted short
                    if not (40.0 <= last["rsi"] <= 58.0): rejections["rsi_out_of_bounds"] += 1
                    elif last["rsi"] > prev["rsi"]: rejections["rsi_not_turning"] += 1
                    elif last["macd_hist"] >= 0: rejections["macdh_wrong_sign"] += 1
                    elif not (0.32 <= last["bb_pctb"] <= 0.80): rejections["bb_pctb_out_of_bounds"] += 1
                    elif not (last["close"] < last["open"]): rejections["candle_wrong_color"] += 1
                    elif not (last["close"] < prev["close"]): rejections["candle_not_higher_than_prev"] += 1
                    elif not (last["close"] < last["ema_higher_tf"]): rejections["close_vs_200ema"] += 1
                    elif not (last["close"] < last["ema_trend"]): rejections["close_vs_50ema"] += 1

        await client.unsubscribe(sub)
    await client.close()
    print("="*65)
    print(f"Total bars tested: {total_candles} | Total signals generated: {signals_count}")
    print("Rejection Breakdown:")
    for k, v in rejections.items():
        print(f"  {k:30s}: {v}")

asyncio.run(test_bars())
