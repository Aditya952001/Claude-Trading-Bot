import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deriv_client import DerivClient, CandleFeed
from config import Config
from strategy_engine import compute_indicators

async def test_balanced():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()

    total_candles = 0
    signals = []

    for sc in cfg.symbols:
        if not sc.enabled: continue
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
            
            c = float(last["close"])
            o = float(last["open"])
            rsi = float(last["rsi"])
            macdh = float(last["macd_hist"])
            pctb = float(last["bb_pctb"])
            adx = float(last["adx"])
            ema200 = float(last["ema_higher_tf"])
            ema9 = float(last["ema_fast"])
            ema21 = float(last["ema_slow"])

            v = {"trend": 0, "cross": 0, "rsi": 0, "bbands": 0, "macd": 0}
            if c > ema200 and ema9 > ema21: v["trend"] = 1
            elif c < ema200 and ema9 < ema21: v["trend"] = -1

            diff = (slice_df["ema_fast"] - slice_df["ema_slow"]).iloc[-5:]
            if diff.iloc[-1] > 0 and (diff.iloc[-1] >= diff.iloc[-2] or (diff.iloc[:-1].tail(3) <= 0).any()): v["cross"] = 1
            elif diff.iloc[-1] < 0 and (diff.iloc[-1] <= diff.iloc[-2] or (diff.iloc[:-1].tail(3) >= 0).any()): v["cross"] = -1

            if 38.0 <= rsi <= 64.0 and rsi >= prev["rsi"]: v["rsi"] = 1
            elif 36.0 <= rsi <= 62.0 and rsi <= prev["rsi"]: v["rsi"] = -1

            if 0.15 <= pctb <= 0.70 and c > o: v["bbands"] = 1
            elif 0.30 <= pctb <= 0.85 and c < o: v["bbands"] = -1

            if macdh > 0: v["macd"] = 1
            elif macdh < 0: v["macd"] = -1

            score = sum(v.values())

            # Balanced entry criteria:
            # Score >= 3.0 (at least 3/5 components), macro trend aligned, not overbought/oversold extreme
            if score >= 3.0 and c > ema200 and rsi <= 65.0 and pctb <= 0.80 and macdh >= -0.05 and c > o:
                signals.append((sc.symbol, i, "LONG", score, rsi, pctb, macdh, adx))
            elif score <= -3.0 and c < ema200 and rsi >= 35.0 and pctb >= 0.20 and macdh <= 0.05 and c < o:
                signals.append((sc.symbol, i, "SHORT", score, rsi, pctb, macdh, adx))

        await client.unsubscribe(sub)
    await client.close()
    print("="*60)
    print(f"Total candles tested: {total_candles} | Signals found: {len(signals)}")
    for s in signals[:15]:
        print(f"  {s[0]} bar {s[1]}: {s[2]} score={s[3]} rsi={s[4]:.1f} %B={s[5]:.2f} macdh={s[6]:.4f} adx={s[7]:.1f}")

asyncio.run(test_balanced())
