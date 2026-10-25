#!/usr/bin/env python3
"""
scratch/build_multi_day_dataset.py
Assembles 3-4 continuous days of 1-minute candles for all active synthetic indices:
From Oct 22, 2026 00:00 UTC through Oct 25, 2026 04:00 UTC.
"""
import asyncio
import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from deriv_client import DerivClient

SYMBOLS = [
    'R_10', 'R_25', 'R_50', 'R_75', 'R_100',
    '1HZ10V', '1HZ25V', '1HZ30V', '1HZ50V', '1HZ75V', '1HZ90V', '1HZ100V',
    'JD10', 'JD25', 'JD50', 'JD75', 'JD100',
    'stpRNG', 'stpRNG2', 'stpRNG3', 'stpRNG4', 'stpRNG5'
]

async def build_dataset():
    cfg = Config.from_env()
    client = DerivClient(cfg.deriv)
    await client.connect()

    # Load existing recent cache if available
    recent_cache = {}
    if os.path.exists("scratch/cached_replay_candles.pkl"):
        with open("scratch/cached_replay_candles.pkl", "rb") as f:
            recent_cache = pickle.load(f)

    full_dataset = {}

    print("Fetching and assembling 3-day continuous candles for all symbols...")

    for sym in SYMBOLS:
        frames = []
        # 1. Existing CSV from data/candles/
        csv_path = f"data/candles/{sym}_60.csv"
        if os.path.exists(csv_path):
            csv_df = pd.read_csv(csv_path)
            if "time" in csv_df.columns and "epoch" not in csv_df.columns:
                csv_df["epoch"] = pd.to_datetime(csv_df["time"]).map(lambda t: int(t.timestamp()))
            frames.append(csv_df[["epoch", "open", "high", "low", "close"]])

        # 2. Gap fetch from Deriv (Oct 24 05:00 UTC to Oct 24 14:00 UTC)
        try:
            res = await client.request({
                "ticks_history": sym,
                "style": "candles",
                "granularity": 60,
                "count": 1000,
                "end": 1792848540,
                "start": 1,
                "adjust_start_time": 1
            })
            gap_candles = res.get("candles", [])
            if gap_candles:
                gap_df = pd.DataFrame(gap_candles)
                frames.append(gap_df[["epoch", "open", "high", "low", "close"]])
        except Exception as e:
            print(f"  [WARN] {sym} gap fetch: {e}")

        # 3. Recent cache
        if sym in recent_cache:
            rec_df = recent_cache[sym]
            frames.append(rec_df[["epoch", "open", "high", "low", "close"]])

        if not frames:
            print(f"  [ERROR] No frames for {sym}")
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
        merged["time"] = pd.to_datetime(merged["epoch"], unit="s", utc=True)
        full_dataset[sym] = merged
        print(f"  [OK] {sym}: {len(merged)} bars | Start: {merged['time'].iloc[0]} -> End: {merged['time'].iloc[-1]}")
        await asyncio.sleep(0.1)

    await client.close()

    out_file = "scratch/three_days_candles.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(full_dataset, f)
    print(f"\nSaved full multi-day dataset to {out_file}!")

if __name__ == "__main__":
    asyncio.run(build_dataset())
