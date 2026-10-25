"""
research/deriv_data.py - Read-only historical candle access for research/backtesting.

Uses the same authenticated (demo) WebSocket route the bot already uses, but only
sends READ-ONLY requests (`ticks_history`). No orders are ever placed from here.

Candles are cached under data/candles/<symbol>_<granularity>.csv so a backtest can
be re-run without hitting the API again.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import pandas as pd
import websockets
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "candles"
MAX_PER_REQUEST = 1000   # Deriv caps `ticks_history` candles at 1000 per request


def _load_env() -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    return os.getenv("DERIV_API_TOKEN", ""), os.getenv("DERIV_APP_ID", "")


def _rest(url: str, token: str, app_id: str, data: Optional[bytes] = None,
          method: str = "GET") -> dict:
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {token}", "Deriv-App-ID": app_id,
                 "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def get_ws_url(environment: str = "demo") -> str:
    token, app_id = _load_env()
    if not token:
        raise RuntimeError("DERIV_API_TOKEN missing from .env")
    accounts = _rest("https://api.derivws.com/trading/v1/options/accounts",
                     token, app_id).get("data", [])
    account = next((a for a in accounts if a.get("account_type") == environment), None)
    if not account:
        raise RuntimeError(f"No {environment} account on this token")
    otp = _rest(
        f"https://api.derivws.com/trading/v1/options/accounts/{account['account_id']}/otp",
        token, app_id, data=b"{}", method="POST",
    )
    url = otp.get("data", {}).get("url")
    if not url:
        raise RuntimeError("OTP response had no ws url")
    return url


class CandleClient:
    """Thin read-only request/response client over the authenticated WS."""

    def __init__(self, ws):
        self.ws = ws
        self._req_id = 1000

    @classmethod
    async def connect(cls, environment: str = "demo") -> "CandleClient":
        url = get_ws_url(environment)
        ws = await websockets.connect(url, ping_interval=None, open_timeout=20,
                                      max_size=8 * 2 ** 20)
        return cls(ws)

    async def request(self, payload: dict) -> dict:
        self._req_id += 1
        payload = dict(payload, req_id=self._req_id)
        await self.ws.send(json.dumps(payload))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if msg.get("req_id") == self._req_id:
                if msg.get("error"):
                    raise RuntimeError(f"Deriv error: {msg['error']}")
                return msg

    async def close(self) -> None:
        await self.ws.close()

    async def candles(self, symbol: str, bars: int, granularity: int = 60) -> pd.DataFrame:
        """Fetch up to `bars` 1m candles ending now, paginating backwards."""
        frames: List[pd.DataFrame] = []
        end = "latest"
        remaining = bars
        while remaining > 0:
            count = min(MAX_PER_REQUEST, remaining)
            msg = await self.request({
                "ticks_history": symbol, "adjust_start_time": 1,
                "count": count, "end": end,
                "style": "candles", "granularity": granularity,
            })
            raw = msg.get("candles") or []
            if not raw:
                break
            df = pd.DataFrame(raw)
            df["epoch"] = df["epoch"].astype("int64")
            frames.append(df)
            remaining -= len(df)
            if len(df) < count:
                break
            end = int(df["epoch"].iloc[0]) - granularity
            await asyncio.sleep(0.15)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).drop_duplicates("epoch")
        out = out.sort_values("epoch").reset_index(drop=True)
        out["time"] = pd.to_datetime(out["epoch"], unit="s", utc=True)
        out = out.set_index("time")
        return out[["open", "high", "low", "close"]].astype(float)


def cache_path(symbol: str, granularity: int = 60) -> Path:
    return CACHE_DIR / f"{symbol}_{granularity}.csv"


def load_cached(symbol: str, granularity: int = 60) -> pd.DataFrame:
    p = cache_path(symbol, granularity)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["time"], index_col="time")
    return df[["open", "high", "low", "close"]].astype(float)


def save_cached(symbol: str, df: pd.DataFrame, granularity: int = 60) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(symbol, granularity)
    df.to_csv(p, index_label="time")
    return p


async def fetch_symbols(symbols: List[str], bars: int, granularity: int = 60,
                        force: bool = False) -> dict:
    todo = [s for s in symbols
            if force or load_cached(s, granularity).empty
            or len(load_cached(s, granularity)) < bars * 0.9]
    results = {}
    if not todo:
        return results

    async def attempt(sym: str) -> bool:
        t0 = time.time()
        client = None
        for try_no in range(1, 4):
            try:
                client = await CandleClient.connect()
                df = await client.candles(sym, bars, granularity)
                if df.empty:
                    print(f"  EMPTY {sym} (try {try_no})", flush=True)
                    return False
                save_cached(sym, df, granularity)
                results[sym] = df
                print(f"  {sym:<12} {len(df):>6} bars "
                      f"{df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%H:%M} "
                      f"({time.time()-t0:.1f}s)", flush=True)
                return True
            except Exception as exc:
                print(f"  retry {sym} ({try_no}/3): {type(exc).__name__}: {exc}", flush=True)
                await asyncio.sleep(2 * try_no)
            finally:
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        pass
        return False

    for sym in todo:
        await attempt(sym)
        await asyncio.sleep(0.4)
    return results



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="R_10,R_25,R_50,R_75,R_100")
    ap.add_argument("--bars", type=int, default=20000)
    ap.add_argument("--granularity", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
    print(f"Fetching {len(syms)} symbols x {a.bars} bars @ {a.granularity}s")
    asyncio.run(fetch_symbols(syms, a.bars, a.granularity, a.force))
