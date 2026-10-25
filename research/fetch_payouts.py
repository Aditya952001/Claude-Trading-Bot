"""
research/fetch_payouts.py - Read-only quote snapshot: payout per $1 stake for each
symbol x duration. Saves research/payouts.json = {symbol: {duration: payout}}.

No orders are placed; only `proposal` requests are sent.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from deriv_data import CandleClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "payouts.json"

SYMBOLS = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
    "JD10", "JD25", "JD50", "JD75", "JD100",
    "stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5",
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD",
    "frxUSDCHF", "frxEURJPY", "frxGBPJPY", "frxXAUUSD", "frxXAGUSD",
]
DURATIONS = [1, 3, 4, 5, 6, 7, 8, 10, 15, 20]


async def quote(client, symbol, duration, ctype="CALL"):
    msg = await client.request({
        "proposal": 1, "amount": 1, "basis": "stake", "contract_type": ctype,
        "currency": "USD", "duration": duration, "duration_unit": "m", "underlying_symbol": symbol,
    })
    p = msg.get("proposal", {})
    return float(p.get("payout", 0.0))


async def main():
    out = {}
    client = await CandleClient.connect()
    try:
        for sym in SYMBOLS:
            row = {}
            for dur in DURATIONS:
                try:
                    payout = await quote(client, sym, dur)
                    if payout:
                        row[str(dur)] = round(payout, 4)
                except Exception:
                    pass
                await asyncio.sleep(0.05)
            if row:
                out[sym] = row
                best = max(row.items(), key=lambda kv: kv[1] / 1.0)
                print(f"{sym:<12} " + " ".join(f"{d}m:{v:.3f}" for d, v in row.items()),
                      flush=True)
    finally:
        await client.close()
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("saved", OUT)


if __name__ == "__main__":
    asyncio.run(main())
