"""
deriv_client.py - Async WebSocket wrapper for the Deriv API.

Features
  * Authorization + demo/live account-type cross-check
  * Request/response correlation through `req_id`
  * Streaming subscriptions (candles, open-contract updates) with automatic
    reconnect + resubscribe
  * Historical candles, balance, portfolio, contracts_for
  * Multiplier order placement with server-side stop-loss / take-profit
  * CandleFeed: keeps a rolling window of CLOSED candles and emits a
    callback whenever a bar closes (the strategy only ever sees closed bars)
"""
from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed

from config import DerivConfig, Environment

log = logging.getLogger("deriv_client")

Callback = Callable[[dict], Any]


class DerivAPIError(Exception):
    def __init__(self, code: str, message: str, request: Optional[dict] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.request = request


@dataclass
class _Subscription:
    payload: dict
    callback: Callback
    req_id: int = 0
    sub_id: Optional[str] = None


class DerivClient:
    def __init__(self, cfg: DerivConfig):
        self.cfg = cfg
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_ids = itertools.count(1)
        self._pending: Dict[int, asyncio.Future] = {}
        self._subs: Dict[int, _Subscription] = {}       # req_id -> subscription
        self._reader_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._closing = False
        self.account: Dict[str, Any] = {}
        self.connected = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> None:
        await self._open_and_authorize()
        self._reader_task = asyncio.create_task(self._reader(), name="deriv-reader")
        self._keepalive_task = asyncio.create_task(self._keepalive(), name="deriv-keepalive")

    async def close(self) -> None:
        self._closing = True
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws:
            await self._ws.close()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("client closed"))

    async def _keepalive(self) -> None:
        while not self._closing:
            await asyncio.sleep(20)
            if self._ws and self.connected.is_set():
                try:
                    await self._ws.send(json.dumps({"ping": 1}))
                except Exception:
                    pass

    async def _open_and_authorize(self) -> None:
        """Open the socket and authorize synchronously (reader not running yet)."""
        if self.cfg.api_token.startswith("pat_"):
            await self._open_and_authorize_pat()
            return

        self._ws = await websockets.connect(
            self.cfg.ws_url, ping_interval=None, max_size=16 * 2**20
        )
        await self._ws.send(json.dumps({"authorize": self.cfg.api_token}))
        msg = json.loads(await asyncio.wait_for(self._ws.recv(), 15))
        if "error" in msg:
            raise DerivAPIError(msg["error"].get("code", "AuthError"), msg["error"]["message"])
        self.account = msg["authorize"]
        self._check_account_type()
        self.connected.set()
        log.info("Authorized loginid=%s virtual=%s balance=%s %s",
                 self.account.get("loginid"), self.account.get("is_virtual"),
                 self.account.get("balance"), self.account.get("currency"))

    async def _open_and_authorize_pat(self) -> None:
        """Deriv Personal Access Token (PAT) authentication via REST OTP endpoint."""
        loop = asyncio.get_running_loop()

        # 1. Fetch account list
        acc_req = urllib.request.Request(
            "https://api.derivws.com/trading/v1/options/accounts",
            headers={
                "Authorization": f"Bearer {self.cfg.api_token}",
                "Deriv-App-ID": str(self.cfg.app_id),
            },
            method="GET",
        )
        try:
            acc_res_bytes = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(acc_req, timeout=15).read()
            )
            accounts = json.loads(acc_res_bytes.decode()).get("data", [])
        except Exception as exc:
            raise DerivAPIError("PATAuthError", f"Failed to fetch accounts via Deriv REST API: {exc}")

        target_type = "demo" if self.cfg.environment is Environment.DEMO else "real"
        account = next((a for a in accounts if a.get("account_type") == target_type), None)
        if not account:
            available_types = [a.get("account_type") for a in accounts]
            raise RuntimeError(
                f"No {target_type} account found for this token (available: {available_types})."
            )

        account_id = account.get("account_id")

        # 2. Request OTP WebSocket URL
        otp_req = urllib.request.Request(
            f"https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp",
            data=b"{}",
            headers={
                "Authorization": f"Bearer {self.cfg.api_token}",
                "Deriv-App-ID": str(self.cfg.app_id),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            otp_res_bytes = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(otp_req, timeout=15).read()
            )
            otp_data = json.loads(otp_res_bytes.decode())
            ws_url = otp_data.get("data", {}).get("url")
            if not ws_url:
                raise ValueError("No WebSocket URL returned in OTP response")
        except Exception as exc:
            raise DerivAPIError("OTPError", f"Failed to acquire OTP WebSocket URL: {exc}")

        # 3. Connect to authenticated WebSocket URL
        self._ws = await websockets.connect(
            ws_url, ping_interval=None, max_size=16 * 2**20
        )

        # 4. Fetch live balance to verify connection
        await self._ws.send(json.dumps({"balance": 1, "req_id": 1}))
        raw_bal = await asyncio.wait_for(self._ws.recv(), 15)
        bal_msg = json.loads(raw_bal)
        live_bal = float((bal_msg.get("balance") or {}).get("balance", account.get("balance", 0.0)))
        currency = (bal_msg.get("balance") or {}).get("currency", account.get("currency", "USD"))

        self.account = {
            "loginid": account_id,
            "is_virtual": (account.get("account_type") == "demo"),
            "balance": live_bal,
            "currency": currency,
        }
        self._check_account_type()
        self.connected.set()
        log.info(
            "Authorized via PAT loginid=%s virtual=%s balance=%.2f %s",
            self.account.get("loginid"),
            self.account.get("is_virtual"),
            self.account.get("balance"),
            self.account.get("currency"),
        )

    def _check_account_type(self) -> None:
        """Refuse to run if the token's account type contradicts DERIV_ENV."""
        is_virtual = bool(self.account.get("is_virtual"))
        if self.cfg.environment is Environment.DEMO and not is_virtual:
            raise RuntimeError("DERIV_ENV=demo but the API token belongs to a REAL account. Aborting.")
        if self.cfg.environment is Environment.LIVE and is_virtual:
            raise RuntimeError("DERIV_ENV=live but the API token belongs to a DEMO account.")

    async def _reader(self) -> None:
        while not self._closing:
            try:
                async for raw in self._ws:
                    self._dispatch(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                log.warning("WebSocket closed: %s", exc)
            except Exception:
                log.exception("Reader error")
            if self._closing:
                return
            await self._reconnect()

    async def _reconnect(self) -> None:
        self.connected.clear()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("connection lost"))
        self._pending.clear()
        delay = 1.0
        while not self._closing:
            try:
                await self._open_and_authorize()
                asyncio.create_task(self._resubscribe(), name="deriv-resubscribe")
                return
            except Exception as exc:
                log.warning("Reconnect failed (%s); retry in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def _resubscribe(self) -> None:
        old = list(self._subs.values())
        self._subs.clear()
        for sub in old:
            try:
                await self.subscribe(sub.payload, sub.callback)
                log.info("Resubscribed: %s", {k: v for k, v in sub.payload.items() if k != "subscribe"})
            except Exception:
                log.exception("Resubscribe failed for %s", sub.payload)

    # ------------------------------------------------------------------ #
    # Low-level request / subscribe
    # ------------------------------------------------------------------ #
    def _dispatch(self, msg: dict) -> None:
        rid = msg.get("req_id")
        fut = self._pending.get(rid)
        sub = self._subs.get(rid)
        if fut is not None and not fut.done():
            if "error" in msg:
                err = msg["error"]
                fut.set_exception(DerivAPIError(err.get("code", "?"), err.get("message", "?")))
            else:
                fut.set_result(msg)
        if sub is not None and "error" not in msg:
            self._invoke(sub.callback, msg)
        elif fut is None and sub is None:
            if "error" in msg:
                log.warning("Unsolicited API error: %s", msg["error"])
            elif msg.get("msg_type") == "ping":
                pass
            else:
                log.debug("Unrouted message: %s", msg.get("msg_type"))

    @staticmethod
    def _invoke(cb: Callback, msg: dict) -> None:
        try:
            if inspect.iscoroutinefunction(cb):
                asyncio.create_task(cb(msg))
            else:
                cb(msg)
        except Exception:
            log.exception("Subscription callback failed")

    async def _request(self, payload: dict, req_id: Optional[int] = None) -> dict:
        if self._ws is None:
            raise ConnectionError("not connected")
        await asyncio.wait_for(self.connected.wait(), 60)
        req_id = req_id or next(self._req_ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps({**payload, "req_id": req_id}))
            return await asyncio.wait_for(fut, self.cfg.request_timeout_sec)
        except DerivAPIError as exc:
            exc.request = payload
            raise
        finally:
            self._pending.pop(req_id, None)

    async def request(self, payload: dict) -> dict:
        return await self._request(payload)

    async def subscribe(self, payload: dict, callback: Callback) -> _Subscription:
        """Send a request with subscribe=1; later stream messages go to `callback`."""
        payload = {**payload, "subscribe": 1}
        req_id = next(self._req_ids)
        sub = _Subscription(payload=payload, callback=callback, req_id=req_id)
        self._subs[req_id] = sub                    # register BEFORE sending
        try:
            first = await self._request(payload, req_id)
        except Exception:
            self._subs.pop(req_id, None)
            raise
        sub.sub_id = (first.get("subscription") or {}).get("id")
        return sub

    async def unsubscribe(self, sub: _Subscription) -> None:
        self._subs.pop(sub.req_id, None)
        if sub.sub_id:
            try:
                await self._request({"forget": sub.sub_id})
            except Exception as exc:
                log.debug("forget failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Account / market data
    # ------------------------------------------------------------------ #
    async def get_balance(self) -> float:
        msg = await self._request({"balance": 1})
        return float(msg["balance"]["balance"])

    async def get_candles(self, symbol: str, granularity: int, count: int) -> List[dict]:
        msg = await self._request({
            "ticks_history": symbol, "adjust_start_time": 1, "count": count,
            "end": "latest", "start": 1, "style": "candles", "granularity": granularity,
        })
        return msg.get("candles", [])

    async def subscribe_candles(self, symbol: str, granularity: int, count: int,
                                callback: Callback) -> _Subscription:
        return await self.subscribe({
            "ticks_history": symbol, "adjust_start_time": 1, "count": count,
            "end": "latest", "start": 1, "style": "candles", "granularity": granularity,
        }, callback)

    async def contracts_for(self, symbol: str) -> dict:
        msg = await self._request({"contracts_for": symbol})
        return msg["contracts_for"]

    async def portfolio(self) -> List[dict]:
        msg = await self._request({"portfolio": 1})
        return msg.get("portfolio", {}).get("contracts", [])

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    async def buy_multiplier(self, *, symbol: str, direction: str, stake: float,
                             multiplier: int, stop_loss: float, take_profit: float) -> dict:
        """
        Open a multiplier contract with server-side SL/TP (currency amounts).
        direction: 'long' -> MULTUP, 'short' -> MULTDOWN.
        Executes via Deriv 2-step Proposal -> Buy flow.
        """
        ctype = "MULTUP" if direction == "long" else "MULTDOWN"
        prop_payload = {
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": ctype,
            "currency": self.cfg.currency,
            "multiplier": multiplier,
            "underlying_symbol": symbol,
            "limit_order": {
                "stop_loss": round(max(stop_loss, 0.10), 2),
                "take_profit": round(max(take_profit, 0.10), 2),
            },
        }
        prop = await self._request(prop_payload)
        proposal_id = prop["proposal"]["id"]
        ask_price = prop["proposal"]["ask_price"]
        buy_msg = await self._request({"buy": proposal_id, "price": ask_price})
        return buy_msg["buy"]

    async def buy_rise_fall(self, *, symbol: str, direction: str, stake: float,
                            duration: int = 3, duration_unit: str = "m") -> dict:
        """
        Open a fixed-expiry Rise/Fall (CALL/PUT) digital option.
        direction: 'long' -> CALL, 'short' -> PUT.
        Executes via Deriv 2-step Proposal -> Buy flow.
        """
        ctype = "CALL" if direction == "long" else "PUT"
        prop_payload = {
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": ctype,
            "currency": self.cfg.currency,
            "underlying_symbol": symbol,
            "duration": duration,
            "duration_unit": duration_unit,
        }
        prop = await self._request(prop_payload)
        proposal_id = prop["proposal"]["id"]
        ask_price = prop["proposal"]["ask_price"]
        buy_msg = await self._request({"buy": proposal_id, "price": ask_price})
        return buy_msg["buy"]

    async def sell_contract(self, contract_id: int) -> dict:
        """Market-close an open contract (price=0 accepts the current bid)."""
        msg = await self._request({"sell": contract_id, "price": 0})
        return msg["sell"]

    async def subscribe_contract(self, contract_id: int, callback: Callback) -> _Subscription:
        return await self.subscribe({"proposal_open_contract": 1, "contract_id": contract_id}, callback)


# ======================================================================== #
# CandleFeed
# ======================================================================== #
class CandleFeed:
    """
    Rolling window of CLOSED candles for one symbol.

    Feed `handle()` with the raw subscription messages:
      * msg_type 'candles' -> history (also used to backfill after reconnect)
      * msg_type 'ohlc'    -> live forming candle; a new `open_time` means the
                              previous candle closed -> on_bar_close(symbol)
    """

    def __init__(self, symbol: str, granularity: int, max_bars: int = 600,
                 on_bar_close: Optional[Callable[[str], None]] = None):
        self.symbol = symbol
        self.granularity = granularity
        self.max_bars = max_bars
        self.on_bar_close = on_bar_close
        self._closed: Dict[int, dict] = {}
        self._forming: Optional[dict] = None
        self.last_price: float = float("nan")
        self.last_tick_ts: float = 0.0            # wall clock of last stream message

    def handle(self, msg: dict) -> None:
        mt = msg.get("msg_type")
        if mt == "candles":
            self._load_history(msg.get("candles", []))
        elif mt == "ohlc":
            self._on_ohlc(msg["ohlc"])

    def _load_history(self, candles: List[dict]) -> None:
        if not candles:
            return
        for c in candles[:-1]:
            self._closed[int(c["epoch"])] = self._norm(c, int(c["epoch"]))
        last = candles[-1]
        self._forming = self._norm(last, int(last["epoch"]))
        self.last_price = self._forming["close"]
        self.last_tick_ts = time.time()
        self._trim()

    def _on_ohlc(self, o: dict) -> None:
        open_time = int(o["open_time"])
        bar = self._norm(o, open_time)
        self.last_price = bar["close"]
        self.last_tick_ts = time.time()
        if self._forming and self._forming["epoch"] != open_time:
            finished = self._forming
            self._closed[finished["epoch"]] = finished
            self._trim()
            self._forming = bar
            if self.on_bar_close:
                self.on_bar_close(self.symbol)
        else:
            self._forming = bar

    @staticmethod
    def _norm(c: dict, epoch: int) -> dict:
        return {"epoch": epoch, "open": float(c["open"]), "high": float(c["high"]),
                "low": float(c["low"]), "close": float(c["close"])}

    def _trim(self) -> None:
        if len(self._closed) > self.max_bars:
            for k in sorted(self._closed)[: len(self._closed) - self.max_bars]:
                del self._closed[k]

    @property
    def tick_age_sec(self) -> float:
        return time.time() - self.last_tick_ts if self.last_tick_ts else float("inf")

    def frame(self) -> pd.DataFrame:
        if not self._closed:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame([self._closed[k] for k in sorted(self._closed)])
        df.index = pd.to_datetime(df.pop("epoch"), unit="s", utc=True)
        return df
