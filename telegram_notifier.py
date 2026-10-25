"""
telegram_notifier.py - Asynchronous Telegram notification service for autonomous trading.

Sends real-time alerts for:
1. Trade Entry (Open): Contract ID, asset, direction, dynamic duration, stake, indicator checklist.
2. Trade Settlement (Close): Outcome (WIN/LOSS), PnL, duration held, payout %, updated equity.
3. Periodic Live Engine Audit Update (every 15 minutes): Session metrics, win rate, open positions.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger("telegram")


def _fmt_day(day: Optional[Dict[str, Any]]) -> str:
    """One-line record for a get_day_stats() payload, or 'no trades'."""
    if not day or not day.get("total_trades"):
        return "no trades"
    return (f"{day.get('wins', 0)}W / {day.get('losses', 0)}L "
            f"({day.get('win_rate', 0.0):.1f}%) {day.get('pnl', 0.0):+.2f} USD")


def _cash_line(opened: Dict[str, Any], settled: Optional[Dict[str, Any]],
               label: str = "today") -> str:
    """Cash-settlement line, only when it differs from the opened-trades view.

    A day's opened trades and the money that settled on it diverge when a
    position crosses UTC midnight or the bot is down at expiry. Saying so
    explicitly stops the two figures from looking like a data error.
    """
    if not settled or not settled.get("total_trades"):
        return ""
    diff = int(settled.get("total_trades", 0)) - int(opened.get("total_trades", 0))
    if diff == 0:
        return ""
    note = (f"incl. {diff} opened earlier" if diff > 0
            else f"{abs(diff)} opened that day settled after midnight")
    return (f"Settled {label} (cash): {settled.get('total_trades', 0)} "
            f"({settled.get('wins', 0)}W / {settled.get('losses', 0)}L) "
            f"{settled.get('pnl', 0.0):+.2f} USD, {note}")


class TelegramNotifier:
    def __init__(self, bot_token: str = "", chat_id: str = "", enabled: bool = True):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = bool(enabled and self.bot_token and self.chat_id)
        if self.enabled:
            # Mask token for logging: 123456:ABC...XYZ
            masked = self.bot_token[:6] + "..." + self.bot_token[-4:] if len(self.bot_token) > 12 else "***"
            log.info("Telegram notifications ENABLED (Bot: %s, Chat: %s)", masked, self.chat_id)
        else:
            log.info("Telegram notifications DISABLED (missing token or chat_id)")

    def _ensure_enabled(self) -> bool:
        if self.enabled:
            return True
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv(".env", override=False)
            tok = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            cid = os.getenv("TELEGRAM_CHAT_ID", "").strip()
            if tok and cid:
                self.bot_token = tok
                self.chat_id = cid
                self.enabled = True
                masked = self.bot_token[:6] + "..." + self.bot_token[-4:] if len(self.bot_token) > 12 else "***"
                log.info("Telegram notifications DYNAMICALLY ENABLED from .env (Bot: %s, Chat: %s)", masked, self.chat_id)
                return True
        except Exception:
            pass
        return False

    async def send_message(self, text: str, parse_mode: Optional[str] = "HTML") -> bool:
        """Sends a message to Telegram via async executor (non-blocking)."""
        if not self._ensure_enabled():
            return False

        def _post(mode: Optional[str] = parse_mode) -> bool:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if mode:
                payload["parse_mode"] = mode
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True
                    log.warning("Telegram API response code: %s", resp.status)
                    return False
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and mode:
                    # Retry immediately as plain text if HTML tags were unparseable
                    try:
                        payload.pop("parse_mode", None)
                        data_plain = json.dumps(payload).encode("utf-8")
                        req_plain = urllib.request.Request(
                            url, data=data_plain, headers={"Content-Type": "application/json"}, method="POST"
                        )
                        with urllib.request.urlopen(req_plain, timeout=10) as resp2:
                            return resp2.status == 200
                    except Exception as exc2:
                        log.warning("Failed plain text retry: %s", exc2)
                log.warning("Failed to send Telegram message: %s", exc)
                return False
            except Exception as exc:
                log.warning("Failed to send Telegram message: %s", exc)
                return False

        try:
            return await asyncio.to_thread(_post)
        except Exception as exc:
            log.warning("Telegram sending error: %s", exc)
            return False

    async def notify_trade_open(self, trade: Dict[str, Any]) -> None:
        """Formats and dispatches a Trade Open alert."""
        if not self._ensure_enabled():
            return

        symbol = trade.get("symbol", "UNKNOWN")
        direction = trade.get("direction", "long").upper()
        contract_type = trade.get("contract_type", "rise_fall").upper()
        cid = trade.get("contract_id", 0)
        stake = trade.get("stake", 1.0)
        entry_price = trade.get("entry_price", 0.0)
        duration = trade.get("duration", 0)
        unit = trade.get("duration_unit", "m")
        reason = html.escape(str(trade.get("duration_reason", "Dynamic Market Adaptive")))
        score = trade.get("score", 0.0)
        ind = trade.get("indicators", {}) or {}

        dir_emoji = "🟢 <b>RISE / CALL</b>" if direction in ("LONG", "CALL") else "🔴 <b>FALL / PUT</b>"
        adx_val = ind.get("adx", 0.0)
        rsi_val = ind.get("rsi", 0.0)
        pct_b = ind.get("bb_pctb", ind.get("pct_b", 0.0))
        ema_f = ind.get("ema_fast", 0.0)
        ema_s = ind.get("ema_slow", 0.0)

        msg = (
            f"🚀 <b>TRADE OPENED [{contract_type}]</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Asset:</b> <code>{symbol}</code>\n"
            f"• <b>Direction:</b> {dir_emoji}\n"
            f"• <b>Contract ID:</b> <code>#{cid}</code>\n"
            f"• <b>Stake:</b> ${stake:.2f} USD\n"
            f"• <b>Entry Spot:</b> <code>{entry_price:.4f}</code>\n"
            f"• <b>Expiry Duration:</b> <b>{duration}{unit}</b>\n"
            f"  <i>({reason})</i>\n"
            f"• <b>Confluence Score:</b> {score:+.2f} / 5.0\n"
            f"• <b>Indicators:</b>\n"
            f"  - ADX: <code>{adx_val:.2f}</code> (Trend Strength)\n"
            f"  - RSI: <code>{rsi_val:.1f}</code>\n"
            f"  - Bollinger %B: <code>{pct_b:.3f}</code>\n"
            f"  - EMA Fast/Slow: <code>{ema_f:.4f} / {ema_s:.4f}</code>\n"
            f"⏱ <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )
        await self.send_message(msg)

    async def notify_trade_close(self, close_data: Dict[str, Any]) -> None:
        """Formats and dispatches a Trade Closed alert."""
        if not self._ensure_enabled():
            return

        cid = close_data.get("contract_id", 0)
        symbol = close_data.get("symbol", "UNKNOWN")
        direction = close_data.get("direction", "long").upper()
        pnl = close_data.get("pnl", 0.0)
        entry_price = close_data.get("entry_price", 0.0)
        exit_price = close_data.get("exit_price", 0.0)
        stake = close_data.get("stake", 1.0)
        duration_sec = close_data.get("duration_sec", 0.0)
        win_streak = close_data.get("win_streak", 0)
        total_trades = close_data.get("total_trades", 0)
        wins = close_data.get("wins", 0)
        losses = close_data.get("losses", 0)
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        balance = close_data.get("balance", 0.0)
        day_pnl = close_data.get("day_pnl", 0.0)
        yesterday = close_data.get("yesterday") or {}
        settled = close_data.get("settled_today") or {}

        is_win = pnl > 0
        if is_win:
            payout_pct = (pnl / stake * 100.0) if stake > 0 else 0.0
            header = f"✅ <b>TRADE WON (+{payout_pct:.1f}% Payout)</b>"
            pnl_line = f"• <b>Realized PnL:</b> <b>+${pnl:.2f} USD</b> 💰"
        else:
            # Was hardcoded to "-$1.00 Stake" even when the stake was different.
            header = (f"❌ <b>TRADE LOSS (-${abs(pnl):.2f} of "
                      f"${stake:.2f} stake)</b>")
            pnl_line = f"• <b>Realized PnL:</b> <b>-${abs(pnl):.2f} USD</b>"

        streak_str = f"🔥 {win_streak} Consecutive Wins" if win_streak > 1 else ("1 Win" if win_streak == 1 else "Reset")
        dur_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
        cash = _cash_line({"total_trades": total_trades, "wins": wins,
                           "losses": losses, "pnl": day_pnl}, settled)
        carry_line = f"• <b>{cash}</b>\n" if cash else ""

        msg = (
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Asset:</b> <code>{symbol}</code> ({direction})\n"
            f"• <b>Contract ID:</b> <code>#{cid}</code>\n"
            f"• <b>Entry ➔ Exit:</b> <code>{entry_price:.4f}</code> ➔ <code>{exit_price:.4f}</code>\n"
            f"• <b>Hold Time:</b> {dur_str}\n"
            f"{pnl_line}\n"
            f"• <b>Active Win Streak:</b> {streak_str}\n"
            f"• <b>Updated Balance:</b> <b>${balance:.2f} USD</b>\n"
            f"• <b>Today (opened, UTC):</b> <b>{wins}W / {losses}L ({win_rate:.1f}% Win Rate)</b>\n"
            f"{carry_line}"
            f"• <b>Yesterday:</b> {_fmt_day(yesterday)}\n"
            f"• <b>Today's PnL (opened today):</b> <b>{day_pnl:+.2f} USD</b>\n"
            f"⏱ <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>"
        )
        await self.send_message(msg)

    async def notify_daily_summary(self, day: Dict[str, Any],
                                   all_time: Optional[Dict[str, Any]] = None,
                                   settled: Optional[Dict[str, Any]] = None) -> None:
        """Formats the end-of-day (UTC rollover) summary for the completed day."""
        if not self._ensure_enabled():
            return

        n = int(day.get("total_trades", 0))
        wins = int(day.get("wins", 0))
        losses = int(day.get("losses", 0))
        wr = float(day.get("win_rate", 0.0))
        pnl = float(day.get("pnl", 0.0))
        avg = (pnl / n) if n else 0.0
        mark = "🟢" if pnl >= 0 else "🔴"
        allt = all_time or {}
        settled = settled or {}
        cash = _cash_line({"total_trades": n, "wins": wins, "losses": losses,
                           "pnl": pnl}, settled, label="that day")
        carry_line = f"• <b>{cash}</b>\n" if cash else ""

        all_line = ""
        if allt.get("total_trades"):
            all_line = (f"• <b>All-time Journal:</b> {allt.get('total_trades', 0)} Trades | "
                        f"{allt.get('win_rate', 0.0):.1f}% | {allt.get('pnl', 0.0):+.2f} USD\n")

        msg = (
            f"🗓 <b>DAILY SUMMARY — {day.get('date', 'previous day')} (UTC)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Trades Opened:</b> <b>{n} ({wins}W / {losses}L)</b>\n"
            f"• <b>Win Rate:</b> <b>{wr:.1f}%</b> (breakeven ~53% at this payout)\n"
            f"• <b>Net PnL:</b> <b>{pnl:+.2f} USD</b> {mark}\n"
            f"• <b>Average per Trade:</b> {avg:+.3f} USD\n"
            f"{carry_line}"
            f"{all_line}"
            f"⏱ <i>Day closed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
        )
        await self.send_message(msg)

    async def notify_engine_audit(self, audit: Dict[str, Any]) -> None:
        """Formats and dispatches the 15-minute Periodic Live Engine Audit Update."""
        if not self._ensure_enabled():
            return

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        open_pos = audit.get("open_positions", 0)
        balance = audit.get("balance", 0.0)
        total_trades = audit.get("total_trades", 0)
        wins = audit.get("wins", 0)
        losses = audit.get("losses", 0)
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        day_pnl = audit.get("day_pnl", 0.0)
        win_streak = audit.get("win_streak", 0)
        day_label = audit.get("day_label", "today")
        yesterday = audit.get("yesterday") or {}
        settled = audit.get("settled_today") or {}
        all_time = audit.get("all_time") or {}
        pnl_mark = audit.get("pnl_mark") or ("🟢" if day_pnl >= 0 else "🔴")
        symbol_count = audit.get("symbol_count", 0)
        loss_clamp = audit.get("loss_clamp", 4)
        cash = _cash_line({"total_trades": total_trades, "wins": wins,
                           "losses": losses, "pnl": day_pnl}, settled)
        carry_line = f"• <b>{cash}</b>\n" if cash else ""

        msg = (
            f"📊 <b>LIVE ENGINE AUDIT UPDATE (15-Min Report)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Scan Engine Status:</b> {ts} bar close nominal across {symbol_count} symbols\n"
            f"• <b>Active Open Contracts:</b> <b>{open_pos}</b>\n"
            f"• <b>Current Account Balance:</b> <b>${balance:.2f} USD</b>\n"
            f"• <b>Today (opened, UTC, {day_label}):</b> <b>{total_trades} Trades ({wins}W / {losses}L)</b>\n"
            f"• <b>Today's Win Rate:</b> <b>{win_rate:.1f}%</b> (breakeven ~53% at this payout)\n"
            f"• <b>Today's PnL (opened today):</b> <b>{day_pnl:+.2f} USD</b> {pnl_mark}\n"
            f"{carry_line}"
            f"• <b>Yesterday:</b> {_fmt_day(yesterday)}\n"
            f"• <b>All-time Journal:</b> {all_time.get('total_trades', 0)} Trades "
            f"({all_time.get('wins', 0)}W / {all_time.get('losses', 0)}L) | "
            f"{all_time.get('win_rate', 0.0):.1f}% | {all_time.get('pnl', 0.0):+.2f} USD\n"
            f"• <b>Active Win Streak:</b> {win_streak} Consecutive Wins\n"
            f"• <b>Loss Clamp:</b> {loss_clamp} consecutive losses → cooldown\n"
            f"• <b>Active Filters:</b> regime + ATR gates, Rise/Fall edge guard\n"
            f"• <b>System Health:</b> Streaming continuous 1m bars cleanly"
        )
        await self.send_message(msg)
