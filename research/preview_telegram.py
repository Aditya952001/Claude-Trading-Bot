"""
research/preview_telegram.py - Render the Telegram messages offline.

No network calls: send_message is stubbed, so this never posts to the chat.
Use it to verify that the numbers in each alert are right before a session.

    python research/preview_telegram.py           # plain text
    python research/preview_telegram.py --html    # raw HTML payloads
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory_and_learning import MemoryStore          # noqa: E402
from telegram_notifier import TelegramNotifier       # noqa: E402

DB = ROOT / "data" / "bot_memory.sqlite3"


def snapshot_db() -> tuple[str, Path]:
    """Consistent copy of the live journal (read-only source, WAL included).

    Messages are rendered from the copy so previewing can never write to, or
    lock, the database the trading bot is using.
    """
    tmp = Path(tempfile.mkdtemp(prefix="tg_preview_")) / "copy.sqlite3"
    src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    return str(tmp), tmp


class PreviewNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__(enabled=False)
        self.sent: list[str] = []

    def _ensure_enabled(self) -> bool:
        return True

    async def send_message(self, text: str, parse_mode: str | None = "HTML") -> bool:
        self.sent.append(text)
        return True


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


async def main(show_html: bool, balance: float | None = None) -> None:
    db_path, _tmp = snapshot_db()
    store = MemoryStore(db_path)
    stats = store.get_today_audit_stats()
    # Offline preview has no broker call; --balance lets you mirror the real one.
    shown_balance = balance if balance is not None else 10000.0 + stats["day_pnl"]
    trade = store._db.execute(
        "SELECT * FROM trades WHERE status='closed' AND pnl IS NOT NULL "
        "ORDER BY COALESCE(exit_time, entry_time) DESC LIMIT 1"
    ).fetchone()
    row = dict(trade) if trade else {}

    settled_today = stats.get("settled_today") or {}
    carried = max(0, int(settled_today.get("total_trades", 0)) - int(stats["total_trades"]))

    audit = {
        "open_positions": 1,
        "balance": shown_balance,
        "total_trades": stats["total_trades"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "day_pnl": stats["day_pnl"],
        "win_streak": stats["win_streak"],
        "day_label": stats.get("date"),
        "yesterday": stats.get("yesterday"),
        "all_time": stats.get("all_time"),
        "settled_today": settled_today,
        "carried": carried,
        "symbol_count": 23,
        "loss_clamp": 4,
    }
    close = {
        "contract_id": row.get("contract_id", 0),
        "symbol": row.get("symbol", "1HZ100V"),
        "direction": row.get("direction", "long"),
        "pnl": float(row.get("pnl") or 0.0),
        "entry_price": float(row.get("entry_price") or 0.0),
        "exit_price": float(row.get("exit_price") or 0.0),
        "stake": float(row.get("stake") or 1.0),
        "duration_sec": float(row.get("duration_sec") or 240),
        "win_streak": stats["win_streak"],
        "total_trades": stats["total_trades"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "balance": audit["balance"],
        "day_pnl": stats["day_pnl"],
        "yesterday": stats.get("yesterday"),
        "settled_today": settled_today,
        "carried": carried,
    }
    open_ = {
        "contract_id": 999,
        "symbol": row.get("symbol", "1HZ90V"),
        "direction": row.get("direction", "long"),
        "contract_type": "rise_fall",
        "stake": float(row.get("stake") or 1.0),
        "entry_price": float(row.get("entry_price") or 0.0),
        "duration": 5,
        "duration_unit": "m",
        "duration_reason": "5m [1s High-Freq Index]",
        "score": 4.0,
        # mirrors a real journal snapshot (bb_pctb, not pct_b)
        "indicators": {},
    }
    raw_ind = store._db.execute(
        "SELECT indicators FROM trades WHERE indicators IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if raw_ind and raw_ind["indicators"]:
        open_["indicators"] = json.loads(raw_ind["indicators"])

    n = PreviewNotifier()
    await n.notify_trade_open(open_)
    await n.notify_trade_close(close)
    await n.notify_engine_audit(audit)
    await n.notify_daily_summary(stats.get("yesterday") or {}, stats.get("all_time"),
                                 store.get_day_stats(1, "settled"))

    for i, msg in enumerate(n.sent, 1):
        print(f"\n{'-' * 70}\nMESSAGE {i}\n{'-' * 70}")
        print(msg if show_html else strip_html(msg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--balance", type=float, default=None, help="live broker balance to display")
    args = ap.parse_args()
    asyncio.run(main(args.html, args.balance))
