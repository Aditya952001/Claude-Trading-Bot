"""
memory_and_learning.py - Trade journal (SQLite) and the self-learning review loop.

What "learning" means here (deliberately conservative)
  * Every closed trade is stored with the indicators, component votes and
    market regime that existed at entry, plus its R-multiple (PnL / risk).
  * Periodically, `SelfLearner.review()` looks for statistically-supported
    failure patterns and nudges StrategyParams:
        - risk weight per asset class  (can only shrink / restore, never > 1)
        - temporary block of an (asset class, regime) combination
        - tighter / looser RSI entry gates
        - global min_score
        - per-component confluence weights
  * Safety rails:
        - minimum sample sizes before any change
        - small step sizes and hard PARAM_BOUNDS
        - at most one adjustment batch per `rollback_min_trades` trades
        - automatic ROLLBACK if performance after a change is clearly worse
        - it can never touch risk_manager limits
  With few trades, a review will mostly do nothing - that is intended. Treat
  the output as a filter against obviously failing patterns, not as an edge.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from config import LearningConfig
from strategy_engine import COMPONENTS, StrategyParams

log = logging.getLogger("learning")

PARAM_BOUNDS = {
    "rsi_long_max": (58.0, 72.0),
    "rsi_short_min": (28.0, 42.0),
    "min_score": (3.0, 4.5),
    "component_weight": (0.3, 1.5),
    "asset_weight": (0.25, 1.0),
}
BLOCK_DURATION_SEC = 7 * 24 * 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER UNIQUE,
    symbol TEXT, asset_class TEXT, direction TEXT,
    status TEXT DEFAULT 'open',
    entry_time REAL, exit_time REAL,
    entry_price REAL, exit_price REAL,
    stake REAL, multiplier INTEGER, risk_amount REAL,
    sl_amount REAL, tp_amount REAL,
    pnl REAL, r_multiple REAL,
    regime TEXT, confidence REAL, score REAL,
    indicators TEXT, components TEXT,
    contract_type TEXT DEFAULT 'multiplier'
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, report TEXT);
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    batch_index INTEGER,
    batch_trades INTEGER,
    batch_wins INTEGER,
    batch_losses INTEGER,
    batch_win_rate REAL,
    batch_gross_profit REAL,
    batch_gross_loss REAL,
    batch_net_pnl REAL,
    batch_profit_factor REAL,
    overall_trades INTEGER,
    overall_wins INTEGER,
    overall_losses INTEGER,
    overall_win_rate REAL,
    overall_net_pnl REAL,
    overall_profit_factor REAL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS loss_post_mortems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER,
    symbol TEXT,
    direction TEXT,
    pnl REAL,
    entry_price REAL,
    exit_price REAL,
    duration_sec REAL,
    peak_profit REAL,
    exit_reason TEXT,
    entry_indicators TEXT,
    exit_indicators TEXT,
    gemini_post_mortem TEXT,
    failure_category TEXT,
    root_cause TEXT,
    recommended_patch TEXT,
    created_at REAL
);
"""


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------
class MemoryStore:
    def __init__(self, db_path: str):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        try:
            self._db.execute("ALTER TABLE trades ADD COLUMN contract_type TEXT DEFAULT 'multiplier'")
        except Exception:
            pass

    # -- key/value (used by RiskManager and the learner) -- #
    def get_kv(self, key: str) -> Optional[str]:
        with self._lock:
            r = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            return r["value"] if r else None

    def set_kv(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO kv(key,value) VALUES(?,?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # -- trades -- #
    def log_open_trade(self, **t) -> None:
        t["indicators"] = json.dumps(t.get("indicators", {}))
        t["components"] = json.dumps(t.get("components", {}))
        cols = ",".join(t)
        qs = ",".join("?" * len(t))
        with self._lock, self._db:
            self._db.execute(f"INSERT OR IGNORE INTO trades({cols}) VALUES({qs})", tuple(t.values()))

    def update_entry_price(self, contract_id: int, entry_price: float) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE trades SET entry_price=? WHERE contract_id=?",
                             (entry_price, contract_id))

    def log_close_trade(self, contract_id: int, exit_price: float, pnl: float,
                        exit_time: Optional[float] = None) -> None:
        with self._lock, self._db:
            row = self._db.execute("SELECT risk_amount FROM trades WHERE contract_id=?",
                                   (contract_id,)).fetchone()
            risk = (row["risk_amount"] if row else None) or 0.0
            r = pnl / risk if risk > 0 else None
            self._db.execute(
                "UPDATE trades SET status='closed', exit_price=?, pnl=?, r_multiple=?, exit_time=? "
                "WHERE contract_id=?", (exit_price, pnl, r, exit_time or time.time(), contract_id))

    def get_trade(self, contract_id: int) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM trades WHERE contract_id=?", (contract_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["indicators"] = json.loads(d["indicators"] or "{}")
            d["components"] = json.loads(d["components"] or "{}")
            return d

    def log_loss_post_mortem(self, **pm) -> None:
        pm["entry_indicators"] = json.dumps(pm.get("entry_indicators", {}))
        pm["exit_indicators"] = json.dumps(pm.get("exit_indicators", {}))
        pm["gemini_post_mortem"] = json.dumps(pm.get("gemini_post_mortem", {}))
        pm["created_at"] = pm.get("created_at") or time.time()
        cols = ",".join(pm)
        qs = ",".join("?" * len(pm))
        with self._lock, self._db:
            self._db.execute(f"INSERT INTO loss_post_mortems({cols}) VALUES({qs})", tuple(pm.values()))

    def recent_loss_post_mortems(self, limit: int = 10) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM loss_post_mortems ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["entry_indicators"] = json.loads(d["entry_indicators"] or "{}")
                d["exit_indicators"] = json.loads(d["exit_indicators"] or "{}")
                d["gemini_post_mortem"] = json.loads(d["gemini_post_mortem"] or "{}")
                out.append(d)
            return out

    def open_trades(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM trades WHERE status='open'")]

    def sync_broker_transactions(self, transactions: List[dict]) -> int:
        """Reconciles closed transactions from Deriv profit table on startup.
        Ensures stats, trade counts, and daily PnL persist across server restarts."""
        inserted = 0
        with self._lock, self._db:
            for t in transactions:
                cid = t.get("contract_id")
                if not cid:
                    continue
                existing = self._db.execute("SELECT id FROM trades WHERE contract_id=?", (cid,)).fetchone()
                if existing:
                    continue
                sym = t.get("underlying_symbol", "")
                buy = float(t.get("buy_price", 1.0))
                sell = float(t.get("sell_price", 0.0))
                pnl = round(sell - buy, 2)
                p_time = float(t.get("purchase_time", 0))
                s_time = float(t.get("sell_time", p_time))
                ctype_raw = str(t.get("contract_type", "")).lower()
                ctype = "rise_fall" if ("call" in ctype_raw or "put" in ctype_raw) else "multiplier"
                direction = "long" if ("call" in ctype_raw or "up" in ctype_raw) else "short"
                self._db.execute(
                    """
                    INSERT OR IGNORE INTO trades (
                        contract_id, symbol, asset_class, direction, status,
                        entry_time, exit_time, entry_price, exit_price,
                        stake, multiplier, risk_amount, pnl, score, contract_type
                    ) VALUES (?, ?, ?, ?, 'closed', ?, ?, 0.0, 0.0, ?, 0, ?, ?, 4.0, ?)
                    """,
                    (cid, sym, "synced", direction, p_time, s_time, buy, buy, pnl, ctype)
                )
                inserted += 1
        return inserted

    def recent_closed(self, limit: int) -> List[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM trades WHERE status='closed' AND r_multiple IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["indicators"] = json.loads(d["indicators"] or "{}")
            d["components"] = json.loads(d["components"] or "{}")
            out.append(d)
        return out

    def closed_since(self, ts: float) -> List[dict]:
        return [t for t in self.recent_closed(10_000) if (t["exit_time"] or 0) > ts]

    def save_review(self, report: dict) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO reviews(ts, report) VALUES(?,?)",
                             (time.time(), json.dumps(report)))

    def get_symbol_stats(self, symbol: str, limit: int = 15) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT pnl FROM trades WHERE symbol=? AND status='closed' AND pnl IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT ?", (symbol, limit)
            ).fetchall()
        if not rows:
            return {"trades": 0, "wins": 0, "win_rate": 0.5, "net_pnl": 0.0, "consecutive_losses": 0}
        pnls = [r["pnl"] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        consec_loss = 0
        for p in pnls:
            if p < 0:
                consec_loss += 1
            else:
                break
        return {
            "trades": len(pnls),
            "wins": wins,
            "win_rate": wins / len(pnls),
            "net_pnl": sum(pnls),
            "consecutive_losses": consec_loss,
        }

    def rf_edge_stats(self, limit: int = 120,
                      asset_classes: Optional[Sequence[str]] = None) -> dict:
        """
        Evidence summary over recent closed Rise/Fall trades.

        `breakeven` is implied by the payout actually received on winners
        (pnl/stake), so it adapts to symbol/expiry differences. `wr_lo` is the
        95% Wilson lower bound - the honest number to compare against breakeven.
        """
        sql = ("SELECT symbol, direction, pnl, stake, asset_class FROM trades "
               "WHERE contract_type='rise_fall' AND status='closed' AND pnl IS NOT NULL ")
        args: list = []
        acs = tuple(a.lower() for a in (asset_classes or ()) if a)
        if acs:
            sql += f"AND lower(asset_class) IN ({','.join('?' * len(acs))}) "
            args.extend(acs)
        sql += "ORDER BY exit_time DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, tuple(args)).fetchall()
        data = [dict(r) for r in rows]
        if not data:
            return {}

        def wilson_lo(p, n, z=1.96):
            if n <= 0:
                return 0.0
            d = 1 + z * z / n
            c = p + z * z / (2 * n)
            s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            return max(0.0, (c - s) / d)

        def summarise(sub):
            n = len(sub)
            wins = sum(1 for t in sub if t["pnl"] > 0)
            wr = wins / n if n else 0.0
            payouts = [t["pnl"] / t["stake"] for t in sub
                       if t["pnl"] > 0 and (t["stake"] or 0) > 0]
            be = 1.0 / (1.0 + (statistics.median(payouts) if payouts else 0.88))
            return {"n": n, "wr": wr, "wr_lo": wilson_lo(wr, n), "breakeven": be,
                    "pnl": sum(t["pnl"] for t in sub)}

        by_symbol = defaultdict(list)
        by_asset = defaultdict(list)
        for t in data:
            by_symbol[t["symbol"]].append(t)
            by_asset[(t.get("asset_class") or "").lower()].append(t)
        out = summarise(data)
        out["by_symbol"] = {s: summarise(v) for s, v in by_symbol.items()}
        out["by_asset_class"] = {a: summarise(v) for a, v in by_asset.items()}
        return out

    def current_win_streak(self, limit: int = 200) -> int:
        """Consecutive wins ending at the most recent closed trade (any day)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT pnl FROM trades WHERE status='closed' AND pnl IS NOT NULL "
                "ORDER BY COALESCE(exit_time, entry_time) DESC LIMIT ?", (int(limit),)
            ).fetchall()
        streak = 0
        for r in rows:
            if r["pnl"] > 0:
                streak += 1
            else:
                break
        return streak

    def get_day_stats(self, offset: int = 0, basis: str = "entry") -> dict:
        """Closed-trade aggregate for one UTC day.

        offset=0 -> today, 1 -> yesterday.

        basis="entry"    -> trades TAKEN that day (what "today's 4 trades" means
                            to a human and to the journal's trade list).
        basis="settled"  -> trades whose money SETTLED that day. This can differ
                            when a position is opened late and settles after
                            midnight, or when the bot was down overnight.

        Returns zeros for an empty day; it never substitutes an older sample (the
        previous implementation reported every trade since id 278 as "today"
        when the day had no trades).
        """
        offset = max(0, int(offset))
        # Indian Standard Time (IST, UTC+5:30): day boundary flips at 12:00 AM midnight IST
        day_expr = "date('now', '+330 minutes')" if offset == 0 else f"date('now', '+330 minutes', '-{offset} day')"
        time_col = "entry_time" if basis == "entry" else "COALESCE(exit_time, entry_time)"
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(pnl > 0), 0) AS w, "
                "COALESCE(SUM(pnl < 0), 0) AS l, "
                "COALESCE(SUM(pnl), 0.0) AS pnl, "
                f"{day_expr} AS day "
                "FROM trades WHERE status='closed' AND pnl IS NOT NULL "
                f"AND date({time_col}, 'unixepoch', '+330 minutes') = {day_expr}"
            ).fetchone()
        n = int(row["n"] or 0)
        wins = int(row["w"] or 0)
        losses = int(row["l"] or 0)
        return {
            "date": row["day"],
            "basis": basis,
            "total_trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / n * 100.0) if n else 0.0,
            "pnl": float(row["pnl"] or 0.0),
        }

    def get_all_time_stats(self) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(pnl > 0), 0) AS w, "
                "COALESCE(SUM(pnl < 0), 0) AS l, COALESCE(SUM(pnl), 0.0) AS pnl "
                "FROM trades WHERE status='closed' AND pnl IS NOT NULL"
            ).fetchone()
        n = int(row["n"] or 0)
        wins = int(row["w"] or 0)
        return {
            "total_trades": n, "wins": wins, "losses": int(row["l"] or 0),
            "win_rate": (wins / n * 100.0) if n else 0.0,
            "pnl": float(row["pnl"] or 0.0),
        }

    def get_today_audit_stats(self) -> dict:
        """Today's results (trades opened today, UTC) plus cash-settled context.

        Top-level keys keep their original names (total_trades/wins/losses/
        win_rate/day_pnl/win_streak) so existing callers are unaffected.
        `settled_today` covers positions opened earlier that settled today; it is
        only different when a trade crosses the UTC boundary or the bot restarts
        after expiry.
        """
        today = self.get_day_stats(0, "entry")
        return {
            **today,
            "day_pnl": today["pnl"],
            "win_streak": self.current_win_streak(),
            "yesterday": self.get_day_stats(1, "entry"),
            "settled_today": self.get_day_stats(0, "settled"),
            "all_time": self.get_all_time_stats(),
        }

    def log_milestone(self, data: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO milestones (
                    ts, batch_index, batch_trades, batch_wins, batch_losses,
                    batch_win_rate, batch_gross_profit, batch_gross_loss, batch_net_pnl,
                    batch_profit_factor, overall_trades, overall_wins, overall_losses,
                    overall_win_rate, overall_net_pnl, overall_profit_factor, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    data.get("ts", time.time()),
                    data.get("batch_index", 1),
                    data.get("batch_trades", 10),
                    data.get("batch_wins", 0),
                    data.get("batch_losses", 0),
                    data.get("batch_win_rate", 0.0),
                    data.get("batch_gross_profit", 0.0),
                    data.get("batch_gross_loss", 0.0),
                    data.get("batch_net_pnl", 0.0),
                    data.get("batch_profit_factor", 0.0),
                    data.get("overall_trades", 0),
                    data.get("overall_wins", 0),
                    data.get("overall_losses", 0),
                    data.get("overall_win_rate", 0.0),
                    data.get("overall_net_pnl", 0.0),
                    data.get("overall_profit_factor", 0.0),
                    json.dumps(data.get("notes", [])),
                ),
            )

    def get_milestones(self, limit: int = 10) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM milestones ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]


# ----------------------------------------------------------------------------
# Learner
# ----------------------------------------------------------------------------
@dataclass
class ReviewReport:
    ts: float = field(default_factory=time.time)
    n_trades: int = 0
    overall: dict = field(default_factory=dict)
    changes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _stats(trades: List[dict]) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pnl": 0.0}
    rs = [t["r_multiple"] for t in trades]
    return {"n": n, "win_rate": sum(t["pnl"] > 0 for t in trades) / n,
            "avg_r": statistics.fmean(rs), "pnl": sum(t["pnl"] for t in trades)}


def _clamp(x: float, lo_hi) -> float:
    return max(lo_hi[0], min(lo_hi[1], x))


class SelfLearner:
    def __init__(self, store: MemoryStore, cfg: LearningConfig):
        self.store, self.cfg = store, cfg
        self.params = self.load_params()
        self._last_review_ts = float(store.get_kv("last_review_ts") or time.time())
        self._closed_since_review = int(store.get_kv("closed_since_review") or 0)
        self._closed_since_milestone = int(store.get_kv("closed_since_milestone") or 0)

    # -- params persistence -- #
    def load_params(self) -> StrategyParams:
        raw = self.store.get_kv("strategy_params")
        return StrategyParams.from_json(raw) if raw else StrategyParams()

    def save_params(self, p: StrategyParams) -> None:
        self.store.set_kv("strategy_params", p.to_json())
        self.params = p

    # -- scheduling -- #
    def note_trade_closed(self) -> None:
        self._closed_since_review += 1
        self._closed_since_milestone += 1
        self.store.set_kv("closed_since_review", str(self._closed_since_review))
        self.store.set_kv("closed_since_milestone", str(self._closed_since_milestone))

    def check_and_run_milestone(self) -> Optional[dict]:
        """
        Evaluates if a 10-trade milestone was reached.
        If yes: calculates batch metrics, overall metrics, performs deep loss learning,
        saves the milestone to sqlite, and returns the milestone report dict.
        """
        if self._closed_since_milestone < self.cfg.review_every_n_trades:
            return None

        # Milestone reached!
        self._closed_since_milestone = 0
        self.store.set_kv("closed_since_milestone", "0")
        batch_idx = int(self.store.get_kv("milestone_batch_index") or 0) + 1
        self.store.set_kv("milestone_batch_index", str(batch_idx))

        # 1. Fetch batch trades (last 10 closed)
        batch_trades = self.store.recent_closed(self.cfg.review_every_n_trades)
        b_wins = [t for t in batch_trades if t.get("pnl", 0) > 0]
        b_losses = [t for t in batch_trades if t.get("pnl", 0) < 0]
        b_be = [t for t in batch_trades if t.get("pnl", 0) == 0]
        b_gross_profit = sum(t.get("pnl", 0) for t in b_wins)
        b_gross_loss = abs(sum(t.get("pnl", 0) for t in b_losses))
        b_net_pnl = b_gross_profit - b_gross_loss
        b_wr = len(b_wins) / (len(b_wins) + len(b_losses)) if (b_wins or b_losses) else 0.0
        b_pf = (b_gross_profit / b_gross_loss) if b_gross_loss > 0 else (999.0 if b_gross_profit > 0 else 0.0)

        # 2. Fetch overall closed trades
        all_trades = self.store.recent_closed(self.cfg.lookback_trades)
        o_wins = [t for t in all_trades if t.get("pnl", 0) > 0]
        o_losses = [t for t in all_trades if t.get("pnl", 0) < 0]
        o_gross_profit = sum(t.get("pnl", 0) for t in o_wins)
        o_gross_loss = abs(sum(t.get("pnl", 0) for t in o_losses))
        o_net_pnl = o_gross_profit - o_gross_loss
        o_wr = len(o_wins) / (len(o_wins) + len(o_losses)) if (o_wins or o_losses) else 0.0
        o_pf = (o_gross_profit / o_gross_loss) if o_gross_loss > 0 else (999.0 if o_gross_profit > 0 else 0.0)

        notes = []

        # 3. Deep Loss Learning & Auto-Adaptation from this 10-trade batch:
        p = StrategyParams.from_json(self.params.to_json())
        before = p.to_json()

        # A. Analyze long losses for high RSI (overbought top chasing).
        # Evidence floor: a 10-trade batch with 2 unlucky losses is noise. Require
        # a real sample (>=20 closed trades, >=8 losses, >=8 matching losses)
        # before moving a gate that changes every future entry.
        long_losses = [t for t in b_losses if t.get("direction") == "long" and "rsi" in t.get("indicators", {})]
        high_rsi_long_losses = [t for t in long_losses if t["indicators"]["rsi"] >= p.rsi_long_max - 2]
        if len(batch_trades) >= 20 and len(b_losses) >= 8 and len(high_rsi_long_losses) >= 8:
            new_rsi_max = _clamp(p.rsi_long_max - 1.5, PARAM_BOUNDS["rsi_long_max"])
            if new_rsi_max != p.rsi_long_max:
                notes.append(f"AI Loss Learning: {len(high_rsi_long_losses)} long losses had high RSI -> lowered rsi_long_max from {p.rsi_long_max:.1f} to {new_rsi_max:.1f} to block overbought traps")
                p.rsi_long_max = new_rsi_max

        # B. Analyze short losses for low RSI (oversold bottom chasing)
        short_losses = [t for t in b_losses if t.get("direction") == "short" and "rsi" in t.get("indicators", {})]
        low_rsi_short_losses = [t for t in short_losses if t["indicators"]["rsi"] <= p.rsi_short_min + 2]
        if len(batch_trades) >= 20 and len(b_losses) >= 8 and len(low_rsi_short_losses) >= 8:
            new_rsi_min = _clamp(p.rsi_short_min + 1.5, PARAM_BOUNDS["rsi_short_min"])
            if new_rsi_min != p.rsi_short_min:
                notes.append(f"AI Loss Learning: {len(low_rsi_short_losses)} short losses had low RSI -> raised rsi_short_min from {p.rsi_short_min:.1f} to {new_rsi_min:.1f} to block oversold traps")
                p.rsi_short_min = new_rsi_min

        # C. Confluence-score ratchet: REMOVED.
        # Raising min_score whenever a batch wins <60% is a one-way ratchet on a
        # coin flip; it permanently shrinks trade count without improving the
        # win rate (14-day replay: score buckets had no monotonic relationship
        # with outcome). min_score now only changes if review() can support it.

        if p.to_json() != before:
            p.version += 1
            self.save_params(p)

        milestone_data = {
            "ts": time.time(),
            "batch_index": batch_idx,
            "batch_trades": len(batch_trades),
            "batch_wins": len(b_wins),
            "batch_losses": len(b_losses),
            "batch_be": len(b_be),
            "batch_win_rate": b_wr,
            "batch_gross_profit": b_gross_profit,
            "batch_gross_loss": b_gross_loss,
            "batch_net_pnl": b_net_pnl,
            "batch_profit_factor": b_pf,
            "overall_trades": len(all_trades),
            "overall_wins": len(o_wins),
            "overall_losses": len(o_losses),
            "overall_win_rate": o_wr,
            "overall_gross_profit": o_gross_profit,
            "overall_gross_loss": o_gross_loss,
            "overall_net_pnl": o_net_pnl,
            "overall_profit_factor": o_pf,
            "notes": notes,
            "params_version": p.version,
            "min_score": p.min_score,
            "rsi_long_max": p.rsi_long_max,
            "rsi_short_min": p.rsi_short_min,
        }
        self.store.log_milestone(milestone_data)
        return milestone_data

    def should_review(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        if self._closed_since_review >= self.cfg.review_every_n_trades:
            return True
        return (now - self._last_review_ts) >= self.cfg.review_every_hours * 3600 \
            and self._closed_since_review > 0

    # -- the review -- #
    def review(self) -> ReviewReport:
        now = time.time()
        rep = ReviewReport()
        self._last_review_ts, self._closed_since_review = now, 0
        self.store.set_kv("last_review_ts", str(now))
        self.store.set_kv("closed_since_review", "0")

        trades = self.store.recent_closed(self.cfg.lookback_trades)
        rep.n_trades = len(trades)
        rep.overall = _stats(trades)
        if len(trades) < self.cfg.min_trades_for_review:
            rep.notes.append(f"only {len(trades)} trades (< {self.cfg.min_trades_for_review}); no changes")
            return self._finish(rep)

        # 1) Judge the previous adjustment (rollback or let it settle)
        last_adj = self.store.get_kv("last_adjustment")
        if last_adj:
            adj = json.loads(last_adj)
            since = self.store.closed_since(adj["ts"])
            if len(since) < self.cfg.rollback_min_trades:
                rep.notes.append(f"last change still settling ({len(since)}/{self.cfg.rollback_min_trades} trades)")
                return self._finish(rep)
            s = _stats(since)
            if s["avg_r"] < 0 and s["avg_r"] < adj["baseline_avg_r"] - 0.3:
                self.save_params(StrategyParams.from_json(adj["prev_params"]))
                self.store.set_kv("last_adjustment", "")
                rep.changes.append(f"ROLLBACK: avg R since change {s['avg_r']:.2f} vs baseline "
                                   f"{adj['baseline_avg_r']:.2f}")
                return self._finish(rep)
            self.store.set_kv("last_adjustment", "")      # change judged acceptable

        # 2) Compute adjustments
        p = StrategyParams.from_json(self.params.to_json())     # deep copy
        before = p.to_json()
        self._tune_asset_weights(p, trades, rep)
        self._tune_regime_blocks(p, trades, rep, now)
        self._tune_rsi(p, trades, rep)
        self._tune_min_score(p, rep)
        self._tune_components(p, trades, rep)

        if p.to_json() != before:
            p.version += 1
            self.store.set_kv("last_adjustment", json.dumps({
                "ts": now, "prev_params": before, "baseline_avg_r": rep.overall["avg_r"]}))
            self.save_params(p)
        else:
            rep.notes.append("no statistically supported changes")
        return self._finish(rep)

    # -- individual tuners -- #
    def _tune_asset_weights(self, p, trades, rep):
        groups = defaultdict(list)
        for t in trades:
            groups[t["asset_class"]].append(t)
        for ac, ts in groups.items():
            s = _stats(ts)
            if s["n"] < self.cfg.min_group_trades:
                continue
            w = p.asset_class_weights.get(ac, 1.0)
            if s["avg_r"] < -0.3:
                new = _clamp(w * 0.8, PARAM_BOUNDS["asset_weight"])
            elif s["avg_r"] > 0.2 and w < 1.0:
                new = _clamp(w * 1.15, PARAM_BOUNDS["asset_weight"])
            else:
                continue
            if abs(new - w) > 1e-9:
                p.asset_class_weights[ac] = round(new, 3)
                rep.changes.append(f"{ac}: risk weight {w:.2f} -> {new:.2f} "
                                   f"(n={s['n']}, avgR={s['avg_r']:.2f})")

    def _tune_regime_blocks(self, p, trades, rep, now):
        p.blocked = {k: v for k, v in p.blocked.items() if v > now}          # purge expired
        groups = defaultdict(list)
        for t in trades:
            groups[(t["asset_class"], t["regime"])].append(t)
        for (ac, rg), ts in groups.items():
            s = _stats(ts)
            key = f"{ac}|{rg}"
            if s["n"] >= 6 and s["avg_r"] < -0.5 and s["win_rate"] < 0.35 and key not in p.blocked:
                p.blocked[key] = now + BLOCK_DURATION_SEC
                rep.changes.append(f"blocked {key} for 7d (n={s['n']}, win={s['win_rate']:.0%}, "
                                   f"avgR={s['avg_r']:.2f})")

    def _tune_rsi(self, p, trades, rep):
        # Samples of 5-8 trades produced random gate moves. Require 20 for a
        # tightening and 30 for a loosening so a single streak cannot reshape
        # every future entry.
        longs = [t for t in trades if t["direction"] == "long" and "rsi" in t["indicators"]]
        hi = [t for t in longs if t["indicators"]["rsi"] >= p.rsi_long_max - 8]
        if len(hi) >= 20 and _stats(hi)["avg_r"] < -0.2:
            new = _clamp(p.rsi_long_max - 3, PARAM_BOUNDS["rsi_long_max"])
            if new != p.rsi_long_max:
                rep.changes.append(f"rsi_long_max {p.rsi_long_max:.0f} -> {new:.0f} "
                                   f"(high-RSI longs avgR={_stats(hi)['avg_r']:.2f}, n={len(hi)})")
                p.rsi_long_max = new
        elif len(hi) >= 30 and _stats(hi)["avg_r"] > 0.3:
            new = _clamp(p.rsi_long_max + 1, PARAM_BOUNDS["rsi_long_max"])
            if new != p.rsi_long_max:
                rep.changes.append(f"rsi_long_max {p.rsi_long_max:.0f} -> {new:.0f} (high-RSI longs profitable)")
                p.rsi_long_max = new

        shorts = [t for t in trades if t["direction"] == "short" and "rsi" in t["indicators"]]
        lo = [t for t in shorts if t["indicators"]["rsi"] <= p.rsi_short_min + 8]
        if len(lo) >= 20 and _stats(lo)["avg_r"] < -0.2:
            new = _clamp(p.rsi_short_min + 3, PARAM_BOUNDS["rsi_short_min"])
            if new != p.rsi_short_min:
                rep.changes.append(f"rsi_short_min {p.rsi_short_min:.0f} -> {new:.0f} "
                                   f"(low-RSI shorts avgR={_stats(lo)['avg_r']:.2f}, n={len(lo)})")
                p.rsi_short_min = new
        elif len(lo) >= 30 and _stats(lo)["avg_r"] > 0.3:
            new = _clamp(p.rsi_short_min - 1, PARAM_BOUNDS["rsi_short_min"])
            if new != p.rsi_short_min:
                rep.changes.append(f"rsi_short_min {p.rsi_short_min:.0f} -> {new:.0f} (low-RSI shorts profitable)")
                p.rsi_short_min = new

    def _tune_min_score(self, p, rep):
        """
        Intentionally a no-op.

        min_score used to ratchet upward whenever overall avgR dipped, which
        permanently removed trades without evidence that confluence strength
        predicts the outcome. The 14-day replay (862 trades) showed score
        buckets 2-5 all within noise of each other and of breakeven, so
        changing this gate cannot manufacture an edge - it only trades less.
        Keep the configured value; revisit only with a pre-registered test.
        """
        return

    def _tune_components(self, p, trades, rep):
        for comp in COMPONENTS:
            agree = [t for t in trades
                     if t["components"].get(comp) == (1 if t["direction"] == "long" else -1)]
            if len(agree) < 10:
                continue
            s = _stats(agree)
            w = p.component_weights.get(comp, 1.0)
            if s["avg_r"] < -0.4:
                new = _clamp(w * 0.9, PARAM_BOUNDS["component_weight"])
            elif s["avg_r"] > 0.4:
                new = _clamp(w * 1.05, PARAM_BOUNDS["component_weight"])
            else:
                continue
            if abs(new - w) > 1e-9:
                p.component_weights[comp] = round(new, 3)
                rep.changes.append(f"component '{comp}' weight {w:.2f} -> {new:.2f} "
                                   f"(n={s['n']}, avgR={s['avg_r']:.2f})")

    def _finish(self, rep: ReviewReport) -> ReviewReport:
        self.store.save_review(rep.__dict__)
        log.info("REVIEW n=%d avgR=%.2f win=%.0f%% | changes=%s | notes=%s",
                 rep.n_trades, rep.overall.get("avg_r", 0), 100 * rep.overall.get("win_rate", 0),
                 rep.changes or "none", rep.notes or "-")
        return rep
