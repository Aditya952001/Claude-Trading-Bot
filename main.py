"""
main.py - Autonomous orchestrator.

Pipeline per closed candle:
    CandleFeed -> StrategyEngine -> RiskManager (veto/sizing) -> DerivClient
              -> contract tracking -> MemoryStore -> SelfLearner (periodic)

Usage:
    python main.py --preflight      # connect, authorize, validate symbols, exit
    python main.py --dry-run        # full pipeline, but NO orders are sent
    python main.py                  # trade (demo unless DERIV_ENV=live + confirmation)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field, replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from config import Config, Environment, RiskConfig, SymbolConfig, TradeMode
from deriv_client import CandleFeed, DerivAPIError, DerivClient
from gemini_advisor import GeminiAdvisor
from memory_and_learning import MemoryStore, SelfLearner
from risk_manager import (HARD_MAX_TOTAL_OPEN_RISK, OpenPosition, RiskManager,
                          TradeIntent)
from strategy_engine import (StrategyEngine, compute_indicators,
                             rf_symbol_excluded)
from telegram_notifier import TelegramNotifier

log = logging.getLogger("bot")


@dataclass
class ActiveTradeTracker:
    contract_id: int
    symbol: str
    direction: str
    stake: float
    entry_price: float
    entry_time: float
    contract_type: str = "multiplier"     # 'multiplier' | 'rise_fall'
    peak_profit: float = 0.0
    current_profit: float = 0.0
    bars_held: int = 0
    consecutive_loss_bars: int = 0
    selling: bool = False
    recent_profits: list = field(default_factory=list)
    green_shield_active: bool = False
    exit_reason: str = "Broker settlement / market close"
    entry_indicators: dict = field(default_factory=dict)
    pre_trade_veto: str = ""


def setup_logging(log_dir: str, level: str) -> None:
    import sys
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-14s %(message)s")
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


class TradingBot:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.client = DerivClient(cfg.deriv)
        self.strategy = StrategyEngine(cfg.strategy)
        self.store = MemoryStore(cfg.learning.db_path)
        self.learner = SelfLearner(self.store, cfg.learning)
        # Two isolated risk ledgers so Multiplier and Rise/Fall cannot veto,
        # halt or force-close each other. Only HYBRID needs the split; the
        # single-strategy modes keep the legacy shared ledger (and its
        # persisted `risk_state` counters) so an upgrade does not reset them.
        if cfg.trade_mode is TradeMode.HYBRID and cfg.hybrid_isolated_risk:
            self.risk_books = {
                "multiplier": RiskManager(cfg.risk, store=self.store,
                                          rf_asset_classes=cfg.rf_allowed_assets,
                                          book="multiplier"),
                "rise_fall": RiskManager(cfg.risk, store=self.store,
                                         rf_asset_classes=cfg.rf_allowed_assets,
                                         book="rise_fall"),
            }
        else:
            shared = RiskManager(cfg.risk, store=self.store,
                                 rf_asset_classes=cfg.rf_allowed_assets)
            self.risk_books = {"multiplier": shared, "rise_fall": shared}
        # Legacy alias: used for book-agnostic tasks (kill switch, flattening,
        # global status). Position accounting always goes through _book_for().
        self.risk = self.risk_books["multiplier"]
        # Equity for sizing/limits must include the other book's open stakes.
        for name, book in self.risk_books.items():
            others = [b for n, b in self.risk_books.items() if n != name and b is not book]
            if others:
                book.other_stakes_fn = (
                    lambda others=others: sum(p.stake for b in others
                                              for p in b.open_positions.values()))
        # Dynamic symbol ownership in HYBRID mode is derived from each book's
        # open positions: the book that opens first holds the symbol until the
        # contract closes. This is not a static split — both books scan their
        # full capability sets.
        self.gemini = GeminiAdvisor()
        self.telegram = TelegramNotifier(
            bot_token=cfg.telegram.bot_token,
            chat_id=cfg.telegram.chat_id,
            enabled=cfg.telegram.enabled,
        )
        self._last_audit_report_time: float = 0.0
        self._last_bar_close_minute: int = -1

        self.symbols: Dict[str, SymbolConfig] = {}
        self.feeds: Dict[str, CandleFeed] = {}
        self._feed_subs = []
        self._contract_subs: Dict[int, object] = {}
        self._active_trackers: Dict[int, ActiveTradeTracker] = {}
        self._finalized: set[int] = set()
        self._entry_set: set[int] = set()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._flattened = False

    # ------------------------------------------------------------------ #
    # Strategy isolation helpers
    # ------------------------------------------------------------------ #
    def _book_for(self, contract_type: str) -> RiskManager:
        """Return the risk ledger that owns this contract type."""
        return self.risk_books["rise_fall" if contract_type == "rise_fall" else "multiplier"]

    def _leg_enabled(self, book: str, symbol: str) -> bool:
        """Apply the optional per-book allow-list (empty = all capable symbols)."""
        wl = self.cfg.hybrid_rf_symbols if book == "rise_fall" else self.cfg.hybrid_mult_symbols
        return not wl or symbol.upper() in {s.upper() for s in wl}

    def _scan_contract_type(self, symbol: str, sc: SymbolConfig) -> str:
        """Contract type used for the engine scan (affects Rise/Fall exclusions).

        Dual-capable symbols are scanned as Multiplier so the Rise/Fall-only
        defect filter cannot hide a valid Multiplier signal; the Rise/Fall leg
        is gated separately.
        """
        if self.cfg.trade_mode is TradeMode.RISE_FALL:
            return "rise_fall"
        if self.cfg.trade_mode is TradeMode.MULTIPLIER:
            return "multiplier"
        if sc.supports_multiplier and self._leg_enabled("multiplier", symbol):
            return "multiplier"
        if sc.supports_rise_fall and self._leg_enabled("rise_fall", symbol):
            return "rise_fall"
        return "multiplier"

    def _cross_strategy_holder(self, symbol: str, contract_type: str) -> Optional[RiskManager]:
        """Return the OTHER book if it holds this symbol, else None.

        Dynamic ownership guard: whichever book opens first on a symbol keeps
        it until that contract closes, so the two strategies never hold the
        same instrument at the same time.
        """
        if not self.cfg.hybrid_block_cross_symbol:
            return None
        other = self._book_for("multiplier" if contract_type == "rise_fall" else "rise_fall")
        if other is self._book_for(contract_type):
            # Legacy shared-ledger mode: there is no "other" book to guard.
            return None
        return other if other.has_open_on_symbol(symbol) else None

    def _risk_snapshot(self) -> dict:
        """Aggregate status across the isolated books."""
        mult = self.risk_books["multiplier"].status()
        rf = self.risk_books["rise_fall"].status()
        if self.cfg.trade_mode is TradeMode.HYBRID and self.cfg.hybrid_isolated_risk:
            return {
                "day": mult["day"],
                "day_start_equity": mult["day_start_equity"],
                "realized_pnl_today": mult["realized_pnl_today"] + rf["realized_pnl_today"],
                "trades_today": mult["trades_today"] + rf["trades_today"],
                "open": mult["open"] + rf["open"],
                "consecutive_losses": max(mult["consecutive_losses"], rf["consecutive_losses"]),
                "halted": mult["halted"] or rf["halted"],
                "daily_limit_hit": mult["daily_limit_hit"] or rf["daily_limit_hit"],
                "daily_profit_target_hit": (mult["daily_profit_target_hit"]
                                            or rf["daily_profit_target_hit"]),
                "multiplier": mult,
                "rise_fall": rf,
            }
        return mult

    async def preflight(self) -> None:
        await self.client.connect()
        bal = await self.client.get_balance()
        log.info("Account %s | balance %.2f | env=%s", self.client.account.get("loginid"),
                 bal, self.cfg.deriv.environment.value)
        errors, warnings = self._account_economics(bal)
        for w in warnings:
            log.warning("Account economics: %s", w)
        for e in errors:
            log.error("Account economics: %s", e)
        await self._validate_symbols()
        log.info("Tradable symbols: %s", list(self.symbols))

    def _account_economics(self, balance: float) -> tuple[list[str], list[str]]:
        """Find configurations that cannot place a single Rise/Fall trade.

        Rise/Fall sizes at `fixed_stake`, but every entry is also capped by
        `max_total_open_risk * equity`. If that cap is below the stake, the
        risk manager vetoes 100% of entries: the bot looks alive while doing
        nothing. On a $10 account the default 3% cap is $0.30 against a $1
        stake, and even the 6% hard ceiling is only $0.60, so $10 is not
        tradeable at a $1 stake under any configuration.

        Returns (errors, warnings). Errors are fatal on a real-money start.
        """
        rc = self.cfg.risk
        stake = rc.fixed_stake if rc.fixed_stake > 0 else rc.min_stake
        cap_pct = min(rc.max_total_open_risk, HARD_MAX_TOTAL_OPEN_RISK)
        cap_usd = cap_pct * balance
        min_funding = stake / cap_pct if cap_pct > 0 else float("inf")
        errors: list[str] = []
        warnings: list[str] = []

        if balance < rc.min_balance:
            errors.append(
                f"balance ${balance:.2f} < MIN_BALANCE ${rc.min_balance:.2f}; "
                f"every entry is vetoed")
        if cap_usd < stake:
            errors.append(
                f"MAX_TOTAL_OPEN_RISK {cap_pct*100:.0f}% of ${balance:.2f} = "
                f"${cap_usd:.2f} < fixed stake ${stake:.2f}, so every Rise/Fall "
                f"entry is vetoed. Fund to >= ${min_funding:.2f}, lower FIXED_STAKE, "
                f"or raise MAX_TOTAL_OPEN_RISK (hard ceiling "
                f"{HARD_MAX_TOTAL_OPEN_RISK*100:.0f}%).")
        if rc.max_daily_drawdown * balance < stake:
            warnings.append(
                f"MAX_DAILY_DRAWDOWN {rc.max_daily_drawdown*100:.0f}% of "
                f"${balance:.2f} = ${rc.max_daily_drawdown*balance:.2f} < one "
                f"${stake:.2f} loss: the day halts after a single loss")

        if self.cfg.deriv.environment is Environment.LIVE:
            if not rc.rf_edge_guard:
                errors.append(
                    "LIVE with RF_EDGE_GUARD_ENABLED=false: no evidence gate on a "
                    "real account")
            elif rc.rf_edge_min_trades > RiskConfig.rf_edge_min_trades:
                errors.append(
                    f"LIVE with RF_EDGE_MIN_TRADES={rc.rf_edge_min_trades} "
                    f"(exploration mode; default {RiskConfig.rf_edge_min_trades}). "
                    f"The guard would not veto entries until that many closed "
                    f"trades. Set RF_EDGE_MIN_TRADES="
                    f"{RiskConfig.rf_edge_min_trades} for real money.")
            if balance < 20 * stake:
                warnings.append(
                    f"LIVE with ${balance:.2f} against a ${stake:.2f} stake: at the "
                    f"measured 49.1% win rate and 0.908 payout, a Monte Carlo of "
                    f"this exact setup is ruined before doubling in ~82% of runs")
        return errors, warnings

    def _rf_asset_allowed(self, sc: SymbolConfig) -> bool:
        """Rise/Fall is restricted to asset classes whose economics we have measured."""
        allowed = tuple(a.lower() for a in (self.cfg.rf_allowed_assets or ()))
        return sc.asset_class.value.lower() in allowed

    async def _validate_symbols(self) -> None:
        """Keep only symbols that are actively open today and supported under the selected trade_mode."""
        market_status = {}
        try:
            res = await self.client.request({"active_symbols": "brief"})
            for s in res.get("active_symbols", []):
                sym = s.get("underlying_symbol", s.get("symbol"))
                market_status[sym] = s
        except Exception as exc:
            log.warning("Could not fetch active_symbols brief (%s); will check individually", exc)

        mode = self.cfg.trade_mode
        log.info("Validating symbols under trade mode: %s (fixed stake: $%.2f)", mode.value.upper(), self.cfg.risk.fixed_stake)

        for sc in self.cfg.symbols:
            if not sc.enabled:
                continue

            # Check if market is open today (e.g. automatically skip closed weekend forex/metals)
            if sc.symbol in market_status:
                s_info = market_status[sc.symbol]
                if s_info.get("exchange_is_open") == 0 or s_info.get("is_trading_suspended") == 1:
                    log.info("%s: market is currently closed or suspended (e.g. weekend); skipping", sc.symbol)
                    continue

            try:
                await asyncio.sleep(0.20)
                info = await self.client.contracts_for(sc.symbol)
                avail = info.get("available", [])

                # 1. Check Multiplier support
                mult_contracts = [c for c in avail if c.get("contract_category") == "multiplier"]
                has_mult = len(mult_contracts) > 0
                if has_mult:
                    allowed = mult_contracts[0].get("multiplier_range", [])
                    if allowed and sc.multiplier not in allowed:
                        log.warning("%s: multiplier %s not in %s; multiplier disabled for symbol",
                                    sc.symbol, sc.multiplier, allowed)
                        has_mult = False

                # 2. Check Fixed Expiry Rise/Fall (CALL/PUT) support
                rf_contracts = [c for c in avail if c.get("contract_category") == "callput" and c.get("contract_type") in ("CALL", "PUT")]
                has_rf = len(rf_contracts) > 0
                if has_rf and not self._rf_asset_allowed(sc):
                    log.info("%s: Rise/Fall disabled for asset class '%s' (RF_ALLOWED_ASSETS=%s)",
                             sc.symbol, sc.asset_class.value,
                             ",".join(self.cfg.rf_allowed_assets) or "none")
                    has_rf = False

                # Filter based on trade mode
                if mode == TradeMode.MULTIPLIER:
                    if not has_mult:
                        log.info("%s: does not support multipliers; skipped in MULTIPLIER mode", sc.symbol)
                        continue
                    active_sc = replace(sc, supports_multiplier=True, supports_rise_fall=False)
                    self.symbols[sc.symbol] = active_sc
                    log.debug("%s: enabled for MULTIPLIER (x%d)", sc.symbol, sc.multiplier)

                elif mode == TradeMode.RISE_FALL:
                    if not has_rf:
                        log.info("%s: does not support Rise/Fall (CALL/PUT); skipped in RISE_FALL mode", sc.symbol)
                        continue
                    active_sc = replace(sc, supports_multiplier=False, supports_rise_fall=True)
                    self.symbols[sc.symbol] = active_sc
                    log.debug("%s: enabled for RISE/FALL (%d%s expiry)", sc.symbol, self.cfg.rise_fall_duration, self.cfg.rise_fall_duration_unit)

                elif mode == TradeMode.HYBRID:
                    if not has_mult and not has_rf:
                        log.info("%s: supports neither Multipliers nor Rise/Fall; skipped", sc.symbol)
                        continue
                    # Keep every capability the symbol actually has. Both books
                    # scan their full sets; ownership is decided per entry.
                    active_sc = replace(sc, supports_multiplier=has_mult,
                                        supports_rise_fall=has_rf)
                    self.symbols[sc.symbol] = active_sc
                    log.debug("%s: HYBRID capabilities mult=%s rf=%s",
                              sc.symbol, has_mult, has_rf)

            except Exception as exc:
                if isinstance(exc, DerivAPIError) and (exc.code == "MarketIsClosed" or "closed" in exc.message.lower()):
                    log.info("%s: market is closed (%s); skipping", sc.symbol, exc.message)
                else:
                    log.info("%s: contracts_for check notice (%s); using default config capabilities", sc.symbol, exc)
                    self.symbols[sc.symbol] = sc

        if mode is TradeMode.HYBRID:
            rf_whitelist = {s.upper() for s in self.cfg.hybrid_rf_symbols}
            mult_whitelist = {s.upper() for s in self.cfg.hybrid_mult_symbols}
            mult_syms = sorted(
                s for s, sc in self.symbols.items()
                if sc.supports_multiplier and (not mult_whitelist or s.upper() in mult_whitelist))
            rf_syms = sorted(
                s for s, sc in self.symbols.items()
                if sc.supports_rise_fall and self._rf_asset_allowed(sc)
                and not rf_symbol_excluded(s)
                and (not rf_whitelist or s.upper() in rf_whitelist))
            shared = sorted(set(mult_syms) & set(rf_syms))
            log.info("HYBRID isolated books: risk=%s, ownership=one strategy per symbol "
                     "at a time, tie-break=%s",
                     "SEPARATE" if self.cfg.hybrid_isolated_risk else "SHARED",
                     self.cfg.hybrid_signal_priority)
            log.info("HYBRID Multiplier symbols (%d): %s", len(mult_syms), mult_syms)
            log.info("HYBRID Rise/Fall symbols (%d): %s", len(rf_syms), rf_syms)
            log.info("HYBRID shared by both books (%d): %s", len(shared), shared)
            excluded = sorted(s for s, sc in self.symbols.items()
                              if sc.supports_rise_fall and rf_symbol_excluded(s))
            if excluded:
                log.info("HYBRID Rise/Fall excludes measured-defect symbols: %s", excluded)
            if not rf_syms or not mult_syms:
                log.warning("HYBRID: one book has no symbols (multiplier=%d, rise_fall=%d); "
                            "the other will run alone.",
                            len(mult_syms), len(rf_syms))

    def _recover_open_trades(self) -> None:
        """Re-attach contracts that were open when we last stopped.

        Each contract goes back to the book that owns its contract type, so a
        Rise/Fall position can never be settled against the Multiplier ledger
        (or vice versa) after a restart.
        """
        for t in self.store.open_trades():
            cid = t["contract_id"]
            ctype = t.get("contract_type", "multiplier") or "multiplier"
            self._book_for(ctype).open_positions[cid] = OpenPosition(
                cid, t["symbol"], t["risk_amount"], t["stake"], ctype)
            if self.cfg.trade_mode is TradeMode.HYBRID:
                other = self._book_for("multiplier" if ctype == "rise_fall" else "rise_fall")
                if other is not self._book_for(ctype) and other.has_open_on_symbol(t["symbol"]):
                    log.warning(
                        "Recovered %s contract %s on %s while the other book also holds "
                        "that symbol; both will be managed to close and no new entry "
                        "will open there until they settle.",
                        ctype, cid, t["symbol"])
            self._active_trackers[cid] = ActiveTradeTracker(
                contract_id=cid,
                symbol=t["symbol"],
                direction=t.get("direction", "long"),
                stake=t["stake"],
                entry_price=t.get("entry_price", 0.0),
                entry_time=t.get("entry_time", time.time()),
                contract_type=ctype,
            )

    async def _startup(self) -> None:
        await self.preflight()
        if not self.symbols:
            raise RuntimeError("No tradable symbols after validation.")
        balance = await self.client.get_balance()
        errors, warnings = self._account_economics(balance)
        for w in warnings:
            log.warning("Account economics: %s", w)
        if errors:
            raise RuntimeError("Refusing to start: " + " | ".join(errors))
        self.risk.roll_day_if_needed(balance)
        self.risk.reconcile_equity(balance)
        for name, book in self.risk_books.items():
            if book is self.risk:
                continue
            book.roll_day_if_needed(balance)
            book.reconcile_equity(balance)

        # Recover trades that were open when we last stopped
        self._recover_open_trades()
        for cid, tracker in list(self._active_trackers.items()):
            await self._track_contract(cid)
            log.info("Re-attached to open contract %s (%s, %s)",
                     cid, tracker.symbol, tracker.contract_type)

        for sym, sc in list(self.symbols.items()):
            feed = CandleFeed(sym, sc.granularity, on_bar_close=self._on_bar_close)
            subscribed = False
            for attempt in range(4):
                try:
                    await asyncio.sleep(0.4)
                    sub = await self.client.subscribe_candles(sym, sc.granularity,
                                                              self.cfg.strategy.history_candles, feed.handle)
                    self.feeds[sym] = feed
                    self._feed_subs.append(sub)
                    log.info("Streaming %s (%ss candles), %d bars loaded", sym, sc.granularity, len(feed.frame()))
                    subscribed = True
                    break
                except Exception as exc:
                    if isinstance(exc, DerivAPIError) and exc.code == "RateLimit" and attempt < 3:
                        wait_sec = 3.5 * (attempt + 1)
                        log.info("%s: rate limit hit on candle history, waiting %.1fs before retry (attempt %d/3)...",
                                 sym, wait_sec, attempt + 1)
                        await asyncio.sleep(wait_sec)
                        continue
                    if isinstance(exc, DerivAPIError) and (exc.code == "MarketIsClosed" or "closed" in exc.message.lower()):
                        log.warning("%s: market is presently closed (%s); skipping for this session", sym, exc.message)
                    else:
                        log.warning("%s: subscribe_candles failed (%s); skipping for this session", sym, exc)
                    break
            if not subscribed:
                self.symbols.pop(sym, None)

        p = self.learner.params
        log.info("Strategy params v%d: rsi_long_max=%.0f rsi_short_min=%.0f min_score=%.2f "
                 "asset_w=%s blocked=%s", p.version, p.rsi_long_max, p.rsi_short_min, p.min_score,
                 p.asset_class_weights, list(p.blocked))
        self._log_rf_edge_status()

    def _log_rf_edge_status(self) -> None:
        """Print the objective evidence the Rise/Fall guard will act on."""
        rc = self.cfg.risk
        log.info("Rise/Fall policy: assets=%s | edge_guard=%s (min_n=%d, lookback=%d) | "
                 "confidence_staking=%s | gemini_veto=%s",
                 ",".join(self.cfg.rf_allowed_assets) or "none",
                 "ON" if rc.rf_edge_guard else "OFF", rc.rf_edge_min_trades,
                 rc.rf_edge_lookback, "ON" if rc.confidence_staking else "OFF",
                 "ON" if self.cfg.gemini_veto_enabled else "OFF (advisory)")
        if not hasattr(self.store, "rf_edge_stats"):
            return
        try:
            s = self.store.rf_edge_stats(limit=rc.rf_edge_lookback,
                                         asset_classes=self.cfg.rf_allowed_assets)
        except Exception as exc:
            log.warning("Rise/Fall edge stats unavailable: %s", exc)
            return
        if not s:
            log.info("Rise/Fall evidence: no closed RF trades yet; exploration allowed at "
                     "minimum stake until n>=%d", rc.rf_edge_min_trades)
            return
        log.info("Rise/Fall evidence: n=%d WR=%.1f%% (95%% LB %.1f%%) breakeven=%.1f%% "
                 "netPnL=%+.2f -> %s", s["n"], s["wr"] * 100, s["wr_lo"] * 100,
                 s["breakeven"] * 100, s["pnl"],
                 "EDGE SUPPORTED" if s["wr_lo"] >= s["breakeven"] else
                 ("INSUFFICIENT SAMPLE" if s["n"] < rc.rf_edge_min_trades else "NO EDGE -> entries vetoed"))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:          # Windows
                pass
        try:
            await self._startup()
            st = self._risk_snapshot()
            log.info("=" * 65)
            log.info("Bot ACTIVE (dry_run=%s) | Account: %s | Day Equity: $%.2f",
                     self.dry_run, self.client.account.get("loginid"), st.get("day_start_equity", 0.0))
            if self.cfg.trade_mode is TradeMode.HYBRID:
                for name in ("multiplier", "rise_fall"):
                    b = self.risk_books[name]
                    log.info("Book %-10s: trades_today=%d open=%d realised=%+.2f halted=%s",
                             name, b.trades_today, len(b.open_positions),
                             b.realized_pnl_today, b.halted_until > time.time())
            log.info("Limits: Risk/Trade: %.1f%% | Max Daily Loss: -%.1f%% | Profit Target: +%.1f%%",
                     self.cfg.risk.max_risk_per_trade * 100,
                     self.cfg.risk.max_daily_drawdown * 100,
                     self.cfg.risk.target_daily_profit * 100)
            log.info("Streaming active markets: %s", list(self.symbols.keys()))
            log.info("Create the file '%s' or press Ctrl+C to halt trading.", self.cfg.risk.kill_switch_file)
            log.info("=" * 65)
            self._tasks = [asyncio.create_task(self._signal_worker(), name="signals"),
                           asyncio.create_task(self._maintenance_loop(), name="maintenance")]
            await self._stop.wait()
        finally:
            await self._shutdown()

    def _on_bar_close(self, symbol: str) -> None:
        self._queue.put_nowait(symbol)

    async def _signal_worker(self) -> None:
        while True:
            symbol = await self._queue.get()
            try:
                await self._process_symbol(symbol)
            except Exception:
                log.exception("Error processing %s", symbol)

    # ------------------------------------------------------------------ #
    # Decision pipeline
    # ------------------------------------------------------------------ #
    async def _process_symbol(self, symbol: str) -> None:
        sc, feed = self.symbols[symbol], self.feeds[symbol]
        params = self.learner.params

        # 1. Manage active positions on this symbol (Dynamic Profit & Dynamic Loss exits)
        for cid, tracker in list(self._active_trackers.items()):
            if tracker.symbol == symbol and not tracker.selling:
                tracker.bars_held += 1
                profit = tracker.current_profit
                if profit < 0:
                    tracker.consecutive_loss_bars += 1
                else:
                    tracker.consecutive_loss_bars = 0

                # Fixed expiry Rise/Fall contracts settle automatically with Deriv at expiry; skip manual early cuts
                if tracker.contract_type != "multiplier":
                    continue

                # --- Dynamic Loss Cuts (Pillar 1 & Pillar 3) ---
                if self.cfg.risk.dynamic_loss_enabled and profit < 0:
                    # Pillar 3: Stagnant loser cut (time-decay in loss)
                    if tracker.consecutive_loss_bars >= self.cfg.risk.max_bars_in_loss and profit <= -0.18:
                        tracker.selling = True
                        tracker.exit_reason = f"Stagnant negative for {tracker.consecutive_loss_bars} consecutive bars"
                        log.info("DYNAMIC LOSS CUT [%s #%s]: Stagnant in negative for %d consecutive bars (loss -$%.2f) -> Cutting loss early!",
                                 symbol, cid, tracker.consecutive_loss_bars, abs(profit))
                        try:
                            await self.client.sell_contract(cid)
                        except Exception as exc:
                            log.warning("sell_contract #%s failed: %s", cid, exc)
                            tracker.selling = False
                        continue

                    # Pillar 1: Thesis invalidation cut (candle breaks contrary to Fast EMA)
                    if self.cfg.risk.early_cut_on_invalidation and profit <= -0.20:
                        raw_df = feed.frame()
                        if len(raw_df) >= self.cfg.strategy.min_candles:
                            try:
                                ind_df = compute_indicators(raw_df, self.cfg.strategy)
                                last_bar = ind_df.iloc[-1]
                                invalidated = False
                                inv_reason = ""
                                if tracker.direction == "long":
                                    if last_bar["close"] < last_bar["ema_fast"] and last_bar["close"] < last_bar["open"]:
                                        invalidated = True
                                        inv_reason = f"bearish candle broke below Fast EMA ({last_bar['close']:.4f} < {last_bar['ema_fast']:.4f})"
                                elif tracker.direction == "short":
                                    if last_bar["close"] > last_bar["ema_fast"] and last_bar["close"] > last_bar["open"]:
                                        invalidated = True
                                        inv_reason = f"bullish candle broke above Fast EMA ({last_bar['close']:.4f} > {last_bar['ema_fast']:.4f})"

                                if invalidated:
                                    tracker.selling = True
                                    tracker.exit_reason = f"Thesis invalidated ({inv_reason})"
                                    log.info("DYNAMIC LOSS CUT [%s #%s]: Thesis invalidated (%s, loss -$%.2f) -> Cutting loss early!",
                                             symbol, cid, inv_reason, abs(profit))
                                    try:
                                        await self.client.sell_contract(cid)
                                    except Exception as exc:
                                        log.warning("sell_contract #%s failed: %s", cid, exc)
                                        tracker.selling = False
                                    continue
                            except Exception as exc:
                                log.debug("Indicator check error for loss cut: %s", exc)

                # --- Dynamic Profit Takes ---
                if self.cfg.risk.dynamic_profit_enabled and profit >= 0.060:
                    # Time-decay / stagnation: take profit after N bars in green
                    if tracker.bars_held >= self.cfg.risk.max_hold_bars_in_profit:
                        tracker.selling = True
                        log.info("DYNAMIC PROFIT TAKE [%s #%s]: Stagnating in profit after %d bars ($+%.2f) -> Closing now!",
                                 symbol, cid, tracker.bars_held, profit)
                        try:
                            await self.client.sell_contract(cid)
                        except Exception as exc:
                            log.warning("sell_contract #%s failed: %s", cid, exc)
                            tracker.selling = False
                        continue

                    # Candlestick reversal while in strong profit (>= $0.070): lock in strong gain
                    if self.cfg.risk.early_exit_on_reversal and profit >= 0.070:
                        df = feed.frame()
                        if len(df) >= 2:
                            last_bar = df.iloc[-1]
                            reversal = False
                            rev_reason = ""
                            if tracker.direction == "long" and last_bar["close"] < last_bar["open"]:
                                reversal = True
                                rev_reason = f"bearish candle close ({last_bar['close']:.4f} < {last_bar['open']:.4f})"
                            elif tracker.direction == "short" and last_bar["close"] > last_bar["open"]:
                                reversal = True
                                rev_reason = f"bullish candle close ({last_bar['close']:.4f} > {last_bar['open']:.4f})"

                            if reversal:
                                tracker.selling = True
                                log.info("DYNAMIC PROFIT TAKE [%s #%s]: %s while in profit ($+%.2f) -> Securing win!",
                                         symbol, cid, rev_reason, profit)
                                try:
                                    await self.client.sell_contract(cid)
                                except Exception as exc:
                                    log.warning("sell_contract #%s failed: %s", cid, exc)
                                    tracker.selling = False

        # Both books scan their full capability sets. The scan contract type
        # only controls contract-specific exclusions (e.g. the Rise/Fall
        # stpRNG4 defect); it does not restrict which books may trade.
        scan_type = self._scan_contract_type(symbol, sc)
        sig = self.strategy.evaluate(symbol, sc.asset_class.value, feed.frame(), params,
                                     contract_type=scan_type)
        if sig is None:
            current_min = int(time.time() // 60)
            if current_min != self._last_bar_close_minute:
                self._last_bar_close_minute = current_min
                stats = self.store.get_today_audit_stats()
                open_cnt = len(self._active_trackers)
                utc_time = time.strftime("%H:%M:%S UTC", time.gmtime())
                log.info("📊 [BAR CLOSE] %s | %d instruments evaluated | Open: %d | Today PnL: %+.2f USD (%dW / %dL)",
                         utc_time, len(self.symbols), open_cnt, stats.get("day_pnl", 0.0), stats.get("wins", 0), stats.get("losses", 0))
            log.debug("[%s] Bar closed @ %.4f | Evaluated -> No trade signal", symbol, feed.last_price)
            return

        # Pillar 2: Opposing Signal Cut
        # If an opposing signal appears against an open trade on this symbol, cut the existing position immediately.
        # Multiplier trades only: a Rise/Fall signal must never cut a
        # Multiplier trade (that was the cross-strategy conflict).
        if self.cfg.risk.cut_on_opposing_signal:
            for cid, tracker in list(self._active_trackers.items()):
                if tracker.symbol == symbol and not tracker.selling and tracker.contract_type == "multiplier":
                    opposing = (
                        (tracker.direction == "long" and sig.direction == "short") or
                        (tracker.direction == "short" and sig.direction == "long")
                    )
                    if opposing:
                        tracker.selling = True
                        tracker.exit_reason = f"Opposing {sig.direction.upper()} signal fired"
                        log.info("DYNAMIC LOSS CUT [%s #%s]: Opposing %s signal fired against open %s (PnL $%.2f) -> Cutting trade now!",
                                 symbol, cid, sig.direction.upper(), tracker.direction.upper(), tracker.current_profit)
                        try:
                            await self.client.sell_contract(cid)
                        except Exception as exc:
                            log.warning("sell_contract #%s failed: %s", cid, exc)
                            tracker.selling = False

        if Path("STOP_AFTER_TRADE").exists():
            return

        log.info("SIGNAL %s %s score=%.2f conf=%.2f regime=%s rsi=%.1f votes=%s",
                 symbol, sig.direction, sig.score, sig.confidence, sig.regime,
                 sig.indicators["rsi"], sig.components)

        balance = await self.client.get_balance()
        price = feed.last_price if math.isfinite(feed.last_price) else sig.price

        mode = self.cfg.trade_mode
        if mode == TradeMode.HYBRID:
            # Both books scan their FULL capability sets; there is no static
            # symbol split. A symbol can be held by only one book at a time.
            want_multiplier = (sc.supports_multiplier
                               and self._leg_enabled("multiplier", symbol))
            want_rise_fall = (sc.supports_rise_fall
                              and self._rf_asset_allowed(sc)
                              and not rf_symbol_excluded(symbol)
                              and self._leg_enabled("rise_fall", symbol))
            log.debug("%s: HYBRID candidates mult=%s rf=%s", symbol,
                      want_multiplier, want_rise_fall)
        else:
            want_multiplier = (mode == TradeMode.MULTIPLIER) and sc.supports_multiplier
            want_rise_fall = (mode == TradeMode.RISE_FALL) and sc.supports_rise_fall \
                and self._rf_asset_allowed(sc)

        if not want_multiplier and not want_rise_fall:
            log.debug("%s: signal generated but neither multiplier nor rise_fall is enabled for this symbol", symbol)
            return

        weight = params.asset_class_weights.get(sc.asset_class.value, 1.0)
        m_order = None
        rf_order = None

        if want_multiplier:
            if self._cross_strategy_holder(symbol, "multiplier") is not None:
                log.info("HYBRID OWNERSHIP [%s]: Rise/Fall book holds this symbol; "
                         "Multiplier waits for it to close", symbol)
            else:
                m_intent = TradeIntent(
                    symbol=symbol, asset_class=sc.asset_class.value, direction=sig.direction,
                    entry_price=price, atr=sig.atr, confidence=sig.confidence, regime=sig.regime,
                    multiplier=sc.multiplier, tick_age_sec=feed.tick_age_sec, granularity=sc.granularity,
                    contract_type="multiplier"
                )
                m_decision = self._book_for("multiplier").evaluate(m_intent, balance, asset_weight=weight)
                if m_decision.approved and m_decision.order:
                    m_order = m_decision.order
                else:
                    log.info("VETO [Multiplier] %s %s: %s", symbol, sig.direction, "; ".join(m_decision.reasons))

        if want_rise_fall:
            if self._cross_strategy_holder(symbol, "rise_fall") is not None:
                log.info("HYBRID OWNERSHIP [%s]: Multiplier book holds this symbol; "
                         "Rise/Fall waits for it to close", symbol)
            else:
                rf_duration = sig.duration if hasattr(sig, "duration") and sig.duration else self.cfg.rise_fall_duration
                rf_reason = getattr(sig, "duration_reason", "Dynamic Market Adaptive")
                log.info("DYNAMIC EXPIRY [%s %s]: %d%s (%s)", symbol, sig.direction, rf_duration, self.cfg.rise_fall_duration_unit, rf_reason)
                rf_intent = TradeIntent(
                    symbol=symbol, asset_class=sc.asset_class.value, direction=sig.direction,
                    entry_price=price, atr=sig.atr, confidence=sig.confidence, regime=sig.regime,
                    multiplier=0, tick_age_sec=feed.tick_age_sec, granularity=sc.granularity,
                    contract_type="rise_fall",
                    duration=rf_duration,
                    duration_unit=self.cfg.rise_fall_duration_unit
                )
                rf_decision = self._book_for("rise_fall").evaluate(rf_intent, balance, asset_weight=weight)
                if rf_decision.approved and rf_decision.order:
                    rf_order = rf_decision.order
                else:
                    log.info("VETO [Rise/Fall] %s %s: %s", symbol, sig.direction, "; ".join(rf_decision.reasons))

        if m_order and rf_order:
            # Both books approve on the same bar and the symbol is free. Only
            # one may take it, otherwise we would recreate the double-entry
            # conflict this mode is designed to prevent.
            if self.cfg.hybrid_signal_priority == "rise_fall":
                m_order = None
                winner = "Rise/Fall"
            else:
                rf_order = None
                winner = "Multiplier"
            log.info("HYBRID TIE-BREAK [%s]: both books approved; %s takes this entry, "
                     "the other waits until it closes", symbol, winner)

        orders_to_place = [o for o in (m_order, rf_order) if o]
        if not orders_to_place:
            return

        # AI Sanity Filter: Gemini Pre-Trade Veto (runs once for the confluent signal)
        # Advisory by default: its rules were derived from a 65-trade sample with
        # no out-of-sample edge, and the 429/timeout latency occasionally cost a
        # valid entry. With GEMINI_VETO_ENABLED=false its opinion is logged only.
        ind_5m = {"close": sig.indicators.get("close"), "ema50": sig.indicators.get("ema_higher_tf")}
        ai_reason = "Approved"
        try:
            ai_approved, ai_reason = await self.gemini.veto_signal(
                symbol=symbol, direction=sig.direction, score=sig.score,
                ind_1m=sig.indicators, ind_5m=ind_5m
            )
            if not ai_approved:
                if self.cfg.gemini_veto_enabled:
                    log.info("GEMINI AI VETO [%s %s]: %s -> Trade rejected!", symbol, sig.direction, ai_reason)
                    return
                log.info("GEMINI AI ADVISORY [%s %s]: %s (veto disabled; continuing)",
                         symbol, sig.direction, ai_reason)
            else:
                log.info("GEMINI AI APPROVED [%s %s]: %s", symbol, sig.direction, ai_reason)
        except Exception as exc:
            log.warning("Gemini veto check notice (%s); failing open to allow validated trade", exc)

        if self.dry_run:
            for o in orders_to_place:
                if o.contract_type == "multiplier":
                    log.info("[DRY RUN] Approved MULTIPLIER order not sent: %s %s stake=%.2f x%d",
                             o.symbol, o.direction, o.stake, o.multiplier)
                else:
                    log.info("[DRY RUN] Approved RISE_FALL order not sent: %s %s stake=%.2f duration=%d%s (%s)",
                             o.symbol, o.direction, o.stake, o.duration, o.duration_unit, getattr(sig, "duration_reason", "dynamic"))
            return

        for o in orders_to_place:
            try:
                if o.contract_type == "multiplier":
                    log.info("OPENING [Multiplier] %s %s stake=%.2f x%d SL=%.2f TP=%.2f",
                             o.symbol, o.direction, o.stake, o.multiplier, o.stop_loss_amount, o.take_profit_amount)
                    buy = await self.client.buy_multiplier(
                        symbol=o.symbol, direction=o.direction, stake=o.stake, multiplier=o.multiplier,
                        stop_loss=o.stop_loss_amount, take_profit=o.take_profit_amount)
                else:
                    log.info("OPENING [Rise/Fall] %s %s stake=%.2f duration=%d%s (%s)",
                             o.symbol, o.direction, o.stake, o.duration, o.duration_unit, getattr(sig, "duration_reason", "dynamic"))
                    buy = await self.client.buy_rise_fall(
                        symbol=o.symbol, direction=o.direction, stake=o.stake,
                        duration=o.duration, duration_unit=o.duration_unit)

                cid = int(buy["contract_id"])
                rf_dur_str = f"{o.duration}{o.duration_unit}" if hasattr(o, "duration") and o.duration else "N/A"
                rf_reason = getattr(sig, "duration_reason", "Dynamic")
                open_card = (
                    f"\n╔════════════════════════════════════════════════════════════════════════╗\n"
                    f"║  🚀 TRADE OPENED [{o.contract_type.upper()}]\n"
                    f"║  Asset: {o.symbol:<14} | Direction: {o.direction.upper():<6} | Stake: ${o.stake:.2f} USD\n"
                    f"║  Contract ID: #{cid} | Entry Spot: {price:.4f}\n"
                    f"║  Duration: {rf_dur_str} ({rf_reason})\n"
                    f"║  Confluence: Score {sig.score:+.2f}/5.0 | ADX: {sig.indicators.get('adx', 0):.2f} | RSI: {sig.indicators.get('rsi', 0):.1f}\n"
                    f"╚════════════════════════════════════════════════════════════════════════╝"
                )
                log.info(open_card)
                self._book_for(o.contract_type).register_open(cid, o.symbol, o.risk_amount, o.stake, contract_type=o.contract_type)
                self._active_trackers[cid] = ActiveTradeTracker(
                    contract_id=cid, symbol=o.symbol, direction=o.direction,
                    stake=o.stake, entry_price=price, entry_time=time.time(),
                    contract_type=o.contract_type,
                    entry_indicators=dict(sig.indicators),
                    pre_trade_veto=ai_reason,
                )
                self.store.log_open_trade(
                    contract_id=cid, symbol=o.symbol, asset_class=sc.asset_class.value,
                    direction=o.direction, entry_time=time.time(), entry_price=price,
                    stake=o.stake, multiplier=o.multiplier, risk_amount=o.risk_amount,
                    sl_amount=o.stop_loss_amount, tp_amount=o.take_profit_amount,
                    regime=sig.regime, confidence=sig.confidence, score=sig.score,
                    indicators=sig.indicators, components=sig.components,
                    contract_type=o.contract_type)

                # Dispatch real-time Telegram alert
                asyncio.create_task(self.telegram.notify_trade_open({
                    "contract_id": cid,
                    "symbol": o.symbol,
                    "direction": o.direction,
                    "contract_type": o.contract_type,
                    "stake": o.stake,
                    "entry_price": price,
                    "duration": getattr(o, "duration", 0),
                    "duration_unit": getattr(o, "duration_unit", "m"),
                    "duration_reason": rf_reason,
                    "score": sig.score,
                    "indicators": dict(sig.indicators),
                }))
                await self._track_contract(cid)
            except DerivAPIError as exc:
                log.error("Order rejected by Deriv [%s %s]: %s", o.contract_type, o.symbol, exc)
            except Exception as exc:
                log.error("Unexpected error opening %s order: %s", o.contract_type, exc)

    # ------------------------------------------------------------------ #
    # Contract lifecycle
    # ------------------------------------------------------------------ #
    async def _track_contract(self, cid: int) -> None:
        async def cb(msg: dict, _cid=cid):
            await self._on_contract_update(_cid, msg)
        try:
            sub = await self.client.subscribe_contract(cid, cb)
            if cid in self._finalized:
                await self.client.unsubscribe(sub)
            else:
                self._contract_subs[cid] = sub
        except DerivAPIError as exc:
            if "subscription limit" in exc.message.lower():
                log.info("Contract %s: subscription limit reached; tracking via polling fallback every 1.5s", cid)
                asyncio.create_task(self._poll_contract(cid))
            else:
                raise

    async def _poll_contract(self, cid: int) -> None:
        while cid not in self._finalized and not self._stop.is_set():
            try:
                poc_res = await self.client._request({"proposal_open_contract": 1, "contract_id": cid})
                await self._on_contract_update(cid, poc_res)
                poc = poc_res.get("proposal_open_contract", {})
                if poc.get("is_sold"):
                    break
            except Exception as exc:
                log.debug("Poll contract %s: %s", cid, exc)
            await asyncio.sleep(1.5)

    def _evaluate_realtime_trade_ai(self, tracker: ActiveTradeTracker, current_profit: float) -> tuple[bool, str, str]:
        """
        Real-time AI trade management evaluated on EVERY TICK.
        Returns: (should_close, reason, action_type: 'profit' | 'loss')
        """
        rc = self.cfg.risk
        if tracker.selling or tracker.contract_type != "multiplier":
            return False, "", ""

        # Update tick state
        tracker.current_profit = current_profit
        if current_profit > tracker.peak_profit:
            tracker.peak_profit = current_profit
        tracker.recent_profits.append(current_profit)
        if len(tracker.recent_profits) > 8:
            tracker.recent_profits.pop(0)

        # -------------------------------------------------------------
        # 1. GREEN SHIELD: Activate zero-loss lock if trade hits green_shield_trigger_usd (+$0.150)
        # -------------------------------------------------------------
        if tracker.peak_profit >= rc.green_shield_trigger_usd:
            tracker.green_shield_active = True

        # If Green Shield is active, lock in at least +$0.050 (guarantees a meaningful green close!)
        if tracker.green_shield_active and current_profit <= 0.050:
            return True, (f"GREEN SHIELD TRIGGERED: Secured profit ${current_profit:+.2f} "
                          f"(peak was ${tracker.peak_profit:+.2f}) -> Gain locked!"), "profit"

        # -------------------------------------------------------------
        # 2. REAL-TIME AI PROFIT LOCKING (Fast Highest Profit Capture)
        # -------------------------------------------------------------
        if rc.dynamic_profit_enabled and current_profit > 0:
            # A. Dynamic Trailing Tier 1 (Peak >= breakeven_trigger_usd, $0.250)
            if tracker.peak_profit >= rc.breakeven_trigger_usd:
                floor = max(0.150, round(tracker.peak_profit * rc.trail_profit_fraction, 2))
                if current_profit <= floor:
                    return True, (f"DYNAMIC TRAIL LOCK: Secured ${current_profit:+.2f} "
                                  f"(Floor ${floor:.2f} hit, peak was ${tracker.peak_profit:+.2f})"), "profit"

            # B. Dynamic Trailing Tier 2 (Peak >= min_profit_take_usd, $0.180)
            elif tracker.peak_profit >= rc.min_profit_take_usd:
                floor = max(0.100, round(tracker.peak_profit * 0.70, 2))
                if current_profit <= floor:
                    return True, (f"PROFIT TRAIL SECURED: Captured ${current_profit:+.2f} "
                                  f"from peak ${tracker.peak_profit:+.2f}"), "profit"

            # C. AI Momentum Exhaustion Cut on Tick:
            # Triggers only once a runner has unfolded (peak >= $0.200) and 3 consecutive ticks give back >= 25%
            if rc.fast_exhaustion_cut and tracker.peak_profit >= 0.200 and len(tracker.recent_profits) >= 4:
                p3, p2, p1 = tracker.recent_profits[-3], tracker.recent_profits[-2], tracker.recent_profits[-1]
                if p3 > p2 > p1 and current_profit <= round(tracker.peak_profit * 0.75, 2) and current_profit >= 0.080:
                    return True, (f"AI MOMENTUM EXHAUSTION: 3 consecutive tick decline detected "
                                  f"(${tracker.peak_profit:+.2f} -> ${current_profit:+.2f}) -> Locking peak gain now!"), "profit"

        # -------------------------------------------------------------
        # 3. REAL-TIME AI LOSS PREVENTION & CAPITAL PROTECTION
        # -------------------------------------------------------------
        if rc.dynamic_loss_enabled and current_profit < 0:
            abs_loss = abs(current_profit)

            # A. Real-Time Software Loss Cap: Hard emergency stop at -$0.28 (rc.max_loss_cut_usd)
            # Protects capital well before broker hard SL (-$0.40) while giving normal breathing room past tick noise
            if current_profit <= -rc.max_loss_cut_usd:
                tracker.exit_reason = f"Hit software loss limit (-${abs_loss:.2f})"
                return True, (f"REAL-TIME LOSS CAP: Hit software limit (loss -${abs_loss:.2f} "
                              f"<= -${rc.max_loss_cut_usd:.2f}) -> Cutting loss!"), "loss"

            # B. Stagnant Loser Cut: Only evaluate after trade had >= 5 full bars to develop
            if tracker.bars_held >= rc.max_bars_in_loss and current_profit <= -0.18:
                tracker.exit_reason = f"Stagnant negative after {tracker.bars_held} bars (-${abs_loss:.2f})"
                return True, (f"AI STAGNANT LOSER CUT: Negative after {tracker.bars_held} bars "
                              f"(loss -${abs_loss:.2f}) without upward traction -> Exiting clean!"), "loss"

        return False, "", ""

    async def _on_contract_update(self, cid: int, msg: dict) -> None:
        poc = msg.get("proposal_open_contract")
        if not poc:
            return
        entry = poc.get("entry_tick") or poc.get("entry_spot")
        if entry and cid not in self._entry_set:
            self._entry_set.add(cid)
            self.store.update_entry_price(cid, float(entry))
            if cid in self._active_trackers:
                self._active_trackers[cid].entry_price = float(entry)

        # Active trade live real-time AI profit & loss evaluation on every tick
        if not poc.get("is_sold"):
            profit = float(poc.get("profit", 0.0))
            tracker = self._active_trackers.get(cid)
            if tracker and not tracker.selling:
                should_close, reason, action_type = self._evaluate_realtime_trade_ai(tracker, profit)
                if should_close:
                    tracker.selling = True
                    log_tag = "DYNAMIC PROFIT LOCK" if action_type == "profit" else "DYNAMIC LOSS CUT"
                    log.info("%s [%s #%s]: %s -> Market-closing now!",
                             log_tag, tracker.symbol, cid, reason)
                    try:
                        await self.client.sell_contract(cid)
                    except Exception as exc:
                        log.warning("sell_contract #%s failed: %s", cid, exc)
                        tracker.selling = False
                    return
            return

        if cid in self._finalized:
            return

        self._finalized.add(cid)
        tracker = self._active_trackers.pop(cid, None)
        pnl = float(poc.get("profit", 0.0))
        exit_price = float(poc.get("exit_tick") or poc.get("sell_spot") or poc.get("current_spot") or 0.0)
        # Journal the BROKER's settlement time, not the moment our websocket
        # noticed it. After a restart the difference can be hours, which used to
        # corrupt hold-time analysis in the learning journal.
        try:
            exit_ts = float(poc.get("sell_time") or poc.get("date_expiry") or 0.0) or None
        except (TypeError, ValueError):
            exit_ts = None
        self.store.log_close_trade(cid, exit_price, pnl, exit_time=exit_ts)
        self._book_for(tracker.contract_type if tracker else "multiplier").register_close(cid, pnl)

        stats = self.store.get_today_audit_stats()
        entry_price = tracker.entry_price if tracker else 0.0
        symbol = tracker.symbol if tracker else "UNKNOWN"
        direction = tracker.direction if tracker else "UNKNOWN"
        stake = tracker.stake if tracker else 1.0
        dur_sec = (exit_ts or time.time()) - (tracker.entry_time if tracker else time.time())
        # Prefer the broker's real balance. The previous reconstruction
        # (day_start_equity + day PnL) printed $9960.63 while the account held
        # $10001.72, because the stored day-start equity was stale.
        try:
            bal = await self.client.get_balance()
            for book in {id(b): b for b in self.risk_books.values()}.values():
                book.reconcile_equity(bal)
        except Exception as exc:
            log.debug("Balance fetch failed (%s); using derived equity", exc)
            risk_status = self._risk_snapshot()
            bal = risk_status.get("day_start_equity", 0.0) + risk_status.get("realized_pnl_today", 0.0)

        yesterday = stats.get("yesterday", {})
        prev_str = (f"{yesterday.get('wins', 0)}W / {yesterday.get('losses', 0)}L "
                    f"({yesterday.get('win_rate', 0.0):.1f}%) "
                    f"{yesterday.get('pnl', 0.0):+.2f} USD"
                    if yesterday.get("total_trades") else "no trades")
        # Positions opened earlier that settled today (crossed UTC midnight, or
        # the bot was down at expiry). Shown separately so "today's trades" stays
        # what was actually taken today.
        settled = stats.get("settled_today", {})
        carried = max(0, int(settled.get("total_trades", 0)) - int(stats.get("total_trades", 0)))
        carry_str = ""
        if carried:
            carry_str = (f"║  Settled today (cash): {settled.get('total_trades', 0)} "
                         f"({settled.get('wins', 0)}W / {settled.get('losses', 0)}L) "
                         f"{settled.get('pnl', 0.0):+.2f} USD, incl. {carried} opened earlier\n")

        outcome_tag = f"WIN (+{(pnl/stake*100):.1f}%)" if pnl > 0 else f"LOSS (-${abs(pnl):.2f})"
        dur_str = f"{int(dur_sec//60)}m {int(dur_sec%60)}s"
        close_card = (
            f"\n╔════════════════════════════════════════════════════════════════════════╗\n"
            f"║  {'✅' if pnl > 0 else '❌'} TRADE CLOSED [{outcome_tag}]\n"
            f"║  Asset: {symbol:<14} | Direction: {direction.upper():<6} | Contract: #{cid}\n"
            f"║  Entry: {entry_price:.4f} ➔ Exit: {exit_price:.4f} | Hold Time: {dur_str}\n"
            f"║  PnL: {pnl:+.2f} USD | Active Win Streak: {stats.get('win_streak', 0)} Wins\n"
            f"║  Updated Balance: ${bal:.2f} USD | Today (opened): {stats.get('wins', 0)}W / {stats.get('losses', 0)}L ({stats.get('win_rate', 0):.1f}%)\n"
            f"║  Yesterday: {prev_str}\n"
            f"{carry_str}"
            f"╚════════════════════════════════════════════════════════════════════════╝"
        )
        log.info(close_card)
        sub = self._contract_subs.pop(cid, None)
        if sub:
            await self.client.unsubscribe(sub)

        # Dispatch real-time Telegram alert
        asyncio.create_task(self.telegram.notify_trade_close({
            "contract_id": cid,
            "symbol": symbol,
            "direction": direction,
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stake": stake,
            "duration_sec": dur_sec,
            "win_streak": stats.get("win_streak", 0),
            "total_trades": stats.get("total_trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "balance": bal,
            "day_pnl": stats.get("day_pnl", 0.0),
            "yesterday": yesterday,
            "settled_today": settled,
            "carried": carried,
        }))

        # Automated Loss Forensic Post-Mortem Hook
        if pnl < 0:
            asyncio.create_task(self._conduct_loss_post_mortem(
                cid=cid, tracker=tracker, pnl=pnl, exit_price=exit_price,
                status=poc.get("status", "sold")
            ))

        # 10-Trade Automated Milestone & Loss Learning Report
        milestone = self.learner.check_and_run_milestone()
        if milestone:
            self._log_milestone_banner(milestone)
            asyncio.create_task(self._run_gemini_milestone_audit(milestone))

        await self._maybe_review()

        if Path("STOP_AFTER_TRADE").exists() and len(self._active_trackers) == 0:
            log.info("🏁 DRAIN COMPLETE: All active trades have closed. Shutting down bot gracefully as requested.")
            self._stop.set()

    async def _conduct_loss_post_mortem(self, cid: int, tracker: Optional[ActiveTradeTracker],
                                         pnl: float, exit_price: float, status: str) -> None:
        """
        Captures exact indicators, asset state, and Gemini analysis upon any losing trade.
        """
        try:
            trade = self.store.get_trade(cid) or {}
            symbol = tracker.symbol if tracker else trade.get("symbol", "UNKNOWN")
            direction = tracker.direction if tracker else trade.get("direction", "UNKNOWN")
            entry_price = tracker.entry_price if tracker else trade.get("entry_price", 0.0)
            entry_time = tracker.entry_time if tracker else trade.get("entry_time", time.time())
            duration_sec = time.time() - entry_time
            peak_profit = tracker.peak_profit if tracker else 0.0
            exit_reason = (tracker.exit_reason if tracker and tracker.exit_reason != "Broker settlement / market close"
                           else f"Broker settlement ({status})")
            entry_ind = (tracker.entry_indicators if tracker and tracker.entry_indicators
                         else trade.get("indicators", {}))

            # Capture latest exit indicators from candle feed if available
            exit_ind = {}
            if symbol in self.feeds:
                try:
                    df = self.feeds[symbol].frame()
                    if len(df) >= self.cfg.strategy.min_candles:
                        ind_df = compute_indicators(df, self.cfg.strategy)
                        last_bar = ind_df.iloc[-1]
                        exit_ind = {
                            "rsi": float(last_bar["rsi"]),
                            "macd_hist": float(last_bar["macd_hist"]),
                            "bb_pctb": float(last_bar["bb_pctb"]),
                            "adx": float(last_bar["adx"]),
                            "atr": float(last_bar["atr"]),
                            "close": float(last_bar["close"]),
                            "ema_fast": float(last_bar["ema_fast"]),
                            "ema_slow": float(last_bar["ema_slow"]),
                            "ema_higher_tf": float(last_bar["ema_higher_tf"]),
                        }
                except Exception as exc:
                    log.debug("Exit indicator capture notice: %s", exc)

            pre_veto = tracker.pre_trade_veto if tracker else ""
            trade_id = trade.get("id", cid)

            # Ground truth for the audit trail. The LLM post-mortem used to be
            # handed the fill price alongside signal-bar indicators and then
            # "discover" a rule breach that never happened (e.g. #322: fill
            # 7328.50 vs EMA9 7328.97, while the signal bar closed 7329.10,
            # comfortably above EMA9). Record what the rules actually saw.
            sig_close = float(entry_ind.get("close") or entry_price)
            ema9, ema21 = entry_ind.get("ema_fast"), entry_ind.get("ema_slow")
            ema50, macd = entry_ind.get("ema_trend"), entry_ind.get("macd_hist")
            facts = []
            if ema9 is not None:
                ok = sig_close >= float(ema9) if direction == "long" else sig_close <= float(ema9)
                facts.append(f"close{'>=' if direction == 'long' else '<='}ema9 {'ok' if ok else 'FAILED'}")
            if ema50 is not None:
                ok = sig_close >= float(ema50) if direction == "long" else sig_close <= float(ema50)
                facts.append(f"close{'>=' if direction == 'long' else '<='}ema50 {'ok' if ok else 'FAILED'}")
            if ema9 is not None and ema21 is not None:
                ok = float(ema9) > float(ema21) if direction == "long" else float(ema9) < float(ema21)
                facts.append(f"ema9{'>' if direction == 'long' else '<'}ema21 {'ok' if ok else 'FAILED'}")
            if macd is not None:
                ok = float(macd) > 0 if direction == "long" else float(macd) < 0
                facts.append(f"macd {float(macd):+.4f} {'ok' if ok else 'FAILED'}")
            log.info("  * Signal-bar facts: close=%.4f ema9=%.4f ema21=%.4f ema50=%.4f | %s | fill=%.4f",
                     sig_close, float(entry_ind.get("ema_fast") or 0),
                     float(entry_ind.get("ema_slow") or 0),
                     float(entry_ind.get("ema_trend") or 0),
                     ", ".join(facts) or "n/a", entry_price)

            pm = await self.gemini.analyze_loss(
                trade_id=trade_id, symbol=symbol, direction=direction, pnl=pnl,
                entry_price=entry_price, exit_price=exit_price, duration_sec=duration_sec,
                peak_profit=peak_profit, exit_reason=exit_reason,
                entry_indicators=entry_ind, exit_indicators=exit_ind,
                pre_trade_veto=pre_veto, signal_close=sig_close,
            )

            # Store in database
            self.store.log_loss_post_mortem(
                contract_id=cid, symbol=symbol, direction=direction, pnl=pnl,
                entry_price=entry_price, exit_price=exit_price, duration_sec=duration_sec,
                peak_profit=peak_profit, exit_reason=exit_reason,
                entry_indicators=entry_ind, exit_indicators=exit_ind,
                gemini_post_mortem=pm, failure_category=pm.get("failure_category", "Unknown"),
                root_cause=pm.get("root_cause", ""),
                recommended_patch=str(pm.get("recommended_patch", "")),
            )

            # Append to persistent JSONL log
            pm_record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "trade_id": trade_id,
                "contract_id": cid,
                "symbol": symbol,
                "direction": direction,
                "pnl": pnl,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "duration_sec": round(duration_sec, 1),
                "peak_profit": round(peak_profit, 3),
                "exit_reason": exit_reason,
                "entry_indicators": entry_ind,
                "exit_indicators": exit_ind,
                "gemini_analysis": pm,
            }
            pm_path = Path(self.cfg.log_dir) / "loss_post_mortems.jsonl"
            with open(pm_path, "a") as f:
                f.write(json.dumps(pm_record) + "\n")

            # Structured console/file log
            log.info("=" * 72)
            log.info("🚨 POST-MORTEM FORENSIC ANALYSIS: TRADE LOSS [Trade #%s | %s %s | -$%.2f]",
                     trade_id, symbol, direction.upper(), abs(pnl))
            log.info("  * Duration: %.1fs | Peak Profit: +$%.2f | Exit Reason: %s",
                     duration_sec, peak_profit, exit_reason)
            log.info("  * Entry Indicators: RSI=%.1f | MACDh=%.4f | BB%%B=%.2f | ADX=%.1f | ATR=%.4f",
                     float(entry_ind.get("rsi") or 0), float(entry_ind.get("macd_hist") or 0),
                     float(entry_ind.get("bb_pctb") or 0), float(entry_ind.get("adx") or 0),
                     float(entry_ind.get("atr") or 0))
            if exit_ind:
                log.info("  * Exit Indicators:  RSI=%.1f | MACDh=%.4f | BB%%B=%.2f | ADX=%.1f",
                         float(exit_ind.get("rsi") or 0), float(exit_ind.get("macd_hist") or 0),
                         float(exit_ind.get("bb_pctb") or 0), float(exit_ind.get("adx") or 0))
            log.info("  * Category: %s", pm.get("failure_category", "N/A"))
            log.info("  * Root Cause: %s", pm.get("root_cause", "N/A"))
            log.info("  * Flawed Indicators: %s", pm.get("flawed_indicators", []))
            log.info("  * Recommended Strategy Patch: %s", pm.get("recommended_patch", "N/A"))
            log.info("=" * 72)
        except Exception as exc:
            log.exception("Error in _conduct_loss_post_mortem for cid %s: %s", cid, exc)

    async def _run_gemini_milestone_audit(self, milestone: dict) -> None:
        try:
            b_idx = milestone["batch_index"]
            batch_trades = self.store.recent_closed(self.cfg.learning.review_every_n_trades)
            params = self.learner.params.to_json()
            audit_res = await self.gemini.audit_milestone_batch(b_idx, batch_trades, json.loads(params))
            if audit_res.get("audit"):
                log.info("=" * 72)
                log.info("🤖 GEMINI AI MILESTONE AUDIT (Batch #%d):", b_idx)
                for line in audit_res["audit"].split("\n"):
                    if line.strip():
                        log.info("  * %s", line.strip())
                log.info("=" * 72)
        except Exception as exc:
            log.debug("Gemini milestone audit error: %s", exc)

    def _log_milestone_banner(self, m: dict) -> None:
        b_idx = m["batch_index"]
        b_win = m["batch_wins"]
        b_loss = m["batch_losses"]
        b_wr = m["batch_win_rate"] * 100
        b_net = m["batch_net_pnl"]
        b_prof = m["batch_gross_profit"]
        b_loss_amt = m["batch_gross_loss"]
        b_pf = m["batch_profit_factor"]

        o_trades = m["overall_trades"]
        o_win = m["overall_wins"]
        o_loss = m["overall_losses"]
        o_wr = m["overall_win_rate"] * 100
        o_net = m["overall_net_pnl"]
        o_pf = m["overall_profit_factor"]

        lines = [
            "=" * 72,
            f"📊 MILESTONE: 10 TRADES COMPLETED (Batch #{b_idx} | Total {o_trades} Trades)",
            "-" * 72,
            f"BATCH PERFORMANCE (Last 10 Trades):",
            f"  * Win Rate:      {b_wr:.1f}% ({b_win} Wins, {b_loss} Losses, {m.get('batch_be', 0)} Breakeven)",
            f"  * Gross Profit:  +${b_prof:.2f}",
            f"  * Gross Loss:    -${b_loss_amt:.2f}",
            f"  * Batch Net PnL: {'+' if b_net >= 0 else ''}${b_net:.2f}",
            f"  * Profit Factor: {b_pf:.2f}",
            "",
            f"SESSION PERFORMANCE ({o_trades} Total Trades in Memory):",
            f"  * Win Rate:      {o_wr:.1f}% ({o_win} Wins, {o_loss} Losses)",
            f"  * Total Gross:   +${m['overall_gross_profit']:.2f} | Gross Loss: -${m['overall_gross_loss']:.2f}",
            f"  * Total Net PnL: {'+' if o_net >= 0 else ''}${o_net:.2f}",
            f"  * Profit Factor: {o_pf:.2f}",
            "",
            f"AI PARAMETERS & CONTINUOUS LEARNING:",
            f"  * Active Strategy: v{m['params_version']} | Min Score: {m['min_score']:.2f} | RSI Bounds: [{m['rsi_short_min']:.1f} - {m['rsi_long_max']:.1f}]",
        ]
        if m.get("notes"):
            for note in m["notes"]:
                lines.append(f"  * {note}")
        lines.append("=" * 72)
        for line in lines:
            log.info(line)

    # ------------------------------------------------------------------ #
    # Self-learning + housekeeping
    # ------------------------------------------------------------------ #
    async def _maybe_review(self) -> None:
        if self.learner.should_review():
            report = await asyncio.to_thread(self.learner.review)
            if report.changes:
                log.warning("STRATEGY ADJUSTED: %s", report.changes)

    async def _maintenance_loop(self) -> None:
        last_status = 0.0
        last_portfolio_sync = 0.0
        while True:
            await asyncio.sleep(15)
            try:
                if self.risk.kill_switch_active():
                    if self.cfg.risk.flatten_on_kill_switch and not self._flattened:
                        await self._flatten_all()

                # Periodic portfolio reconciliation with Deriv broker state
                if not self.dry_run and self._active_trackers and time.time() - last_portfolio_sync >= 45:
                    last_portfolio_sync = time.time()
                    try:
                        port_res = await self.client._request({"portfolio": 1})
                        active_ids = {c.get("contract_id") for c in port_res.get("portfolio", {}).get("contracts", [])}
                        for cid in list(self._active_trackers.keys()):
                            if cid not in active_ids:
                                poc_res = await self.client._request({"proposal_open_contract": 1, "contract_id": cid})
                                poc = poc_res.get("proposal_open_contract", {})
                                pnl = float(poc.get("profit", 0.0))
                                exit_price = float(poc.get("exit_tick") or poc.get("sell_spot") or poc.get("current_spot") or 0.0)
                                log.info("RECONCILED: Contract %s was closed externally on Deriv (status=%s, pnl=%.2f, exit=%s)",
                                         cid, poc.get("status"), pnl, exit_price)
                                self.store.log_close_trade(cid, exit_price, pnl)
                                ext_tracker = self._active_trackers.get(cid)
                                self._book_for(
                                    ext_tracker.contract_type if ext_tracker else "multiplier"
                                ).register_close(cid, pnl)
                                self.learner.note_trade_closed()
                                milestone = self.learner.check_and_run_milestone()
                                if milestone:
                                    self._log_milestone_banner(milestone)
                                    asyncio.create_task(self._run_gemini_milestone_audit(milestone))
                                tracker = self._active_trackers.pop(cid, None)
                                self._finalized.add(cid)
                                sub = self._contract_subs.pop(cid, None)
                                if sub:
                                    try:
                                        await self.client.unsubscribe(sub)
                                    except Exception:
                                        pass
                                if pnl < 0:
                                    asyncio.create_task(self._conduct_loss_post_mortem(
                                        cid=cid, tracker=tracker, pnl=pnl, exit_price=exit_price,
                                        status=poc.get("status", "sold")
                                    ))
                    except Exception as exc:
                        log.debug("Portfolio reconciliation skipped: %s", exc)

                if Path("STOP_AFTER_TRADE").exists():
                    if not getattr(self, "_drain_mode_announced", False):
                        self._drain_mode_announced = True
                        log.info("🛑 DRAIN MODE ACTIVE: No new trades will be opened. Waiting for %d active trade(s) to close before shutdown.",
                                 len(self._active_trackers))
                    if len(self._active_trackers) == 0:
                        log.info("🏁 DRAIN COMPLETE: All active trades have closed. Shutting down bot gracefully as requested.")
                        self._stop.set()
                        break

                # 15-Minute Periodic Live Engine Audit Report (Log + Telegram)
                audit_interval_sec = max(60, self.cfg.telegram.report_interval_min * 60)
                if self._last_audit_report_time == 0.0 or (time.time() - self._last_audit_report_time >= audit_interval_sec):
                    self._last_audit_report_time = time.time()
                    await self._emit_engine_audit_report()

                await self._maybe_send_daily_summary()

                if time.time() - last_status >= 15:
                    last_status = time.time()
                    if self._active_trackers:
                        pos_summary = ", ".join(
                            f"{t.symbol}: ${t.current_profit:+.2f} (pk ${t.peak_profit:+.2f})"
                            for t in self._active_trackers.values()
                        )
                        log.info("📊 LIVE ACTIVE POSITIONS (%d): %s", len(self._active_trackers), pos_summary)
                    else:
                        prices = " | ".join(
                            f"{s}: {self.feeds[s].last_price:.4f}"
                            for s in self.symbols
                            if s in self.feeds and math.isfinite(self.feeds[s].last_price)
                        )
                        log.debug("SCAN [%s] Balance: $%.2f | Ticks: %s",
                                  "DRY-RUN" if self.dry_run else "LIVE",
                                  self._risk_snapshot().get("day_start_equity", 0.0),
                                  prices or "streaming...")

                await self._maybe_review()
            except Exception:
                log.exception("maintenance error")

    async def _emit_engine_audit_report(self) -> None:
        """Emits beautified Live Engine Audit Report to console/log and Telegram."""
        stats = self.store.get_today_audit_stats()
        open_pos = len(self._active_trackers)
        st = self._risk_snapshot()
        # Live broker balance is authoritative; reconcile the day-start equity
        # so the daily loss/profit limits cannot drift from reality.
        try:
            bal = await self.client.get_balance()
            for book in {id(b): b for b in self.risk_books.values()}.values():
                book.reconcile_equity(bal)
            st = self._risk_snapshot()
        except Exception as exc:
            log.debug("Audit balance fetch failed (%s); using derived equity", exc)
            bal = st.get("day_start_equity", 0.0) + st.get("realized_pnl_today", 0.0)
        utc_time = time.strftime("%H:%M:%S UTC", time.gmtime())
        total = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        wr = stats.get("win_rate", 0.0)
        day_pnl = stats.get("day_pnl", 0.0)
        streak = stats.get("win_streak", 0)
        yday = stats.get("yesterday", {})
        allt = stats.get("all_time", {})
        settled = stats.get("settled_today", {})
        carried = max(0, int(settled.get("total_trades", 0)) - int(total))
        prev_str = (f"{yday.get('wins', 0)}W / {yday.get('losses', 0)}L "
                    f"({yday.get('win_rate', 0.0):.1f}%) | {yday.get('pnl', 0.0):+.2f} USD"
                    if yday.get("total_trades") else "no trades")
        carry_line = ""
        if carried:
            carry_line = (f"║  Settled today (cash): {settled.get('total_trades', 0)} Trades "
                          f"({settled.get('wins', 0)}W / {settled.get('losses', 0)}L) | "
                          f"{settled.get('pnl', 0.0):+.2f} USD, incl. {carried} opened earlier\n")
        day_label = stats.get("date", "today")
        pnl_mark = "🟢" if day_pnl >= 0 else "🔴"

        audit_card = (
            f"\n╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║                         LIVE ENGINE AUDIT UPDATE                             ║\n"
            f"╠══════════════════════════════════════════════════════════════════════════════╣\n"
            f"║  Scan Engine Status: {utc_time} bar close nominal across {len(self.symbols)} symbols\n"
            f"║  Active Open Contracts: {open_pos:<4}                                               \n"
            f"║  Current Account Balance: ${bal:.2f} USD                                      \n"
            f"║  Today (opened, UTC, {day_label}): {total} Trades ({wins}W / {losses}L) | Win Rate: {wr:.1f}% | {day_pnl:+.2f} USD {pnl_mark}\n"
            f"{carry_line}"
            f"║  Yesterday: {prev_str}\n"
            f"║  All-time Journal: {allt.get('total_trades', 0)} Trades ({allt.get('wins', 0)}W / {allt.get('losses', 0)}L) | Win Rate: {allt.get('win_rate', 0.0):.1f}% | {allt.get('pnl', 0.0):+.2f} USD\n"
            f"║  Active Win Streak: {streak} Consecutive Wins                                  \n"
            f"║  Loss Clamp: {self.cfg.risk.max_consecutive_losses} consecutive losses -> cooldown\n"
            f"║  Active Filters: regime + ATR gates, Rise/Fall edge guard (see startup log)   \n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝"
        )
        log.info(audit_card)

        # Dispatch 15-min update to Telegram
        audit_payload = {
            "open_positions": open_pos,
            "balance": bal,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "day_pnl": day_pnl,
            "win_streak": streak,
            "day_label": day_label,
            "yesterday": yday,
            "settled_today": settled,
            "carried": carried,
            "all_time": allt,
            "pnl_mark": pnl_mark,
            "symbol_count": len(self.symbols),
            "loss_clamp": self.cfg.risk.max_consecutive_losses,
        }
        await self.telegram.notify_engine_audit(audit_payload)

    async def _maybe_send_daily_summary(self) -> None:
        """Send a summary of the completed UTC day once, right after rollover.

        Without this, the first alert of a new day reports "today: 0 trades" and
        yesterday's result is never announced on its own. The marker is stored in
        the journal so restarts do not duplicate or skip the summary.
        """
        today = time.strftime("%Y-%m-%d", time.gmtime())
        now = time.time()
        if now - getattr(self, "_last_summary_check", 0.0) < 60:
            return
        self._last_summary_check = now
        last = self.store.get_kv("last_daily_summary_day")
        if last == today:
            return
        if last is None:
            # First run with this feature: do not retro-send an arbitrary day.
            self.store.set_kv("last_daily_summary_day", today)
            return
        prev = self.store.get_day_stats(1)
        if prev.get("total_trades"):
            allt = self.store.get_all_time_stats()
            await self.telegram.notify_daily_summary(
                prev, allt, self.store.get_day_stats(1, "settled"))
            log.info("🗓 Daily summary sent for %s: %d trades, %dW/%dL, %+.2f USD",
                     prev.get("date"), prev.get("total_trades", 0), prev.get("wins", 0),
                     prev.get("losses", 0), prev.get("pnl", 0.0))
        else:
            log.info("🗓 No trades settled on %s; daily summary skipped", prev.get("date"))
        self.store.set_kv("last_daily_summary_day", today)

    def _all_open_positions(self) -> Dict[int, OpenPosition]:
        """Union of open positions across the isolated books."""
        merged: Dict[int, OpenPosition] = {}
        for book in self.risk_books.values():
            merged.update(book.open_positions)
        return merged

    async def _flatten_all(self) -> None:
        self._flattened = True
        log.warning("Kill switch: closing all open contracts")
        for cid in list(self._all_open_positions()):
            try:
                await self.client.sell_contract(cid)
            except DerivAPIError as exc:
                log.error("Could not close %s: %s", cid, exc)

    async def _shutdown(self) -> None:
        log.info("Shutting down. Open contracts keep their server-side SL/TP: %s",
                 list(self._all_open_positions()))
        for t in self._tasks:
            t.cancel()
        for sub in self._feed_subs:
            try:
                await self.client.unsubscribe(sub)
            except Exception:
                pass
        await self.client.close()


async def _amain(args) -> None:
    cfg = Config.from_env()
    if args.mode:
        cfg = replace(cfg, trade_mode=TradeMode(args.mode.lower()))
    setup_logging(cfg.log_dir, args.log_level)
    bot = TradingBot(cfg, dry_run=args.dry_run)
    if args.preflight:
        try:
            await bot.preflight()
        finally:
            await bot.client.close()
        return
    await bot.run()


def _acquire_instance_lock(path: Path = Path("data/bot.lock")):
    """Refuse to start a second bot against the same journal/account.

    Two concurrent instances produce two contracts on the same symbol in the
    same minute (the journal has such pairs: stpRNG 2026-10-18 06:03, 1HZ50V
    2026-10-22 15:01) and duplicate every log line. O_EXCL makes the check
    atomic, so a race between two starts cannot both win.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            f = os.fdopen(fd, "w")
            f.write(str(os.getpid()))
            f.flush()
            return f
        except FileExistsError:
            try:
                other = int(path.read_text().strip() or 0)
            except (ValueError, OSError):
                other = 0
            if other and other != os.getpid():
                try:
                    os.kill(other, 0)
                except ProcessLookupError:
                    path.unlink(missing_ok=True)   # stale lock
                    continue
                except PermissionError:
                    pass
                print(f"Another bot instance is already running (pid {other}). "
                      f"Refusing to start. Remove {path} if that is wrong.",
                      file=sys.stderr)
                return None
            path.unlink(missing_ok=True)
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Deriv autonomous trading bot")
    ap.add_argument("--mode", choices=["multiplier", "rise_fall", "hybrid"], default=None,
                    help="Trading mode: 'multiplier' (only multipliers), 'rise_fall' (only fixed expiry), 'hybrid' (both)")
    ap.add_argument("--preflight", action="store_true", help="connect, validate, exit")
    ap.add_argument("--dry-run", action="store_true", help="evaluate everything, send no orders")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    # Preflight is read-only; it may run alongside a live instance.
    lock = None if args.preflight else _acquire_instance_lock()
    if lock is None and not args.preflight:
        raise SystemExit(1)
    try:
        asyncio.run(_amain(args))
    finally:
        if lock is not None:
            try:
                Path("data/bot.lock").unlink(missing_ok=True)
            except OSError:
                pass
            lock.close()


if __name__ == "__main__":
    main()
