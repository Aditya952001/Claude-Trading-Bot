"""
risk_manager.py - Deterministic, hardcoded risk guardrail layer.

Design rules
  * No machine learning, no adaptive logic, no imports from the learning module.
  * Never trusts the strategy: it recomputes stop distance, size, SL and TP
    from raw ATR + price + account equity.
  * Every signal passes through `evaluate()`; only an approved ApprovedOrder
    may reach the Deriv client.
  * Absolute ceilings below override ANY config / env / learned value.

Position sizing for Deriv multipliers
  Profit/loss on a multiplier = stake * multiplier * (price change %).
  We want the loss at the stop-loss to equal `risk_amount` (a % of equity):

      stop_pct   = (ATR * sl_atr_mult) / entry_price
      stake      = risk_amount / (multiplier * stop_pct)
      stop_loss  = risk_amount            (currency amount sent to Deriv)
      take_profit= risk_amount * reward_risk

  Deriv also stops the contract out when loss reaches the stake, so we require
  multiplier * stop_pct < max_stop_out_fraction, i.e. our stop is well inside
  the stop-out level.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import RiskConfig

log = logging.getLogger("risk_manager")

# --- Absolute ceilings: cannot be exceeded by configuration ---------------- #
HARD_MAX_RISK_PER_TRADE = 0.02
HARD_MAX_DAILY_DRAWDOWN = 0.10
HARD_MAX_OPEN_POSITIONS = 5
HARD_MAX_TOTAL_OPEN_RISK = 0.06
HARD_MAX_STAKE_PCT = 0.40


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    asset_class: str
    direction: str            # 'long' | 'short'
    entry_price: float
    atr: float
    confidence: float
    regime: str
    multiplier: int
    tick_age_sec: float
    granularity: int
    contract_type: str = "multiplier"     # 'multiplier' | 'rise_fall'
    duration: int = 3
    duration_unit: str = "m"


@dataclass(frozen=True)
class ApprovedOrder:
    symbol: str
    direction: str
    stake: float
    multiplier: int
    stop_loss_amount: float
    take_profit_amount: float
    risk_amount: float
    sl_price: float
    tp_price: float
    contract_type: str = "multiplier"     # 'multiplier' | 'rise_fall'
    duration: int = 3
    duration_unit: str = "m"


@dataclass
class RiskDecision:
    approved: bool
    order: Optional[ApprovedOrder] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class OpenPosition:
    contract_id: int
    symbol: str
    risk_amount: float
    stake: float
    contract_type: str = "multiplier"


class RiskManager:
    def __init__(self, cfg: RiskConfig, store=None, clock=time.time,
                 rf_asset_classes: tuple = (), book: str = ""):
        """
        store: optional object with get_kv(key)->Optional[str] and set_kv(key, str)
               so daily counters survive restarts.
        rf_asset_classes: asset classes actually traded for Rise/Fall; evidence
               stats are scoped to them so retired instruments (e.g. forex
               digitals) cannot distort the guard.
        book: name of the isolated strategy ledger. Empty means the legacy
               single shared ledger. Separate books keep independent
               trades/day, drawdown, loss-streak and symbol-cooldown state, so
               one strategy's losses cannot halt or veto the other.
        """
        self.cfg = cfg
        self.store = store
        self.clock = clock
        self.book = book
        self.rf_asset_classes = tuple(rf_asset_classes or ())
        # Optional callable returning stakes held by OTHER isolated books. Equity
        # must include every open stake or the daily limits drift when the other
        # strategy has positions open.
        self.other_stakes_fn = None
        # effective (clamped) limits
        self.max_risk = min(cfg.max_risk_per_trade, HARD_MAX_RISK_PER_TRADE)
        self.max_dd = min(cfg.max_daily_drawdown, HARD_MAX_DAILY_DRAWDOWN)
        self.max_positions = min(cfg.max_open_positions, HARD_MAX_OPEN_POSITIONS)
        self.max_open_risk = min(cfg.max_total_open_risk, HARD_MAX_TOTAL_OPEN_RISK)
        self.max_stake_pct = min(cfg.max_stake_pct_of_equity, HARD_MAX_STAKE_PCT)
        if self.max_risk != cfg.max_risk_per_trade or self.max_dd != cfg.max_daily_drawdown:
            log.warning("Configured risk limits exceeded hard ceilings and were clamped.")

        self.open_positions: Dict[int, OpenPosition] = {}
        self.day: str = ""
        self.day_start_equity: float = 0.0
        self.realized_pnl_today: float = 0.0
        self.trades_today: int = 0
        self.consecutive_losses: int = 0
        self.halted_until: float = 0.0
        self.last_close_ts: Dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    @property
    def _state_key(self) -> str:
        return f"risk_state:{self.book}" if self.book else "risk_state"

    def _load(self) -> None:
        if not self.store:
            return
        raw = self.store.get_kv(self._state_key)
        if raw:
            s = json.loads(raw)
            for k in ("day", "day_start_equity", "realized_pnl_today", "trades_today",
                      "consecutive_losses", "halted_until", "last_close_ts"):
                if k in s:
                    setattr(self, k, s[k])

    def _save(self) -> None:
        if not self.store:
            return
        self.store.set_kv(self._state_key, json.dumps({
            "day": self.day, "day_start_equity": self.day_start_equity,
            "realized_pnl_today": self.realized_pnl_today, "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses, "halted_until": self.halted_until,
            "last_close_ts": self.last_close_ts,
        }))

    # ------------------------------------------------------------------ #
    # State updates (called by the orchestrator)
    # ------------------------------------------------------------------ #
    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def _equity(self, balance: float) -> float:
        # Deriv deducts open stakes from balance, so add them back for equity.
        # Include stakes held by the other isolated book(s) too.
        other = self.other_stakes_fn() if callable(self.other_stakes_fn) else 0.0
        return balance + sum(p.stake for p in self.open_positions.values()) + other

    def roll_day_if_needed(self, balance: float) -> None:
        today = self._today()
        if self.day != today:
            self.day = today
            self.day_start_equity = self._equity(balance)
            self.realized_pnl_today = 0.0
            self.trades_today = 0
            log.info("New risk day %s, start equity %.2f", today, self.day_start_equity)
            self._save()

    def register_open(self, contract_id: int, symbol: str, risk_amount: float, stake: float, contract_type: str = "multiplier") -> None:
        self.open_positions[contract_id] = OpenPosition(contract_id, symbol, risk_amount, stake, contract_type)
        self.trades_today += 1
        self._save()

    def has_open_on_symbol(self, symbol: str) -> bool:
        """True if this book currently holds any contract on `symbol`."""
        return any(p.symbol == symbol for p in self.open_positions.values())

    def register_close(self, contract_id: int, pnl: float) -> None:
        pos = self.open_positions.pop(contract_id, None)
        self.realized_pnl_today += pnl
        if pos:
            self.last_close_ts[pos.symbol] = self.clock()
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.halted_until = self.clock() + self.cfg.loss_cooldown_minutes * 60
                log.warning("%d consecutive losses -> trading halted for %d min",
                            self.consecutive_losses, self.cfg.loss_cooldown_minutes)
        elif pnl > 0:
            self.consecutive_losses = 0
        self._save()

    def reconcile_equity(self, balance: float, tol: float = 0.5) -> None:
        """Align the persisted day-start equity with the broker's real balance.

        The stored value can go stale when the demo account is reset (Deriv
        restores the $10k starting balance) or when the process restarts on a
        new day without rolling. Every displayed balance and every daily limit
        is derived from this number, so a stale value silently reports the
        wrong equity. Implied start = current equity - realised PnL today.
        """
        if balance <= 0:
            return
        equity = self._equity(balance)
        implied = equity - self.realized_pnl_today
        if self.day_start_equity <= 0 or abs(implied - self.day_start_equity) > tol:
            log.warning(
                "Equity reconcile: stored day_start_equity %.2f != implied %.2f "
                "(broker balance %.2f, realised today %+.2f) -> resetting",
                self.day_start_equity, implied, balance, self.realized_pnl_today,
            )
            self.day_start_equity = implied
            self._save()

    def kill_switch_active(self) -> bool:
        return os.path.exists(self.cfg.kill_switch_file)

    def daily_limit_hit(self) -> bool:
        return (self.day_start_equity > 0 and
                self.realized_pnl_today <= -self.max_dd * self.day_start_equity)

    def daily_target_profit_hit(self) -> bool:
        return (self.cfg.target_daily_profit > 0 and
                self.day_start_equity > 0 and
                self.realized_pnl_today >= self.cfg.target_daily_profit * self.day_start_equity)

    def status(self) -> dict:
        return {"day": self.day, "day_start_equity": round(self.day_start_equity, 2),
                "realized_pnl_today": round(self.realized_pnl_today, 2),
                "trades_today": self.trades_today, "open": len(self.open_positions),
                "consecutive_losses": self.consecutive_losses,
                "halted": self.halted_until > self.clock(),
                "daily_limit_hit": self.daily_limit_hit(),
                "daily_profit_target_hit": self.daily_target_profit_hit()}

    # ------------------------------------------------------------------ #
    # Evidence gate for fixed-expiry contracts
    # ------------------------------------------------------------------ #
    def _rf_edge_veto(self, intent: TradeIntent) -> Optional[str]:
        """
        Stop paying for a negative-expectancy digitals book.

        Fixed-expiry Rise/Fall pays ~0.88 profit per 1.00 staked, so the
        breakeven win rate is ~53%. A 14-day replay of this exact engine on
        real 1m candles produced 48% (and 43% out-of-sample), i.e. every trade
        is a negative-EV lottery ticket. Rather than pretend a filter fixes
        that, this guard requires the journal to *prove* an edge:

          * fewer than `rf_edge_min_trades` closed RF trades -> exploration
            is allowed at the fixed (minimum) stake;
          * afterwards the 95% Wilson lower bound of the realised win rate
            must exceed the payout-implied breakeven, otherwise RF entries
            are vetoed (set RF_EDGE_GUARD_ENABLED=false to override).

        The evidence is scoped per asset class, because the classes are
        economically different: forex/metals pay ~0.79-0.82 (breakeven
        ~55%), synthetics pay ~0.88-0.95 (breakeven ~51-53%), and each class
        is allowed its own exploration window. A class that has not yet
        reached `rf_edge_min_trades` trades is explored at the fixed stake;
        an older class that fails its own breakeven test is vetoed.

        A per-symbol version of the same test is applied once that symbol has
        its own sample, so one persistently losing index is dropped even if
        the book as a whole is marginal.
        """
        try:
            stats = self.store.rf_edge_stats(limit=self.cfg.rf_edge_lookback,
                                             asset_classes=self.rf_asset_classes)
        except Exception as exc:  # never let a journal read block trading
            log.warning("RF edge guard: journal stats unavailable (%s); allowing entry", exc)
            return None
        if not stats:
            return None

        def verdict(s: dict, min_n: int, who: str) -> Optional[str]:
            if not s or s.get("n", 0) < min_n:
                return None
            if s["wr_lo"] >= s["breakeven"]:
                return None
            return (f"{who}edge not proven: {s['n']} trades, WR {s['wr']*100:.1f}% "
                    f"(95% LB {s['wr_lo']*100:.1f}%) < breakeven {s['breakeven']*100:.1f}%")

        # Per asset class: each class proves its own edge (or explores at the
        # fixed stake). A class that fails its own breakeven test is vetoed
        # without dragging down a class that is still in exploration.
        cls = (intent.asset_class or "").lower()
        by_class = stats.get("by_asset_class") or {}
        cls_veto = verdict(by_class.get(cls), self.cfg.rf_edge_min_trades, f"{cls} RF ")
        if cls_veto:
            return cls_veto

        sym = (stats.get("by_symbol") or {}).get(intent.symbol)
        return verdict(sym, self.cfg.rf_edge_symbol_min_trades, f"{intent.symbol} ")

    # ------------------------------------------------------------------ #
    # The gate
    # ------------------------------------------------------------------ #
    def evaluate(self, intent: TradeIntent, balance: float, asset_weight: float = 1.0) -> RiskDecision:
        now = self.clock()
        self.roll_day_if_needed(balance)
        veto: List[str] = []

        # ---- state-based vetoes ----
        if self.kill_switch_active():
            veto.append("kill switch file present")
        if self.daily_limit_hit():
            veto.append(f"daily loss limit hit ({self.realized_pnl_today:.2f} of "
                        f"-{self.max_dd * self.day_start_equity:.2f})")
        if self.daily_target_profit_hit():
            veto.append(f"daily profit target hit (+{self.realized_pnl_today:.2f} of "
                        f"+{self.cfg.target_daily_profit * self.day_start_equity:.2f})")
        if self.halted_until > now:
            veto.append(f"loss-streak cooldown until {datetime.fromtimestamp(self.halted_until, tz=timezone.utc):%H:%M}Z")
        elif self.halted_until > 0:
            self.halted_until = 0.0
            self.consecutive_losses = 0
            self._save()

        if self.trades_today >= self.cfg.max_trades_per_day:
            veto.append("max trades per day reached")
        if len(self.open_positions) >= self.max_positions:
            veto.append("max open positions reached")
        ctype = getattr(intent, "contract_type", "multiplier")
        if any(p.symbol == intent.symbol and getattr(p, "contract_type", "multiplier") == ctype for p in self.open_positions.values()):
            veto.append(f"{ctype} position already open on symbol")
        cooldown = self.cfg.symbol_cooldown_bars * intent.granularity
        if now - self.last_close_ts.get(intent.symbol, 0) < cooldown:
            veto.append("symbol cooldown after last close")

        # ---- AI Dynamic Asset Filter & Pruner ----
        # In demo exploration mode, do not hard-block assets so we can gather telemetry and optimize them
        ctype = getattr(intent, "contract_type", "multiplier")
        if ctype != "rise_fall" and self.store and hasattr(self.store, "get_symbol_stats"):
            stats = self.store.get_symbol_stats(intent.symbol, limit=15)
            if stats["trades"] >= 5 and stats["win_rate"] < 0.30 and stats["net_pnl"] < -0.50:
                veto.append(f"AI asset pruner: {intent.symbol} win rate {stats['win_rate']*100:.0f}% (< 30%) with net loss in memory")
            elif stats["consecutive_losses"] >= 3:
                last_loss_time = self.last_close_ts.get(intent.symbol, 0)
                if now - last_loss_time < 3600:
                    veto.append(f"AI asset cooldown: {intent.symbol} in cooldown after {stats['consecutive_losses']} consecutive losses")
        elif ctype == "rise_fall":
            # A v11 hard blacklist (1HZ15V + forex tokens) was removed here:
            # 1HZ15V is average in the 14-day replay (18 trades, 50% WR,
            # -$0.06/trade) and the blacklist came from 3 live losses. Forex
            # is already excluded by RF_ALLOWED_ASSETS, so those tokens were
            # dead code. Asset-level pruning for RF belongs in the edge guard
            # (rf_edge_symbol_min_trades), which uses a sample-size threshold.
            pass

        if ctype == "rise_fall" and self.cfg.rf_edge_guard and self.store \
                and hasattr(self.store, "rf_edge_stats"):
            edge_veto = self._rf_edge_veto(intent)
            if edge_veto:
                veto.append(edge_veto)

        # ---- input sanity ----
        if intent.direction not in ("long", "short"):
            veto.append("invalid direction")
        if not all(map(math.isfinite, (intent.entry_price, intent.atr, balance))) \
                or intent.entry_price <= 0 or intent.atr <= 0:
            veto.append("non-finite or non-positive price/ATR/balance")
        if intent.tick_age_sec > self.cfg.max_tick_age_sec:
            veto.append(f"stale data ({intent.tick_age_sec:.0f}s old)")
        if ctype == "rise_fall":
            # Was 0.40 (score >= 2.0/5). A v11 attempt to hard-code 0.50 was
            # reverted: it improved validation (+0.062) but degraded training
            # (-0.026), i.e. one half only. Keep the value the baseline used.
            min_conf = 0.40
        else:
            min_conf = self.cfg.min_confidence
        if intent.confidence < min_conf:
            veto.append(f"confidence {intent.confidence:.2f} < {min_conf:.2f}")
        if balance < self.cfg.min_balance:
            veto.append("balance below minimum")
        weight = min(1.0, max(0.0, asset_weight))      # learning may only REDUCE risk
        if weight <= 0:
            veto.append("asset class weight is zero")
        if veto:
            return RiskDecision(False, reasons=veto)

        # ---- independent sizing ----
        equity = self._equity(balance)
        if ctype == "rise_fall":
            base_stake = self.cfg.fixed_stake if self.cfg.fixed_stake > 0 else 1.0
            max_stake = self.cfg.max_stake if self.cfg.max_stake > 0 else base_stake

            # --- Confidence-Tiered Dynamic Staking ---
            # OFF by default (CONFIDENCE_STAKING=false). Scaling stake with
            # confidence only helps if the confidence score is actually
            # predictive of the outcome; on this book it is not (train-only
            # filters that used it lost their edge out-of-sample). While there
            # is no proven edge, staking more on "high conviction" simply
            # increases variance and drawdown. Opt in only after an edge has
            # been demonstrated on unseen data.
            conf = intent.confidence
            if not self.cfg.confidence_staking:
                stake = base_stake
            elif conf >= 0.80 and max_stake > base_stake:
                stake = max_stake
                log.info("🔥 ELITE SIGNAL [conf=%.2f] → MAX_STAKE $%.2f applied", conf, stake)
            elif conf >= 0.70 and max_stake > base_stake:
                lerp = 0.75
                stake = round(base_stake + lerp * (max_stake - base_stake), 2)
                log.info("⚡ HIGH-CONF SIGNAL [conf=%.2f] → boosted stake $%.2f", conf, stake)
            elif conf >= 0.60 and max_stake > base_stake:
                lerp = 0.50
                stake = round(base_stake + lerp * (max_stake - base_stake), 2)
                log.info("✅ GOOD SIGNAL [conf=%.2f] → moderate stake $%.2f", conf, stake)
            else:
                stake = base_stake

            # Clamp: never exceed max_stake from env
            stake = min(stake, max_stake)
            stake = math.floor(stake * 100) / 100  # round down to cent

            risk_amount = stake
            total_open = sum(p.risk_amount for p in self.open_positions.values()) + risk_amount
            if total_open > self.max_open_risk * equity:
                return RiskDecision(False, reasons=[
                    f"total open risk {total_open:.2f} > cap {self.max_open_risk * equity:.2f}"])
            order = ApprovedOrder(
                symbol=intent.symbol, direction=intent.direction, stake=stake,
                multiplier=0,
                stop_loss_amount=risk_amount,
                take_profit_amount=round(stake * 0.90, 2),
                risk_amount=risk_amount,
                sl_price=intent.entry_price,
                tp_price=intent.entry_price,
                contract_type="rise_fall",
                duration=intent.duration,
                duration_unit=intent.duration_unit,
            )
            return RiskDecision(True, order=order)

        risk_amount = equity * self.max_risk * weight
        stop_dist = intent.atr * self.cfg.sl_atr_mult
        stop_pct = stop_dist / intent.entry_price
        loss_fraction = intent.multiplier * stop_pct        # fraction of stake lost at stop
        if loss_fraction >= self.cfg.max_stop_out_fraction:
            return RiskDecision(False, reasons=[
                f"stop too close to stop-out (mult*stop%={loss_fraction:.2f}); lower the multiplier"])

        if self.cfg.fixed_stake > 0:
            stake = self.cfg.fixed_stake
            risk_amount = stake * loss_fraction
        else:
            stake = risk_amount / loss_fraction

        max_stake = equity * self.max_stake_pct
        if self.cfg.max_stake > 0:
            max_stake = min(max_stake, self.cfg.max_stake)
        if stake > max_stake:
            stake = max_stake
        stake = math.floor(stake * 100) / 100

        # Provide sufficient breathing room to survive tick/spread noise:
        # A 10-cent stop on 80x-100x multiplier chokes on 0.1% noise.
        # Ensure stop loss is at least 40% of stake (e.g. $0.40 on $1 stake),
        # allowing 0.4%-0.7% price room, capped at 70% of stake.
        effective_loss_fraction = max(loss_fraction, 0.40)
        effective_loss_fraction = min(effective_loss_fraction, 0.70)
        risk_amount = round(stake * effective_loss_fraction, 2)
        if risk_amount < 0.20 and stake >= 1.00:
            risk_amount = 0.40

        if stake < self.cfg.min_stake:
            # Rounding UP to min stake would exceed the risk budget -> refuse.
            return RiskDecision(False, reasons=[
                f"required stake {stake:.2f} below minimum {self.cfg.min_stake}; "
                "risk budget too small for this volatility"])
        if risk_amount <= 0 or risk_amount >= stake:
            return RiskDecision(False, reasons=["degenerate risk amount"])

        total_open = sum(p.risk_amount for p in self.open_positions.values()) + risk_amount
        if total_open > self.max_open_risk * equity:
            return RiskDecision(False, reasons=[
                f"total open risk {total_open:.2f} > cap {self.max_open_risk * equity:.2f}"])

        sign = 1 if intent.direction == "long" else -1
        order = ApprovedOrder(
            symbol=intent.symbol, direction=intent.direction, stake=stake,
            multiplier=intent.multiplier,
            stop_loss_amount=risk_amount,
            take_profit_amount=round(risk_amount * self.cfg.reward_risk, 2),
            risk_amount=risk_amount,
            sl_price=intent.entry_price - sign * stop_dist,
            tp_price=intent.entry_price + sign * stop_dist * self.cfg.reward_risk,
        )
        return RiskDecision(True, order=order)
