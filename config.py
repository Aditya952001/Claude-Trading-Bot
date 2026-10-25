"""
config.py - Central configuration for the Deriv autonomous trading bot.

Secrets and environment toggles come from environment variables / .env.
Everything else is typed, frozen dataclasses so a running process cannot
mutate its own risk limits by accident.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple

try:  # optional dependency
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THE_RISKS"


class Environment(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class AssetClass(str, Enum):
    SYNTHETIC = "synthetic"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"


class TradeMode(str, Enum):
    MULTIPLIER = "multiplier"      # Trades only Multipliers
    RISE_FALL = "rise_fall"        # Trades only Fixed Expiry Rise/Fall (CALL/PUT)
    HYBRID = "hybrid"              # Trades both Multipliers and Rise/Fall concurrently


# ----------------------------------------------------------------------------
# Connection
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DerivConfig:
    app_id: str
    api_token: str
    environment: Environment
    live_confirmation: str = ""
    # Legacy public WebSocket endpoint. Deriv has been evolving its API; if the
    # endpoint changes, override with DERIV_WS_URL (see developers.deriv.com).
    ws_base_url: str = "wss://ws.derivws.com/websockets/v3"
    currency: str = "USD"
    request_timeout_sec: float = 15.0

    @property
    def ws_url(self) -> str:
        return f"{self.ws_base_url}?app_id={self.app_id}"


# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SymbolConfig:
    symbol: str                 # Deriv symbol code
    asset_class: AssetClass
    multiplier: int             # Deriv multiplier; MUST be valid for the symbol
    granularity: int = 60       # candle size in seconds (60 = 1 min)
    enabled: bool = True
    supports_multiplier: bool = True
    supports_rise_fall: bool = True


# NOTE: valid multipliers differ per symbol. The bot verifies them against
# Deriv's `contracts_for` at startup and automatically skips any symbol that is
# closed today (e.g. weekend forex) or doesn't match.
DEFAULT_SYMBOLS: Tuple[SymbolConfig, ...] = (
    # Continuous Volatility Indices (24/7) - Highest statistical predictability & smoothness
    SymbolConfig("R_10", AssetClass.SYNTHETIC, multiplier=400),
    SymbolConfig("R_25", AssetClass.SYNTHETIC, multiplier=160),
    SymbolConfig("R_50", AssetClass.SYNTHETIC, multiplier=80),
    SymbolConfig("R_75", AssetClass.SYNTHETIC, multiplier=50),
    SymbolConfig("R_100", AssetClass.SYNTHETIC, multiplier=40),

    # Jump Indices (24/7) - Clean trend following with strong momentum
    SymbolConfig("JD10", AssetClass.SYNTHETIC, multiplier=100),
    SymbolConfig("JD25", AssetClass.SYNTHETIC, multiplier=50),
    SymbolConfig("JD50", AssetClass.SYNTHETIC, multiplier=20),
    SymbolConfig("JD75", AssetClass.SYNTHETIC, multiplier=15),
    SymbolConfig("JD100", AssetClass.SYNTHETIC, multiplier=10),

    # Major Forex Pairs & Metals
    SymbolConfig("frxEURUSD", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxGBPUSD", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxUSDJPY", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxAUDUSD", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxUSDCAD", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxUSDCHF", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxEURJPY", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxGBPJPY", AssetClass.FOREX, multiplier=100),
    SymbolConfig("frxXAUUSD", AssetClass.COMMODITIES, multiplier=100),
    SymbolConfig("frxXAGUSD", AssetClass.COMMODITIES, multiplier=100),
)


# ----------------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50                 # 1-minute trend filter (50 periods)
    ema_higher_tf: int = 200            # higher-timeframe macro trend filter (200 1m periods = 40 5m bars)
    rsi_length: int = 14
    bb_length: int = 20
    bb_std: float = 2.0
    atr_length: int = 14
    adx_length: int = 14
    adx_trend_threshold: float = 20.0
    high_vol_atr_percentile: float = 0.90
    trade_high_vol: bool = True
    history_candles: int = 250
    min_candles: int = 150


# ----------------------------------------------------------------------------
# Risk (enforced by risk_manager.py, which also applies absolute hard ceilings)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = 0.01        # 1% of equity risked per trade
    max_daily_drawdown: float = 0.03        # halt for the day at -3% realized loss
    target_daily_profit: float = 0.05       # halt for the day at +5% realized profit (0 to disable)
    max_total_open_risk: float = 0.03       # sum of risk across open trades
    max_open_positions: int = 3
    max_trades_per_day: int = 30
    max_consecutive_losses: int = 4
    loss_cooldown_minutes: int = 10         # 10m cooldown
    symbol_cooldown_bars: int = 3
    sl_atr_mult: float = 1.5                # stop distance = ATR * this
    reward_risk: float = 2.0                # take-profit = stop * this
    min_confidence: float = 0.60
    min_stake: float = 1.0                  # minimum stake in USD
    max_stake: float = 50.0                 # maximum stake cap in USD per trade
    fixed_stake: float = 1.0                # fixed stake in USD (set to 1.0 for capital safety & $10 testing)
    max_stake_pct_of_equity: float = 0.25
    max_tick_age_sec: float = 60.0
    min_balance: float = 10.0
    max_stop_out_fraction: float = 0.80     # multiplier * stop% must stay < this
    kill_switch_file: str = "KILL_SWITCH"
    flatten_on_kill_switch: bool = False

    # --- Rise/Fall evidence guard (capital preservation) ---
    # Only keep placing fixed-expiry contracts while the bot's own closed-trade
    # history proves a positive edge: the 95% Wilson lower bound of the win rate
    # must sit above the payout breakeven. With n < rf_edge_min_trades it allows
    # exploration at the fixed (minimum) stake.
    rf_edge_guard: bool = True
    rf_edge_min_trades: int = 40
    rf_edge_symbol_min_trades: int = 20
    rf_edge_lookback: int = 120
    # Confidence-tiered stake boosting is OFF by default: scaling a 0-edge system
    # only multiplies variance. Opt in with CONFIDENCE_STAKING=true once an edge
    # has been demonstrated out-of-sample.
    confidence_staking: bool = False

    # Dynamic Profit Booking & Real-Time AI Trailing System (Targeting +$0.15 to +$0.50+ runners)
    dynamic_profit_enabled: bool = True
    green_shield_trigger_usd: float = 0.200  # Activates zero-loss green shield once profit touches +$0.200
    min_profit_take_usd: float = 0.220       # Minimum profit threshold to activate dynamic lock (+70% floor)
    breakeven_trigger_usd: float = 0.250     # Tier 1 runner lock threshold (locks in >= 75% of peak)
    trail_profit_fraction: float = 0.75      # Lock in at least 75% of peak profit on pullback
    early_exit_on_reversal: bool = True      # Close in green if momentum/candlestick reverses
    max_hold_bars_in_profit: int = 5         # Allow runner room: hold up to 5 bars in profit
    fast_exhaustion_cut: bool = True         # AI tick exhaustion on momentum breakdown

    # Dynamic Loss Management & Capital Protection (Breathing room for tick noise; stops below noise floor)
    dynamic_loss_enabled: bool = True
    max_loss_cut_usd: float = 0.200          # Tightened software loss cap: cuts trade at -$0.20 max
    max_bars_in_loss: int = 4                # Stagnant loser cut: exit if trade stays in loss for 4 consecutive bars
    early_cut_on_invalidation: bool = True   # Cut if candle breaks decisively contrary to EMA trend
    cut_on_opposing_signal: bool = True      # Cut immediately if an opposing trade signal is detected
    tighten_loss_bars: int = 3               # After 3 bars, tighten loss threshold
    adverse_tick_cut: bool = False           # Disabled: 1m synthetic tick noise triggers false exits on ticks


# ----------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LearningConfig:
    db_path: str = "data/bot_memory.sqlite3"
    review_every_n_trades: int = 10
    review_every_hours: float = 6.0
    lookback_trades: int = 60
    min_trades_for_review: int = 20
    min_group_trades: int = 8
    rollback_min_trades: int = 15          # trades needed to judge last change


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    report_interval_min: int = 15


# ----------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    deriv: DerivConfig
    symbols: Tuple[SymbolConfig, ...] = DEFAULT_SYMBOLS
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    trade_mode: TradeMode = TradeMode.HYBRID
    rise_fall_duration: int = 3
    rise_fall_duration_unit: str = "m"
    log_dir: str = "logs"

    # Rise/Fall asset policy. Forex/metal digitals pay ~0.79-0.82 with a spread,
    # i.e. a 55.5% breakeven on a 1-minute random walk; the backtest measured 40%
    # out-of-sample. They are excluded unless explicitly opted in.
    rf_allowed_assets: Tuple[str, ...] = ("synthetic",)
    # The LLM pre-trade veto is advisory by default (log-only). It added latency
    # (429s) and its rules were fitted to a 65-trade sample with no OOS edge.
    gemini_veto_enabled: bool = False

    # --- Hybrid isolation ---------------------------------------------------
    # In HYBRID mode Multiplier and Rise/Fall run as two separate books. Each
    # book owns its own risk ledger (trades/day, daily drawdown, loss-streak
    # cooldown, per-symbol cooldown) and its own open positions, so a loss in
    # one strategy can never veto, halt or force-close the other.
    #
    # Both books scan their FULL capability set: every symbol that supports
    # Multipliers is evaluated by the Multiplier book, and every symbol that
    # supports Rise/Fall is evaluated by the Rise/Fall book. There is no
    # static symbol split.
    #
    # Conflict prevention is dynamic: a symbol can be held by only one book
    # at a time. Whichever book opens first owns that symbol until the
    # contract closes; the other book is blocked on that symbol meanwhile.
    # If both books approve on the same bar, `hybrid_signal_priority` decides
    # which one takes the entry.
    #
    # HYBRID_MULT_SYMBOLS / HYBRID_RF_SYMBOLS are optional per-book
    # allow-lists (comma separated). Empty means "all capable symbols".
    hybrid_isolated_risk: bool = True
    hybrid_block_cross_symbol: bool = True
    hybrid_signal_priority: str = "rise_fall"   # "rise_fall" | "multiplier"
    hybrid_rf_symbols: Tuple[str, ...] = ()
    hybrid_mult_symbols: Tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Config":
        env = Environment(os.getenv("DERIV_ENV", "demo").lower())
        base_url = os.getenv("DERIV_WS_URL", DerivConfig.ws_base_url)
        app_id = os.getenv("DERIV_APP_ID", "34DyB1Yk9MY6apfuxbUQV").strip().strip('"').strip("'")
        api_token = os.getenv("DERIV_API_TOKEN", "").strip().strip('"').strip("'")
        deriv = DerivConfig(
            app_id=app_id,
            api_token=api_token,
            environment=env,
            live_confirmation=os.getenv("DERIV_LIVE_CONFIRMATION", "").strip(),
            ws_base_url=base_url,
        )
        risk = RiskConfig(
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", RiskConfig.max_risk_per_trade)),
            max_daily_drawdown=float(os.getenv("MAX_DAILY_DRAWDOWN", RiskConfig.max_daily_drawdown)),
            target_daily_profit=float(os.getenv("TARGET_DAILY_PROFIT", RiskConfig.target_daily_profit)),
            min_stake=float(os.getenv("MIN_STAKE", RiskConfig.min_stake)),
            max_stake=float(os.getenv("MAX_STAKE", RiskConfig.max_stake)),
            fixed_stake=float(os.getenv("FIXED_STAKE", "1.0")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", RiskConfig.max_open_positions)),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", RiskConfig.max_trades_per_day)),
            reward_risk=float(os.getenv("REWARD_RISK", RiskConfig.reward_risk)),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", RiskConfig.max_consecutive_losses)),
            dynamic_profit_enabled=os.getenv("DYNAMIC_PROFIT_ENABLED", "true").lower() in ("true", "1", "yes"),
            green_shield_trigger_usd=float(os.getenv("GREEN_SHIELD_TRIGGER_USD", RiskConfig.green_shield_trigger_usd)),
            min_profit_take_usd=float(os.getenv("MIN_PROFIT_TAKE_USD", RiskConfig.min_profit_take_usd)),
            breakeven_trigger_usd=float(os.getenv("BREAKEVEN_TRIGGER_USD", RiskConfig.breakeven_trigger_usd)),
            trail_profit_fraction=float(os.getenv("TRAIL_PROFIT_FRACTION", RiskConfig.trail_profit_fraction)),
            early_exit_on_reversal=os.getenv("EARLY_EXIT_ON_REVERSAL", "true").lower() in ("true", "1", "yes"),
            max_hold_bars_in_profit=int(os.getenv("MAX_HOLD_BARS_IN_PROFIT", RiskConfig.max_hold_bars_in_profit)),
            fast_exhaustion_cut=os.getenv("FAST_EXHAUSTION_CUT", "true").lower() in ("true", "1", "yes"),
            dynamic_loss_enabled=os.getenv("DYNAMIC_LOSS_ENABLED", "true").lower() in ("true", "1", "yes"),
            max_loss_cut_usd=float(os.getenv("MAX_LOSS_CUT_USD", RiskConfig.max_loss_cut_usd)),
            max_bars_in_loss=int(os.getenv("MAX_BARS_IN_LOSS", RiskConfig.max_bars_in_loss)),
            early_cut_on_invalidation=os.getenv("EARLY_CUT_ON_INVALIDATION", "true").lower() in ("true", "1", "yes"),
            cut_on_opposing_signal=os.getenv("CUT_ON_OPPOSING_SIGNAL", "true").lower() in ("true", "1", "yes"),
            tighten_loss_bars=int(os.getenv("TIGHTEN_LOSS_BARS", RiskConfig.tighten_loss_bars)),
            adverse_tick_cut=os.getenv("ADVERSE_TICK_CUT", "true").lower() in ("true", "1", "yes"),
            rf_edge_guard=os.getenv("RF_EDGE_GUARD_ENABLED", "true").lower() in ("true", "1", "yes"),
            rf_edge_min_trades=int(os.getenv("RF_EDGE_MIN_TRADES", RiskConfig.rf_edge_min_trades)),
            rf_edge_symbol_min_trades=int(os.getenv("RF_EDGE_SYMBOL_MIN_TRADES", RiskConfig.rf_edge_symbol_min_trades)),
            confidence_staking=os.getenv("CONFIDENCE_STAKING", "false").lower() in ("true", "1", "yes"),
        )
        mode_str = os.getenv("TRADE_MODE", "hybrid").lower()
        try:
            trade_mode = TradeMode(mode_str)
        except ValueError:
            trade_mode = TradeMode.HYBRID

        rf_duration = int(os.getenv("RISE_FALL_DURATION", "3"))
        rf_unit = os.getenv("RISE_FALL_DURATION_UNIT", "m")

        tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        tg_interval = int(os.getenv("TELEGRAM_REPORT_INTERVAL_MIN", "15"))
        telegram = TelegramConfig(
            enabled=bool(tg_token and tg_chat_id),
            bot_token=tg_token,
            chat_id=tg_chat_id,
            report_interval_min=tg_interval,
        )

        cfg = cls(deriv=deriv, risk=risk, learning=LearningConfig(), telegram=telegram,
                  trade_mode=trade_mode, rise_fall_duration=rf_duration,
                  rise_fall_duration_unit=rf_unit,
                  rf_allowed_assets=tuple(
                      a.strip().lower() for a in
                      os.getenv("RF_ALLOWED_ASSETS", "synthetic").split(",") if a.strip()
                  ),
                  gemini_veto_enabled=os.getenv("GEMINI_VETO_ENABLED", "false").lower()
                  in ("true", "1", "yes"),
                  hybrid_isolated_risk=os.getenv("HYBRID_ISOLATED_RISK", "true").lower()
                  in ("true", "1", "yes"),
                  hybrid_block_cross_symbol=os.getenv("HYBRID_BLOCK_CROSS_SYMBOL", "true").lower()
                  in ("true", "1", "yes"),
                  hybrid_signal_priority=os.getenv("HYBRID_SIGNAL_PRIORITY", "rise_fall").strip().lower(),
                  hybrid_rf_symbols=tuple(
                      s.strip() for s in os.getenv("HYBRID_RF_SYMBOLS", "").split(",") if s.strip()),
                  hybrid_mult_symbols=tuple(
                      s.strip() for s in os.getenv("HYBRID_MULT_SYMBOLS", "").split(",") if s.strip()))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.deriv.api_token:
            raise ValueError("DERIV_API_TOKEN is not set (see .env.example).")
        if self.deriv.environment is Environment.LIVE:
            if self.deriv.live_confirmation != LIVE_CONFIRMATION_PHRASE:
                raise ValueError(
                    "Live trading requires DERIV_LIVE_CONFIRMATION="
                    f"{LIVE_CONFIRMATION_PHRASE}. Run on demo first."
                )
        if not any(s.enabled for s in self.symbols):
            raise ValueError("No enabled symbols configured.")
        if self.hybrid_signal_priority not in ("multiplier", "rise_fall"):
            raise ValueError(
                "HYBRID_SIGNAL_PRIORITY must be 'multiplier' or 'rise_fall', "
                f"got {self.hybrid_signal_priority!r}.")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.learning.db_path).parent.mkdir(parents=True, exist_ok=True)
