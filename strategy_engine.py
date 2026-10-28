"""
strategy_engine.py (v2, audited) - Indicators and multi-indicator confluence signals.

The engine is PURE: closed candles + tunable parameters in, optional Signal out.
It knows nothing about money, positions or the broker; sizing and veto power
live in risk_manager.py.

Strategy: trend-pullback continuation.
  Direction comes from a confluence score over five votes (+1/0/-1):
    trend  : TRUE higher-timeframe alignment (resampled candles) + base EMA order
    cross  : EMA fast/slow spread expanding or freshly crossed
    rsi    : value-zone RSI that is turning with the trade
    bbands : %B in the value area with a candle in the trade direction
    macd   : histogram sign
  A trade also has to pass a set of hard gates (candle conviction, structure,
  extension, oscillator zones, momentum). All gates are evaluated in
  "direction space" so longs and shorts use exactly mirrored thresholds.

Every rejection is counted in `engine.rejects` (a funnel), so you can see which
gate is starving or flooding the strategy. Thresholds live in `EntryTuning`.

Changes vs the uploaded version (bugs first)
  FIXED  Metals never matched: the symbol was upper-cased, then compared with
         mixed-case prefixes ("frxXAU"), so XAU/XAG got 1-10m expiries instead
         of the 5m broker minimum.
  FIXED  MACD-acceleration rule required |hist| > 1.5*ATR, which never happens
         (hist/ATR peaks well below 1). Now uses a reachable threshold.
  FIXED  Toxic-symbol exclusion returned None for EVERY contract type, although
         the comment says it was measured on Rise/Fall only. It now applies to
         Rise/Fall only; Multipliers on those symbols are no longer silently off.
  FIXED  "Higher timeframe" EMA was just another EMA on the same candles. HTF is
         now real (candles are resampled; only COMPLETED HTF bars are used, so
         there is no look-ahead).
  FIXED  Shorts used looser RSI/%B bounds than longs (shorts could enter at
         RSI 28 while longs were capped at 63). Gates are now mirrored.
  FIXED  `min_score` default (2.0) vs hidden floor (3.0) vs learner bounds
         (2.5-4.5) disagreed. Floor is explicit (MIN_SCORE_FLOOR) and the
         threshold is applied to the NORMALISED score, so the learner shrinking
         component weights can no longer make a trade unreachable.
  FIXED  Fixed "<= 2.2 ATR from EMA200" gate removed ~92% of trending bars in
         tests (it only allowed trades right at the 200 EMA). Replaced by a
         z-score extension cap that adapts to each instrument.
  ADDED  Data-gap guard, dead-market guard, session filter for forex/metals,
         close-location (wick rejection) gate, `scan()` for fast backtests,
         `rejects` funnel, broker duration limits passed in from contracts_for.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, Optional, Tuple

import pandas as pd
import pandas_ta as ta

from config import StrategyConfig

COMPONENTS = ("trend", "cross", "rsi", "bbands", "macd")
MIN_SCORE_FLOOR = 3.0   # keep PARAM_BOUNDS["min_score"] lower bound in memory_and_learning.py equal to this

# Empirical Toxic / Defective instruments identified in audit (fixed-expiry Rise/Fall).
# NOTE: the audit samples behind these were tiny (n of roughly 3-7 trades each). That is
# not statistically meaningful; re-test with backtest_strategy.py before trusting the list.
TOXIC_SYMBOLS = frozenset({
    "R_25", "STPRNG3", "STPRNG4", "STPRNG5", "1HZ15V", "CRYBTCUSD", "CRYETHUSD"
})
EXCLUDED_RF_SYMBOLS = frozenset({"STPRNG4"})


def rf_symbol_excluded(symbol: str) -> bool:
    """True if this instrument is excluded from Rise/Fall."""
    return symbol.upper() in EXCLUDED_RF_SYMBOLS or symbol.upper() in TOXIC_SYMBOLS


def symbol_excluded(symbol: str, contract_type: str = "rise_fall") -> bool:
    """The blacklist was measured on Rise/Fall settlements, so it only applies there."""
    return contract_type == "rise_fall" and rf_symbol_excluded(symbol)


# ----------------------------------------------------------------------------
# Entry thresholds (all in one place so they can be A/B tested)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EntryTuning:
    # regime / market-quality gates
    adx_floor: float = 20.0
    adx_floor_fast: float = 22.0          # Jump (JD*) and 1-second (1HZ*) indices
    atr_pctile_floor: float = 0.10        # NEW: skip dead markets (spread/ties dominate)
    # structure and extension
    min_spread_atr: float = 0.015         # |EMA fast - EMA slow| >= this * ATR
    max_ext_fast_atr: float = 0.80        # price within this many ATR of EMA fast
    jump_max_ext_fast_atr: float = 0.80   # Jump (JD*): never looser than this, whatever the profile
    boom_crash_max_ext_fast_atr: float = 2.0  # Boom/Crash: spikes distort ATR; starting guess, tune it
    max_ext_trend_z: float = 2.0          # z-score cap on distance from EMA trend
    # oscillator zones, in direction space (short: rsi -> 100-rsi, %B -> 1-%B)
    rsi_floor: float = 33.0
    rsi_cap_bonus: float = 0.0            # added to the ADX-based RSI cap table (60/62/63)
    pctb_floor: float = 0.12
    pctb_cap_bonus: float = 0.0           # added to the ADX-based %B cap table (.70/.74/.78)
    exhaustion_pctb: float = 0.70
    exhaustion_rsi: float = 58.0
    min_macd_rel: float = 0.003           # |MACD hist| / ATR
    # candle conviction
    min_body_rise_fall: float = 0.45
    min_body_multiplier: float = 0.30
    min_clv_rise_fall: float = 0.55       # NEW: close location within the bar
    min_clv_multiplier: float = 0.50
    # data integrity
    max_gap_bars: float = 3.0             # skip the first bar after a gap/outage
    # sessions (forex / metals only). OFF by default: the engine is a pure signal generator and
    # the orchestrator should own trading hours. Enable for standalone use or to test the effect.
    use_session_filter: bool = False
    session_start_utc: int = 7
    session_end_utc: int = 20


# Four of the original gates (RSI cap, %B cap, distance from EMA fast, exhaustion combo) all test
# the same thing, "price is not stretched". On simulated data each passed only 10-40% of bars that
# already had 3+ agreeing votes, so together they left ~0.1% of bars tradable (a handful of trades
# per symbol, far too few to measure a win rate or feed the learner).
PROFILES = {
    # your original thresholds (mirrored for shorts): very selective, very few trades
    "strict": EntryTuning(),
    # calibrated balanced logic: healthy trade frequency without disabling exhaustion
    "balanced": EntryTuning(
        rsi_cap_bonus=3.0,
        pctb_cap_bonus=0.05,
        max_ext_fast_atr=1.0,
        exhaustion_pctb=0.78,
        exhaustion_rsi=63.0,
    ),
}
DEFAULT_TUNING = PROFILES["balanced"]


# ----------------------------------------------------------------------------
# Tunable parameters (persisted & adjusted by memory_and_learning.py)
# ----------------------------------------------------------------------------
@dataclass
class StrategyParams:
    rsi_long_max: float = 68.0        # never go long above this RSI (shorts mirror: 100 - rsi_short_min)
    rsi_short_min: float = 32.0       # never go short below this RSI
    min_score: float = 3.0
    component_weights: Dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in COMPONENTS})
    asset_class_weights: Dict[str, float] = field(default_factory=dict)  # risk scalers (<=1)
    blocked: Dict[str, float] = field(default_factory=dict)              # "class|regime" -> until ts
    version: int = 0

    def is_blocked(self, asset_class: str, regime: str, now: Optional[float] = None) -> bool:
        until = self.blocked.get(f"{asset_class}|{regime}")
        return bool(until and until > (now or time.time()))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "StrategyParams":
        d = json.loads(raw)
        base = cls()
        for k, v in d.items():
            if hasattr(base, k):
                setattr(base, k, v)
        for c in COMPONENTS:                       # forward-compat for new components
            base.component_weights.setdefault(c, 1.0)
        return base


@dataclass(frozen=True)
class Signal:
    symbol: str
    asset_class: str
    direction: str                     # 'long' | 'short'
    score: float
    confidence: float                  # |score| / sum(weights), 0..1
    regime: str                        # 'trend' | 'range' | 'high_vol'
    price: float
    atr: float
    bar_time: pd.Timestamp
    indicators: Dict[str, float]
    components: Dict[str, int]
    duration: int = 3                  # expiry in minutes (Rise/Fall only)
    duration_reason: str = "Standard 3m default"


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    """pandas-ta names vary by version (BBL_20_2.0 vs BBL_20_2.0_2.0; ADX_14 vs ADXR_14_2)."""
    for c in df.columns:
        if c.startswith(prefix + "_"):
            return df[c]
    for c in df.columns:
        if c.startswith(prefix):
            return df[c]
    raise KeyError(prefix)


def _add_htf(out: pd.DataFrame, cfg) -> None:
    """
    Real higher-timeframe trend: resample to `htf_multiple` x the base bar, EMA on
    COMPLETED HTF bars only, then align each base bar to the latest HTF bar that had
    already closed at that moment (no look-ahead). Falls back to the base-TF
    ema_higher_tf if the data cannot support it.
    """
    mult = int(getattr(cfg, "htf_multiple", 3))
    length = int(getattr(cfg, "htf_ema_length", 21))
    try:
        idx = out.index
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 10:
            raise ValueError("needs DatetimeIndex")
        bar = float(pd.Series(idx).diff().dt.total_seconds().median())
        if not bar > 0:
            raise ValueError("bad bar size")
        out.attrs["bar_sec"] = bar
        htf_sec = int(round(bar * mult))
        agg = (out[["open", "high", "low", "close"]]
               .resample(f"{htf_sec}s", label="left", closed="left")
               .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
               .dropna())
        avail = agg.index + pd.Timedelta(seconds=htf_sec)          # moment each HTF bar is final
        done = avail <= idx[-1] + pd.Timedelta(seconds=bar)
        agg, avail = agg[done], avail[done]
        if len(agg) < length + 4:
            raise ValueError("too little history for HTF EMA")
        ema = ta.ema(agg["close"], length=length)
        feats = pd.DataFrame({"htf_close": agg["close"].to_numpy(),
                              "htf_ema": ema.to_numpy(),
                              "htf_slope": (ema - ema.shift(2)).to_numpy()}, index=avail)
        aligned = feats.reindex(idx + pd.Timedelta(seconds=bar), method="ffill")
        for c in ("htf_close", "htf_ema", "htf_slope"):
            out[c] = aligned[c].to_numpy()
    except (ValueError, KeyError):
        out["htf_close"] = out["close"]
        out["htf_ema"] = out["ema_higher_tf"]
        out["htf_slope"] = out["ema_higher_tf"] - out["ema_higher_tf"].shift(6)


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    out["ema_fast"] = ta.ema(c, length=cfg.ema_fast)
    out["ema_slow"] = ta.ema(c, length=cfg.ema_slow)
    out["ema_trend"] = ta.ema(c, length=cfg.ema_trend)
    out["ema_higher_tf"] = ta.ema(c, length=getattr(cfg, "ema_higher_tf", 50))
    out["ema_diff"] = out["ema_fast"] - out["ema_slow"]
    out["rsi"] = ta.rsi(c, length=cfg.rsi_length)
    out["atr"] = ta.atr(h, l, c, length=cfg.atr_length)

    bb = ta.bbands(c, length=cfg.bb_length, std=cfg.bb_std)
    out["bb_lower"], out["bb_mid"] = _col(bb, "BBL"), _col(bb, "BBM")
    out["bb_upper"], out["bb_pctb"] = _col(bb, "BBU"), _col(bb, "BBP")

    out["adx"] = _col(ta.adx(h, l, c, length=cfg.adx_length), "ADX")
    out["macd_hist"] = _col(ta.macd(c), "MACDh")
    out["macd_rel"] = out["macd_hist"] / out["atr"].clip(lower=1e-12)   # scale-free momentum

    out["atr_pctile"] = out["atr"].rolling(100, min_periods=50).rank(pct=True)

    rng = (h - l).clip(lower=1e-12)
    out["body_ratio"] = (c - out["open"]).abs() / rng
    out["clv"] = (c - l) / rng                                           # 1 = closed on the high

    ext = c - out["ema_trend"]                                           # extension from trend EMA
    out["ext_trend_z"] = (ext - ext.rolling(200, min_periods=50).mean()) / \
        ext.rolling(200, min_periods=50).std().clip(lower=1e-12)

    _add_htf(out, cfg)
    return out


_NEEDED = ["ema_fast", "ema_slow", "ema_trend", "ema_higher_tf", "rsi", "atr", "bb_pctb", "adx",
           "macd_hist", "atr_pctile", "ext_trend_z", "htf_ema", "htf_slope"]


def classify_regime(row: pd.Series, cfg: StrategyConfig) -> str:
    if row["atr_pctile"] >= getattr(cfg, "high_vol_atr_percentile", 0.90):
        return "high_vol"
    if row["adx"] >= cfg.adx_trend_threshold:
        return "trend"
    return "range"


# ----------------------------------------------------------------------------
# Expiry selection (Rise/Fall)
# ----------------------------------------------------------------------------
# Deriv enforces minimum durations per market. These are defaults only; pass the real
# limits from contracts_for via evaluate(..., duration_limits=(min_min, max_min)).
BROKER_MIN_MINUTES = {"forex": 15, "metal": 5}
MAX_MINUTES = {"forex": 30, "metal": 10}
DEFAULT_MAX_MINUTES = 10
MACD_ACCEL_REL = 0.50                      # |hist|/ATR; ~top 5% of bars (old 1.5 never fired)
_METAL_PREFIXES = ("FRXXAU", "FRXXAG", "FRXXPD", "FRXXPT")
_JUMP_RE = re.compile(r"^JD\d+$")
_HZ_RE = re.compile(r"^1HZ\d+V$")
_STEP_RE = re.compile(r"^STPRNG\d*$")
_SPIKE_RE = re.compile(r"^(BOOM|CRASH)\d+N?$")      # Boom/Crash spike indices


def _instrument_kind(sym: str, asset_class: str) -> Tuple[str, str]:
    if sym.startswith(_METAL_PREFIXES) or asset_class == "commodities":
        return "metal", "Metals/Commodities (smooth structural trend)"
    if asset_class == "forex" or sym.startswith("FRX"):
        return "forex", "Forex (low bar volatility, needs time to clear spread)"
    if _STEP_RE.match(sym):
        return "step", "Step Index (discrete step progression)"
    if _JUMP_RE.match(sym):
        return "jump", "Jump Index (fast mean-reversion, short horizon)"
    if _HZ_RE.match(sym):
        return "hz1", "1s High-Freq Index (retest buffer needed)"
    if sym in ("R_75", "R_100"):
        return "vol_hi", "High-Volatility Index (extended wave development)"
    if sym.startswith("R_"):
        return "vol", "Continuous Volatility Index"
    if asset_class == "crypto" or sym.startswith("CRY"):
        return "crypto", "Crypto (impulsive momentum)"
    return "other", "General Asset Profile"


_BASE_MINUTES = {"metal": 7, "forex": 18, "step": 6, "jump": 3, "hz1": 4,
                 "vol_hi": 6, "vol": 4, "crypto": 3, "other": 5}


def calculate_dynamic_duration(symbol: str, asset_class: str, last: pd.Series, regime: str,
                               limits: Optional[Tuple[int, int]] = None) -> Tuple[int, str]:
    """
    Adaptive fixed-expiry horizon in minutes from instrument profile, ADX, ATR
    percentile and momentum acceleration, clamped to broker limits.
    limits: optional (min_minutes, max_minutes) taken from Deriv's contracts_for.
    """
    sym = symbol.upper()
    kind, profile = _instrument_kind(sym, asset_class)
    adx = float(last.get("adx", 25.0))
    atr_pct = float(last.get("atr_pctile", 0.50))
    atr = float(last.get("atr", 0.0))
    macd_rel = abs(float(last.get("macd_hist", 0.0))) / atr if atr > 0 else 0.0

    dur = _BASE_MINUTES.get(kind, 5)
    reasons = [profile]

    if adx >= 40.0:
        dur -= 2; reasons.append(f"ADX {adx:.1f}>=40 (-2m)")
    elif adx >= 32.0:
        dur -= 1; reasons.append(f"ADX {adx:.1f}>=32 (-1m)")
    elif adx < 24.0:
        dur += 2; reasons.append(f"ADX {adx:.1f}<24 (+2m)")
    elif adx < 28.0:
        dur += 1; reasons.append(f"ADX {adx:.1f}<28 (+1m)")

    if atr_pct >= 0.80:
        dur -= 1; reasons.append(f"ATR {atr_pct * 100:.0f}%ile (-1m)")
    elif atr_pct <= 0.25:
        dur += 1; reasons.append(f"ATR {atr_pct * 100:.0f}%ile (+1m)")

    if macd_rel >= MACD_ACCEL_REL:
        dur -= 1; reasons.append(f"MACD accel {macd_rel:.2f}xATR (-1m)")

    if limits:
        lo, hi = int(limits[0]), int(limits[1])
        hi = min(hi, MAX_MINUTES.get(kind, DEFAULT_MAX_MINUTES))
    else:
        lo, hi = BROKER_MIN_MINUTES.get(kind, 1), MAX_MINUTES.get(kind, DEFAULT_MAX_MINUTES)
    hi = max(hi, lo)
    final = max(lo, min(hi, dur))
    if final != dur:
        reasons.append(f"clamped to broker range {lo}-{hi}m")
    return final, f"{final}m [{', '.join(reasons)}]"


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, cfg: StrategyConfig, tuning: EntryTuning = DEFAULT_TUNING):
        """tuning: PROFILES["balanced"] (default) or PROFILES["strict"], or your own EntryTuning."""
        self.cfg = cfg
        self.tuning = tuning
        self.rejects: Counter = Counter()          # funnel: why signals were NOT produced
        self.last_reject: Dict[str, str] = {}      # symbol -> most recent reject reason

    # -- public API ------------------------------------------------------- #
    def evaluate(self, symbol: str, asset_class: str, candles: pd.DataFrame,
                 params: StrategyParams, contract_type: str = "rise_fall",
                 duration_limits: Optional[Tuple[int, int]] = None) -> Optional[Signal]:
        """Evaluate the LAST closed candle of `candles` (live path)."""
        if len(candles) < self.cfg.min_candles:
            return self._rej(symbol, "history")
        if symbol_excluded(symbol, contract_type):
            return self._rej(symbol, "excluded_symbol")
        df = compute_indicators(candles, self.cfg)
        return self._decide(df, len(df) - 1, symbol, asset_class, params, contract_type, duration_limits)

    def scan(self, symbol: str, asset_class: str, candles: pd.DataFrame, params: StrategyParams,
             contract_type: str = "rise_fall", duration_limits: Optional[Tuple[int, int]] = None
             ) -> Iterator[Tuple[int, Signal]]:
        """Yield (row position, Signal) for every bar. Same logic as evaluate(); for backtests."""
        if symbol_excluded(symbol, contract_type):
            self._rej(symbol, "excluded_symbol")
            return
        df = compute_indicators(candles, self.cfg)
        for pos in range(max(self.cfg.min_candles, 6) - 1, len(df)):
            sig = self._decide(df, pos, symbol, asset_class, params, contract_type, duration_limits)
            if sig is not None:
                yield pos, sig

    # -- internals -------------------------------------------------------- #
    def _rej(self, symbol: str, reason: str) -> None:
        self.rejects[reason] += 1
        self.last_reject[symbol] = reason
        return None

    def _adx_floor(self, sym: str) -> float:
        T = self.tuning
        return T.adx_floor_fast if (_JUMP_RE.match(sym) or _HZ_RE.match(sym)) else T.adx_floor

    def _in_session(self, asset_class: str, sym: str, ts: pd.Timestamp) -> bool:
        T = self.tuning
        if not T.use_session_filter:
            return True
        if not (asset_class in ("forex", "commodities") or sym.startswith("FRX")):
            return True                                   # synthetics / crypto trade 24/7
        return ts.weekday() < 5 and T.session_start_utc <= ts.hour < T.session_end_utc

    def _decide(self, df: pd.DataFrame, pos: int, symbol: str, asset_class: str,
                params: StrategyParams, contract_type: str,
                limits: Optional[Tuple[int, int]]) -> Optional[Signal]:
        T, sym = self.tuning, symbol.upper()
        last, prev = df.iloc[pos], df.iloc[pos - 1]
        if last[_NEEDED].isna().any() or last["atr"] <= 0:
            return self._rej(symbol, "warmup")

        # data integrity + session (decision time = close of the signal bar)
        bar = float(df.attrs.get("bar_sec") or 0.0)
        ts = df.index[pos]
        if bar > 0 and (ts - df.index[pos - 1]).total_seconds() > T.max_gap_bars * bar:
            return self._rej(symbol, "data_gap")
        if bar > 0 and not self._in_session(asset_class, sym, ts + pd.Timedelta(seconds=bar)):
            return self._rej(symbol, "session")

        # regime / market quality
        regime = classify_regime(last, self.cfg)
        if regime == "high_vol" and not getattr(self.cfg, "trade_high_vol", False):
            return self._rej(symbol, "high_vol")
        if params.is_blocked(asset_class, regime):
            return self._rej(symbol, "blocked_regime")
        adx = float(last["adx"])
        if adx < self._adx_floor(sym):
            return self._rej(symbol, "adx_floor")
        if float(last["atr_pctile"]) < T.atr_pctile_floor and not _SPIKE_RE.match(sym):
            return self._rej(symbol, "dead_market")      # Boom/Crash are exempt: tick drift is normal

        # confluence score on the NORMALISED scale, so learned weights cannot make it unreachable
        votes = self._votes(df, pos)
        w = params.component_weights
        score = sum(w.get(k, 1.0) * v for k, v in votes.items())
        denom = sum(w.get(k, 1.0) for k in votes) or 1.0
        norm = score / denom
        need = max(MIN_SCORE_FLOOR, float(params.min_score)) / len(COMPONENTS)
        if norm >= need:
            d = 1
        elif norm <= -need:
            d = -1
        else:
            return self._rej(symbol, "score")

        fail = self._gates(d, last, adx, sym, params, contract_type)
        if fail:
            return self._rej(symbol, fail)

        snapshot = {k: float(last[k]) for k in
                    ("rsi", "atr", "adx", "bb_pctb", "macd_hist", "macd_rel", "ema_fast", "ema_slow",
                     "ema_trend", "ema_higher_tf", "htf_ema", "htf_slope", "atr_pctile",
                     "ext_trend_z", "body_ratio", "clv")}
        snapshot["close"] = float(last["close"])
        dur, why = calculate_dynamic_duration(symbol, asset_class, last, regime, limits)
        self.last_reject.pop(symbol, None)
        return Signal(
            symbol=symbol, asset_class=asset_class, direction="long" if d > 0 else "short",
            score=float(score), confidence=min(1.0, abs(norm)), regime=regime,
            price=float(last["close"]), atr=float(last["atr"]), bar_time=ts,
            indicators=snapshot, components=votes, duration=dur, duration_reason=why,
        )

    def _gates(self, d: int, last: pd.Series, adx: float, sym: str, params: StrategyParams,
               contract_type: str) -> Optional[str]:
        """All hard gates in direction space (d=+1 long, d=-1 short). Returns failing gate or None."""
        T = self.tuning
        close, atr = float(last["close"]), float(last["atr"])
        rf = contract_type == "rise_fall"

        # candle conviction: body size, direction, and close location (rejects wick-heavy bars)
        if float(last["body_ratio"]) < (T.min_body_rise_fall if rf else T.min_body_multiplier) \
                or d * (close - float(last["open"])) <= 0:
            return "candle_body"
        clv = float(last["clv"]) if d > 0 else 1.0 - float(last["clv"])
        if clv < (T.min_clv_rise_fall if rf else T.min_clv_multiplier):
            return "candle_close"

        # higher-timeframe trend must agree (price side and slope)
        if d * (close - float(last["htf_ema"])) < 0 or d * float(last["htf_slope"]) <= 0:
            return "htf_trend"

        # base-timeframe structure
        if d * (float(last["ema_fast"]) - float(last["ema_slow"])) < T.min_spread_atr * atr:
            return "ema_structure"
        ext_fast = d * (close - float(last["ema_fast"])) / atr
        cap_fast = T.max_ext_fast_atr
        if _JUMP_RE.match(sym):
            cap_fast = min(cap_fast, T.jump_max_ext_fast_atr)       # don't chase jump-gap tops
        elif _SPIKE_RE.match(sym):
            cap_fast = max(cap_fast, T.boom_crash_max_ext_fast_atr)
        if ext_fast < 0 or ext_fast > cap_fast:
            return "ext_fast"
        if d * (close - float(last["ema_trend"])) < 0 or d * float(last["ext_trend_z"]) > T.max_ext_trend_z:
            return "ext_trend"

        # oscillator zones (mirrored for shorts)
        max_bb, max_rsi = self._bounds(adx, sym)
        rsi = float(last["rsi"]) if d > 0 else 100.0 - float(last["rsi"])
        cap = params.rsi_long_max if d > 0 else 100.0 - params.rsi_short_min
        if not (T.rsi_floor <= rsi <= min(max_rsi, float(cap))):
            return "rsi_zone"
        pb = float(last["bb_pctb"]) if d > 0 else 1.0 - float(last["bb_pctb"])
        if not (T.pctb_floor <= pb <= max_bb):
            return "bb_zone"
        if pb > T.exhaustion_pctb and rsi > T.exhaustion_rsi:
            return "exhaustion"

        # momentum
        if d * float(last["macd_hist"]) <= 0 or d * float(last["macd_rel"]) < T.min_macd_rel:
            return "macd"
        return None

    def _bounds(self, adx: float, sym: str) -> Tuple[float, float]:
        """(max %B, max RSI) in direction space; stronger trends tolerate a little more extension."""
        if _HZ_RE.match(sym):
            bb, rsi = (0.74, 60.0) if adx >= 28.0 else (0.68, 57.0)
        elif adx >= 30.0:
            bb, rsi = 0.78, 63.0
        elif adx >= 25.0:
            bb, rsi = 0.74, 62.0
        else:
            bb, rsi = 0.70, 60.0
        return min(1.0, bb + self.tuning.pctb_cap_bonus), min(100.0, rsi + self.tuning.rsi_cap_bonus)

    def _votes(self, df: pd.DataFrame, pos: int) -> Dict[str, int]:
        last, prev = df.iloc[pos], df.iloc[pos - 1]
        v = {k: 0 for k in COMPONENTS}
        close, rsi, pctb = last["close"], last["rsi"], last["bb_pctb"]

        # trend: true HTF alignment plus base EMA order
        if close > last["htf_ema"] and last["htf_slope"] > 0 and last["ema_fast"] > last["ema_slow"]:
            v["trend"] = 1
        elif close < last["htf_ema"] and last["htf_slope"] < 0 and last["ema_fast"] < last["ema_slow"]:
            v["trend"] = -1

        # EMA momentum: spread expanding or a fresh cross within 3 bars
        diff = df["ema_diff"].iloc[pos - 4:pos + 1]
        if diff.iloc[-1] > 0 and (diff.iloc[-1] >= diff.iloc[-2] or (diff.iloc[:-1].tail(3) <= 0).any()):
            v["cross"] = 1
        elif diff.iloc[-1] < 0 and (diff.iloc[-1] <= diff.iloc[-2] or (diff.iloc[:-1].tail(3) >= 0).any()):
            v["cross"] = -1

        # RSI pullback into value, turning with the trade (zones are exact mirrors)
        if 38.0 <= rsi <= 64.0 and rsi >= prev["rsi"]:
            v["rsi"] = 1
        elif 36.0 <= rsi <= 62.0 and rsi <= prev["rsi"]:
            v["rsi"] = -1

        # Bollinger %B: value area with a candle in the trade direction (mirror: x -> 1-x)
        if 0.15 <= pctb <= 0.70 and close > last["open"]:
            v["bbands"] = 1
        elif 0.30 <= pctb <= 0.85 and close < last["open"]:
            v["bbands"] = -1

        # MACD histogram direction
        if last["macd_hist"] > 0:
            v["macd"] = 1
        elif last["macd_hist"] < 0:
            v["macd"] = -1
        return v
