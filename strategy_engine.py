"""
strategy_engine.py - Indicator computation and multi-indicator confluence signals.

The engine is PURE: it takes closed candles + tunable parameters and returns an
optional Signal. It knows nothing about money, positions or the broker; sizing
and veto power live exclusively in risk_manager.py.

Confluence model
  Five components each vote +1 / 0 / -1:
    trend  : close vs EMA200 and EMA fast vs slow       (disabled in 'range')
    cross  : fresh EMA fast/slow cross within 3 bars
    rsi    : momentum zone (trend) or oversold/overbought (range)
    bbands : %B position (trend: with-the-band, range: mean reversion)
    macd   : histogram sign and slope
  score = sum(weight_i * vote_i). A trade needs |score| >= params.min_score,
  plus a hard RSI gate so we never chase an exhausted move.
  Weights / thresholds live in StrategyParams, which the self-learning loop
  may adjust within hard bounds.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import pandas as pd
import pandas_ta as ta

from config import StrategyConfig

COMPONENTS = ("trend", "cross", "rsi", "bbands", "macd")

# Instruments excluded from Rise/Fall on measured evidence.
#
# stpRNG4: 46 replayed trades, 12 wins (26.1%), binomial p=0.0008 against a fair
# coin (family-wise 1.9% across the 23 symbols), and bad in both halves of the
# 14 days (28.6% then 22.2%). Unlike the rejected v11 blacklist (1HZ15V, 3 live
# losses), this survives a multiple-testing correction and has a real sample.
# It is a LOSS REDUCER, not an edge: excluding it moves the 14-day replay from
# -0.063 to about -0.034 per trade, which is still below breakeven.
EXCLUDED_RF_SYMBOLS = frozenset({"STPRNG4"})


def rf_symbol_excluded(symbol: str) -> bool:
    """True if this instrument is excluded from Rise/Fall (measured defect).

    Scoped to Rise/Fall only: the same symbol may still trade Multipliers.
    """
    return symbol.upper() in EXCLUDED_RF_SYMBOLS


# ----------------------------------------------------------------------------
# Tunable parameters (persisted & adjusted by memory_and_learning.py)
# ----------------------------------------------------------------------------
@dataclass
class StrategyParams:
    rsi_long_max: float = 68.0        # never go long above this RSI (avoid overbought tops)
    rsi_short_min: float = 32.0       # never go short below this RSI (avoid oversold bottoms)
    min_score: float = 2.0
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
    duration: int = 3                  # Dynamic expiry duration (1 to 10 minutes)
    duration_reason: str = "Standard 3m default"


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    """pandas-ta column names vary by version (BBL_20_2.0 vs BBL_20_2.0_2.0)."""
    for c in df.columns:
        if c.startswith(prefix):
            return df[c]
    raise KeyError(prefix)


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    out["ema_fast"] = ta.ema(c, length=cfg.ema_fast)
    out["ema_slow"] = ta.ema(c, length=cfg.ema_slow)
    out["ema_trend"] = ta.ema(c, length=cfg.ema_trend)
    out["ema_higher_tf"] = ta.ema(c, length=cfg.ema_higher_tf)
    out["rsi"] = ta.rsi(c, length=cfg.rsi_length)
    out["atr"] = ta.atr(h, l, c, length=cfg.atr_length)

    bb = ta.bbands(c, length=cfg.bb_length, std=cfg.bb_std)
    out["bb_lower"], out["bb_mid"] = _col(bb, "BBL"), _col(bb, "BBM")
    out["bb_upper"], out["bb_pctb"] = _col(bb, "BBU"), _col(bb, "BBP")

    adx = ta.adx(h, l, c, length=cfg.adx_length)
    out["adx"] = _col(adx, "ADX")

    macd = ta.macd(c)
    out["macd_hist"] = _col(macd, "MACDh")

    out["atr_pctile"] = out["atr"].rolling(100, min_periods=50).rank(pct=True)
    return out


def classify_regime(row: pd.Series, cfg: StrategyConfig) -> str:
    if row["atr_pctile"] >= cfg.high_vol_atr_percentile:
        return "high_vol"
    if row["adx"] >= cfg.adx_trend_threshold:
        return "trend"
    return "range"


def calculate_dynamic_duration(symbol: str, asset_class: str, last: pd.Series, regime: str) -> tuple[int, str]:
    """
    Computes an adaptive fixed-expiry horizon (1 to 10 minutes) based on:
    - Asset class speed and noise characteristics
    - ADX trend strength & velocity
    - ATR percentile (expansion vs compression)
    - Momentum acceleration (MACD histogram)
    """
    sym = symbol.upper()
    adx = float(last.get("adx", 25.0))
    atr_pct = float(last.get("atr_pctile", 0.50))
    macd_hist = float(last.get("macd_hist", 0.0))

    # 1. Base duration by asset class & symbol profile
    if asset_class == "forex":
        base_dur = 8
        profile = "Forex (low bar volatility, needs time to clear spread)"
    elif asset_class == "commodities":
        base_dur = 7
        profile = "Commodities (smooth structural trend)"
    elif "STPRNG" in sym:
        base_dur = 6
        profile = "Step Index (discrete step progression)"
    elif "JD" in sym:
        base_dur = 3
        profile = "Jump Index (fast mean-reversion, short horizon)"
    elif "1HZ" in sym:
        base_dur = 4
        profile = "1s High-Freq Index (retest buffer needed)"
    elif sym in ("R_75", "R_100"):
        base_dur = 6
        profile = "High-Volatility Index (extended wave development)"
    elif sym.startswith("R_"):
        base_dur = 4
        profile = "Continuous Volatility Index"
    elif asset_class == "crypto":
        base_dur = 3
        profile = "Crypto (impulsive momentum)"
    else:
        base_dur = 5
        profile = "General Asset Profile"

    dur = base_dur
    reasons = [profile]

    # 2. ADX Trend Velocity Adjustment
    if adx >= 40.0:
        dur -= 2
        reasons.append(f"ADX {adx:.1f}>=40 (-2m)")
    elif adx >= 32.0:
        dur -= 1
        reasons.append(f"ADX {adx:.1f}>=32 (-1m)")
    elif adx < 24.0:
        dur += 2
        reasons.append(f"ADX {adx:.1f}<24 (+2m)")
    elif adx < 28.0:
        dur += 1
        reasons.append(f"ADX {adx:.1f}<28 (+1m)")

    # 3. ATR Volatility Expansion vs Compression
    if atr_pct >= 0.80:
        dur -= 1
        reasons.append(f"ATR {atr_pct*100:.0f}%ile (-1m)")
    elif atr_pct <= 0.25:
        dur += 1
        reasons.append(f"ATR {atr_pct*100:.0f}%ile (+1m)")

    # 4. Momentum Acceleration
    atr = float(last.get("atr", 1.0))
    if atr > 0 and abs(macd_hist) > 1.5 * atr:
        dur -= 1
        reasons.append("MACD acceleration (-1m)")

    # 5. Broker Allowed Duration Minimums:
    # Deriv enforces a strict 15m minimum on Major Forex pairs, 5m on Commodities/Metals, and 1m on Synthetics
    if asset_class == "forex" and not any(sym.startswith(x) for x in ("frxXAU", "frxXAG")):
        final_dur = max(15, min(30, dur + 10))
        reasons.append("Broker 15m min for Forex")
    elif any(sym.startswith(x) for x in ("frxXAU", "frxXAG")):
        final_dur = max(5, min(10, dur))
        reasons.append("Broker 5m min for Metals")
    else:
        # Synthetics (Continuous, Jump, Step) allow 1m to 10m dynamic expiry
        final_dur = max(1, min(10, dur))

    summary_reason = f"{final_dur}m [{', '.join(reasons)}]"
    return final_dur, summary_reason


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def evaluate(self, symbol: str, asset_class: str, candles: pd.DataFrame,
                 params: StrategyParams, contract_type: str = "rise_fall") -> Optional[Signal]:
        if len(candles) < self.cfg.min_candles:
            return None
        df = compute_indicators(candles, self.cfg)
        last, prev = df.iloc[-1], df.iloc[-2]
        needed = ["ema_fast", "ema_slow", "ema_trend", "ema_higher_tf", "rsi", "atr", "bb_pctb",
                  "adx", "macd_hist", "atr_pctile"]
        if last[needed].isna().any() or last["atr"] <= 0:
            return None

        regime = classify_regime(last, self.cfg)
        if regime == "high_vol" and not self.cfg.trade_high_vol:
            return None
        if params.is_blocked(asset_class, regime):
            return None

        # Measured-defect instrument (see EXCLUDED_RF_SYMBOLS above).
        # This exclusion was measured on fixed-expiry Rise/Fall settlements, so
        # it must not silently disable the Multiplier strategy on the same
        # symbol. The caller passes the contract_type it is evaluating for.
        if contract_type == "rise_fall" and rf_symbol_excluded(symbol):
            return None

        # Confluence Score
        votes = self._votes(df, last, prev, regime, params)
        w = params.component_weights
        score = sum(w.get(k, 1.0) * v for k, v in votes.items())
        denom = sum(w.get(k, 1.0) for k in votes) or 1.0

        adx_val = float(last["adx"])
        # PATCH v10 (Forensic): Raise global ADX floor from 18 → 20.
        # Data: ADX<22 trades had only 33% win rate; loosened slightly to 20 to allow more
        # qualifying setups while still filtering the worst noise (ADX < 18 was original).
        # NOTE: a v11 attempt to raise this to 22 was reverted (2026-10-25) after an
        # A/B on identical candles: it improved the training half (+0.003) but made
        # the validation half worse (-0.011) and cut 47% of trades.
        if adx_val < 20.0:
            return None

        # Jump Indices (JD10..JD100): Require ADX >= 22.0 (slightly tighter than global floor)
        if "JD" in symbol.upper() and adx_val < 22.0:
            return None

        # 1-Second Synthetics (1HZ10V..1HZ100V): Require ADX >= 22.0 to filter intra-bar noise
        if "1HZ" in symbol.upper() and adx_val < 22.0:
            return None

        # Step Indices (stpRNG..stpRNG5) and R_75 remain UNBLOCKED for full market analysis

        # Relative MACD Momentum: Normalized by ATR to work universally across Forex, Commodities, and Synthetics
        atr_val = max(1e-9, float(last["atr"]))
        macd_rel = float(last["macd_hist"]) / atr_val

        direction = None
        # Calibrated Confluence Entry for High-Probability Rise/Fall Trading:
        # Long requires:
        # - Score >= min_score (concurring momentum components)
        # - EMA 9 > EMA 21 (fast moving average momentum up)
        # - Close >= EMA 9 & Close >= EMA 50 (above intermediate trend)
        # - STRICT Macro Alignment: Close >= EMA 200 (no buying into overhead macro resistance!)
        # - Relative MACD momentum: positive and macd_rel >= 0.010 (meaningful trend velocity)
        # - Bullish confirmation candle: close > open
        # - Gemini Momentum Exhaustion Guard: Reject extended entries near upper band without strong MACD expansion
        req_score = max(2.0, min(4.5, float(params.min_score)))

        # Dynamic BB and RSI bounds for LONGS
        if "1HZ" in symbol.upper():
            max_bb_long = 0.74 if adx_val >= 28.0 else 0.68
            max_rsi_long = 60.0 if adx_val >= 28.0 else 57.0
        elif adx_val >= 30.0:
            max_bb_long = 0.78  # Strong trend: allow slightly higher BB
            max_rsi_long = 63.0  # Strong trend: allow slightly extended RSI
        elif adx_val >= 25.0:
            max_bb_long = 0.74  # Moderate-strong: slightly looser than v10 (was 0.72)
            max_rsi_long = 62.0  # Moderate-strong: loosened from 60
        else:
            max_bb_long = 0.70  # Moderate (22-25)
            max_rsi_long = 60.0

        # Short BB floor — bottom-bounce trap protection
        if adx_val >= 30.0:
            min_bb_short = 0.18
            min_rsi_short = 28.0
        elif adx_val >= 25.0:
            min_bb_short = 0.32
            min_rsi_short = 33.0
        else:
            min_bb_short = 0.38
            min_rsi_short = 35.0

        # Learned/global caps
        max_rsi_long = min(max_rsi_long, float(params.rsi_long_max))
        min_rsi_short = max(min_rsi_short, float(params.rsi_short_min))

        if (score >= req_score
                and last["ema_fast"] > last["ema_slow"]
                and (last["ema_fast"] - last["ema_slow"]) >= 0.015 * atr_val
                and last["close"] >= last["ema_fast"]
                and (last["close"] - last["ema_fast"]) <= 0.80 * atr_val
                and last["close"] >= last["ema_trend"]
                and (last["close"] - last["ema_trend"]) <= 2.20 * atr_val
                # Macro trend alignment: ADX>=25 bypass retained (v10). Requiring
                # close>=EMA200 unconditionally was tested as v11 and made the
                # validation half worse (-0.133 vs -0.157 only on a single rule,
                # and -0.053 on the training half).
                and (adx_val >= 25.0 or last["close"] >= last["ema_higher_tf"])
                and last["rsi"] <= max_rsi_long
                and 33.0 <= last["rsi"]  # Loosened: 35 → 33 (mild momentum allowed)
                and 0.12 <= last["bb_pctb"] <= max_bb_long  # BB lower bound loosened: 0.15→0.12
                # PATCH: Hard block — if BB > 0.70 AND RSI > 60, it's an extended entry, reject it
                and not (last["bb_pctb"] > 0.70 and last["rsi"] > 60.0)
                and (last["macd_hist"] > 0 and macd_rel >= 0.003)
                and last["close"] > last["open"]):
            direction = "long"

        # Short requires:
        # - Score <= -min_score (concurring downward momentum components)
        # - EMA 9 < EMA 21 (fast moving average momentum down)
        # - Close <= EMA 9 & Close <= EMA 50 (below intermediate trend)
        # - Macro Trend Alignment: Close <= EMA 200 (unless ADX >= 25.0)
        # - Adaptive Bollinger Floor: %B down to 0.12 on strong trends
        # - Relative MACD momentum: negative and macd_rel <= -0.003
        # - Bearish confirmation candle: close < open
        elif (score <= -req_score
                and last["ema_fast"] < last["ema_slow"]
                and (last["ema_slow"] - last["ema_fast"]) >= 0.015 * atr_val
                and last["close"] <= last["ema_fast"]
                and (last["ema_fast"] - last["close"]) <= 0.80 * atr_val
                and last["close"] <= last["ema_trend"]
                and (last["ema_trend"] - last["close"]) <= 2.20 * atr_val
                # Macro trend alignment for shorts (ADX>=25 bypass retained, v10)
                and (adx_val >= 25.0 or last["close"] <= last["ema_higher_tf"])
                and min_rsi_short <= last["rsi"] <= 65.0  # Loosened RSI ceiling: was 63.0
                and min_bb_short <= last["bb_pctb"] <= 0.85  # Loosened BB ceiling: was 0.82
                # PATCH: Hard block — if BB < min_bb_short+0.05, it's a floor-bounce trap, reject
                and not (last["bb_pctb"] < (min_bb_short + 0.05) and last["rsi"] < 42.0)
                and (last["macd_hist"] < 0 and macd_rel <= -0.003)
                and last["close"] < last["open"]):
            direction = "short"

        if direction is None:
            return None

        snapshot = {k: float(last[k]) for k in
                    ("rsi", "atr", "adx", "bb_pctb", "macd_hist", "ema_fast", "ema_slow",
                     "ema_trend", "ema_higher_tf", "atr_pctile")}
        snapshot["close"] = float(last["close"])

        dyn_dur, dyn_reason = calculate_dynamic_duration(symbol, asset_class, last, regime)

        return Signal(
            symbol=symbol, asset_class=asset_class, direction=direction,
            score=float(score), confidence=min(1.0, abs(score) / denom), regime=regime,
            price=float(last["close"]), atr=float(last["atr"]), bar_time=df.index[-1],
            indicators=snapshot, components=votes,
            duration=dyn_dur, duration_reason=dyn_reason,
        )

    # -- component voting -------------------------------------------------- #
    def _votes(self, df: pd.DataFrame, last: pd.Series, prev: pd.Series,
               regime: str, p: StrategyParams) -> Dict[str, int]:
        v = {k: 0 for k in COMPONENTS}
        close, rsi, pctb = last["close"], last["rsi"], last["bb_pctb"]
        prev_rsi = prev["rsi"]

        # 1. Macro & Intermediate Trend Alignment
        if close > last["ema_higher_tf"] and last["ema_fast"] > last["ema_slow"]:
            v["trend"] = 1
        elif close < last["ema_higher_tf"] and last["ema_fast"] < last["ema_slow"]:
            v["trend"] = -1

        # 2. Moving Average Momentum (crossover or expanding spread)
        diff = (df["ema_fast"] - df["ema_slow"]).iloc[-5:]
        if diff.iloc[-1] > 0 and (diff.iloc[-1] >= diff.iloc[-2] or (diff.iloc[:-1].tail(3) <= 0).any()):
            v["cross"] = 1
        elif diff.iloc[-1] < 0 and (diff.iloc[-1] <= diff.iloc[-2] or (diff.iloc[:-1].tail(3) >= 0).any()):
            v["cross"] = -1

        # 3. RSI: Pullback into value, avoiding exhausted extremes
        if 38.0 <= rsi <= 64.0 and rsi >= prev_rsi:
            v["rsi"] = 1
        elif 36.0 <= rsi <= 62.0 and rsi <= prev_rsi:
            v["rsi"] = -1

        # 4. Bollinger %B: Bounce off value bands; reject outer band chase
        if 0.15 <= pctb <= 0.70 and last["close"] > last["open"]:
            v["bbands"] = 1
        elif 0.30 <= pctb <= 0.85 and last["close"] < last["open"]:
            v["bbands"] = -1

        # 5. MACD histogram momentum: Requires directional momentum
        h = last["macd_hist"]
        if h > 0:
            v["macd"] = 1
        elif h < 0:
            v["macd"] = -1
        return v
