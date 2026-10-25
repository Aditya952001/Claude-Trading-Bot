"""
research/rf_backtest.py - Offline replay of the Rise/Fall engine on real 1m candles.

Pipeline
  candles -> indicators -> base signals (same gates as strategy_engine.py)
          -> optional filter layer -> trade simulation at fixed expiry -> metrics

Entries are taken at the OPEN of the bar AFTER the signal bar (the bot acts on the
tick right after a closed candle), and exit at the OPEN `duration` minutes later.
That is deliberately a little more conservative than using the signal bar close.

Payouts are taken from real proposal quotes (research/payouts.json) when present,
otherwise from the per-symbol defaults observed in the trade journal.

No orders are placed. Read-only.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import StrategyConfig
from strategy_engine import (EXCLUDED_RF_SYMBOLS, calculate_dynamic_duration,
                             classify_regime, compute_indicators)

CANDLE_DIR = ROOT / "data" / "candles"
PAYOUT_FILE = ROOT / "research" / "payouts.json"

CFG = StrategyConfig()

# Fallback payouts observed in the journal (profit per $1 stake).
DEFAULT_PAYOUT = {
    "R_10": 0.88, "R_25": 0.88, "R_50": 0.88, "R_75": 0.88, "R_100": 0.88,
    "1HZ10V": 0.88, "1HZ15V": 0.88, "1HZ25V": 0.88, "1HZ30V": 0.88,
    "1HZ50V": 0.88, "1HZ75V": 0.88, "1HZ90V": 0.88, "1HZ100V": 0.88,
    "JD10": 0.92, "JD25": 0.92, "JD50": 0.91, "JD75": 0.91, "JD100": 0.90,
    "stpRNG": 0.95, "stpRNG2": 0.95, "stpRNG3": 0.95, "stpRNG4": 0.95, "stpRNG5": 0.95,
    "frxEURUSD": 0.80, "frxGBPUSD": 0.79, "frxUSDJPY": 0.82, "frxAUDUSD": 0.82,
    "frxUSDCAD": 0.82, "frxUSDCHF": 0.80, "frxEURJPY": 0.79, "frxGBPJPY": 0.79,
    "frxXAUUSD": 0.75, "frxXAGUSD": 0.72,
}

# Price cost charged against the trade (spread/slippage). Synthetics quote a single
# price so 0; forex gets ~1 pip round-trip.
SPREAD = {
    "frxEURUSD": 0.00010, "frxGBPUSD": 0.00012, "frxUSDJPY": 0.012,
    "frxAUDUSD": 0.00012, "frxUSDCAD": 0.00014, "frxUSDCHF": 0.00014,
    "frxEURJPY": 0.014, "frxGBPJPY": 0.018, "frxXAUUSD": 0.30, "frxXAGUSD": 0.03,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def available_symbols() -> List[str]:
    return sorted(p.name[:-4].rsplit("_", 1)[0] for p in CANDLE_DIR.glob("*_60.csv"))


def load(symbol: str) -> pd.DataFrame:
    p = CANDLE_DIR / f"{symbol}_60.csv"
    df = pd.read_csv(p, parse_dates=["time"], index_col="time")
    return df[["open", "high", "low", "close"]].astype(float)


def payout_table() -> Dict[str, Dict[int, float]]:
    table: Dict[str, Dict[int, float]] = {}
    if PAYOUT_FILE.exists():
        raw = json.loads(PAYOUT_FILE.read_text())
        for sym, v in raw.items():
            if isinstance(v, dict) and v:
                table[sym] = {int(k): float(x) - 1.0 for k, x in v.items()}
    return table


def pay_for(table: Dict[str, Dict[int, float]], sym: str, duration: int) -> float:
    d = table.get(sym)
    if d:
        if duration in d:
            return d[duration]
        return d[min(d, key=lambda k: abs(k - duration))]
    return DEFAULT_PAYOUT.get(sym, 0.88)


def payouts() -> Dict[str, float]:
    """Best available profit per $1 for each symbol (used for diagnostics)."""
    table = payout_table()
    return {**DEFAULT_PAYOUT, **{s: max(v.values()) for s, v in table.items()}}


# ---------------------------------------------------------------------------
# Signal generation (mirrors strategy_engine.evaluate, vectorised)
# ---------------------------------------------------------------------------
SIGNAL_COLS = [
    "bar", "direction", "duration", "adx", "rsi", "bb_pctb", "atr_pctile",
    "macd_rel", "ema_gap", "close_vs_ema", "close_vs_trend", "close_vs_higher",
    "score", "atr", "price", "hour", "regime", "candle_body_atr", "rsi_slope",
]


def build_signals(symbol: str, df: pd.DataFrame,
                  cfg: StrategyConfig = CFG,
                  variant: str = "v10",
                  exclude_toxic: bool = True) -> pd.DataFrame:
    """Vectorised mirror of StrategyEngine.evaluate.

    variant="v10" -> the rules the 862-trade baseline was measured on.
    variant="v11" -> the rules introduced on 2026-10-24 22:51 (ADX 22 floor,
                     1HZ15V blacklist, mandatory close>=EMA200 on longs,
                     macd_rel 0.008, bb>0.72&rsi>59 block, exhaustion guard,
                     short floor-trap block). Kept as a flag so both can be
                     measured on identical candles.
    """
    if len(df) < cfg.min_candles:
        return pd.DataFrame(columns=SIGNAL_COLS)
    d = compute_indicators(df, cfg)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    ef, es, et, eh = (d["ema_fast"].values, d["ema_slow"].values,
                      d["ema_trend"].values, d["ema_higher_tf"].values)
    rsi, atr, pctb, adx = d["rsi"].values, d["atr"].values, d["bb_pctb"].values, d["adx"].values
    hist, pctile = d["macd_hist"].values, d["atr_pctile"].values
    n = len(d)
    if n < cfg.min_candles:
        return pd.DataFrame(columns=SIGNAL_COLS)

    diff = ef - es
    prev_diff = np.roll(diff, 1)
    cross_up = (diff > 0) & ((diff >= prev_diff) | (np.roll(diff, 2) <= 0) |
                             (np.roll(diff, 3) <= 0) | (np.roll(diff, 4) <= 0))
    cross_dn = (diff < 0) & ((diff <= prev_diff) | (np.roll(diff, 2) >= 0) |
                             (np.roll(diff, 3) >= 0) | (np.roll(diff, 4) >= 0))
    prev_rsi = np.roll(rsi, 1)

    votes = {}
    votes["trend"] = np.where((c > eh) & (ef > es), 1,
                              np.where((c < eh) & (ef < es), -1, 0))
    votes["cross"] = np.where(cross_up, 1, np.where(cross_dn, -1, 0))
    votes["rsi"] = np.where((rsi >= 38) & (rsi <= 64) & (rsi >= prev_rsi), 1,
                            np.where((rsi >= 36) & (rsi <= 62) & (rsi <= prev_rsi), -1, 0))
    votes["bbands"] = np.where((pctb >= 0.15) & (pctb <= 0.70) & (c > o), 1,
                               np.where((pctb >= 0.30) & (pctb <= 0.85) & (c < o), -1, 0))
    votes["macd"] = np.sign(hist)
    score = sum(votes.values())

    with np.errstate(invalid="ignore", divide="ignore"):
        macd_rel = hist / atr
        ema_gap = (ef - es) / atr
        close_vs_ema = (c - ef) / atr
        close_vs_trend = (c - et) / atr
        close_vs_higher = (c - eh) / atr
        body = (c - o) / atr

    # --- direction gates, exactly as in strategy_engine.evaluate ---
    req = 2.0
    sym = symbol.upper()

    adx_ge30 = adx >= 30
    adx_ge25 = adx >= 25

    max_rsi_long = np.where(adx_ge25, 62.0, 60.0)
    max_bb_long = np.where(adx_ge25, 0.74, 0.70)
    max_rsi_long = np.where(adx_ge30, 63.0, max_rsi_long)
    max_bb_long = np.where(adx_ge30, 0.78, max_bb_long)

    min_rsi_short = np.where(adx_ge25, 33.0, 35.0)
    min_bb_short = np.where(adx_ge25, 0.32, 0.38)
    min_rsi_short = np.where(adx_ge30, 28.0, min_rsi_short)
    min_bb_short = np.where(adx_ge30, 0.18, min_bb_short)

    if "1HZ" in sym:
        max_bb_long = np.where(adx_ge28 := (adx >= 28), 0.74, 0.68)
        max_rsi_long = np.where(adx_ge28, 60.0, 57.0)

    v11 = variant == "v11"
    if v11:
        # v11 tightened the ADX>=30 RSI cap from 63 to 62.
        max_rsi_long = np.where(adx_ge30, 62.0, max_rsi_long)

    if v11:
        long_ok = (
            (score >= req)
            & (ef > es)
            & (ema_gap >= 0.015)
            & (c >= ef)
            & (close_vs_ema <= 0.80)
            & (c >= et)
            & (close_vs_trend <= 2.20)
            & (c >= eh)                              # mandatory macro alignment
            & (rsi <= max_rsi_long)
            & (rsi >= 33.0)
            & (pctb >= 0.12) & (pctb <= max_bb_long)
            & ~((pctb > 0.72) & (rsi > 59.0))        # extended-entry block
            & ~((pctb > 0.70) & (macd_rel < 0.015))  # exhaustion guard
            & (hist > 0) & (macd_rel >= 0.008)
            & (c > o)
        )
        short_ok = (
            (score <= -req)
            & (ef < es)
            & (-ema_gap >= 0.015)
            & (c <= ef)
            & (-close_vs_ema <= 0.80)
            & (c <= et)
            & (-close_vs_trend <= 2.20)
            & ((adx >= 30.0) | (c <= eh))
            & (rsi >= min_rsi_short) & (rsi <= 65.0)
            & (pctb >= min_bb_short) & (pctb <= 0.85)
            & ~((pctb < (min_bb_short + 0.05)) & (rsi < 42.0))
            & ~((pctb < 0.28) & (rsi < 40.0))        # floor-trap block
            & (hist < 0) & (macd_rel <= -0.008)
            & (c < o)
        )
    else:
        long_ok = (
            (score >= req)
            & (ef > es)
            & (ema_gap >= 0.015)
            & (c >= ef)
            & (close_vs_ema <= 0.80)
            & (c >= et)
            & (close_vs_trend <= 2.20)
            & (adx_ge25 | (c >= eh))
            & (rsi <= max_rsi_long)
            & (rsi >= 33.0)
            & (pctb >= 0.12) & (pctb <= max_bb_long)
            & ~((pctb > 0.70) & (rsi > 60.0))
            & (hist > 0) & (macd_rel >= 0.003)
            & (c > o)
        )
        short_ok = (
            (score <= -req)
            & (ef < es)
            & (-ema_gap >= 0.015)
            & (c <= ef)
            & (-close_vs_ema <= 0.80)
            & (c <= et)
            & (-close_vs_trend <= 2.20)
            & (adx_ge25 | (c <= eh))
            & (rsi >= min_rsi_short) & (rsi <= 65.0)
            & (pctb >= min_bb_short) & (pctb <= 0.85)
            & ~((pctb < (min_bb_short + 0.05)) & (rsi < 42.0))
            & (hist < 0) & (macd_rel <= -0.003)
            & (c < o)
        )

    # ADX floors + regime veto
    valid = ~np.isnan(adx) & ~np.isnan(rsi) & ~np.isnan(pctb) & ~np.isnan(atr) & (atr > 0)
    valid &= adx >= (22.0 if v11 else 20.0)
    if "JD" in sym or "1HZ" in sym:
        valid &= adx >= 22.0
    if v11 and "1HZ15V" in sym:
        valid &= False        # v11 blacklist
    if exclude_toxic and sym.upper() in EXCLUDED_RF_SYMBOLS:
        valid &= False        # measured defect (stpRNG4), see strategy_engine
    high_vol = pctile >= cfg.high_vol_atr_percentile
    valid &= ~np.isnan(pctile)
    # cfg.trade_high_vol is True, so high-vol regimes are *allowed*.

    # ignore the very first bars (rolling warm-up)
    valid &= np.arange(n) >= 60

    long_ok &= valid
    short_ok &= valid
    long_ok &= ~short_ok

    rows = []
    asset_class = ("forex" if sym.startswith("FRX") and not sym.startswith(("FRXXAU", "FRXXAG"))
                   else "commodities" if sym.startswith(("FRXXAU", "FRXXAG"))
                   else "synthetic")
    for i in np.flatnonzero(long_ok | short_ok):
        direction = "long" if long_ok[i] else "short"
        dur, _ = calculate_dynamic_duration(symbol, asset_class, d.iloc[i], "trend")
        rows.append((
            i, direction, dur, adx[i], rsi[i], pctb[i], pctile[i], macd_rel[i],
            ema_gap[i], close_vs_ema[i], close_vs_trend[i], close_vs_higher[i],
            score[i], atr[i], c[i],
            int(d.index[i].hour), ("high_vol" if pctile[i] >= cfg.high_vol_atr_percentile
                                    else "trend" if adx[i] >= 20 else "range"),
            body[i], rsi[i] - prev_rsi[i],
        ))
    return pd.DataFrame(rows, columns=SIGNAL_COLS)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class Filters:
    """Post-signal filter layer. Empty/None = inactive."""
    symbols: Optional[List[str]] = None
    directions: Optional[List[str]] = None
    durations: Optional[List[int]] = None
    adx_min: float = 0.0
    adx_max: float = 1e9
    rsi_long: tuple = (0.0, 100.0)
    rsi_short: tuple = (0.0, 100.0)
    bb_long: tuple = (0.0, 1.0)
    bb_short: tuple = (0.0, 1.0)
    atr_pctile: tuple = (0.0, 1.0)
    macd_rel_long: tuple = (-1e9, 1e9)
    macd_rel_short: tuple = (-1e9, 1e9)
    ema_gap_long: tuple = (-1e9, 1e9)
    ema_gap_short: tuple = (-1e9, 1e9)
    body_long_min: float = -1e9
    body_short_max: float = 1e9
    min_score: float = -1e9
    max_score: float = 1e9
    hours: Optional[tuple] = None
    bar_min: int = -1
    bar_max: int = 10 ** 9
    one_per_symbol: bool = True
    cooldown_bars: int = 3
    invert: bool = False
    duration_override: Optional[int] = None
    # Price must be on the correct side of the higher-timeframe EMA, not merely
    # past the ADX>=25 bypass. Long: close > ema_higher_tf, short: close < it.
    require_htf_agreement: bool = False

    def mask(self, s: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(s), dtype=bool)
        if self.symbols is not None:
            m &= s["symbol"].isin(self.symbols).values
        if self.directions is not None:
            m &= s["direction"].isin(self.directions).values
        if self.durations is not None:
            m &= s["duration"].isin(self.durations).values
        if self.hours is not None:
            m &= s["hour"].between(self.hours[0], self.hours[1]).values
        m &= (s["adx"] >= self.adx_min).values & (s["adx"] <= self.adx_max).values
        m &= (s["atr_pctile"] >= self.atr_pctile[0]).values & (s["atr_pctile"] <= self.atr_pctile[1]).values
        m &= (s["score"] >= self.min_score).values & (s["score"] <= self.max_score).values
        m &= (s["bar"] >= self.bar_min).values & (s["bar"] <= self.bar_max).values

        is_long = (s["direction"] == "long").values

        def per_dir(col, rng_long, rng_short):
            v = s[col].values
            return np.where(is_long, (v >= rng_long[0]) & (v <= rng_long[1]),
                            (v >= rng_short[0]) & (v <= rng_short[1]))

        m &= per_dir("rsi", self.rsi_long, self.rsi_short)
        m &= per_dir("bb_pctb", self.bb_long, self.bb_short)
        m &= per_dir("macd_rel", self.macd_rel_long, self.macd_rel_short)
        m &= per_dir("ema_gap", self.ema_gap_long, self.ema_gap_short)
        m &= per_dir("candle_body_atr", (self.body_long_min, 1e9), (-1e9, self.body_short_max))
        if self.require_htf_agreement:
            v = s["close_vs_higher"].values
            m &= np.where(is_long, v > 0.0, v < 0.0)
        return m


def simulate(sig: pd.DataFrame, bars: pd.DataFrame, payout: float,
             spread: float = 0.0, f: Optional[Filters] = None) -> pd.DataFrame:
    """Sequential replay with one position per symbol and a post-close cooldown."""
    if sig.empty:
        return pd.DataFrame()
    mask = f.mask(sig) if f is not None else np.ones(len(sig), dtype=bool)
    s = sig[mask].sort_values("bar")
    cooldown = f.cooldown_bars if f is not None else 3
    one_per_symbol = f.one_per_symbol if f is not None else True
    o = bars["open"].values
    n = len(o)
    last_exit_bar = -10 ** 9
    open_until = -1
    trades = []
    for r in s.itertuples():
        i = int(r.bar)
        if i + 1 >= n:
            continue
        if one_per_symbol and open_until >= 0 and i <= open_until + cooldown:
            continue
        entry_bar = i + 1
        dur = int(f.duration_override) if (f is not None and f.duration_override) else int(r.duration)
        exit_bar = entry_bar + dur
        if exit_bar >= n:
            continue
        entry = o[entry_bar]
        raw_exit = o[exit_bar]
        direction = r.direction
        if f is not None and getattr(f, "invert", False):
            direction = "short" if direction == "long" else "long"
        if spread:
            entry = entry + spread / 2 * (1 if direction == "long" else -1)
        move = (raw_exit - entry) if direction == "long" else (entry - raw_exit)
        win = move > 0
        p = payout(dur) if callable(payout) else payout
        pnl = p if win else -1.0
        open_until = exit_bar
        trades.append((r.bar, entry_bar, exit_bar, r.direction, dur,
                       r.score, r.adx, r.rsi, r.bb_pctb, r.atr_pctile, r.hour,
                       r.macd_rel, r.ema_gap, r.candle_body_atr, r.close_vs_higher,
                       win, pnl))
    return pd.DataFrame(trades, columns=[
        "signal_bar", "entry_bar", "exit_bar", "direction", "duration", "score",
        "adx", "rsi", "bb_pctb", "atr_pctile", "hour", "macd_rel", "ema_gap",
        "body", "close_vs_higher", "win", "pnl"])


def metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if trades.empty:
        return {"label": label, "n": 0, "wr": 0.0, "pnl": 0.0, "exp": 0.0,
                "pf": 0.0, "max_dd": 0.0}
    pnl = trades["pnl"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    eq = np.cumsum(pnl)
    dd = np.maximum.accumulate(np.concatenate([[0.0], eq])) - np.concatenate([[0.0], eq])
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() else float("inf")
    return {
        "label": label, "n": len(trades),
        "wr": float(trades["win"].mean()), "pnl": float(pnl.sum()),
        "exp": float(pnl.mean()), "pf": float(pf), "max_dd": float(dd.max()),
    }


def fmt(m: dict) -> str:
    return (f"{m['label']:<28} n={m['n']:>5}  WR={m['wr']*100:>5.1f}%  "
            f"PnL={m['pnl']:>8.2f}  exp/trade={m['exp']:>+6.3f}  "
            f"PF={m['pf']:>4.2f}  maxDD={m['max_dd']:>6.2f}")


def load_all(symbols: Optional[List[str]] = None):
    symbols = symbols or available_symbols()
    data, sigs = {}, []
    for sym in symbols:
        p = CANDLE_DIR / f"{sym}_60.csv"
        if not p.exists():
            continue
        df = load(sym)
        s = build_signals(sym, df)
        if s.empty:
            continue
        s.insert(0, "symbol", sym)
        s.insert(1, "epoch", df.index.view("int64")[s["bar"].values] // 1_000_000)
        data[sym] = df
        sigs.append(s)
    if not sigs:
        return data, pd.DataFrame(columns=["symbol"] + SIGNAL_COLS)
    return data, pd.concat(sigs, ignore_index=True)


def run_all(data: dict, sigs: pd.DataFrame, f: Optional[Filters] = None,
            pay: Optional[Dict[str, float]] = None,
            symbol_subset: Optional[List[str]] = None) -> pd.DataFrame:
    table = payout_table()
    out = []
    for sym, bars in data.items():
        if symbol_subset is not None and sym not in symbol_subset:
            continue
        s = sigs[sigs["symbol"] == sym]
        if s.empty:
            continue
        tr = simulate(s, bars, lambda d, _s=sym: pay_for(table, _s, d),
                      SPREAD.get(sym, 0.0), f)
        if not tr.empty:
            tr.insert(0, "symbol", sym)
            out.append(tr)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
