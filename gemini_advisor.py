"""
gemini_advisor.py - AI Trade Auditor and Pre-Trade Sanity Veto using Gemini API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Explicit environment variables must win over .env (12-factor precedence).
# override=True here silently clobbered CLI overrides such as
# `RF_EDGE_MIN_TRADES=1000 python run_rise_fall.py`.
load_dotenv(override=False)
log = logging.getLogger("gemini_advisor")

MODEL_CANDIDATES = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
]

class GeminiAdvisor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.enabled = bool(self.api_key)
        self.active_model = MODEL_CANDIDATES[0]
        if not self.enabled:
            log.warning("GeminiAdvisor initialized WITHOUT api_key; AI advisory disabled.")
        else:
            log.info("GeminiAdvisor initialized with model '%s'", self.active_model)

    def _call_gemini_sync(self, prompt: str, timeout: float = 12.0) -> Optional[str]:
        if not self.enabled:
            return None
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            }
        }).encode("utf-8")

        models = [self.active_model] + [m for m in MODEL_CANDIDATES if m != self.active_model]
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    self.active_model = model
                    return text
            except urllib.error.HTTPError as exc:
                log.warning("Gemini model %s HTTP error (%s: %s); failing over to next model candidate", model, exc.code, exc)
                continue
            except Exception as exc:
                log.warning("Gemini model %s call failed (%s: %s); failing over", model, type(exc).__name__, exc)
                continue
        return None

    async def call_gemini(self, prompt: str, timeout: float = 6.0) -> Optional[str]:
        return await asyncio.to_thread(self._call_gemini_sync, prompt, timeout)

    async def analyze_loss(
        self,
        trade_id: int,
        symbol: str,
        direction: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
        duration_sec: float,
        peak_profit: float,
        exit_reason: str,
        entry_indicators: dict,
        exit_indicators: Optional[dict] = None,
        pre_trade_veto: Optional[str] = None,
        signal_close: Optional[float] = None,
    ) -> dict:
        """
        Forensic post-mortem analysis of a losing trade.
        Captures the exact indicators, asset state, and determines the strategy flaw.

        `entry_price` is the FILL, which happens a few seconds after the signal
        bar closed; `signal_close` is the close of the bar the entry rules were
        actually evaluated on. They differ, and conflating them made the model
        report "entered below EMA9" for trades whose signal bar closed above it.
        """
        if not self.enabled:
            return {"status": "disabled", "summary": "Gemini AI not enabled"}

        sig_close = signal_close if signal_close is not None else entry_indicators.get("close")
        prompt = f"""You are a Quantitative Trading Auditor. Perform an immediate forensic post-mortem on this losing trade:
Trade #{trade_id} {symbol} {direction.upper()} | Loss: ${pnl:.2f} | Entry fill: {entry_price} -> Exit: {exit_price} | Duration: {duration_sec:.1f}s | Exit: {exit_reason}
Signal bar close (the price the entry rules were tested on): {sig_close}
Fill price (0-8 seconds AFTER that bar closed; slippage is expected): {entry_price}
1-Minute Signal-Bar Indicators:
- RSI: {entry_indicators.get('rsi')}
- MACD Histogram: {entry_indicators.get('macd_hist')}
- Bollinger %B: {entry_indicators.get('bb_pctb')}
- ADX: {entry_indicators.get('adx')}
- ATR: {entry_indicators.get('atr')}
- Fast EMA (9): {entry_indicators.get('ema_fast')}
- Slow EMA (21): {entry_indicators.get('ema_slow')}
- Macro EMA (200): {entry_indicators.get('ema_higher_tf')}

Rules for your analysis:
- Judge whether an entry rule was violated using the SIGNAL BAR CLOSE and the signal-bar indicators, never the fill price.
- A fill price slightly below EMA9 on a LONG is normal execution drift, not a rule breach, if the signal bar closed at or above EMA9.
- Do not propose a patch that merely restates an existing rule.
- One losing trade is not evidence that a rule is wrong; prefer "Uncertain" over inventing a flaw.

Respond in EXACTLY this JSON structure:
{{
  "failure_category": "Chop Trap | False Breakout | Momentum Exhaustion | Floor/Ceiling Trap | Counter-Trend | Uncertain",
  "root_cause": "<1-sentence direct explanation>",
  "flawed_indicators": ["<indicators that misled the entry>"],
  "recommended_patch": "<exact rule/condition to add to strategy_engine.py, or 'none'>",
  "preventable_by": "<how this rule prevents this loss>"
}}"""


        res = await self.call_gemini(prompt, timeout=12.0)
        if not res:
            return {
                "status": "timeout",
                "failure_category": "Uncertain (AI Timeout)",
                "root_cause": f"Loss of ${abs(pnl):.2f} on {symbol} {direction} held {duration_sec:.0f}s",
                "flawed_indicators": [],
                "recommended_patch": "Review MACD histogram sign, ADX chop threshold, and Bollinger %B boundaries.",
            }

        # Clean JSON markdown if wrapped
        clean_text = res.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            parsed = json.loads(clean_text)
            parsed["status"] = "success"
            return parsed
        except Exception:
            return {
                "status": "raw_text",
                "raw_response": res,
                "failure_category": "Forensic Analysis",
                "root_cause": res[:200],
            }


    async def veto_signal(self, symbol: str, direction: str, score: float,
                          ind_1m: dict, ind_5m: dict) -> Tuple[bool, str]:
        """
        Pre-trade AI sanity filter.
        Returns: (approved: bool, reason: str)
        If Gemini is unreachable or times out, it fails open (approved=True) to avoid blocking trading.
        """
        if not self.enabled:
            return True, "Gemini not enabled"

        close_5m = float(ind_5m.get("close") or ind_1m.get("close") or 0.0)
        ema_5m = float(ind_5m.get("ema50") or ind_1m.get("ema_trend") or close_5m)
        trend_5m = "BULLISH" if close_5m >= ema_5m else "BEARISH"

        prompt = f"""You are an intelligent quantitative risk filter for an algorithmic trading bot executing fast digital Rise/Fall contracts (1m to 20m duration).
Your goal is to reject high-risk traps (exhaustion peaks/troughs, BB extended entries, mean-reversion bounces, weak trend noise) while approving genuine high-probability momentum waves.

Signal Details:
- Instrument: {symbol}
- Direction: {direction.upper()}
- Confluence Score: {score:.2f} / 5.0
- 1-Minute State: Close={ind_1m.get('close')}, EMA9={ind_1m.get('ema_fast')}, EMA21={ind_1m.get('ema_slow')}, RSI={ind_1m.get('rsi'):.1f}, BB%B={ind_1m.get('bb_pctb'):.3f}, MACD_Hist={ind_1m.get('macd_hist')}, ADX={ind_1m.get('adx'):.1f}
- Intermediate/Macro Context: Close={close_5m}, Reference_EMA={ema_5m}, Alignment={trend_5m}

Decision Rules (apply in order, first match wins):
1. REJECT if Direction is LONG and RSI > 63.0 (momentum exhaustion — overbought entry, historical 41% win rate).
2. REJECT if Direction is LONG and BB%B > 0.70 and RSI > 57.0 (extended entry trap — price at top of range with elevated RSI, historical 25% win rate).
3. REJECT if Direction is SHORT and RSI < 30.0 (extreme panic climax; selling into hard floor causes instant bounce).
4. REJECT if Direction is SHORT and BB%B < 0.35 and RSI < 42.0 (floor-bounce trap — price at bottom of range, mean-reversion risk, caused R_25 and JD50 losses).
5. REJECT if ADX < 22.0 (weak trend — no directional conviction, only 33% win rate historically).
6. REJECT if EMA9 and EMA21 are completely flat/entangled with zero momentum spread.
7. APPROVE if EMA9/21 are aligned in signal direction, RSI is healthy (35-63 for LONG, 37-63 for SHORT), BB%B is in valid range (0.15-0.70 for LONG, 0.35-0.82 for SHORT), and ADX >= 22.
8. Otherwise APPROVE (do not over-filter valid setups).

Output EXACTLY one line in this format:
APPROVED: <brief 1-sentence reason>
OR
REJECTED: <brief 1-sentence reason>"""

        res = await self.call_gemini(prompt, timeout=6.0)
        if not res:
            return True, "Gemini timeout/offline; allowed by fallback"

        lines = [line.strip() for line in res.strip().split("\n") if line.strip()]
        decision_line = lines[0] if lines else ""

        if "REJECTED" in decision_line.upper():
            return False, decision_line
        return True, decision_line or "Approved by Gemini"


    async def audit_milestone_batch(self, batch_index: int, batch_trades: list, current_params: dict) -> dict:
        """
        Post-milestone deep loss audit across a 10-trade batch.
        """
        if not self.enabled:
            return {"advice": "Gemini not enabled"}

        trade_summaries = []
        for t in batch_trades:
            pnl = t.get("pnl", 0.0)
            res = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
            dur = int(t.get("exit_time", 0) - t.get("entry_time", 0)) if t.get("exit_time") else 0
            trade_summaries.append(
                f"- #{t.get('id')}: {t.get('symbol')} {t.get('direction').upper()} | PnL: ${pnl:+.2f} ({res}) | "
                f"Dur: {dur}s | Score: {t.get('score')} | RSI: {t.get('indicators', {}).get('rsi')}"
            )
        trades_text = "\n".join(trade_summaries)

        prompt = f"""You are the Chief Quantitative Strategist reviewing Batch #{batch_index} (10 trades).
Here are the trades:
{trades_text}

Current Strategy Settings:
- RSI Long Max: {current_params.get('rsi_long_max')}
- RSI Short Min: {current_params.get('rsi_short_min')}
- Min Score: {current_params.get('min_score')}

Analyze the loss patterns. Identify which symbols or behaviors caused losses.
Output your findings in 2 short bullet points:
1. Root cause of losses in this batch
2. Suggested focus for next 10 trades"""

        res = await self.call_gemini(prompt, timeout=5.0)
        return {"audit": res or "No audit response"}
