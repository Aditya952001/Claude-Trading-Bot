"""
research/v11_rules.py - Marginal effect of each individual v11 rule.

Each rule is applied on its own to the v10 signal set, then replayed, so the
cooldown/one-position logic is re-evaluated after the filter. This answers
"which of the v11 changes actually helped, and on which half of the data?".

Usage: .venv/bin/python research/v11_rules.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from v11_ab import UNIVERSE, SPLIT_DATE, wilson  # noqa: E402

RULES = {
    "baseline v10":      lambda s: pd.Series(True, index=s.index),
    "ADX floor 22":      lambda s: s["adx"] >= 22,
    "drop 1HZ15V":       lambda s: s["symbol"] != "1HZ15V",
    "long close>=EMA200": lambda s: ~((s["direction"] == "long") & (s["close_vs_higher"] < 0)),
    "macd_rel 0.008":    lambda s: s["macd_rel"].abs() >= 0.008,
    "score>=3 (conf .5)": lambda s: s["score"].abs() >= 3,
    "bb>0.72&rsi>59":    lambda s: ~((s["direction"] == "long") & (s["bb_pctb"] > 0.72) & (s["rsi"] > 59)),
    "exhaustion guard":  lambda s: ~((s["direction"] == "long") & (s["bb_pctb"] > 0.70) & (s["macd_rel"] < 0.015)),
    "short floor block": lambda s: ~((s["direction"] == "short") & (s["bb_pctb"] < 0.28) & (s["rsi"] < 40)),
}


def main() -> None:
    data, frames = {}, []
    for sym in UNIVERSE:
        if not (rb.CANDLE_DIR / f"{sym}_60.csv").exists():
            continue
        df = rb.load(sym)
        s = rb.build_signals(sym, df, variant="v10")
        if s.empty:
            data[sym] = df
            continue
        s.insert(0, "symbol", sym)
        s.insert(1, "epoch", df.index.view("int64")[s["bar"].values] // 1_000_000)
        s["time"] = pd.to_datetime(s["epoch"], unit="s", utc=True)
        data[sym] = df
        frames.append(s)
    sigs = pd.concat(frames, ignore_index=True)

    print(f"{'rule':<20}{'n':>6}{'train n':>9}{'train exp':>11}{'valid n':>9}{'valid exp':>11}{'delta(valid)':>14}")
    print("-" * 82)
    base_valid = None
    for name, fn in RULES.items():
        keep = fn(sigs)
        sub = sigs[keep]
        tr = rb.run_all(data, sub)
        tr = tr.merge(sigs[["symbol", "bar", "time"]].rename(columns={"bar": "signal_bar"}),
                      on=["symbol", "signal_bar"], how="left")
        trn = tr[tr["time"] < SPLIT_DATE]
        val = tr[tr["time"] >= SPLIT_DATE]
        e_tr = trn["pnl"].mean() if len(trn) else 0.0
        e_va = val["pnl"].mean() if len(val) else 0.0
        if base_valid is None:
            base_valid = e_va
            delta = 0.0
        else:
            delta = e_va - base_valid
        print(f"{name:<20}{len(sub):>6}{len(trn):>9}{e_tr:>+11.3f}{len(val):>9}{e_va:>+11.3f}{delta:>+14.3f}")

    # Per-symbol record for the one symbol that was blacklisted.
    tr = rb.run_all(data, sigs)
    tr = tr.merge(sigs[["symbol", "bar", "time"]].rename(columns={"bar": "signal_bar"}),
                  on=["symbol", "signal_bar"], how="left")
    print("\nPer-symbol backtest record (v10 signals, all 14 days):")
    g = tr.groupby("symbol").agg(n=("win", "size"), wr=("win", "mean"),
                                 exp=("pnl", "mean"), pnl=("pnl", "sum")).sort_values("exp")
    for sym, r in g.iterrows():
        lo, hi = wilson(r["wr"], int(r["n"]))
        print(f"  {sym:<9}{int(r['n']):>4}  {r['wr']*100:>5.1f}%  "
              f"CI {lo*100:>4.1f}-{hi*100:<4.1f}  exp {r['exp']:+.3f}  pnl {r['pnl']:+.2f}")


if __name__ == "__main__":
    main()
