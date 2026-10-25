"""
research/v11_ab.py - A/B the v10 entry rules against the v11 rules that were
shipped on 2026-10-24 22:51, on identical candles, with a train/validation split.

Why: v11 was derived from the live journal (ADX<22 losses, one 1HZ15V sample,
one loss post-mortem) and was never measured out-of-sample. The backtest still
encoded v10, so nothing checked it. This does.

Split: train = candles before SPLIT_DATE, validation = on/after it. The rules
are not re-fitted here; both variants are fixed and simply scored on both halves.

Usage: .venv/bin/python research/v11_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402

UNIVERSE = ["R_10", "R_25", "R_50", "R_75", "R_100",
            "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
            "JD10", "JD25", "JD50", "JD75", "JD100",
            "stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"]

SPLIT_DATE = pd.Timestamp("2026-10-20", tz="UTC")
# v11's risk manager requires confidence >= 0.50 on rise_fall; confidence is
# |score| / sum(weights) = |score| / 5, so that is |score| >= 2.5 -> 3 (integers).
MIN_ABS_SCORE = {"v10": 2, "v11": 3}


def build(variant: str) -> tuple[dict, pd.DataFrame]:
    data, frames = {}, []
    for sym in UNIVERSE:
        if not (rb.CANDLE_DIR / f"{sym}_60.csv").exists():
            continue
        df = rb.load(sym)
        s = rb.build_signals(sym, df, variant=variant)
        if s.empty:
            data[sym] = df
            continue
        s.insert(0, "symbol", sym)
        s.insert(1, "epoch", df.index.view("int64")[s["bar"].values] // 1_000_000)
        s["time"] = pd.to_datetime(s["epoch"], unit="s", utc=True)
        s = s[s["score"].abs() >= MIN_ABS_SCORE[variant]]
        data[sym] = df
        frames.append(s)
    sigs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, sigs


def score(trades: pd.DataFrame, label: str) -> dict:
    m = rb.metrics(trades, label)
    if trades.empty:
        return m
    wins = int(trades["win"].sum())
    n = len(trades)
    lo, hi = wilson(wins / n, n)
    m.update(win_lo=lo, win_hi=hi)
    return m


def wilson(p: float, n: int, z: float = 1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - s) / d, (c + s) / d)


def main() -> None:
    results = {}
    for variant in ("v10", "v11"):
        data, sigs = build(variant)
        trades = rb.run_all(data, sigs)
        trades = trades.merge(
            sigs[["symbol", "bar", "time"]].rename(columns={"bar": "signal_bar"}),
            on=["symbol", "signal_bar"], how="left")
        # one trade per symbol per entry bar already; time comes from the signal bar
        for split, mask in (("train", trades["time"] < SPLIT_DATE),
                            ("valid", trades["time"] >= SPLIT_DATE),
                            ("all", pd.Series(True, index=trades.index))):
            results[(variant, split)] = score(trades[mask], f"{variant} {split}")
        print(f"\n{variant}: {len(sigs)} signals -> {len(trades)} trades "
              f"({trades['time'].min():%Y-%m-%d} .. {trades['time'].max():%Y-%m-%d})")

    print(f"\n{'variant':<7}{'split':<7}{'n':>6}{'WR':>8}{'95% CI':>16}"
          f"{'exp/trade':>11}{'netPnL':>10}{'maxDD':>9}")
    print("-" * 76)
    for (v, split), m in results.items():
        if m["n"] == 0:
            print(f"{v:<7}{split:<7}{0:>6}")
            continue
        print(f"{v:<7}{split:<7}{m['n']:>6}{m['wr']*100:>7.1f}%"
              f"{m['win_lo']*100:>7.1f}-{m['win_hi']*100:<7.1f}"
              f"{m['exp']:>+11.3f}{m['pnl']:>+10.2f}{m['max_dd']:>9.2f}")

    print("\nbreakeven at 0.88 payout = 53.2%; a variant only helps if it lifts")
    print("expectancy on the VALIDATION half, not just on the training half.")


if __name__ == "__main__":
    main()
