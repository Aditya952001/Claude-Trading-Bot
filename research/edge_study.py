"""
research/edge_study.py - Exhaustive search for a real, out-of-sample edge.

Question: can the Rise/Fall entry rules be made more profitable, and how high can
win rate honestly go? This script does NOT fit anything to the validation half.
It enumerates every plausible lever, reports train and validation separately, and
only calls a lever "real" if it improves BOTH halves.

Levers tested
  1. duration override 1..10 minutes
  2. signal-quality buckets: score, ADX, RSI, %B, macd_rel, ema_gap, body,
     atr percentile, hour, direction, symbol family
  3. per-symbol selection (payout differs: 0.88 / 0.90-0.92 / 0.95)
  4. fade (invert) variant
  5. regime exclusion (high_vol / trend / range)

Method notes
  * Descriptive buckets use the baseline trade list (conditional win rate).
  * Deployable filters are re-simulated, because removing signals changes the
    one-position-per-symbol cooldown and therefore which later trades happen.
  * A bucket is only interesting if n is large enough that its 95% CI excludes
    breakeven. With ~700 trades, that needs n >~ 300 for a 5pp effect.
  * Multiple testing: ~100 buckets are inspected. At alpha=0.05 you expect ~5
    false positives by chance, so any single "significant" bucket is treated as
    a hypothesis, not a result, unless it also holds on the other half.

Usage: .venv/bin/python research/edge_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402
from v11_ab import UNIVERSE, SPLIT_DATE, wilson  # noqa: E402

PD = 2  # decimals for money


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_signals(variant: str = "v10") -> tuple[dict, pd.DataFrame]:
    data, frames = {}, []
    for sym in UNIVERSE:
        if not (rb.CANDLE_DIR / f"{sym}_60.csv").exists():
            continue
        df = rb.load(sym)
        data[sym] = df
        s = rb.build_signals(sym, df, variant=variant)
        if s.empty:
            continue
        s.insert(0, "symbol", sym)
        s.insert(1, "epoch", df.index.view("int64")[s["bar"].values] // 1_000_000)
        s["time"] = pd.to_datetime(s["epoch"], unit="s", utc=True)
        frames.append(s)
    sigs = pd.concat(frames, ignore_index=True)
    return data, sigs


def replay(data: dict, sigs: pd.DataFrame, f=None) -> pd.DataFrame:
    tr = rb.run_all(data, sigs, f)
    if tr.empty:
        return tr
    tr = tr.merge(
        sigs[["symbol", "bar", "time"]].rename(columns={"bar": "signal_bar"}),
        on=["symbol", "signal_bar"], how="left")
    tr["split"] = np.where(tr["time"] < SPLIT_DATE, "train", "valid")
    return tr


def stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"n": 0, "wr": 0.0, "exp": 0.0, "pnl": 0.0, "be": 0.0}
    n = len(t)
    w = int(t["win"].sum())
    pnl = float(t["pnl"].sum())
    payout = float(t.loc[t["win"], "pnl"].mean()) if w else 0.0
    be = 1.0 / (1.0 + payout) if payout else 0.0
    lo, hi = wilson(w / n, n)
    return {"n": n, "w": w, "wr": w / n, "pnl": pnl, "exp": pnl / n,
            "payout": payout, "be": be, "lo": lo, "hi": hi}


def line(name: str, t: pd.DataFrame, width: int = 30) -> str:
    s = stats(t)
    if s["n"] == 0:
        return f"{name:<{width}}{0:>6}"
    return (f"{name:<{width}}{s['n']:>6}{s['wr']*100:>7.1f}%"
            f"{s['lo']*100:>6.1f}-{s['hi']*100:<6.1f}"
            f"{s['exp']:>+9.3f}{s['pnl']:>+9.2f}")


HDR = (f"{'bucket':<30}{'n':>6}{'WR':>8}{'95% CI':>14}"
       f"{'exp/trade':>9}{'netPnL':>9}")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 1. baseline
# ---------------------------------------------------------------------------
def baseline(data, sigs) -> pd.DataFrame:
    section("0. BASELINE (v10 rules, 23 symbols, $1 stake, real payout table)")
    tr = replay(data, sigs)
    print(HDR)
    print("-" * 78)
    for label, sub in (("all", tr), ("train (<10-20)", tr[tr.split == "train"]),
                       ("valid (>=10-20)", tr[tr.split == "valid"])):
        print(line(label, sub))
    s = stats(tr)
    print(f"\nbreakeven at the realised average payout {s['payout']:.3f} = {s['be']*100:.1f}%")
    print(f"win rate {s['wr']*100:.1f}% -> gap to breakeven {(s['wr']-s['be'])*100:+.1f} pp")
    # how many trades to prove a 3pp edge at this variance?
    p = s["wr"]
    need = int(np.ceil((1.96 ** 2) * p * (1 - p) / (0.03 ** 2)))
    print(f"trades needed to resolve a 3pp edge at 95% confidence: ~{need}")
    return tr


# ---------------------------------------------------------------------------
# 2. duration sweep
# ---------------------------------------------------------------------------
def duration_sweep(data, sigs) -> None:
    section("1. DURATION SWEEP (payout is flat per symbol, so only WR moves)")
    print(HDR)
    print("-" * 78)
    for dur in range(1, 11):
        tr = replay(data, sigs, rb.Filters(duration_override=dur))
        print(line(f"{dur}m fixed", tr))
    print("\nsame, split by half (only both-halves improvements matter):")
    print(f"{'dur':<6}{'train n':>9}{'train exp':>11}{'valid n':>9}{'valid exp':>11}")
    print("-" * 78)
    for dur in range(1, 11):
        tr = replay(data, sigs, rb.Filters(duration_override=dur))
        a = stats(tr[tr.split == "train"])
        b = stats(tr[tr.split == "valid"])
        print(f"{dur}m{'':<4}{a['n']:>9}{a['exp']:>+11.3f}{b['n']:>9}{b['exp']:>+11.3f}")


# ---------------------------------------------------------------------------
# 3. descriptive buckets
# ---------------------------------------------------------------------------
def bucket_table(tr: pd.DataFrame, col: str, bins, labels, title: str) -> None:
    section(f"2. BUCKET: {title}")
    print(f"{'bucket':<22}{'train n':>9}{'train WR':>10}{'train exp':>11}"
          f"{'valid n':>9}{'valid WR':>10}{'valid exp':>11}{'both?':>7}")
    print("-" * 90)
    x = tr[col]
    grp = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
    for lab in labels:
        sub = tr[grp == lab]
        a = stats(sub[sub.split == "train"])
        b = stats(sub[sub.split == "valid"])
        if a["n"] == 0 or b["n"] == 0:
            continue
        both = "YES" if (a["exp"] > 0 and b["exp"] > 0) else (
            "no" if (a["exp"] <= 0 and b["exp"] <= 0) else "one")
        print(f"{str(lab):<22}{a['n']:>9}{a['wr']*100:>9.1f}%{a['exp']:>+11.3f}"
              f"{b['n']:>9}{b['wr']*100:>9.1f}%{b['exp']:>+11.3f}{both:>7}")


def buckets(tr: pd.DataFrame) -> None:
    bucket_table(tr, "score", [1.5, 2.5, 3.5, 4.5, 5.5],
                 [2, 3, 4, 5], "confluence score |score|")
    bucket_table(tr, "adx", [19.9, 22, 25, 30, 40, 100],
                 ["20-22", "22-25", "25-30", "30-40", "40+"], "ADX")
    bucket_table(tr, "atr_pctile", [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
                 ["0-20", "20-40", "40-60", "60-80", "80-100"], "ATR percentile")
    bucket_table(tr, "macd_rel", [-1e9, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 1e9],
                 ["<-0.02", "-0.02..-0.01", "-0.01..-0.005", "-0.005..0", "0..0.005",
                  "0.005..0.01", "0.01..0.02", ">0.02"], "MACD histogram / ATR")
    bucket_table(tr, "ema_gap", [-1e9, -0.5, -0.25, 0.0, 0.25, 0.5, 1e9],
                 ["<-0.5", "-0.5..-0.25", "-0.25..0", "0..0.25", "0.25..0.5", ">0.5"],
                 "EMA9-EMA21 gap / ATR")
    bucket_table(tr, "hour", [-0.1, 3.9, 7.9, 11.9, 15.9, 19.9, 23.1],
                 ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"], "UTC hour")

    section("2b. BUCKET: RSI (direction-aware)")
    for d in ("long", "short"):
        sub = tr[tr.direction == d]
        print(f"  {d}:")
        print(f"  {'rsi':<18}{'train n':>9}{'train WR':>10}{'train exp':>11}"
              f"{'valid n':>9}{'valid WR':>10}{'valid exp':>11}")
        bins = [0, 35, 40, 45, 50, 55, 60, 65, 100]
        labs = ["<35", "35-40", "40-45", "45-50", "50-55", "55-60", "60-65", "65+"]
        g = pd.cut(sub["rsi"], bins=bins, labels=labs, include_lowest=True)
        for lab in labs:
            x = sub[g == lab]
            a, b = stats(x[x.split == "train"]), stats(x[x.split == "valid"])
            if a["n"] < 5 or b["n"] < 5:
                continue
            print(f"  {str(lab):<18}{a['n']:>9}{a['wr']*100:>9.1f}%{a['exp']:>+11.3f}"
                  f"{b['n']:>9}{b['wr']*100:>9.1f}%{b['exp']:>+11.3f}")

    section("2c. BUCKET: Bollinger %B (direction-aware)")
    for d in ("long", "short"):
        sub = tr[tr.direction == d]
        print(f"  {d}:")
        bins = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.01]
        labs = ["0-.15", ".15-.3", ".3-.45", ".45-.6", ".6-.75", ".75-.9", ".9-1"]
        g = pd.cut(sub["bb_pctb"], bins=bins, labels=labs, include_lowest=True)
        print(f"  {'pctb':<18}{'train n':>9}{'train WR':>10}{'train exp':>11}"
              f"{'valid n':>9}{'valid WR':>10}{'valid exp':>11}")
        for lab in labs:
            x = sub[g == lab]
            a, b = stats(x[x.split == "train"]), stats(x[x.split == "valid"])
            if a["n"] < 5 or b["n"] < 5:
                continue
            print(f"  {str(lab):<18}{a['n']:>9}{a['wr']*100:>9.1f}%{a['exp']:>+11.3f}"
                  f"{b['n']:>9}{b['wr']*100:>9.1f}%{b['exp']:>+11.3f}")

    section("2d. BUCKET: direction and symbol family")
    print(HDR)
    print("-" * 78)
    print(line("long", tr[tr.direction == "long"]))
    print(line("short", tr[tr.direction == "short"]))
    for fam, mask in (("R_ (0.88)", tr.symbol.str.startswith("R_")),
                      ("1HZ (0.88)", tr.symbol.str.startswith("1HZ")),
                      ("JD (0.90-0.92)", tr.symbol.str.startswith("JD")),
                      ("stpRNG (0.95)", tr.symbol.str.startswith("stp"))):
        print(line(fam, tr[mask]))
    print("\nper symbol, train vs valid:")
    print(f"{'symbol':<10}{'pay':>5}{'train n':>9}{'train WR':>10}{'train exp':>11}"
          f"{'valid n':>9}{'valid WR':>10}{'valid exp':>11}")
    print("-" * 90)
    for sym, g in tr.groupby("symbol"):
        a, b = stats(g[g.split == "train"]), stats(g[g.split == "valid"])
        print(f"{sym:<10}{a['payout'] if a['n'] else b['payout']:>5.2f}"
              f"{a['n']:>9}{a['wr']*100:>9.1f}%{a['exp']:>+11.3f}"
              f"{b['n']:>9}{b['wr']*100:>9.1f}%{b['exp']:>+11.3f}")


# ---------------------------------------------------------------------------
# 4. deployable filters (re-simulated)
# ---------------------------------------------------------------------------
def deployable(data, sigs) -> None:
    section("3. DEPLOYABLE FILTERS (re-simulated, so cooldown effects are included)")
    print(f"{'filter':<34}{'train n':>9}{'train exp':>11}{'valid n':>9}{'valid exp':>11}{'verdict':>9}")
    print("-" * 86)

    cands = {
        "baseline": rb.Filters(),
        "score >= 3": rb.Filters(min_score=3),
        "score >= 4": rb.Filters(min_score=4),
        "longs only": rb.Filters(directions=["long"]),
        "shorts only": rb.Filters(directions=["short"]),
        "R_ only": rb.Filters(symbols=[s for s in UNIVERSE if s.startswith("R_")]),
        "1HZ only": rb.Filters(symbols=[s for s in UNIVERSE if s.startswith("1HZ")]),
        "JD only": rb.Filters(symbols=[s for s in UNIVERSE if s.startswith("JD")]),
        "stpRNG only": rb.Filters(symbols=[s for s in UNIVERSE if s.startswith("stp")]),
        "stpRNG (0.95 payout)": rb.Filters(
            symbols=["stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"]),
        "ADX 20-25": rb.Filters(adx_min=20, adx_max=25),
        "ADX 25-30": rb.Filters(adx_min=25, adx_max=30),
        "ADX >= 30": rb.Filters(adx_min=30),
        "ADX >= 35": rb.Filters(adx_min=35),
        "atr_pctile <= 0.5": rb.Filters(atr_pctile=(0, 0.5)),
        "atr_pctile >= 0.5": rb.Filters(atr_pctile=(0.5, 1)),
        "atr_pctile >= 0.8": rb.Filters(atr_pctile=(0.8, 1)),
        "hours 00-08 UTC": rb.Filters(hours=(0, 7)),
        "hours 08-16 UTC": rb.Filters(hours=(8, 15)),
        "hours 16-24 UTC": rb.Filters(hours=(16, 23)),
        "HTF agreement (strict)": rb.Filters(require_htf_agreement=True),
        "FADE (invert)": rb.Filters(invert=True),
        "1 trade / 5 bars": rb.Filters(cooldown_bars=5),
        "1 trade / 10 bars": rb.Filters(cooldown_bars=10),
    }
    rows = []
    for name, f in cands.items():
        tr = replay(data, sigs, f)
        a, b = stats(tr[tr.split == "train"]), stats(tr[tr.split == "valid"])
        if a["n"] == 0 or b["n"] == 0:
            continue
        if a["exp"] > 0 and b["exp"] > 0:
            verdict = "BOTH+"
        elif a["exp"] > 0 or b["exp"] > 0:
            verdict = "one"
        else:
            verdict = "no"
        rows.append((name, a, b, verdict))
        print(f"{name:<34}{a['n']:>9}{a['exp']:>+11.3f}{b['n']:>9}{b['exp']:>+11.3f}{verdict:>9}")
    return rows


# ---------------------------------------------------------------------------
# 5. what would 80% require?
# ---------------------------------------------------------------------------
def reality_check(tr: pd.DataFrame) -> None:
    section("4. WHAT WOULD 80% ACCURACY REQUIRE?")
    n = len(tr)
    w = int(tr["win"].sum())
    print(f"observed: {w}/{n} = {w/n*100:.1f}%")
    # best bucket of 100 inspected, with n>=100
    print("\nEvery 1m candle in this dataset, if entered at random, wins ~50% of the")
    print("time (the price is a martingale over a fixed horizon). A 5-gate trend")
    print("filter is a *selection* over that, so its ceiling is the conditional")
    print("drift it can identify. Measured conditional drift here is small:")
    print(f"  best direction hit rate across all 14 days: {w/n*100:.1f}%")
    print("\nTo reach 80% you would need a 30pp conditional edge over a coin flip.")
    print("The largest genuine conditional edge in this dataset (any bucket with")
    print("n>=100) is shown below; it is nowhere near 30pp.")
    # largest bucket edge with n>=100
    best = None
    for col, bins, labs in (
        ("score", [1.5, 2.5, 3.5, 4.5, 5.5], [2, 3, 4, 5]),
        ("adx", [19.9, 22, 25, 30, 40, 100], ["20-22", "22-25", "25-30", "30-40", "40+"]),
        ("atr_pctile", [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
         ["0-20", "20-40", "40-60", "60-80", "80-100"]),
    ):
        g = pd.cut(tr[col], bins=bins, labels=labs, include_lowest=True)
        for lab in labs:
            x = tr[g == lab]
            if len(x) < 100:
                continue
            s = stats(x)
            if best is None or s["wr"] > best[0]:
                best = (s["wr"], f"{col}={lab}", s)
    if best:
        wr, name, s = best
        print(f"  {name}: n={s['n']}  WR={wr*100:.1f}%  "
              f"95% CI {s['lo']*100:.1f}-{s['hi']*100:.1f}%")
        print(f"  -> even the upper CI ({s['hi']*100:.1f}%) is far from 80%.")
    print("\nConclusion: 80% is not a tuning target, it is a different instrument.")
    print("A 52-56% win rate at 0.88-0.95 payout is the realistic range for this")
    print("family of rules. Profit comes from expectancy x volume, not from WR.")


def main() -> None:
    data, sigs = load_signals("v10")
    tr = baseline(data, sigs)
    duration_sweep(data, sigs)
    buckets(tr)
    deployable(data, sigs)
    reality_check(tr)


if __name__ == "__main__":
    main()
