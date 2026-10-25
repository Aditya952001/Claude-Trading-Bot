"""
research/live_vs_backtest.py - Does the mirror reproduce the real live trades?

The 14-day replay says -0.063/trade. The live journal says +0.092/trade over 68
RF synthetic trades. Before trusting either, check that the backtest mirror
predicts the SAME direction and the SAME outcome for trades the bot actually
placed. If it does, the replay is a fair proxy and the live +0.09 is just a
small-sample fluctuation. If it does not, the replay is measuring something else.

For each closed live RF synthetic trade:
  * find the 1m candle whose close is the signal bar (entry_time - 1 minute)
  * run build_signals on that symbol's cached candles
  * compare the mirror's direction at that bar with the live direction
  * compare the mirror's simulated outcome with the realised pnl sign

Usage: .venv/bin/python research/live_vs_backtest.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import rf_backtest as rb  # noqa: E402

DB = ROOT / "data" / "bot_memory.sqlite3"


def main() -> None:
    con = sqlite3.connect(DB)
    tr = pd.read_sql_query(
        """select id, symbol, direction, entry_time, exit_time, entry_price,
                  exit_price, stake, pnl, status
           from trades
           where contract_type='rise_fall' and asset_class='synthetic'
             and status in ('closed','sold') and pnl is not null
           order by entry_time""", con)
    con.close()
    print(f"live closed RF synthetic trades: {len(tr)}")

    rows = []
    for r in tr.itertuples():
        p = rb.CANDLE_DIR / f"{r.symbol}_60.csv"
        if not p.exists():
            continue
        df = rb.load(r.symbol)
        entry = pd.Timestamp(int(r.entry_time), unit="s", tz="UTC")
        # signal bar = the last bar that closed before entry
        sig_bar_time = entry.floor("min") - pd.Timedelta(minutes=1)
        if sig_bar_time < df.index.min() or sig_bar_time > df.index.max():
            continue
        s = rb.build_signals(r.symbol, df, variant="v10")
        if s.empty:
            continue
        s = s.copy()
        s["time"] = df.index[s["bar"].values]
        hit = s[s["time"] == sig_bar_time]
        if hit.empty:
            rows.append({"id": r.id, "symbol": r.symbol, "time": sig_bar_time,
                         "live_dir": r.direction, "mirror_dir": None,
                         "live_win": r.pnl > 0, "mirror_win": None,
                         "pnl": r.pnl, "reason": "no mirror signal"})
            continue
        m = hit.iloc[0]
        # simulate from that signal bar forward with the same duration
        dur = int(m["duration"])
        i = int(m["bar"])
        o = df["open"].values
        if i + 1 + dur >= len(o):
            continue
        entry_px = o[i + 1]
        exit_px = o[i + 1 + dur]
        mwin = (exit_px > entry_px) if m["direction"] == "long" else (exit_px < entry_px)
        rows.append({"id": r.id, "symbol": r.symbol, "time": sig_bar_time,
                     "live_dir": r.direction, "mirror_dir": m["direction"],
                     "live_win": r.pnl > 0, "mirror_win": bool(mwin),
                     "pnl": r.pnl, "reason": ""})

    x = pd.DataFrame(rows)
    print(f"matched to cached candles: {len(x)}")
    if x.empty:
        return

    both = x.dropna(subset=["mirror_dir"])
    print(f"with a mirror signal at the same bar: {len(both)}")

    dir_agree = (both["live_dir"] == both["mirror_dir"]).mean()
    print(f"\ndirection agreement (live vs mirror): {dir_agree*100:.1f}% "
          f"({int((both['live_dir']==both['mirror_dir']).sum())}/{len(both)})")

    out_agree = (both["live_win"] == both["mirror_win"]).mean()
    print(f"outcome agreement (win/loss):        {out_agree*100:.1f}% "
          f"({int((both['live_win']==both['mirror_win']).sum())}/{len(both)})")

    print("\ndisagreements:")
    bad = both[both["live_dir"] != both["mirror_dir"]]
    for r in bad.head(20).itertuples():
        print(f"  #{r.id} {r.symbol:<9}{r.time:%m-%d %H:%M}  live={r.live_dir:<6}"
              f"mirror={r.mirror_dir:<6} live_win={r.live_win} mirror_win={r.mirror_win}")

    print("\nlive vs mirror PnL on the matched set:")
    print(f"  live   n={len(both):>3}  WR={both['live_win'].mean()*100:.1f}%  "
          f"PnL={both['pnl'].sum():+.2f}  exp={both['pnl'].mean():+.3f}")
    mp = both["mirror_win"].map({True: 0.9, False: -1.0})
    print(f"  mirror n={len(both):>3}  WR={both['mirror_win'].mean()*100:.1f}%  "
          f"PnL={mp.sum():+.2f}  exp={mp.mean():+.3f}")

    # Where there was no mirror signal, what was the live record?
    nom = x[x["mirror_dir"].isna()]
    if len(nom):
        print(f"\nlive trades with NO mirror signal ({len(nom)}): "
              f"WR={nom['live_win'].mean()*100:.1f}%  PnL={nom['pnl'].sum():+.2f}")
        print("  (these are signals the current rules would not take at all)")


if __name__ == "__main__":
    main()
