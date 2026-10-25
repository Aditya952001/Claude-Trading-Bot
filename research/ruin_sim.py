"""
research/ruin_sim.py - What actually happens to a $10 account at $1 stake?

The user plans to move to a real account with $10 once confident. This answers
the only question that matters for that plan: given the measured win rate and
the payout, what is the probability of doubling the account before losing it?

Model: start at $10, stake $1 per trade, win -> +payout, lose -> -1. Stop when
balance < $1 (cannot place the minimum stake = ruin) or balance >= $20.

Scenarios
  * measured     p = 49.1%, payout 0.908   (the 14-day replay)
  * optimistic   p = 55.0%, payout 0.88    (better than anything measured)
  * dream        p = 80.0%, payout 0.20    (what 80% "accuracy" would look like
                                            if it came from a far barrier)
  * breakeven    p = 52.4%, payout 0.908   (zero EV)

Usage: .venv/bin/python research/ruin_sim.py
"""
from __future__ import annotations

import numpy as np

N_SIMS = 200_000
START = 10.0
STAKE = 1.0
TARGET = 20.0
MAX_TRADES = 5_000


def sim(p: float, payout: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bal = np.full(N_SIMS, START)
    alive = np.ones(N_SIMS, dtype=bool)
    doubled = np.zeros(N_SIMS, dtype=bool)
    ruined = np.zeros(N_SIMS, dtype=bool)
    trades_to_end = np.zeros(N_SIMS, dtype=int)

    for t in range(MAX_TRADES):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        win = rng.random(len(idx)) < p
        bal[idx] += np.where(win, payout * STAKE, -STAKE)
        doubled[idx] = bal[idx] >= TARGET
        ruined[idx] = bal[idx] < STAKE
        done = doubled[idx] | ruined[idx]
        trades_to_end[idx[done]] = t + 1
        alive[idx[done]] = False

    # anything still alive at MAX_TRADES is neither doubled nor ruined
    n_d = int(doubled.sum())
    n_r = int(ruined.sum())
    n_a = int(alive.sum())
    return {
        "p": p, "payout": payout, "ev": p * payout - (1 - p),
        "p_double": n_d / N_SIMS, "p_ruin": n_r / N_SIMS,
        "p_open": n_a / N_SIMS,
        "med_trades_ruin": float(np.median(trades_to_end[ruined])) if n_r else float("nan"),
        "med_trades_double": float(np.median(trades_to_end[doubled])) if n_d else float("nan"),
    }


def main() -> None:
    print(f"start ${START:.2f}, stake ${STAKE:.2f}, target ${TARGET:.2f}, "
          f"{N_SIMS:,} simulations\n")
    print(f"{'scenario':<14}{'p':>6}{'payout':>8}{'EV/trade':>10}"
          f"{'P(double)':>11}{'P(ruin)':>9}{'P(open)':>9}"
          f"{'med trades->ruin':>18}")
    print("-" * 90)
    rows = [
        ("measured", 0.491, 0.908),
        ("optimistic", 0.550, 0.880),
        ("breakeven", 0.524, 0.908),
        ("dream 80%", 0.800, 0.200),
    ]
    for name, p, pay in rows:
        r = sim(p, pay, seed=hash(name) % 2**32)
        print(f"{name:<14}{r['p']*100:>5.1f}%{r['payout']:>8.3f}{r['ev']:>+10.4f}"
              f"{r['p_double']*100:>10.1f}%{r['p_ruin']*100:>8.1f}%"
              f"{r['p_open']*100:>8.1f}%{r['med_trades_ruin']:>18.0f}")

    print("\nAt the measured rate the expected loss is "
          f"{0.491*0.908-0.509:+.4f}/trade, so $10 survives about "
          f"{9/abs(0.491*0.908-0.509):.0f} trades on average (before variance).")
    print("At ~2.8 trades/hour that is roughly "
          f"{9/abs(0.491*0.908-0.509)/2.8/24:.1f} days of continuous trading.")
    print("\nNote the 'dream 80%' row: an 80% win rate at a 0.20 payout is still")
    print("a losing game. Accuracy and profitability are not the same thing.")
    print("Breakeven at 0.20 payout is 83.3%; at 0.88 it is 53.2%.")


if __name__ == "__main__":
    main()
