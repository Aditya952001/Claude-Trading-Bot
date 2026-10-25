#!/usr/bin/env python3
"""
run_hybrid.py - Dedicated runner for Hybrid (Multiplier + Rise/Fall) trading.

Both strategies run at the same time, each in its own isolated book:
- No static symbol split. Every symbol that supports Multipliers is scanned by
  the Multiplier book; every symbol that supports Rise/Fall is scanned by the
  Rise/Fall book. Dual-capable symbols are watched by both.
- Ownership is dynamic: the book that opens first on a symbol holds it until
  the contract closes, so the two strategies never hold the same instrument at
  the same time. If both approve on the same bar, HYBRID_SIGNAL_PRIORITY
  decides which one takes the entry (default: multiplier).
- Each book keeps its own risk ledger: trades/day, daily drawdown, loss-streak
  halt, per-symbol cooldown and open positions. One book's losses cannot veto,
  halt or force-close the other.
- Direction comes from the shared confluence engine; execution, sizing,
  expiry/stop management stay strategy-specific.

Optional per-book allow-lists: HYBRID_MULT_SYMBOLS / HYBRID_RF_SYMBOLS.
Legacy single-ledger behaviour: HYBRID_ISOLATED_RISK=false.
"""
import sys
from main import main

if __name__ == "__main__":
    # Ensure --mode hybrid is passed to main
    args = [sys.argv[0], "--mode", "hybrid"] + [a for a in sys.argv[1:] if not a.startswith("--mode")]
    sys.argv = args
    main()
