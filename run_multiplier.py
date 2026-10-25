#!/usr/bin/env python3
"""
run_multiplier.py - Dedicated runner for Multiplier-Only Autonomous Trading.

Trades only Multiplier contracts with dynamic profit-locking (Green Shield,
trailing stops) and real-time loss caps (-$0.28).
"""
import sys
from main import main

if __name__ == "__main__":
    # Ensure --mode multiplier is passed to main
    args = [sys.argv[0], "--mode", "multiplier"] + [a for a in sys.argv[1:] if not a.startswith("--mode")]
    sys.argv = args
    main()
