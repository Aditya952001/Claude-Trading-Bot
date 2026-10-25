#!/usr/bin/env python3
"""
run_rise_fall.py - Dedicated runner for Adaptive Fixed Expiry Rise/Fall (CALL/PUT) Autonomous Trading.

Trades Dynamic Fixed Expiry digital options (1 to 10 minutes, adaptively calculated based
on market speed, ADX trend velocity, and ATR percentile) across all supported volatility,
jump, step, and forex instruments with fixed $1.00 stake for high payout (88-89%).
Contracts settle automatically at expiration with the broker.
"""
import sys
from main import main

if __name__ == "__main__":
    # Ensure --mode rise_fall is passed to main
    args = [sys.argv[0], "--mode", "rise_fall"] + [a for a in sys.argv[1:] if not a.startswith("--mode")]
    sys.argv = args
    main()
