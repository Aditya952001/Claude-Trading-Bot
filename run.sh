#!/usr/bin/env bash
# run.sh - Helper for the Deriv autonomous trading bot
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "[ERROR] venv not found. Please run: make install" >&2
  exit 1
fi

MODE="${1:-}"

if [ -z "$MODE" ]; then
  echo "============================================================"
  echo "       Deriv Autonomous Trading Bot - Launcher             "
  echo "============================================================"
  echo "Select bot mode:"
  echo "  1) Multiplier Bot        (Trades only Multipliers)"
  echo "  2) Fixed Expiry Bot     (Trades only 3m Rise/Fall CALL/PUT)"
  echo "  3) Hybrid Bot (Recommended - Trades Both Concurrently)"
  echo "  4) Preflight Test        (Validate connections and symbols)"
  echo "  5) View Live Logs        (tail logs/bot.log)"
  echo "  6) Exit"
  echo "============================================================"
  read -r -p "Enter choice [1-6]: " choice
  case "$choice" in
    1) MODE="multiplier" ;;
    2) MODE="rise_fall" ;;
    3) MODE="hybrid" ;;
    4) MODE="preflight" ;;
    5) MODE="logs" ;;
    *) echo "Exiting."; exit 0 ;;
  esac
fi

case "$MODE" in
  multiplier)
    echo "Starting [1. Multiplier-Only Bot] (Fixed stake $1.00)..."
    "$PYTHON" run_multiplier.py "${@:2}"
    ;;
  rise_fall)
    echo "Starting [2. Fixed Expiry Rise/Fall Bot] (3m Expiry, Fixed stake $1.00)..."
    "$PYTHON" run_rise_fall.py "${@:2}"
    ;;
  hybrid)
    echo "Starting [3. Hybrid Bot - Multiplier + Rise/Fall] (Fixed stake $1.00)..."
    "$PYTHON" run_hybrid.py "${@:2}"
    ;;
  preflight)
    TARGET_MODE="${2:-hybrid}"
    echo "Running preflight validation for mode: $TARGET_MODE..."
    "$PYTHON" main.py --mode "$TARGET_MODE" --preflight
    ;;
  dry-run)
    TARGET_MODE="${2:-hybrid}"
    echo "Running dry-run simulation for mode: $TARGET_MODE..."
    "$PYTHON" main.py --mode "$TARGET_MODE" --dry-run
    ;;
  logs)
    tail -n 60 -f logs/bot.log
    ;;
  kill)
    touch KILL_SWITCH
    echo "🚨 Kill switch SET (KILL_SWITCH file created)."
    ;;
  unkill)
    rm -f KILL_SWITCH
    echo "✅ Kill switch CLEARED (KILL_SWITCH file removed)."
    ;;
  *)
    echo "Usage: $0 {multiplier|rise_fall|hybrid|preflight|dry-run|logs|kill|unkill}"
    exit 1
    ;;
esac
