#!/bin/zsh
# Demo exploration session for the Rise/Fall bot.
# - Guard exploration window widened so the bot can actually collect data.
# - Stops gracefully after TARGET closed RF trades, or MAXSEC seconds.
# - Never trades live: DERIV_ENV=demo comes from .env.
set -u
cd /Users/aditya/Desktop/TYD/Claude_Ai_Trade_V2_EXT

export RF_EDGE_MIN_TRADES=1000
export RF_EDGE_SYMBOL_MIN_TRADES=20

TARGET=${TARGET:-4}
MAXSEC=${MAXSEC:-6000}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/rf_session_${STAMP}.log"
START=$(date +%s)

echo "session_start=$STAMP target=$TARGET maxsec=$MAXSEC guard_min=1000" >> "$LOG"

.venv/bin/python run_rise_fall.py >> "$LOG" 2>&1 &
PID=$!

stop_bot() {
  echo "drain_requested $(date +%H:%M:%S)" >> "$LOG"
  touch STOP_AFTER_TRADE
  local waited=0
  while kill -0 $PID 2>/dev/null && [ $waited -lt 600 ]; do
    sleep 10
    waited=$((waited + 10))
  done
  if kill -0 $PID 2>/dev/null; then
    echo "drain_timeout -> SIGINT" >> "$LOG"
    kill -INT $PID 2>/dev/null
    sleep 5
    kill -TERM $PID 2>/dev/null
  fi
}

while kill -0 $PID 2>/dev/null; do
  sleep 30
  N=$(sqlite3 data/bot_memory.sqlite3 \
      "SELECT count(*) FROM trades WHERE contract_type='rise_fall' AND status IN ('closed','sold') AND exit_time >= $START;" 2>/dev/null || echo 0)
  echo "progress $(date +%H:%M:%S) closed_since_start=$N" >> "$LOG"
  if [ "${N:-0}" -ge "$TARGET" ]; then
    echo "target_reached=$N" >> "$LOG"
    stop_bot
    break
  fi
  if [ $(( $(date +%s) - START )) -ge "$MAXSEC" ]; then
    echo "time_limit_reached" >> "$LOG"
    stop_bot
    break
  fi
done

wait $PID 2>/dev/null
RC=$?
rm -f STOP_AFTER_TRADE
if [ $RC -ne 0 ]; then
  echo "NOTE bot exit code $RC (check log for halt reason)" >> "$LOG"
fi
echo "SESSION_DONE log=$LOG trades_since_start=$(sqlite3 data/bot_memory.sqlite3 "SELECT count(*) FROM trades WHERE contract_type='rise_fall' AND status IN ('closed','sold') AND exit_time >= $START;" 2>/dev/null || echo '?')" >> "$LOG"
