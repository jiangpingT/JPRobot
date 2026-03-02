#!/bin/bash
# Stable dashboard/viz launcher for JPRobot.
# Default mode is foreground (recommended) to avoid silent background exits.

set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${JPROBOT_DASHBOARD_PORT:-18791}"
PID_FILE="/tmp/jprobot_dashboard.pid"
LOG_FILE="/tmp/jprobot_training_server.log"

start_fg() {
  echo "[dev_dashboard] Starting in foreground on :$PORT"
  exec conda run -n jprobot python scripts/training_server.py --port "$PORT"
}

start_bg() {
  echo "[dev_dashboard] Starting in background on :$PORT"
  nohup conda run -n jprobot python scripts/training_server.py --port "$PORT" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P || true
  echo "[dev_dashboard] PID file: $PID_FILE  log: $LOG_FILE"
}

stop_server() {
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  lsof -ti TCP:"$PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  echo "[dev_dashboard] Stopped :$PORT"
}

status_server() {
  lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P || true
}

cmd="${1:-start}"
case "$cmd" in
  start)
    start_fg
    ;;
  start-bg)
    start_bg
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_fg
    ;;
  status)
    status_server
    ;;
  *)
    echo "Usage: $0 {start|start-bg|stop|restart|status}"
    exit 1
    ;;
esac

