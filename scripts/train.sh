#!/bin/bash
# JPRobot 一键训练栈脚本：训练 + Dashboard/Viz + 健康守护
#
# 用法：
#   ./scripts/train.sh --curriculum multidir_v3_right_refine --auto
#   ./scripts/train.sh --resume --curriculum multidir_v3_right_refine --auto
#
# 说明：
# - 训练进程和 Dashboard 分离
# - Dashboard 掉线会自动拉起，减少手动重启
# - 训练日志固定写入 ./log/train_*.log，便于追踪

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${JPROBOT_DASHBOARD_PORT:-18791}"
LOG_DIR="./log"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
DASH_LOG="/tmp/jprobot_training_server.log"

start_dashboard() {
  lsof -ti TCP:"$PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  conda run -n jprobot python scripts/training_server.py \
    --port "$PORT" \
    --log "$LOGFILE" > "$DASH_LOG" 2>&1 &
  DASH_PID=$!
  sleep 2
  if ! lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null; then
    echo "[train.sh] Dashboard failed to bind :$PORT, retry once..."
    kill "$DASH_PID" 2>/dev/null || true
    conda run -n jprobot python scripts/training_server.py \
      --port "$PORT" \
      --log "$LOGFILE" > "$DASH_LOG" 2>&1 &
    DASH_PID=$!
    sleep 2
  fi
  if ! lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null; then
    echo "[train.sh] ERROR: dashboard failed to start on :$PORT"
    echo "[train.sh] See: $DASH_LOG"
    exit 1
  fi
  echo "[train.sh] Dashboard PID: ${DASH_PID:-unknown}  URL: http://127.0.0.1:${PORT}/dashboard"
}

dashboard_alive() {
  curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${PORT}/dashboard"
}

echo "[train.sh] Log: $LOGFILE"
echo "[train.sh] Starting training..."
KMP_DUPLICATE_LIB_OK=TRUE conda run -n jprobot \
  python -m jprobot.training.progressive "$@" > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "[train.sh] Training PID: $TRAIN_PID"

echo "[train.sh] Starting dashboard/viz..."
start_dashboard

echo "[train.sh] Guard loop running (auto-restart dashboard if needed)..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if ! dashboard_alive; then
    echo "[train.sh] Dashboard unhealthy, restarting..."
    start_dashboard
  fi
  sleep 3
done

wait "$TRAIN_PID"
EXIT_CODE=$?
echo "[train.sh] Training finished with exit code: $EXIT_CODE"
echo "[train.sh] Final log: $LOGFILE"
exit "$EXIT_CODE"
