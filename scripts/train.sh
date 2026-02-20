#!/bin/bash
# JPRobot 一键训练脚本：启动渐进训练 + 自动重启 Dashboard
#
# 用法：
#   ./scripts/train.sh              # 从头开始
#   ./scripts/train.sh --resume     # 从上次断点续训
#   ./scripts/train.sh --auto       # 自动通过所有阶段（无需手动确认）
#   ./scripts/train.sh --resume --auto

set -e
cd "$(dirname "$0")/.."

PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/jprobot/bin/python
LOG_DIR=/private/tmp/claude-501/-Users-mlamp-Workspace-JPRobot/tasks

# ── 1. 停止旧 Dashboard ───────────────────────────────────────────────────────
echo "[train.sh] Stopping old dashboard server..."
lsof -ti :18791 | xargs kill -9 2>/dev/null || true
sleep 1

# ── 2. 启动训练（后台），获取日志文件名 ──────────────────────────────────────
echo "[train.sh] Starting progressive training..."
LOGFILE=$(mktemp "${LOG_DIR}/train_XXXXXX.output")

KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m jprobot.training.progressive \
    --auto "$@" > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "[train.sh] Training PID: $TRAIN_PID  Log: $LOGFILE"

# ── 3. 等训练输出第一行，再启动 Dashboard ───────────────────────────────────
sleep 5
echo "[train.sh] Starting dashboard server → http://127.0.0.1:18791/dashboard"
$PYTHON scripts/training_server.py --log "$LOGFILE" &
DASH_PID=$!
echo "[train.sh] Dashboard PID: $DASH_PID"

# ── 4. 等待训练结束 ──────────────────────────────────────────────────────────
wait $TRAIN_PID
echo "[train.sh] Training finished."
