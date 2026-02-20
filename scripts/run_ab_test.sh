#!/bin/bash
# A/B Test: Run original opencat-gym AND our fixed code simultaneously.
#
# Process A: Original opencat-gym (baseline) → /tmp/ab_test/baseline.log
# Process B: JPRobot with fixes (--curriculum simple) → /tmp/ab_test/jprobot.log
#
# Usage:
#   bash scripts/run_ab_test.sh          # Start both trainings
#   bash scripts/run_ab_test.sh stop     # Stop both trainings
#   bash scripts/run_ab_test.sh status   # Check status

set -e

LOG_DIR="/tmp/ab_test"
BASELINE_LOG="${LOG_DIR}/baseline.log"
JPROBOT_LOG="${LOG_DIR}/jprobot.log"
BASELINE_PID_FILE="${LOG_DIR}/baseline.pid"
JPROBOT_PID_FILE="${LOG_DIR}/jprobot.pid"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JPROBOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_DIR="/Users/mlamp/Workspace/opencat-gym"
CONDA_ENV="jprobot"

# Resolve conda activation
CONDA_BASE="$(conda info --base 2>/dev/null)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p "${LOG_DIR}"

stop_training() {
    echo "Stopping A/B test..."
    for pid_file in "${BASELINE_PID_FILE}" "${JPROBOT_PID_FILE}"; do
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if kill -0 "${pid}" 2>/dev/null; then
                echo "  Killing PID ${pid} ($(basename ${pid_file%.pid}))"
                kill "${pid}" 2>/dev/null || true
                # Also kill child processes
                pkill -P "${pid}" 2>/dev/null || true
            fi
            rm -f "${pid_file}"
        fi
    done
    echo "Done."
}

show_status() {
    echo "=== A/B Test Status ==="
    echo ""
    for name in baseline jprobot; do
        pid_file="${LOG_DIR}/${name}.pid"
        log_file="${LOG_DIR}/${name}.log"
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if kill -0 "${pid}" 2>/dev/null; then
                echo "${name}: RUNNING (PID ${pid})"
            else
                echo "${name}: DEAD (stale PID ${pid})"
            fi
        else
            echo "${name}: NOT STARTED"
        fi
        if [ -f "${log_file}" ]; then
            echo "  Log: ${log_file} ($(wc -l < "${log_file}") lines)"
            # Show last reward line
            last_reward=$(grep -o "ep_rew_mean[^|]*" "${log_file}" 2>/dev/null | tail -1 || true)
            if [ -n "${last_reward}" ]; then
                echo "  Last: ${last_reward}"
            fi
        fi
        echo ""
    done
}

case "${1:-start}" in
    stop)
        stop_training
        exit 0
        ;;
    status)
        show_status
        exit 0
        ;;
    start)
        ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        exit 1
        ;;
esac

# Check if already running
for pid_file in "${BASELINE_PID_FILE}" "${JPROBOT_PID_FILE}"; do
    if [ -f "${pid_file}" ]; then
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: Training already running (PID ${pid}). Use '$0 stop' first."
            exit 1
        fi
        rm -f "${pid_file}"
    fi
done

echo "============================================"
echo "  A/B Training Test"
echo "============================================"
echo ""
echo "  Process A (Baseline): original opencat-gym"
echo "    Log: ${BASELINE_LOG}"
echo ""
echo "  Process B (JPRobot):  our fixed code"
echo "    Log: ${JPROBOT_LOG}"
echo ""
echo "  Monitor:  python scripts/compare_logs.py"
echo "  Stop:     bash scripts/run_ab_test.sh stop"
echo "  Status:   bash scripts/run_ab_test.sh status"
echo "============================================"
echo ""

# === Process A: Baseline (original opencat-gym) ===
echo "Starting baseline training..."
(
    source "${CONDA_SH}" && conda activate "${CONDA_ENV}"
    export KMP_DUPLICATE_LIB_OK=TRUE
    cd "${BASELINE_DIR}"
    python train.py 2>&1
) > "${BASELINE_LOG}" 2>&1 &
BASELINE_PID=$!
echo "${BASELINE_PID}" > "${BASELINE_PID_FILE}"
echo "  Baseline PID: ${BASELINE_PID}"

# === Process B: JPRobot (our fixed code, simple curriculum) ===
echo "Starting JPRobot training..."
(
    source "${CONDA_SH}" && conda activate "${CONDA_ENV}"
    cd "${JPROBOT_DIR}"
    python -m jprobot.training.progressive --curriculum simple --auto 2>&1
) > "${JPROBOT_LOG}" 2>&1 &
JPROBOT_PID=$!
echo "${JPROBOT_PID}" > "${JPROBOT_PID_FILE}"
echo "  JPRobot PID: ${JPROBOT_PID}"

echo ""
echo "Both trainings started. Use 'bash scripts/run_ab_test.sh status' to check progress."
echo "Use 'python scripts/compare_logs.py' for real-time comparison."
echo ""

# Wait for both processes
wait
echo "Both trainings completed."
