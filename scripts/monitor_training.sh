#!/bin/bash
# 训练监控脚本：每5分钟检查一次训练状态，持续3小时
cd /Users/mlamp/Workspace/JPRobot
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/jprobot/bin/python
LOG=/Users/mlamp/Workspace/JPRobot/trained/monitor.log

for i in $(seq 1 36); do
  {
    echo ""
    echo "========== CHECK #$i at $(date '+%Y-%m-%d %H:%M:%S') =========="
    echo "--- Training Log ---"
    tail -20 /private/tmp/claude-501/-Users-mlamp-Workspace-JPRobot/tasks/train_XXXXXX.output 2>/dev/null \
      | grep -E "(ep_rew_mean|ep_len_mean|total_timesteps|fps)" || echo "No data"
    echo "--- Posture ---"
    cat trained/posture_eval.json 2>/dev/null || echo "No posture data"
    echo "--- Stage Progress ---"
    $PYTHON -c "
import json
with open('trained/progressive_state.json') as f:
    d = json.load(f)
cur = d.get('curriculum', 'default')
idx = d.get('curriculum_stage_idx', d.get('stage_idx', 0))
total = d.get('total_stages', 0)
steps = d.get('total_steps', 0)
stages = d.get('stages', [])
print(f'Mode: {cur}, Stage: {idx}/{total}, Total steps: {steps:,}')
if stages:
    last = stages[-1]
    print(f'Last completed: {last[\"name\"]}, reward={last[\"metrics\"][\"reward_final\"]:.0f}, healthy={last[\"healthy\"]}')
    ec = last.get('env_config', {})
    if ec:
        print(f'  fac_movement={ec.get(\"fac_movement\")}, fac_height_bonus={ec.get(\"fac_height_bonus\")}')
" 2>/dev/null || echo "No state data"
    echo "--- Process ---"
    ps aux | grep "jprobot.training.progressive" | grep -v grep \
      | awk '{print "PID="$2, "CPU="$3"%", "MEM="$4"%"}' || echo "PROCESS DEAD!"
    echo "--- Latest Snapshot ---"
    ls -lt trained/snapshots/*.zip 2>/dev/null | head -1 | awk '{print $6, $7, $8, $9}'
  } >> "$LOG" 2>&1
  [ $i -lt 36 ] && sleep 300
done
echo "========== MONITORING COMPLETE ==========" >> "$LOG"
