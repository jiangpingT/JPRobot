# ADR-0006: 训练快照与断点续训系统

**状态**：已采纳
**日期**：2026-02-19
**作者**：JPRobot Team

---

## 背景

强化学习训练周期长（数十到数百 M 步），存在以下风险：
1. 训练崩溃（奖励突然退化）
2. 超参数需要中途调整
3. 系统意外中断（电源、内存不足）
4. 需要回退到历史最优模型

需要一套系统化的快照与续训机制，支持从任意历史节点恢复训练。

## 决策

实现三层快照系统：

```
trained/
├── snapshots/           # 里程碑快照（每个 stage 一个 + best.zip）
│   ├── stage_1M.zip
│   ├── stage_5M.zip
│   ├── stage_10M.zip
│   └── best.zip         # 历史最高奖励模型（实时更新）
├── checkpoints/         # 周期性检查点（每 2M 步自动保存）
│   └── ckpt_XXXXXXX_steps.zip
└── progressive_state.json  # 阶段进度状态
```

`progressive_state.json` 结构：
```json
{
  "stage_idx": 3,
  "total_stages": 5,
  "total_steps": 10000000,
  "best_model": "trained/snapshots/best.zip",
  "planned_stages": [1000000, 5000000, 10000000, 50000000, 100000000],
  "last_metrics": { "reward_final": 61.5, "ep_len_final": 63.96 },
  "stages": [...]
}
```

## 实现细节

### SnapshotCallback（`jprobot/training/train.py`）

```python
class SnapshotCallback(BaseCallback):
    def _on_step(self):
        reward = self.model.ep_info_buffer[-1]["r"]
        if reward > self.best_reward:
            self.best_reward = reward
            # 关键：路径必须显式包含 ".zip" 扩展名
            snapshot_path = os.path.join(self.snapshot_dir, name + ".zip")
            best_path = os.path.join(self.snapshot_dir, "best.zip")
            self.model.save(snapshot_path)
            shutil.copy2(snapshot_path, best_path)
```

**关键修复**：`model.save()` 传入路径时，SB3 会自动追加 `.zip`，但 `shutil.copy2()` 不会。若路径不含 `.zip`，copy 源文件找不到会报 `FileNotFoundError`。解决方案：路径始终显式包含 `.zip`，`model.save()` 检测到已有扩展名不再重复追加。

### 续训机制

```python
# --resume 参数触发从 best.zip 续训
if resume and os.path.exists(best_zip):
    model = PPO.load(best_zip, env=env)
else:
    model = PPO("MlpPolicy", env=env, ...)
```

### 路径修复（`os.path.abspath`）

`train.py` 位于 `jprobot/training/`，相对路径 `../../trained` 在不同 CWD 下解析结果不同：
```python
# 错误：依赖 CWD
trained_dir = "../../trained"

# 正确：相对于脚本位置的绝对路径
trained_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "trained")
)
```

## 理由

- **best.zip 策略**：始终保存历史最高奖励模型，即使后续训练退化，也能从最优点恢复
- **里程碑快照**：每个 stage 结束时保存，支持回退到任意阶段重新训练
- **分离 checkpoints**：周期性检查点不与快照混用，避免文件夹混乱
- **`.gitignore` 排除**：`.zip` 和 `snapshots/`、`checkpoints/` 均不提交（单个文件 ~150MB）

## 后果

**正面影响**：
- Stage 50M 崩溃后，成功从 `best.zip`（Stage 10M 末尾，reward=61）续训
- 调整 PENALTY_STEPS 后，新参数立即在续训中生效，无需重新训练前3阶段

**负面影响**：
- `progressive_state.json` 文件记录 `stage_idx`，手动续训时需确认该值正确
- 大量快照文件（~150MB 每个）消耗磁盘空间，`checkpoints/` 需定期清理

**已知问题（已修复）**：
- `SnapshotCallback` 中路径不含 `.zip` 导致 `FileNotFoundError`（见 ADR-0006 关键修复）
- `trained_dir` 在不同 CWD 下路径解析错误（通过 `os.path.abspath` 修复）
