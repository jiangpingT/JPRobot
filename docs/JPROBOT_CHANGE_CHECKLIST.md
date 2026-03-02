# JPRobot 变更检查清单

> 目的：每次改动后做最小但完整的联动检查，避免“修了 A，漏了 B”。

## 1. 代码改动联动检查

- `jprobot/training/env.py` 有改动时，检查：
  - `jprobot/training/progressive.py`（课程配置是否匹配）
  - `scripts/training_server.py`（dashboard/viz 是否兼容新观测维度）
  - `TRAINING.md`（启动命令和说明是否更新）
- `jprobot/training/progressive.py` 有改动时，检查：
  - `TRAINING.md`（新增课程命令是否同步）
  - `scripts/train.sh`（默认流程是否需要调整）
- `scripts/training_server.py` 有改动时，检查：
  - `scripts/dev_dashboard.sh`（启动策略一致：优先前台）
  - `scripts/train.sh`（自动拉起和健康检查逻辑）

## 2. 启动流程检查（训练前必做）

- 启动 dashboard/viz（推荐前台）：
  - `conda run -n jprobot python scripts/training_server.py --port 18791`
- 验证端口监听：
  - `lsof -iTCP:18791 -sTCP:LISTEN -n -P`
- 验证页面可达：
  - `http://127.0.0.1:18791/dashboard`
  - `http://127.0.0.1:18791/viz`

## 3. 训练流程检查（训练中）

- 训练命令统一写日志（避免 dashboard 读到旧日志）
- 每次训练确认：
  - `trained/progressive_state.json` 正在更新
  - `trained/direction_eval.json` 更新时间前进
- 若 dashboard 显示异常（例如误报 100%）：
  - 先核对 `progressive_state.json` 再判断 UI 问题

## 4. 验收检查（训练后）

- 固定方向回放评估（四方向各 20 局）
- 备份当前最优模型（带时间戳）
- 记录本轮结论：可用能力 / 不足能力 / 下一轮计划

## 5. 变更交付格式（每次回复必须包含）

- 已修改文件清单
- 已验证命令与结果
- 未修改但已评估的关联文件

