---
active: false
iteration: 5
max_iterations: 5
completion_promise: null
started_at: "2026-03-01T15:00:00Z"
---
四足机器人后空翻落地站立优化循环（已完成）

目标：训练机器人在后空翻落地后能站稳

版本进展：
- V61：ep_len=100✅（post-success机制生效），W_POST_STAND=10太弱→机器人躺平不站
- V62：W_POST_STAND=25（加强），avg uprightness=0.75，viz step96仍躺平，梯度不足
- V63：W_POST_STAND=50（再翻倍），uprightness≈0.95但平趴骗分（无高度激励）
- V64：新增W_POST_HEIGHT=30（高度奖励），height_ratio=0.20，脚着地身体略抬起

最终版本：V64（最佳模型，trained/backflip_v64/full/best.zip）
结果：脚着地、身体水平（不再仰躺），但未完全站立（body高度0.052m vs 目标0.10m）

停止原因：5轮用完
