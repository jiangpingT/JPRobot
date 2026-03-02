"""Progressive training orchestrator for BittleX.

Trains in cumulative stages: 1M → 5M → 10M → 50M → 100M steps.
After each stage an automatic health check decides whether to proceed.
All state is persisted so training can be stopped and resumed at any time.

Curriculum learning is also supported for multi-phase training (e.g.
crawl → stand → walk) where each phase uses different reward weights.

State file:  trained/progressive_state.json
Snapshots:   trained/snapshots/stage_1M.zip, stage_5M.zip, ...
Best model:  trained/snapshots/best.zip  (always current best)

Usage:
    # Start from scratch
    python -m jprobot.training.progressive

    # Resume where we left off (reads state file automatically)
    python -m jprobot.training.progressive --resume

    # Custom stages (cumulative targets in millions)
    python -m jprobot.training.progressive --stages 1 5 10 50

    # Skip health check and always proceed
    python -m jprobot.training.progressive --auto

    # Curriculum learning: stand then walk
    python -m jprobot.training.progressive --curriculum walk --auto
    python -m jprobot.training.progressive --curriculum walk --resume --auto
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from .train import train

# ── Cumulative step targets (M = million) ────────────────────────────────────
DEFAULT_STAGES = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]

# ── Curriculum definitions ───────────────────────────────────────────────────
# Each curriculum is a sequence of stages with env_config overrides.
# "target" is cumulative steps *within the curriculum* (not global).
#
# v5 — Bug-fixed, back to original opencat-gym simplicity.
# All 5 critical bugs fixed in env.py:
#   1. Smoothness uses normalized+squared (was raw degrees+abs → 100x too large)
#   2. arm_link_indices = {1,2,4,5} (was {0,1,4,5} — 0 is battery)
#   3. Termination threshold = 74.5° (was 50°)
#   4. arm_contact = cumulative count * 0.01 (was binary 2.0/step)
#   5. PENALTY_STEPS = 2M (was 100M)

def make_refine_curriculum(weakest: str) -> dict:
    """Generate a 2-stage refinement curriculum targeting the weakest direction.

    Stage 1 (0.5M): The weakest direction gets 0.5 probability; others share 0.5.
    Stage 2 (0.5M): Rebalance to uniform 0.25/0.25/0.25/0.25.
    Both stages continue from trained/snapshots/best.zip.

    Args:
        weakest: One of "forward", "backward", "left", "right".

    Returns:
        Curriculum dict compatible with run_curriculum().
    """
    directions = ["forward", "backward", "left", "right"]
    if weakest not in directions:
        raise ValueError(f"Unknown direction: {weakest!r}. Must be one of {directions}")

    idx = directions.index(weakest)
    remaining = 0.5 / 3
    probs = [remaining if i != idx else 0.5 for i in range(4)]

    return {
        "name": f"refine_{weakest}",
        "description": f"针对 {weakest} 方向的定向精化（从 best.zip 续训 1M 步）",
        "start_model": "trained/snapshots/best.zip",
        "stages": [
            {
                "name": f"focus-{weakest}-0.5M",
                "target": 500_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": probs,
                },
            },
            {
                "name": "rebalance-0.5M",
                "target": 1_000_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    }


def _resolve_env_class(name: str | None):
    """Return the env class for a given string name, or BittleGymEnv if None."""
    if name is None:
        return None  # train() defaults to BittleGymEnv
    if name == "BittleGymEnvV2":
        from .env_v2 import BittleGymEnvV2
        return BittleGymEnvV2
    if name == "BittleGymEnvVelocity":
        from .env_velocity import BittleGymEnvVelocity
        return BittleGymEnvVelocity
    if name == "BittleGymEnvVelocityV2":
        from .env_velocity_v2 import BittleGymEnvVelocityV2
        return BittleGymEnvVelocityV2
    raise ValueError(
        f"Unknown env_class: {name!r}. "
        f"Valid: BittleGymEnvV2, BittleGymEnvVelocity, BittleGymEnvVelocityV2"
    )


CURRICULA = {
    "simple": {
        "name": "simple",
        "description": "Original opencat-gym design: 2M steps produces stable walking",
        "start_model": None,
        "stages": [
            {"name": "Learn-2M", "target": 2_000_000, "ent_coef": 0.0, "env_config": {}},
        ],
    },
    "multidir_v1": {
        "name": "multidir_v1",
        "description": "Direction-conditioned single-policy locomotion (forward/backward/left/right)",
        "start_model": None,
        "stages": [
            {
                "name": "base-forward-0.5M",
                "target": 500_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "fixed",
                    "fixed_direction": "forward",
                },
            },
            {
                "name": "biased-mix-1.2M",
                "target": 1_200_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.5, 0.2, 0.15, 0.15],
                },
            },
            {
                "name": "uniform-mix-2.0M",
                "target": 2_000_000,
                "ent_coef": 0.01,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    "multidir_v2_refine": {
        "name": "multidir_v2_refine",
        "description": "Refine multidir policy with higher left/right/backward sampling to close weak directions",
        "start_model": "trained/snapshots/best.zip",
        "stages": [
            {
                "name": "weak-dir-focus-1.0M",
                "target": 1_000_000,
                "ent_coef": 0.01,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.2, 0.2, 0.3, 0.3],
                },
            },
            {
                "name": "uniform-rebalance-2.0M",
                "target": 2_000_000,
                "ent_coef": 0.01,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    "multidir_v3_right_refine": {
        "name": "multidir_v3_right_refine",
        "description": "Targeted right-direction refinement with short validation run",
        "start_model": "trained/snapshots/best.zip",
        "stages": [
            {
                "name": "right-focus-0.5M",
                "target": 500_000,
                "ent_coef": 0.01,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.15, 0.15, 0.2, 0.5],
                },
            },
            {
                "name": "rebalance-1.0M",
                "target": 1_000_000,
                "ent_coef": 0.01,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    "multidir_v4_allweak": {
        "name": "multidir_v4_allweak",
        "description": "同时改善三个弱方向（backward/left/right），显式保护 forward，从 best.zip 续训 2M 步",
        "start_model": "trained/snapshots/best.zip",
        "stages": [
            {
                "name": "allweak-focus-1.0M",
                "target": 1_000_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.20, 0.27, 0.27, 0.27],
                },
            },
            {
                "name": "rebalance-1.0M",
                "target": 2_000_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    # ── Route A v3 精调：旧 best(5M步) + only_positive_rewards + 向后定向修复 ──
    "route_a_v3_refine": {
        "name": "route_a_v3_refine",
        "description": (
            "Route A v3 精调：从旧 route_a 5M步 best.zip 出发，"
            "加 only_positive_rewards（修复向后 reward hacking）+ 向后方向强化训练。"
            "阶段1：backward=55% + ent=0.015 打破固化；"
            "阶段2：均匀重平衡 + ent=0.005 巩固。共 1M 步。"
        ),
        "env_class": "BittleGymEnvV2",
        "start_model": "route_a/snapshots/best.zip",  # 旧 route_a 的 5M 步模型
        "stages": [
            {
                "name": "backward-focus-0.5M",
                "target": 500_000,
                "ent_coef": 0.015,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.15, 0.55, 0.15, 0.15],  # backward=55%
                    "only_positive_rewards": True,
                },
            },
            {
                "name": "rebalance-1M",
                "target": 1_000_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                    "only_positive_rewards": True,
                },
            },
        ],
    },
    # ── Route A续训：从 route_a run 的 @best 继续 uniform 均匀训练 ───────────
    "env_v2_continue": {
        "name": "env_v2_continue",
        "description": "路线A续训：从 @best（本次run的最佳模型）继续均匀四方向 2.5M 步",
        "env_class": "BittleGymEnvV2",
        "start_model": "@best",   # resolved to trained/{run_id}/snapshots/best.zip
        "stages": [
            {
                "name": "continue-uniform-2.5M",
                "target": 2_500_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    # ── Route A: env_v2 (obs 248→254, +lin_vel_xy +feet_contact_state) ──────
    "env_v2_multidir": {
        "name": "env_v2_multidir",
        "description": "路线A：obs=254（+速度反馈+足端接触），从零训练，随机四方向",
        "env_class": "BittleGymEnvV2",
        "start_model": None,    # 与v1 obs 维度不同，必须从零开始
        "stages": [
            {
                "name": "base-forward-0.5M",
                "target": 500_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "fixed",
                    "fixed_direction": "forward",
                },
            },
            {
                "name": "biased-mix-1.5M",
                "target": 1_500_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.5, 0.2, 0.15, 0.15],
                },
            },
            {
                "name": "uniform-mix-2.5M",
                "target": 2_500_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    # ── Route B (original v1, 已知零速度骗分问题) ───────────────────────────
    "velocity_basic": {
        "name": "velocity_basic",
        "description": "路线B v1（已知缺陷：零速度命令导致机器人爬行骗分）",
        "env_class": "BittleGymEnvVelocity",
        "start_model": None,
        "stages": [
            {"name": "slow-vel-1.0M", "target": 1_000_000, "ent_coef": 0.01,
             "env_config": {"vel_cmd_range_x": [-0.2, 0.2], "vel_cmd_range_y": [-0.1, 0.1],
                            "vel_cmd_min_norm": 0.05}},
            {"name": "mid-vel-2.0M", "target": 2_000_000, "ent_coef": 0.005,
             "env_config": {"vel_cmd_range_x": [-0.35, 0.35], "vel_cmd_range_y": [-0.25, 0.25],
                            "vel_cmd_min_norm": 0.1}},
            {"name": "full-vel-3.0M", "target": 3_000_000, "ent_coef": 0.0,
             "env_config": {"vel_cmd_range_x": [-0.35, 0.35], "vel_cmd_range_y": [-0.25, 0.25],
                            "vel_cmd_min_norm": 0.1}},
        ],
    },
    # ── Route B v2: 修复版（严格tracking + forward bootstrap）──────────────
    "velocity_v2": {
        "name": "velocity_v2",
        "description": (
            "路线B v2：修复零速度骗分问题。"
            "阶段1固定前进指令（必须真正移动），阶段2随机前进命令，阶段3双向。"
            "tracking_sigma=0.15（更严格），arm_contact=0.05（更强爬行惩罚）。"
        ),
        "env_class": "BittleGymEnvVelocity",
        "start_model": None,  # 从零开始
        "stages": [
            {
                # Stage 1: fixed forward command — robot MUST actually move to get reward.
                # fixed_vel_cmd=[0.25, 0.0] means every episode commands 0.25 m/s forward.
                # A stationary robot gets exp(-0.0625/0.15) ≈ exp(-0.42) ≈ 0.66 per step
                # → only 66% of max reward, strong pressure to actually move.
                "name": "fixed-forward-0.5M",
                "target": 500_000,
                "ent_coef": 0.01,
                "env_config": {
                    "fixed_vel_cmd": [0.25, 0.0],
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.05,
                },
            },
            {
                # Stage 2: random forward-only commands (no backward/lateral yet).
                # All commands have positive vx ≥ 0.15 m/s, small vy allowed.
                "name": "forward-random-1.5M",
                "target": 1_500_000,
                "ent_coef": 0.005,
                "env_config": {
                    "vel_cmd_range_x": [0.15, 0.35],
                    "vel_cmd_range_y": [-0.10, 0.10],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.05,
                },
            },
            {
                # Stage 3: full bidirectional commands.
                "name": "bidirectional-2.5M",
                "target": 2_500_000,
                "ent_coef": 0.0,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.15,
                },
            },
        ],
    },
    # ── Route B 保守精调：向后方向改善（不破坏前进）────────────────────────────
    "route_b_v2_conservative": {
        "name": "route_b_v2_conservative",
        "description": (
            "Route B 保守精调：改善向后/侧向实际位移，同时保住向前性能。"
            "失败教训：fixed_vel_cmd=[-0.20,0] 导致 catastrophic forgetting（前进 progress 变负）。"
            "本版修复：用速度范围限制（vel_cmd_range_x=[-0.35,-0.10]）代替固定命令，"
            "让模型见到各种后退速度而非单一命令，避免过拟合。"
            "ent_coef=0.003（极低，只是轻推），sigma=0.12（稍严格），不剧烈改变策略。"
            "从 @best 续训 1M 步。"
        ),
        "env_class": "BittleGymEnvVelocity",
        "start_model": "@best",
        "stages": [
            {
                # Backward-range stage: only sample negative-vx commands.
                # Key difference from failed backward-bootstrap:
                #   - Range [-0.35, -0.10] gives VARIETY of backward speeds (not one fixed speed)
                #   - Robot must generalize across different backward magnitudes
                #   - Much less likely to cause catastrophic forgetting than fixed_vel_cmd
                # ent_coef=0.003: barely above zero, just enough to allow tiny policy updates
                # sigma=0.12: slightly stricter than training (0.15), reduces slow-speed cheating
                "name": "backward-range-0.5M",
                "target": 500_000,
                "ent_coef": 0.003,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, -0.10],
                    "vel_cmd_range_y": [-0.15, 0.15],
                    "vel_cmd_min_norm": 0.10,
                    "tracking_sigma": 0.12,
                    "fac_arm_contact": 0.05,
                },
            },
            {
                # Full rebalance: return to all directions.
                # sigma stays at 0.12 (stricter) to lock in the improvement.
                "name": "full-rebalance-0.5M",
                "target": 1_000_000,
                "ent_coef": 0.001,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.12,
                    "fac_arm_contact": 0.05,
                },
            },
        ],
    },
    # ── Route A 精调：向后方向专项修复 ─────────────────────────────────────────
    "route_a_v2_refine": {
        "name": "route_a_v2_refine",
        "description": (
            "Route A 精调：修复向后方向崩溃（ep_len=42，仅为最强方向11%）。"
            "高探索率（ent=0.015）打破前进步态固化，向后概率55%强制学习新步态，"
            "再重平衡恢复四方向均衡。从 @best 续训 1M 步。"
        ),
        "env_class": "BittleGymEnvV2",
        "start_model": "@best",
        "stages": [
            {
                "name": "backward-focus-0.5M",
                "target": 500_000,
                "ent_coef": 0.015,
                "env_config": {
                    "direction_mode": "random",
                    # backward=55%, forward=15%, left=15%, right=15%
                    "direction_probs": [0.15, 0.55, 0.15, 0.15],
                },
            },
            {
                "name": "rebalance-0.5M",
                "target": 1_000_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                },
            },
        ],
    },
    # ── Route A SOTA: 无前向 bootstrap，第1步全方向，episode=500，奖励截断 ──
    "env_v2_sota": {
        "name": "env_v2_sota",
        "description": (
            "Route A SOTA修复版：删除前向 bootstrap，第1步就全方向均匀采样。"
            "episode_length=500（SOTA的1/4，但远好于原来的250）。"
            "only_positive_rewards=True：reward<0截断为0，根治向后 reward hacking。"
            "阶段1高探索（ent=0.01），阶段2固化（ent=0.0）。共3M步。"
        ),
        "env_class": "BittleGymEnvV2",
        "start_model": None,
        "stages": [
            {
                "name": "alldir-explore-1.5M",
                "target": 1_500_000,
                "ent_coef": 0.01,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                    "episode_length": 500,
                    "only_positive_rewards": True,
                },
            },
            {
                "name": "alldir-refine-3M",
                "target": 3_000_000,
                "ent_coef": 0.0,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                    "episode_length": 500,
                    "only_positive_rewards": True,
                },
            },
        ],
    },
    # ── Route A SOTA v2: episode=700，从 env_v2_sota/best.zip 续训，目标 ep_len > 500 ──
    "env_v2_sota_v2": {
        "name": "env_v2_sota_v2",
        "description": (
            "Route A SOTA 续训：episode_length 500→700，从 env_v2_sota best.zip 热启动。"
            "目标：训练中 ep_len 突破 500 步（真正走路 12.5 秒不倒）。"
            "已有基础：env_v2_sota 达到 ep_len=466/500（93%），差最后 34 步。"
            "ent_coef=0.005（低探索，精化已有策略），共 4M 步。"
        ),
        "env_class": "BittleGymEnvV2",
        "start_model": "trained/route_a_sota/snapshots/best.zip",
        "stages": [
            {
                "name": "ep700-refine-4M",
                "target": 4_000_000,
                "ent_coef": 0.005,
                "env_config": {
                    "direction_mode": "random",
                    "direction_probs": [0.25, 0.25, 0.25, 0.25],
                    "episode_length": 700,
                    "only_positive_rewards": True,
                },
            },
        ],
    },
    # ── Route B SOTA: 无前向 bootstrap，sigma=0.25，episode=500 ─────────────
    "velocity_v4": {
        "name": "velocity_v4",
        "description": (
            "Route B SOTA修复版：删除前向 bootstrap，第1步全范围双向采样。"
            "sigma=0.25（回归 legged_gym 默认值，梯度更平滑）。"
            "episode_length=500。feet_air_time+alive bonus 保留（env_velocity_v2）。"
            "阶段1高探索（ent=0.01），阶段2固化（ent=0.0）。共3M步。"
        ),
        "env_class": "BittleGymEnvVelocityV2",
        "start_model": None,
        "stages": [
            {
                "name": "alldir-explore-1.5M",
                "target": 1_500_000,
                "ent_coef": 0.01,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.25,
                    "fac_arm_contact": 0.05,
                    "episode_length": 500,
                },
            },
            {
                "name": "alldir-refine-3M",
                "target": 3_000_000,
                "ent_coef": 0.0,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.25,
                    "fac_arm_contact": 0.05,
                    "episode_length": 500,
                },
            },
        ],
    },
    # ── Route B v4: 防爬行修复版（从 route_b_v3 best.zip 续训）────────────────
    # 核心修复（对比 route_b_SOTA 失败原因）：
    #   1. height_termination_threshold=0.04 → 躯体趴地立即终止（最关键！）
    #   2. fac_arm_contact: 0.03→0.08（提高 2.5 倍，接近 legged_gym 方向）
    #   3. fac_alive: 0.2→0.0（移除 alive bonus，防止"活着爬行=免费收益"）
    #   4. 保留 forward bootstrap（12 envs 必须先降低方差）
    "velocity_v4_fixed": {
        "name": "velocity_v4_fixed",
        "description": (
            "Route B v4：防爬行修复版。从 route_b_v3 best.zip 续训。"
            "核心修复：高度终止(0.04m) + 强接触惩罚(0.08) + 无 alive bonus。"
            "保留 forward bootstrap 降低 12 envs 的梯度方差。2 阶段共 2M 步。"
        ),
        "env_class": "BittleGymEnvVelocityV2",
        "start_model": "route_b_v3/snapshots/best.zip",
        "stages": [
            {
                # Stage 1：全方向均匀采样，高度终止 + 强接触惩罚生效
                # 去掉 alive bonus → 机器人必须靠真正移动拿奖励
                # 高度终止 → 趴地爬行立即 episode 结束，无法积累 vel_reward
                "name": "anticrash-uniform-1M",
                "target": 1_000_000,
                "ent_coef": 0.005,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.08,
                    "fac_alive": 0.0,
                    "height_termination_threshold": 0.04,
                },
            },
            {
                # Stage 2：固化，降低探索率
                "name": "refine-2M",
                "target": 2_000_000,
                "ent_coef": 0.0,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.08,
                    "fac_alive": 0.0,
                    "height_termination_threshold": 0.04,
                },
            },
        ],
    },
    # ── Route B v3: feet_air_time + alive bonus（SOTA 修复，从零训练）────────
    "velocity_v3": {
        "name": "velocity_v3",
        "description": (
            "路线B v3：引入 feet_air_time 奖励（SOTA 关键修复）+ alive bonus + obs=254。"
            "feet_air_time 梯度清晰（不抬腿就是零），不可被站立/慢速骗分。"
            "与速度追踪奖励互补，解决向后/侧向不移动问题。"
            "obs=254 与 Route A 对齐，为未来 MoE 融合做准备。"
            "课程结构同 velocity_v2（forward bootstrap → forward random → bidirectional）。"
        ),
        "env_class": "BittleGymEnvVelocityV2",
        "start_model": None,  # 从零开始，与旧网络不兼容
        "stages": [
            {
                # Stage 1: 固定前进命令 bootstrap。
                # 站着不动：vel_reward ≈ 0.66 满分 + air_time_reward = 0（脚不离地）
                # 真正前进：vel_reward ≈ 1.0 + air_time_reward > 0
                # 双重压力迫使机器人既要移动又要迈步。
                "name": "fixed-forward-0.5M",
                "target": 500_000,
                "ent_coef": 0.01,
                "env_config": {
                    "fixed_vel_cmd": [0.25, 0.0],
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.05,
                },
            },
            {
                # Stage 2: 随机前进命令（小 vy 允许），巩固步态。
                "name": "forward-random-1.5M",
                "target": 1_500_000,
                "ent_coef": 0.005,
                "env_config": {
                    "vel_cmd_range_x": [0.15, 0.35],
                    "vel_cmd_range_y": [-0.10, 0.10],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.05,
                },
            },
            {
                # Stage 3: 全方向双向命令。
                # feet_air_time 在所有方向均提供清晰梯度，
                # 机器人必须学会各方向的真实步态（不能靠站立骗速度奖励）。
                "name": "bidirectional-2.5M",
                "target": 2_500_000,
                "ent_coef": 0.0,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.15,
                },
            },
        ],
    },
    # ── Route B 精调：向后/侧向实际移动修复 ─────────────────────────────────────
    "route_b_v2_refine": {
        "name": "route_b_v2_refine",
        "description": (
            "Route B 精调：修复向后/侧向几乎不移动问题（progress≈0.0002）。"
            "复制 stage1 的固定命令bootstrap方式：固定向后命令 [-0.20,0] + 高探索率，"
            "然后扩展全范围速度命令，sigma=0.12（更严格，防止慢速骗分）。"
            "从 @best 续训 1.1M 步。"
        ),
        "env_class": "BittleGymEnvVelocity",
        "start_model": "@best",
        "stages": [
            {
                # Backward bootstrap: fixed backward command forces the robot to actually
                # move backward to get reward. High ent_coef explores new gait patterns.
                # Standing still: exp(-0.04/0.15) ≈ 0.76 per step → only 76% max reward.
                # Moving at 0.20 m/s backward: exp(0) = 1.0 → full reward.
                "name": "backward-bootstrap-0.4M",
                "target": 400_000,
                "ent_coef": 0.015,
                "env_config": {
                    "fixed_vel_cmd": [-0.20, 0.0],
                    "tracking_sigma": 0.15,
                    "fac_arm_contact": 0.05,
                },
            },
            {
                # Full rebalance: random velocity in all directions.
                # sigma=0.12 (stricter than training 0.15) to suppress slow-speed cheating.
                # min_norm=0.15 prevents zero-command exploitation.
                "name": "full-rebalance-0.7M",
                "target": 1_100_000,
                "ent_coef": 0.008,
                "env_config": {
                    "vel_cmd_range_x": [-0.35, 0.35],
                    "vel_cmd_range_y": [-0.25, 0.25],
                    "vel_cmd_min_norm": 0.15,
                    "tracking_sigma": 0.12,
                    "fac_arm_contact": 0.05,
                },
            },
        ],
    },
}

# ── Health check thresholds ───────────────────────────────────────────────────
HEALTH = {
    "min_reward_final": 0,       # final reward must be positive
    "min_ep_len": 25,            # robot must survive > 25 steps
    "max_trend_drop": -500,      # end-of-stage reward drop tolerance
                                 # NOTE: was -30 for 250-step episodes; raised to -500
                                 # for 500-step episodes where absolute reward is 2x larger
                                 # and normal PPO oscillation can exceed 300 points.
    "max_cross_stage_drop": -20, # final reward must not fall > 20 vs previous stage
}


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

def check_health(metrics: dict, prev_metrics: dict | None = None,
                  skip_cross_stage: bool = False) -> tuple[bool, list[str]]:
    """Evaluate whether a stage's training result is healthy.

    Args:
        skip_cross_stage: If True, skip cross-stage reward comparison.
            Use this when env_config changes between stages (reward scale differs).

    Returns (is_healthy, list_of_issues).
    """
    issues = []

    final = metrics.get("reward_final", 0)
    trend = metrics.get("reward_trend", 0)
    ep_len = metrics.get("ep_len_final", 0)

    if final < HEALTH["min_reward_final"]:
        issues.append(f"Reward negative: {final:.0f} < 0")

    if ep_len < HEALTH["min_ep_len"]:
        issues.append(f"Episode length too low: {ep_len:.0f} < {HEALTH['min_ep_len']}")

    if trend < HEALTH["max_trend_drop"]:
        issues.append(
            f"Reward declined within stage: end={metrics.get('reward_end', 0):.0f}"
            f" vs start={metrics.get('reward_start', 0):.0f} (Δ{trend:+.0f})"
        )

    if prev_metrics and not skip_cross_stage:
        prev_final = prev_metrics.get("reward_final", 0)
        drop = final - prev_final
        if drop < HEALTH["max_cross_stage_drop"]:
            issues.append(
                f"Reward dropped from previous stage: {prev_final:.0f} → {final:.0f}"
                f" (Δ{drop:+.0f})"
            )

    return len(issues) == 0, issues


# ─────────────────────────────────────────────────────────────────────────────
# Progressive trainer
# ─────────────────────────────────────────────────────────────────────────────

class ProgressiveTrainer:
    """Runs training in progressive stages with automatic health checks."""

    def __init__(self, parallel_envs: int = 8, seed: int = 42, run_id: str = None):
        base_trained = os.path.join(os.path.dirname(__file__), "..", "..", "trained")
        base_trained = os.path.abspath(base_trained)
        # If run_id is set, use trained/{run_id}/ as the isolated output directory.
        # This allows multiple runs (e.g. route_a, route_b) to run in parallel
        # without conflicting on state files and snapshots.
        if run_id:
            self.trained_dir = os.path.join(base_trained, run_id)
        else:
            self.trained_dir = base_trained
        os.makedirs(self.trained_dir, exist_ok=True)
        self.snapshot_dir = os.path.join(self.trained_dir, "snapshots")
        self.state_path = os.path.join(self.trained_dir, "progressive_state.json")
        self.parallel_envs = parallel_envs
        self.seed = seed

    # ── State persistence ────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            print(f"[Progressive] Loaded state: stage {state['stage_idx']}/{state['total_stages']}"
                  f", {state['total_steps']:,} steps so far")
            return state
        return {
            "stage_idx": 0,
            "total_stages": 0,
            "total_steps": 0,
            "best_model": None,
            "last_metrics": None,
            "stages": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_state(self, state: dict) -> None:
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ── Display helpers ───────────────────────────────────────────────────────

    def _print_stage_header(self, idx: int, total: int, name: str,
                            stage_steps: int, best_model: str | None) -> None:
        print(f"\n{'═' * 60}")
        print(f"  STAGE {idx}/{total}  →  {name} cumulative steps")
        print(f"  This stage:  {stage_steps / 1e6:.1f}M additional steps")
        src = best_model if best_model else "scratch"
        print(f"  Starting from: {src}")
        print(f"{'═' * 60}\n")

    def _print_health_report(self, name: str, metrics: dict,
                             healthy: bool, issues: list[str]) -> None:
        status = "HEALTHY ✓" if healthy else "UNHEALTHY ✗"
        bar = "─" * 60
        print(f"\n{bar}")
        print(f"  Stage {name} — {status}")
        print(f"  Reward :  final={metrics.get('reward_final', 0):.0f}"
              f"  best={metrics.get('reward_best', 0):.0f}"
              f"  trend={metrics.get('reward_trend', 0):+.0f}")
        print(f"  ep_len :  {metrics.get('ep_len_final', 0):.0f} steps")
        if issues:
            print(f"  Issues :")
            for issue in issues:
                print(f"    • {issue}")
        print(bar)

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self, stages: list[int] = None, auto_proceed: bool = False) -> None:
        """Execute progressive training.

        Args:
            stages: Cumulative step targets. Defaults to DEFAULT_STAGES.
            auto_proceed: If True, skip interactive prompt and always continue
                          to the next stage when health check passes.
        """
        if stages is None:
            stages = DEFAULT_STAGES

        state = self._load_state()
        state["total_stages"] = len(stages)
        state["planned_stages"] = stages
        self._save_state(state)  # persist plan immediately so dashboard knows current stage target
        start_idx = state["stage_idx"]

        if start_idx >= len(stages):
            print("[Progressive] All stages already completed.")
            return

        os.makedirs(self.snapshot_dir, exist_ok=True)

        for i, target in enumerate(stages[start_idx:], start=start_idx):
            stage_steps = target - state["total_steps"]
            stage_name = f"{target // 1_000_000}M"

            self._print_stage_header(
                idx=i + 1,
                total=len(stages),
                name=stage_name,
                stage_steps=stage_steps,
                best_model=state["best_model"],
            )

            # ── Train ────────────────────────────────────────────────────────
            _, metrics = train(
                total_timesteps=stage_steps,
                parallel_envs=self.parallel_envs,
                resume_from=state["best_model"],
                seed=self.seed,
                return_metrics=True,
                trained_dir=self.trained_dir,
            )

            # ── Stage snapshot ───────────────────────────────────────────────
            # best.zip is already updated by SnapshotCallback inside train().
            # Save an additional immutable stage snapshot.
            best_zip = os.path.join(self.snapshot_dir, "best.zip")
            stage_snap = os.path.join(self.snapshot_dir, f"stage_{stage_name}.zip")
            if os.path.exists(best_zip):
                shutil.copy2(best_zip, stage_snap)
                print(f"[Progressive] Stage snapshot → {stage_snap}")

            # ── Persist state ────────────────────────────────────────────────
            healthy, issues = check_health(metrics, state["last_metrics"])

            state["stage_idx"] = i + 1
            state["total_steps"] = target
            state["best_model"] = best_zip if os.path.exists(best_zip) else None
            state["last_metrics"] = metrics
            state["stages"].append({
                "name": stage_name,
                "target_steps": target,
                "stage_steps": stage_steps,
                "snapshot": stage_snap,
                "metrics": metrics,
                "healthy": healthy,
                "issues": issues,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_state(state)

            # ── Health report ────────────────────────────────────────────────
            self._print_health_report(stage_name, metrics, healthy, issues)

            if not healthy:
                print(
                    "\n[Progressive] Stopped due to unhealthy stage.\n"
                    "  Adjust hyperparameters or reward function, then resume:\n"
                    f"    python -m jprobot.training.progressive --resume\n"
                )
                return

            if i + 1 >= len(stages):
                print("\n[Progressive] All stages completed!")
                return

            # ── Decide whether to proceed ────────────────────────────────────
            next_name = f"{stages[i + 1] // 1_000_000}M"
            if not auto_proceed:
                answer = input(
                    f"\n  Proceed to next stage ({next_name} steps)? [Y/n]: "
                ).strip().lower()
                if answer == "n":
                    print("[Progressive] Stopped by user. Resume anytime with --resume.")
                    return

            print(f"\n[Progressive] Proceeding to stage {next_name}...\n")

    # ── Curriculum run ────────────────────────────────────────────────────

    def run_curriculum(self, curriculum: dict, auto_proceed: bool = False) -> None:
        """Execute a multi-phase curriculum (e.g. stand → walk).

        Each curriculum stage can have its own env_config and ent_coef.
        The start_model is used as the initial model for the first stage.
        """
        cur_name = curriculum["name"]
        stages = curriculum["stages"]
        start_model = curriculum.get("start_model")
        env_cls = _resolve_env_class(curriculum.get("env_class"))

        # Resolve start_model path (None = train from scratch)
        # Supports "@name" syntax to refer to a snapshot within this run's snapshot dir:
        #   "@best"                  → {snapshot_dir}/best.zip
        #   "@curriculum_foo_bar"    → {snapshot_dir}/curriculum_foo_bar.zip
        # (No ".zip" needed — added automatically if missing.)
        start_model_abs = None
        if start_model is not None:
            if start_model.startswith("@"):
                snap_name = start_model[1:]
                if not snap_name.endswith(".zip"):
                    snap_name += ".zip"
                start_model_abs = os.path.join(self.snapshot_dir, snap_name)
                if not os.path.exists(start_model_abs):
                    print(f"[Curriculum] ERROR: '@' snapshot not found: {start_model_abs}")
                    return
            else:
                start_model_abs = os.path.join(self.trained_dir, "..", start_model)
                start_model_abs = os.path.abspath(start_model_abs)
                if not os.path.exists(start_model_abs):
                    if os.path.exists(start_model):
                        start_model_abs = os.path.abspath(start_model)
                    else:
                        print(f"[Curriculum] ERROR: Start model not found: {start_model}")
                        print(f"  Tried: {start_model_abs}")
                        return

        state = self._load_state()

        # If resuming, skip completed stages
        start_idx = 0
        if state.get("curriculum") == cur_name and state.get("curriculum_stage_idx", 0) > 0:
            start_idx = state["curriculum_stage_idx"]
            print(f"[Curriculum] Resuming '{cur_name}' from stage {start_idx}/{len(stages)}")
        else:
            # Fresh curriculum start
            state = {
                "stage_idx": 0,
                "total_stages": len(stages),
                "total_steps": 0,
                "best_model": start_model_abs,  # None = from scratch
                "last_metrics": None,
                "stages": [],
                "curriculum": cur_name,
                "curriculum_stage_idx": 0,
                "curriculum_stages": stages,   # save stage defs for dashboard
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save_state(state)
            # Clear stale posture metrics from previous training
            for fname in (
                "posture_eval.json",
                "posture_eval_history.jsonl",
                "direction_eval.json",
                "direction_eval_history.jsonl",
            ):
                fpath = os.path.join(self.trained_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"[Curriculum] Cleared stale {fname}")
            print(f"[Curriculum] Starting '{cur_name}' with {len(stages)} stages")
            src = start_model_abs if start_model_abs else "scratch (random init)"
            print(f"  Base model: {src}")

        os.makedirs(self.snapshot_dir, exist_ok=True)

        prev_env_config = None
        for i, stage_def in enumerate(stages[start_idx:], start=start_idx):
            stage_name = stage_def["name"]
            env_config = stage_def.get("env_config", {})
            ent_coef = stage_def.get("ent_coef", 0.0)

            # Calculate steps for this stage
            prev_target = stages[i - 1]["target"] if i > 0 else 0
            stage_steps = stage_def["target"] - prev_target

            # Determine whether env_config changed (for skip_cross_stage)
            env_config_changed = (prev_env_config is not None
                                  and env_config != prev_env_config)

            resume_model = state["best_model"]

            print(f"\n{'═' * 60}")
            print(f"  CURRICULUM '{cur_name}' — STAGE {i + 1}/{len(stages)}: {stage_name}")
            print(f"  Steps: {stage_steps / 1e6:.1f}M (cumulative target: {stage_def['target'] / 1e6:.1f}M)")
            print(f"  ent_coef: {ent_coef}")
            print(f"  env_config: {env_config}")
            print(f"  Starting from: {resume_model}")
            if env_config_changed:
                print(f"  NOTE: env_config changed → skip cross-stage reward comparison")
            print(f"{'═' * 60}\n")

            # ── Train ────────────────────────────────────────────────────
            train_kwargs = dict(
                total_timesteps=stage_steps,
                parallel_envs=self.parallel_envs,
                resume_from=resume_model,
                seed=self.seed,
                return_metrics=True,
                env_config=env_config,
                ent_coef=ent_coef,
                trained_dir=self.trained_dir,
            )
            if env_cls is not None:
                train_kwargs["env_class"] = env_cls
            _, metrics = train(**train_kwargs)

            # ── Stage snapshot ───────────────────────────────────────────
            best_zip = os.path.join(self.snapshot_dir, "best.zip")
            stage_snap = os.path.join(self.snapshot_dir, f"curriculum_{cur_name}_{stage_name}.zip")
            if os.path.exists(best_zip):
                shutil.copy2(best_zip, stage_snap)
                print(f"[Curriculum] Stage snapshot → {stage_snap}")

            # ── Health check ─────────────────────────────────────────────
            healthy, issues = check_health(
                metrics,
                state["last_metrics"],
                skip_cross_stage=env_config_changed,
            )

            # ── Persist state ────────────────────────────────────────────
            state["curriculum_stage_idx"] = i + 1
            state["stage_idx"] = i + 1
            state["total_stages"] = len(stages)
            state["total_steps"] = stage_def["target"]
            state["best_model"] = best_zip if os.path.exists(best_zip) else resume_model
            state["last_metrics"] = metrics
            state["stages"].append({
                "name": stage_name,
                "target_steps": stage_def["target"],
                "stage_steps": stage_steps,
                "snapshot": stage_snap,
                "metrics": metrics,
                "healthy": healthy,
                "issues": issues,
                "env_config": env_config,
                "ent_coef": ent_coef,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_state(state)

            # ── Health report ────────────────────────────────────────────
            self._print_health_report(stage_name, metrics, healthy, issues)

            if not healthy:
                print(
                    f"\n[Curriculum] Stopped at stage '{stage_name}' due to unhealthy metrics.\n"
                    f"  Resume with: --curriculum {cur_name} --resume --auto\n"
                )
                return

            if i + 1 >= len(stages):
                print(f"\n[Curriculum] All stages of '{cur_name}' completed!")
                # Auto-run fixed evaluation after all stages finish
                best_zip = os.path.join(self.snapshot_dir, "best.zip")
                if os.path.exists(best_zip):
                    print("\n[Curriculum] 自动运行固定方向验收评估（deterministic=True）...")
                    try:
                        import importlib.util
                        _scripts_dir = os.path.join(
                            os.path.dirname(__file__), "..", "..", "scripts", "fixed_eval.py"
                        )
                        spec = importlib.util.spec_from_file_location(
                            "fixed_eval", os.path.abspath(_scripts_dir)
                        )
                        fixed_eval_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(fixed_eval_mod)
                        _eval_env_cls = _resolve_env_class(
                            curriculum.get("env_class", None)
                        )
                        _eval_output = os.path.join(
                            self.trained_dir, "fixed_direction_eval.json"
                        )
                        eval_result = fixed_eval_mod.run_fixed_eval(
                            model_path=best_zip,
                            episodes_per_direction=20,
                            env_class=_eval_env_cls,
                            output_path=_eval_output,
                        )
                        # Persist fixed_eval result in state
                        state["fixed_eval"] = eval_result
                        self._save_state(state)
                        weakest = eval_result.get("weakest_direction", "?")
                        rec = eval_result.get("recommendation", "")
                        print(f"\n[Curriculum] 验收完成。最弱方向：{weakest}，建议：{rec}")
                        if rec.startswith("refine_"):
                            print(f"  下一步：python -m jprobot.training.progressive "
                                  f"--curriculum {rec} --auto")
                    except Exception as exc:
                        print(f"[Curriculum] 验收评估失败（不影响训练结果）: {exc}")
                return

            # ── Decide whether to proceed ────────────────────────────────
            next_name = stages[i + 1]["name"]
            if not auto_proceed:
                answer = input(
                    f"\n  Proceed to next stage ({next_name})? [Y/n]: "
                ).strip().lower()
                if answer == "n":
                    print("[Curriculum] Stopped by user. Resume with --resume.")
                    return

            prev_env_config = env_config
            print(f"\n[Curriculum] Proceeding to stage {next_name}...\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Progressive PPO training: 1M → 5M → 10M → 50M → 100M steps"
    )
    parser.add_argument(
        "--stages", type=int, nargs="+",
        default=[s // 1_000_000 for s in DEFAULT_STAGES],
        metavar="M",
        help="Cumulative targets in millions (default: 1 5 10 50 100)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved progressive_state.json",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-proceed through all stages without interactive prompt",
    )
    _refine_choices = [f"refine_{d}" for d in ["forward", "backward", "left", "right"]]
    _all_curriculum_choices = list(CURRICULA.keys()) + _refine_choices
    parser.add_argument(
        "--curriculum", type=str, default=None,
        help=(
            "Run a named curriculum instead of default progressive stages. "
            "Static: " + ", ".join(CURRICULA.keys()) + ". "
            "Dynamic refinement: " + ", ".join(_refine_choices) + "."
        ),
    )
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-id", type=str, default=None, dest="run_id",
        help=(
            "Isolate output to trained/{run_id}/ so multiple runs can proceed in parallel. "
            "Example: --run-id route_a  →  trained/route_a/snapshots/, etc."
        ),
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="Print the stage definitions for the chosen curriculum and exit.",
    )
    args = parser.parse_args()

    # Validate curriculum name
    if args.curriculum and args.curriculum not in _all_curriculum_choices:
        parser.error(
            f"Unknown curriculum: {args.curriculum!r}. "
            f"Choose from: {', '.join(_all_curriculum_choices)}"
        )

    trainer = ProgressiveTrainer(parallel_envs=args.envs, seed=args.seed, run_id=args.run_id)

    # ── Curriculum mode ──────────────────────────────────────────────────
    if args.curriculum:
        # Resolve static vs dynamic (refine_*) curriculum
        if args.curriculum in CURRICULA:
            curriculum = CURRICULA[args.curriculum]
        else:
            # Dynamic: refine_<direction>
            direction = args.curriculum[len("refine_"):]
            curriculum = make_refine_curriculum(direction)

        # --list-stages: print and exit
        if args.list_stages:
            import json as _json
            print(f"\nCurriculum: {curriculum['name']}")
            print(f"Description: {curriculum['description']}")
            print(f"Env class:   {curriculum.get('env_class', 'BittleGymEnv (default)')}")
            print(f"Start model: {curriculum.get('start_model', 'scratch')}")
            print(f"\nStages:")
            for i, s in enumerate(curriculum["stages"], 1):
                print(f"  {i}. {s['name']} — {s['target'] / 1e6:.1f}M steps "
                      f"(ent_coef={s.get('ent_coef', 0.0)})")
                ec = s.get("env_config", {})
                if ec:
                    print(f"     env_config: {_json.dumps(ec)}")
            return
        if not args.resume:
            state_path = trainer.state_path
            if os.path.exists(state_path):
                os.remove(state_path)
                print(f"[Curriculum] Cleared previous state (fresh start).")
        trainer.run_curriculum(curriculum, auto_proceed=args.auto)
        return

    # ── Default progressive mode ─────────────────────────────────────────
    stages = [m * 1_000_000 for m in args.stages]

    if not args.resume:
        # Fresh start: wipe old state
        state_path = trainer.state_path
        if os.path.exists(state_path):
            os.remove(state_path)
            print("[Progressive] Cleared previous state (fresh start).")

    trainer.run(stages=stages, auto_proceed=args.auto)


if __name__ == "__main__":
    main()
