"""MoE Policy — Mixture of Experts 多模型路由评估

路由规则（接受现实，按方向特长路由）：
  向前          → Route A（BittleGymEnvV2，direction-reward，progress=0.00576 最快）
  向后/左/右     → Route B（BittleGymEnvVelocity，velocity-tracking，ep_len=251 不崩溃）

观测空间转换（Route A env 254-dim → Route B v2 250-dim）：
  Route A obs layout: [state_robot(6), lin_vel_xy(2), joint_history(240), target_dir(2), feet_contact(4)]
  Route B v2 obs layout: [state_robot(6), lin_vel_xy(2), joint_history(240), vel_cmd(2)]
  共享前 248 维（物理状态完全相同），将后 6 维替换为 scaled vel_cmd（2 维）。

用法：
  # 快速验证（5局/方向，约2分钟）
  python scripts/moe_policy.py --episodes 5

  # 完整评估（20局/方向，约8分钟）
  python scripts/moe_policy.py

  # 指定模型路径
  python scripts/moe_policy.py \\
      --route-a trained/route_a_v3/snapshots/best.zip \\
      --route-b trained/route_b/snapshots/best.zip

结果写入 trained/route_a_v3/moe_eval.json（格式与 fixed_direction_eval.json 兼容）。
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from jprobot.training.env_v2 import BittleGymEnvV2

# ── 常量 ──────────────────────────────────────────────────────────────────────

DIRECTIONS = ["forward", "backward", "left", "right"]

_DIR_CN = {
    "forward": "向前",
    "backward": "向后",
    "left": "向左",
    "right": "向右",
}

# Route B 用这些 vel_cmd 代表各方向（与 fixed_eval.py 保持一致）
_DIR_TO_VEL_CMD = {
    "forward":  np.array([ 0.25,  0.0], dtype=np.float32),
    "backward": np.array([-0.25,  0.0], dtype=np.float32),
    "left":     np.array([ 0.0,   0.25], dtype=np.float32),
    "right":    np.array([ 0.0,  -0.25], dtype=np.float32),
}

# Route B v2 (env_velocity.py) 的速度缩放系数
_VEL_CMD_SCALE_B = 2.5

# Route A old baseline（用于 diff 输出）
_ROUTE_A_BASELINE = {
    "forward":  {"reward": 1446, "progress": 0.00576, "ep_len": 251},
    "backward": {"reward": 155,  "progress": 0.00379, "ep_len": 42},
    "left":     {"reward": 1095, "progress": 0.00437, "ep_len": 251},
    "right":    {"reward": 1239, "progress": 0.00495, "ep_len": 251},
}


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def translate_obs_a_to_b(obs_a: np.ndarray, vel_cmd: np.ndarray, b_obs_dim: int) -> np.ndarray:
    """将 Route A 环境的 254-dim obs 转换为 Route B 模型期望的 obs。

    Route A env obs (254): [state(6), lin_vel(2), joints(240), target_dir(2), feet_contact(4)]

    Route B v2 (250-dim):  [state(6), lin_vel(2), joints(240), vel_cmd(2)]
      → obs_b = obs_a[0:248] + vel_cmd_scaled

    Route B v3/v4 (254-dim): [state(6), lin_vel(2), joints(240), vel_cmd(2), feet_contact(4)]
      → obs_b = obs_a[0:248] + vel_cmd_scaled + obs_a[250:254]
      （feet_contact 两个环境完全相同，直接复用）

    前 248 维（body state + lin_vel + joint_history）在两个环境中物理含义完全一致。
    """
    vel_cmd_scaled = np.clip(vel_cmd * _VEL_CMD_SCALE_B, -1.0, 1.0)
    if b_obs_dim == 254:
        # 254-dim: 替换 target_dir(2)，保留 feet_contact(4)
        return np.concatenate([obs_a[:248], vel_cmd_scaled, obs_a[250:254]]).astype(np.float32)
    else:
        # 250-dim: 丢弃 target_dir + feet_contact，只加 vel_cmd
        return np.concatenate([obs_a[:248], vel_cmd_scaled]).astype(np.float32)


def run_moe_eval(
    model_a_path: str,
    model_b_path: str,
    episodes_per_direction: int = 20,
    output_path: str = None,
) -> dict:
    """运行 MoE 评估：向前用 Model A，其余用 Model B（obs 转换后输入）。

    使用 Route A 环境（BittleGymEnvV2）作为统一物理仿真器：
    - 提供 254-dim obs（直接送给 Model A）
    - 提供方向进度信息（info["direction"]["avg_target_progress"]）
    - 当 Model B 控制时，obs 被转换为 250-dim 后送给 B
    """
    model_a_path = os.path.abspath(model_a_path)
    model_b_path = os.path.abspath(model_b_path)

    if not os.path.exists(model_a_path):
        raise FileNotFoundError(f"Route A 模型不存在: {model_a_path}")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(f"Route B 模型不存在: {model_b_path}")

    print(f"\n[MoE] Route A 模型: {model_a_path}")
    print(f"[MoE] Route B 模型: {model_b_path}")
    print(f"[MoE] 路由规则: 向前→A, 向后/左/右→B")
    print(f"[MoE] 每方向 {episodes_per_direction} 局\n")

    model_a = PPO.load(model_a_path)
    model_b = PPO.load(model_b_path)

    # 自动检测 Route B 模型期望的 obs 维度（250 或 254）
    b_obs_dim = model_b.observation_space.shape[0]
    print(f"[MoE] Route B obs 维度: {b_obs_dim}")

    print("[MoE] 创建 Route A 环境（DIRECT 模式）...")
    env = BittleGymEnvV2(render_mode=None)

    results = {}
    routing = {}

    for direction in DIRECTIONS:
        use_model_id = "A" if direction == "forward" else "B"
        routing[direction] = use_model_id
        vel_cmd = _DIR_TO_VEL_CMD[direction]

        print(f"[MoE] {_DIR_CN[direction]} ({direction}) → Model {use_model_id}, {episodes_per_direction} 局...")

        episode_rewards = []
        episode_lens = []
        posture_heights = []
        posture_arm_contacts = []
        target_progresses = []

        for ep in range(episodes_per_direction):
            obs, _ = env.reset(options={"target_name": direction})
            total_reward = 0.0
            ep_len = 0
            done = False

            while not done:
                if use_model_id == "A":
                    action, _ = model_a.predict(obs, deterministic=True)
                else:
                    obs_b = translate_obs_a_to_b(obs, vel_cmd, b_obs_dim)
                    action, _ = model_b.predict(obs_b, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                ep_len += 1
                done = terminated or truncated

                if done:
                    posture = info.get("posture", {})
                    if posture:
                        posture_heights.append(posture.get("avg_height", 0.0))
                        posture_arm_contacts.append(posture.get("arm_contact_rate", 0.0))

                    direction_info = info.get("direction", {})
                    if direction_info:
                        target_progresses.append(direction_info.get("avg_target_progress", 0.0))
                        episode_lens.append(direction_info.get("ep_len", ep_len))
                    else:
                        episode_lens.append(ep_len)

            episode_rewards.append(total_reward)
            print(f"  ep {ep + 1:2d}/{episodes_per_direction}: reward={total_reward:.0f}", end="\r")

        print()

        median_tp = float(np.median(target_progresses)) if target_progresses else 0.0
        mean_ep_len = float(np.mean(episode_lens)) if episode_lens else 0.0
        mean_rew = float(np.mean(episode_rewards))
        avg_height = float(np.mean(posture_heights)) if posture_heights else 0.0
        arm_contact = float(np.mean(posture_arm_contacts)) if posture_arm_contacts else 0.0

        results[direction] = {
            "episodes": episodes_per_direction,
            "model_used": use_model_id,
            "median_target_progress": round(median_tp, 6),
            "mean_ep_len": round(mean_ep_len, 1),
            "mean_episode_reward": round(mean_rew, 1),
            "arm_contact_rate": round(arm_contact, 3),
            "avg_height": round(avg_height, 4),
        }

        print(
            f"  {_DIR_CN[direction]} [Model {use_model_id}]: "
            f"reward={mean_rew:.0f}  progress={median_tp:.5f}  "
            f"ep_len={mean_ep_len:.0f}  ac={arm_contact:.1%}"
        )

    env.close()

    output = {
        "model_a": model_a_path,
        "model_b": model_b_path,
        "routing": routing,
        "episodes_per_direction": episodes_per_direction,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[MoE] 结果写入: {output_path}")

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MoE 验收结果（对比 Route A 旧版最佳）")
    print("=" * 70)
    print(f"  {'方向':4s}  {'模型':4s}  {'奖励':6s}  {'进度/步':10s}  {'ep_len':6s}  {'手臂触地':8s}")
    print(f"  {'----':4s}  {'----':4s}  {'------':6s}  {'----------':10s}  {'------':6s}  {'--------':8s}")

    for d in DIRECTIONS:
        r = results[d]
        base = _ROUTE_A_BASELINE[d]
        prog_diff = r["median_target_progress"] - base["progress"]
        diff_str = f"({'↑' if prog_diff >= 0 else '↓'}{abs(prog_diff):.5f})"
        print(
            f"  {_DIR_CN[d]:4s}  [  {r['model_used']}]  "
            f"{r['mean_episode_reward']:6.0f}  "
            f"{r['median_target_progress']:10.5f} {diff_str}  "
            f"{r['mean_ep_len']:6.0f}  "
            f"{r['arm_contact_rate']:8.1%}"
        )

    print("=" * 70)

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MoE Policy 评估：向前用 Route A，其余用 Route B"
    )
    parser.add_argument(
        "--route-a",
        default="trained/route_a_v3/snapshots/best.zip",
        help="Route A 模型路径（向前方向使用）",
    )
    parser.add_argument(
        "--route-b",
        default="trained/route_b/snapshots/best.zip",
        help="Route B 模型路径（向后/左/右方向使用）",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="每方向局数（默认 20，快速验证用 5）",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出 JSON 路径（默认：route_a 目录下的 moe_eval.json）",
    )
    args = parser.parse_args()

    # 默认输出路径：route_a 模型目录的上两级 + moe_eval.json
    if args.output is None:
        a_dir = os.path.dirname(os.path.abspath(args.route_a))  # .../snapshots
        run_dir = os.path.dirname(a_dir)                         # .../route_a_v3
        output_path = os.path.join(run_dir, "moe_eval.json")
    else:
        output_path = args.output

    run_moe_eval(
        model_a_path=args.route_a,
        model_b_path=args.route_b,
        episodes_per_direction=args.episodes,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
