#!/usr/bin/env python3
"""后空翻模型验收评估脚本（确定性推理，无探索噪声）。

对标 fixed_eval.py 的设计：
  - deterministic=True 关闭探索噪声（训练时的噪声会掩盖真实能力）
  - 每个阶段训练结束后自动调用，写出 JSON 供对比分析
  - 支持与上次评估结果对比，打印变化趋势

指标体系（7 轮训练总结）：
  ┌─────────────────┬──────────────────────────────────────────────────┐
  │ 类别            │ 指标                                              │
  ├─────────────────┼──────────────────────────────────────────────────┤
  │ 真实后空翻证明  │ mean_max_rotation_deg（核心：旋转了多少度）        │
  │                 │ mean_max_height_m（核心：跳了多高，>0.12m=真起跳） │
  │                 │ pct_over_90/180/270deg（旋转里程碑分布）           │
  │                 │ success_rate（完整后空翻比例，终极目标）            │
  ├─────────────────┼──────────────────────────────────────────────────┤
  │ Gaming 检测     │ mean_ep_len（<50步=疑似 gaming）                  │
  │                 │ mean_liftoff_ang_vel（>6 rad/s=飙角速度 gaming）   │
  │                 │ mean_rot_at_landing_deg（<90°+高落地率=落地 gaming）│
  │                 │ rotation_std_deg（接近0=策略固化/deterministic lock）│
  └─────────────────┴──────────────────────────────────────────────────┘

用法：
    python scripts/fixed_eval_backflip.py                        # 评估 full/best.zip
    python scripts/fixed_eval_backflip.py --stage rotate         # 评估 rotate 阶段
    python scripts/fixed_eval_backflip.py --model path/to/model.zip
    python scripts/fixed_eval_backflip.py --episodes 5           # 快速 5 局
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRAINED_DIR = Path(__file__).parent.parent / "trained" / "backflip"

STAGE_ORDER = ["jump", "rotate", "land", "full"]


def run_backflip_eval(
    model_path: str | Path,
    stage: str = "full",
    episodes: int = 20,
    output_path: str | Path | None = None,
    disable_rsi: bool = True,
) -> dict:
    """对一个阶段的模型做确定性验收评估。

    Args:
        model_path: 模型 zip 路径。
        stage: 环境的 training_phase（决定哪些奖励项生效）。
        episodes: 评估局数。
        output_path: 结果 JSON 写入路径，None 则自动推断。

    Returns:
        结果 dict。
    """
    from stable_baselines3 import PPO
    from jprobot.training.env_backflip import BittleBackflipEnv

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    if output_path is None:
        output_path = model_path.parent / "fixed_eval.json"
    output_path = Path(output_path)

    print(f"[BackflipEval] 模型:    {model_path}")
    print(f"[BackflipEval] 阶段:    {stage}")
    print(f"[BackflipEval] 局数:    {episodes}")
    print(f"[BackflipEval] 输出:    {output_path}")

    model = PPO.load(str(model_path), device="cpu")
    env   = BittleBackflipEnv(render_mode=None, training_phase=stage, disable_rsi=disable_rsi)
    rsi_label = "禁用RSI（地面起跳）" if disable_rsi else "启用RSI（25%空中起步）"
    print(f"[BackflipEval] RSI:     {rsi_label}")

    # ── 每局统计变量 ──────────────────────────────────────────────────────
    ep_rewards       = []
    ep_lens          = []
    max_heights      = []
    max_rots         = []
    successes        = []
    launcheds        = []
    landeds          = []
    liftoff_ang_vels = []   # 腾空瞬间向后角速度（rad/s，正值=向后）
    rot_at_landings  = []   # 落地瞬间已积累旋转角（度）
    post_heights     = []   # v65新增：各成功局的post-success阶段平均body高度（m）
    post_feet_list   = []   # v65新增：各成功局的post-success阶段平均脚接触数

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward      = 0.0
        ep_len            = 0
        done              = False
        ep_max_h          = 0.0
        ep_max_rot        = 0.0
        ep_liftoff_ang_vel = None   # 本局腾空瞬间角速度（只取第一帧）
        ep_rot_at_landing  = None   # 本局落地瞬间旋转角
        ep_post_heights    = []     # v65新增：本局成功后各步body高度（m）
        ep_post_feet       = []     # v65新增：本局成功后各步脚接触数

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_len       += 1
            ep_max_h   = max(ep_max_h,   info.get("max_height_m", 0.0))
            ep_max_rot = max(ep_max_rot, info.get("rotation_deg", 0.0))

            # 捕获腾空瞬间角速度（info["liftoff_ang_vel_y"] 仅在腾空第一帧非 None）
            lv = info.get("liftoff_ang_vel_y")
            if lv is not None and ep_liftoff_ang_vel is None:
                ep_liftoff_ang_vel = -lv   # 转换为"向后"正值（-ang_vel[1]）

            # 捕获落地瞬间旋转角
            if info.get("just_landed") and ep_rot_at_landing is None:
                ep_rot_at_landing = info.get("rotation_deg", 0.0)

            # 收集成功后站稳阶段的身体高度和脚接触（v65新增）
            if info.get("success"):
                ep_post_heights.append(info.get("height_m", 0.0))
                ep_post_feet.append(info.get("n_feet", 0))

            done = terminated or truncated

        ep_rewards.append(total_reward)
        ep_lens.append(ep_len)
        max_heights.append(ep_max_h)
        max_rots.append(ep_max_rot)
        successes.append(info.get("success",  False))
        launcheds.append(info.get("launched", False))
        landeds.append(info.get("landed",    False))

        if ep_liftoff_ang_vel is not None:
            liftoff_ang_vels.append(ep_liftoff_ang_vel)
        if ep_rot_at_landing is not None:
            rot_at_landings.append(ep_rot_at_landing)
        if ep_post_heights:
            post_heights.append(float(np.mean(ep_post_heights)))
            post_feet_list.append(float(np.mean(ep_post_feet)))

        status = "完成" if info.get("success") else "未完成"
        lv_str = f"  lv={ep_liftoff_ang_vel:.1f}r/s" if ep_liftoff_ang_vel is not None else ""
        print(f"  ep {ep+1:2d}/{episodes}: "
              f"rew={total_reward:6.1f}  "
              f"rot={ep_max_rot:5.1f}°  "
              f"h={ep_max_h:.3f}m  "
              f"len={ep_len:3d}步"
              f"{lv_str}  "
              f"{status}")

    env.close()

    # ── 汇总指标 ──────────────────────────────────────────────────────────
    mean_rew     = float(np.mean(ep_rewards))
    mean_len     = float(np.mean(ep_lens))
    std_len      = float(np.std(ep_lens))
    mean_height  = float(np.mean(max_heights))
    mean_rot     = float(np.mean(max_rots))
    std_rot      = float(np.std(max_rots))
    success_rate = float(np.mean(successes))
    launch_rate  = float(np.mean(launcheds))
    landing_rate = float(np.mean(landeds))

    # 旋转里程碑分布（证明"真的在翻"的核心证据）
    pct_90  = float(np.mean([r >= 90  for r in max_rots]))
    pct_180 = float(np.mean([r >= 180 for r in max_rots]))
    pct_270 = float(np.mean([r >= 270 for r in max_rots]))

    # Gaming 检测指标
    mean_liftoff_vel   = float(np.mean(liftoff_ang_vels)) if liftoff_ang_vels else 0.0
    mean_rot_at_land   = float(np.mean(rot_at_landings))  if rot_at_landings  else 0.0
    mean_post_height   = float(np.mean(post_heights))     if post_heights     else 0.0
    mean_post_feet     = float(np.mean(post_feet_list))   if post_feet_list   else 0.0
    mean_height_ratio  = min(1.0, max(0.0, (mean_post_height - 0.04) / (0.10 - 0.04))) if post_heights else 0.0

    result = {
        "model": str(model_path),
        "stage": stage,
        "episodes": episodes,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            # ── 真实后空翻证明 ──────────────────────────────────
            "mean_episode_reward":   round(mean_rew,    1),
            "mean_ep_len":           round(mean_len,    1),
            "mean_max_rotation_deg": round(mean_rot,    1),   # 主指标
            "mean_max_height_m":     round(mean_height, 4),   # 主指标
            "success_rate":          round(success_rate, 3),  # 终极目标
            "launch_rate":           round(launch_rate,  3),
            "landing_rate":          round(landing_rate, 3),
            # 旋转里程碑分布
            "pct_over_90deg":        round(pct_90,  3),
            "pct_over_180deg":       round(pct_180, 3),
            "pct_over_270deg":       round(pct_270, 3),
            # ── Gaming 检测 ─────────────────────────────────────
            "ep_len_std":            round(std_len,          1),
            "rotation_std_deg":      round(std_rot,          1),
            "mean_liftoff_ang_vel":  round(mean_liftoff_vel, 2),  # 正值=向后（正常<6, gaming>8）
            "mean_rot_at_landing_deg": round(mean_rot_at_land, 1),  # <90°=落地gaming
            # ── V65 站立恢复评估指标 ────────────────────────────
            "mean_post_stand_height_m": round(mean_post_height, 4),  # 成功后站稳阶段平均body高度（m）
            "mean_post_height_ratio":   round(mean_height_ratio, 3), # 0=贴地(0.04m), 1=正常站立(0.10m)
            "mean_post_n_feet":         round(mean_post_feet, 2),    # 成功后平均脚接触数（满分4）
        },
        "assessment": _assess(stage, mean_rot, success_rate, launch_rate,
                               mean_len, mean_liftoff_vel, mean_rot_at_land),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[BackflipEval] 结果已写入 {output_path}")

    return result


def _assess(stage: str, mean_rot: float, success_rate: float,
            launch_rate: float, mean_len: float,
            mean_liftoff_vel: float, mean_rot_at_land: float) -> str:
    """根据阶段和指标给出评估结论，含 gaming 检测。"""
    # Gaming 检测优先
    gaming_flags = []
    if mean_len < 50 and launch_rate > 0.5:
        gaming_flags.append(f"ep_len={mean_len:.0f}步过短")
    if mean_liftoff_vel > 8.0:
        gaming_flags.append(f"起飞角速度={mean_liftoff_vel:.1f}rad/s过大")
    if mean_rot_at_land < 90 and (stage in ("land", "full")) and launch_rate > 0.5:
        gaming_flags.append(f"落地时旋转仅{mean_rot_at_land:.0f}°")
    if gaming_flags:
        return "疑似 Gaming: " + "，".join(gaming_flags)

    if stage == "jump":
        if launch_rate > 0.8:
            return "起跳阶段通过：80%+ 局能成功起跳，可进入 rotate 阶段"
        return "起跳阶段待改进：起跳率 %.0f%%，建议延长训练" % (launch_rate * 100)
    if stage == "rotate":
        if mean_rot > 180:
            return "旋转阶段通过：平均旋转 %.0f°（>180°），可进入 land 阶段" % mean_rot
        if mean_rot > 90:
            return "旋转阶段进展中：平均旋转 %.0f°，目标 >180°" % mean_rot
        return "旋转阶段待改进：平均旋转 %.0f°，目标 >180°" % mean_rot
    if stage == "land":
        if mean_rot > 270 and launch_rate > 0.5:
            return "落地阶段通过：旋转 %.0f° + 起跳率 %.0f%%，可进入 full 阶段" % (mean_rot, launch_rate * 100)
        return "落地阶段待改进：旋转 %.0f°，起跳率 %.0f%%" % (mean_rot, launch_rate * 100)
    # full
    if success_rate > 0.5:
        return "优秀：%.0f%% 局完整后空翻成功！" % (success_rate * 100)
    if success_rate > 0.1:
        return "初步可行：%.0f%% 局成功，建议继续训练" % (success_rate * 100)
    if mean_rot > 180:
        return "有进展：平均旋转 %.0f° 但未完整落地，继续 full 阶段" % mean_rot
    return "仍在学习阶段，建议延长训练或调整奖励权重"


def _bar(value: float, total: float, width: int = 24) -> str:
    """生成进度条字符串。"""
    filled = max(0, min(width, int(value / total * width)))
    return "█" * filled + "░" * (width - filled)


def _gaming_flag(condition: bool) -> str:
    return "[WARN]" if condition else "[OK]  "


def _print_summary(result: dict, prev: dict | None = None):
    """打印对比摘要，含 gaming 诊断和旋转里程碑分布。"""
    m = result["metrics"]
    stage = result["stage"]

    print("\n" + "=" * 60)
    print("  后空翻验收评估  [%s]  %s" % (stage, result["evaluated_at"]))
    print("=" * 60)

    # ── 与上次对比 ────────────────────────────────────────────────────────
    if prev and prev.get("metrics"):
        pm = prev["metrics"]
        print("  与上次对比 (%s):" % prev.get("evaluated_at", "?"))
        for key, label in [
            ("mean_episode_reward",   "平均奖励"),
            ("mean_ep_len",           "平均步长"),
            ("mean_max_rotation_deg", "平均最大旋转"),
            ("mean_max_height_m",     "平均最大高度"),
            ("success_rate",          "成功率"),
        ]:
            curr     = m.get(key, 0)
            prev_val = pm.get(key, 0)
            delta    = curr - prev_val
            arrow    = "↑" if delta >= 0 else "↓"
            print("    %12s: %s → %s  (%s%.3g)" % (label, prev_val, curr, arrow, abs(delta)))
        print()

    # ── 旋转量 & 成功率进度条 ───────────────────────────────────────────
    rot = m["mean_max_rotation_deg"]
    print("  旋转量  %s  %.1f° / 360°" % (_bar(rot, 360), rot))
    suc = m["success_rate"] * 100
    print("  成功率  %s  %.1f%%" % (_bar(suc, 100), suc))

    # ── 旋转里程碑分布（核心证据：真的在翻吗？）────────────────────────
    print()
    print("  ── 旋转里程碑分布（各局中，越过该角度的比例）──")
    p90  = m.get("pct_over_90deg",  0) * 100
    p180 = m.get("pct_over_180deg", 0) * 100
    p270 = m.get("pct_over_270deg", 0) * 100
    print("   >90°   %s  %.0f%%" % (_bar(p90,  100, 20), p90))
    print("  >180°   %s  %.0f%%" % (_bar(p180, 100, 20), p180))
    print("  >270°   %s  %.0f%%  ← 目标" % (_bar(p270, 100, 20), p270))

    # ── Gaming 诊断（每项单独判断）──────────────────────────────────────
    mean_len     = m.get("mean_ep_len", 0)
    std_len      = m.get("ep_len_std",  0)
    lv           = m.get("mean_liftoff_ang_vel", 0)
    rot_at_land  = m.get("mean_rot_at_landing_deg", 0)
    rot_std      = m.get("rotation_std_deg", 0)

    gaming_ep_len   = mean_len < 50 and m.get("launch_rate", 0) > 0.5
    gaming_liftoff  = lv > 8.0
    gaming_landing  = (rot_at_land < 90 and stage in ("land", "full")
                       and m.get("landing_rate", 0) > 0.3)
    deterministic   = rot_std < 5.0 and result["episodes"] >= 10

    print()
    print("  ── Gaming 诊断 ──────────────────────────────────────────")
    print("  %s ep_len:      %.0f ± %.0f 步   [正常 60-120, <50 = gaming]"
          % (_gaming_flag(gaming_ep_len), mean_len, std_len))
    print("  %s 起飞角速度:  %.1f rad/s       [正常 <6, >8 = W_SPIN gaming]"
          % (_gaming_flag(gaming_liftoff), lv))
    print("  %s 落地时旋转:  %.1f°            [需 >90° 才有落地奖励]"
          % (_gaming_flag(gaming_landing), rot_at_land))
    print("  %s 旋转一致性:  ±%.1f°           [<5° = 策略固化/deterministic lock]"
          % (_gaming_flag(deterministic), rot_std))

    # ── 站立恢复指标（V65 新增）──────────────────────────────────────────
    post_height_ratio = m.get("mean_post_height_ratio", 0.0)
    post_n_feet = m.get("mean_post_n_feet", 0.0)
    post_stand_h = m.get("mean_post_stand_height_m", 0.0)
    if post_stand_h > 0:
        print()
        print("  ── 站立恢复指标（V65）───────────────────────────────────")
        print("  后空翻后身高:  %s  %.4f m  (ratio=%.2f，目标≥0.09m)"
              % (_bar(post_height_ratio, 1.0, 20), post_stand_h, post_height_ratio))
        feet_flag = "[OK] ≥3脚着地" if post_n_feet >= 3 else "[WARN] 脚接触不足"
        print("  脚接触均值:    %.2f/4 脚  %s" % (post_n_feet, feet_flag))
        stand_ok = post_height_ratio >= 0.833
        print("  站立评估:      %s  (height_ratio≥0.833=达标，当前=%.3f)"
              % ("[PASS]" if stand_ok else "[FAIL]", post_height_ratio))

    # ── 基础指标 ─────────────────────────────────────────────────────────
    print()
    print("  ── 基础指标 ─────────────────────────────────────────────")
    print("  平均奖励:    %s" % m["mean_episode_reward"])
    print("  起跳高度:    %.4f m  %s" % (
        m["mean_max_height_m"],
        "[OK] 真实腾空" if m["mean_max_height_m"] > 0.15 else "[WARN] 高度过低"
    ))
    print("  起跳率:      %.1f%%" % (m["launch_rate"]  * 100))
    print("  落地率:      %.1f%%" % (m["landing_rate"] * 100))

    print()
    print("  评估结论:    %s" % result["assessment"])
    print("=" * 60)
    print()
    print("  【名词备注】")
    print("  旋转角度        机器人腾空后向后翻转的累计度数。目标>270°才算完整后空翻。")
    print("  腾空高度        起跳后离地最高点。>0.15m=真实起跳，<0.12m=没真正跳起来。")
    print("  起跳率          每局中机器人是否成功蹬地离地，100%=每局都跳了。")
    print("  落地率          起跳后是否落回地面，配合旋转角才有意义。")
    print("  成功率          完整后空翻比例（旋转>286°且安全落地），终极目标指标。")
    print("  每局步数        每局游戏平均持续多少步（1步≈25ms）。<50步=找到快速骗分捷径。")
    print("  起飞角速度      脚刚离地那一帧身体向后旋转的速度（rad/s）。正常<6，>8=在飙速度骗分。")
    print("  落地时旋转      落地瞬间已累计旋转了多少度。<90°+落地率高=没翻就落地骗落地分。")
    print("  旋转一致性      20局中旋转角的标准差。≈0°=所有局完全相同，策略死锁在局部最优。")
    print("  >90°/>180°/>270°比例  各局中越过该角度门槛的百分比。全0%=从未突破第一个里程碑。")
    print("  后空翻后身高    成功后站稳阶段平均body离地高度。0.04m=贴地，0.10m=正常站立，目标≥0.09m。")
    print("  脚接触均值      成功后站稳阶段平均接触地面的脚数（0-4）。≥3=能站稳，<2=可能仍在翻滚倒地。")
    print("  站立评估        height_ratio≥0.833（body≥0.09m）视为达标（PASS）。")
    print()


def main():
    parser = argparse.ArgumentParser(description="后空翻模型验收评估")
    parser.add_argument("--stage",    default="full", choices=STAGE_ORDER,
                        help="训练阶段（决定环境的 training_phase）")
    parser.add_argument("--model",    default=None,
                        help="模型路径（默认 trained/backflip/{stage}/best.zip）")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output",   default=None, help="JSON 输出路径")
    parser.add_argument("--with-rsi", action="store_true",
                        help="保留 RSI（随机空中初始化，默认禁用以评估真实地面起跳）")
    args = parser.parse_args()

    model_path = Path(args.model or str(TRAINED_DIR / args.stage / "best.zip"))
    output_path = args.output

    # 载入上次结果做对比（优先从 model 同目录读）
    auto_out = Path(output_path) if output_path else model_path.parent / "fixed_eval.json"
    prev = None
    if auto_out.exists():
        try:
            with open(auto_out) as f:
                prev = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    result = run_backflip_eval(model_path, args.stage, args.episodes, output_path,
                               disable_rsi=not args.with_rsi)
    _print_summary(result, prev)


if __name__ == "__main__":
    main()
