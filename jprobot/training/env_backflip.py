"""BittleX 后空翻专用 RL 环境。

为什么要单独写一个环境，不复用 env.py？

  步态环境（env.py）的核心设计围绕"持续向前移动"：
    - 终止条件：俯仰角 > 1.3rad（74.5°）立即 kill episode
    - 奖励：方向上的位移累积
    - episode 长度：250 步（约 12 秒）

  后空翻的物理需求完全不同：
    - 机器人必须翻转约 360°，中间会经历倒置（pitch = ±π）
      → 不能有俯仰终止条件
    - 奖励分为 4 个阶段：起跳 → 旋转 → 落地 → 站稳
    - episode 只需 ~120 步（约 3 秒）
    - 动作控制模式改为直接位置目标（非增量），支持大幅度快速运动

关节顺序（8 个自由度，与 env.py 一致）：
  [lf_shoulder, lf_knee, rf_shoulder, rf_knee,
   rb_hip, rb_knee, lb_hip, lb_knee]

    lf = left front（左前腿）  rf = right front（右前腿）
    rb = right back（右后腿）  lb = left back（左后腿）
    shoulder/hip = 靠近身体的关节（近端）
    knee = 远端关节

观测空间（23 维）：
  [0:4]   body_quaternion    机体四元数（描述空间姿态，不同于欧拉角，不会万向锁）
  [4:7]   body_ang_velocity  机体角速度 rad/s（检测旋转速率，是关键信号）
  [7:10]  body_lin_velocity  机体线速度 m/s（vz > 0 = 起跳中，< 0 = 下落中）
  [10]    height             离地高度 m（> 0.12 = 腾空）
  [11:19] joint_angles_norm  8 个关节角归一化到 [-1, 1]
  [19:23] foot_contacts      4 只脚的触地状态（1=接触，0=离地）

动作空间（8 维，连续 [-1, 1]）：
  直接映射到关节目标角度 [-110°, +110°]。
  与步态 env 的增量控制不同，这里是绝对目标位置，
  让 agent 能执行固件 bf 关键帧那样的大幅度快速运动。
"""

import math
import os

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


# ── 物理仿真参数 ──────────────────────────────────────────────────────────
EPISODE_LENGTH = 120        # 最长 120 步（约 3 秒）——后空翻远比行走快
JOINT_LIMIT    = 110        # 关节角度限制（度）
MOTOR_FORCE    = 4.5        # 电机力矩 N·m（v25: 4.5N·m 从零开始完整课程）
                            # v11: 3.0→4.0。v21-v22 实验：从 V19(4.0N·m) 热启动切换到 4.5N·m → 三轮均退步。
                            # 根因：4.0N·m 策略无法迁移到 4.5N·m 物理，不是 4.5N·m 本身有问题。
                            # 物理分析：4.0N·m → liftoff ω=14.24 r/s，height=0.232m，
                            # 理论旋转 245°，实际 V19 最佳 231°，永远到不了 286°（需 height≥0.271m）。
                            # v25 策略：4.5N·m 从 jump 阶段从零开始，让策略在新物理下原生学习，
                            # 期望 height 提升到 0.27m+，从而突破 286° 物理门槛。
                            # v23 恢复 4.0N·m 路线：V23(153.9°), V24(155.1°) 均退步，4.0N·m 路线确认到头。
                            # v10 结果：liftoff ω=6.67 rad/s，实际旋转 61.7°，理论 198°。
                            # 空中角动量仍在衰减（平均 ω≈2.1 rad/s，需要 3.0 rad/s 才能到 90°）。
                            # 提高到 4.0 N·m 目标：liftoff ω 达到 ~10 rad/s，
                            # 即使空中衰减，平均 ω 仍能达到 ~3.3 rad/s → 旋转 ≈100°，突破 90°。
                            # 安全性：W_SPIN_AT_LAUNCH_ONCE 已永久删除，无 gaming 激励，
                            # 力矩提升只会带来更大的真实起跳角动量，不会诱发 gaming。
MOTOR_VELOCITY = 15 * math.pi   # 最大关节角速度 rad/s（v17: 恢复 15π）
                                # v16 教训：20π 与 V15 策略（训练于 15π）产生物理错配，
                                # 4M 步不够重学，rotate eval 退步：100°→58°，liftoff ω=0。
                                # 15π 是 v15 验证有效的最佳值，恢复。
PHYSICS_STEPS  = 5          # 每个 env step 运行 5 次 PyBullet 物理仿真
                            # 步态 env 用 3 次；这里用 5 次让快速翻转的积分更精确
_DT = PHYSICS_STEPS / 240.0 # 每个 env step 的物理时间（s）：5次 × (1/240Hz) ≈ 0.0208s
                             # v15 新增：用于角速度积分追踪俯仰旋转（替代 Euler 角差分）

# ── 固件参考关键帧（弧度）────────────────────────────────────────────────
# 来自 skills_keyframes.json 的 bf 技能，已在真机验证能产生后空翻。
# 这 5 帧是 agent 要学习"什么时候做什么动作"的最好参考。
BF_FRAMES_RAD = np.deg2rad([
    [ 54, -18,  54, -18,   54, -18,  54, -18],  # 帧0: 蹲伏（蓄力）
    [ 10, 100,  10, 100,  -44,  50, -44,  50],  # 帧1: 起跳爆发
    [-16, -56, -16, -56,   52, 125,  52, 125],  # 帧2: 腾空收腿（减小转动惯量）
    [ 54, -44,  54, -44,   82, -50,  82, -50],  # 帧3: 伸腿准备落地
    [ 30,  30,  30,  30,   30,  30,  30,  30],  # 帧4: 站稳收势
])

# ── 奖励权重 ──────────────────────────────────────────────────────────────
W_JUMP        = 5.0   # 起跳阶段：奖励向上速度（v2→v3：3.0→5.0，v2 jump 0% 的直接修复）
W_ROTATION    = 20.0  # 旋转阶段：累计旋转量奖励（v2→v3：15.0→20.0，加强旋转信号）
                      # v1→v2：从 5.0 提升到 15.0。
                      # v1 训练结论：agent gaming 问题 —— 跳起后不旋转直接落地
                      # 能同时拿到 W_JUMP + W_LANDING，比费力旋转划算。
                      # W_ROTATION 必须明显压过 "jump+land 不旋转" 的总收益。
W_SPIN_AT_LAUNCH = 0.0  # v4 改动：设为 0（等效删除）。
                         # v3 教训：在地面蹬腿时奖励 backward_ang 可被地面 gaming——
                         # 机器人学会趴地后仰 82° 同时蹬腿，拿巨额奖励但从不起跳。
                         # v4 改为依赖腾空后里程碑奖励（MILESTONES_RAD）来引导旋转，
                         # 里程碑要求真实累积旋转量，无法在地面触发。
W_LANDING     = 2.0   # 落地阶段：奖励脚着地（v33: 10→2）
                      # v32 教训：W_LANDING=10 × 80步 = 800分，提前落地的吸引力压制了 W_SUCCESS=1000。
                      # 提前落地（222°落地）= 800分；成功（286°）= 1000+20×10=1200 → 差价仅200。
                      # v33 修复：W_LANDING=2，提前落地仅 160分；成功 ≈ 1040分；差价 880分，激励明显。
W_STABILITY   = 0.0   # v41 再次关闭每步站稳奖励（v40教训）。
                      # v40 失败复盘：W_STABILITY=2/步 × 60步 = 120分 持续收入。
                      #   agent 发现：落地259°(< 286°)→ 站稳60步拿W_STABILITY+W_LANDING=240分
                      #   vs 冲286°成功拿1000分但episode立即结束（后续0分）。
                      #   gaming占优：持续小收入 >> 一次性大奖（折扣后差距缩小）。
                      # v41 修复：W_STABILITY 绑定到 W_SUCCESS 的一次性时刻。
                      #   成功时: reward += W_SUCCESS × (1 + UPRIGHTNESS_BONUS × uprightness)
                      #   这样站稳是成功的"加分项"，而非独立的持续收入流，彻底杜绝gaming。
W_UPRIGHTNESS_BONUS = 1.5  # v44 调整：成功时直立姿态加成系数（v41=0.5, v42=2.0失败, v43=1.0 → v44=1.5）。
                             # 成功时总奖励 = W_SUCCESS × (1.0 + W_UPRIGHTNESS_BONUS × uprightness)
                             # uprightness = exp(-2*(pitch²+roll²)) ∈ [0,1]
                             # v42 教训：2.0 时 ep_rew=4239 超上限（直立理论=3000），agent 超旋转到343°后反转
                             #   落地倾斜从27.9°退步到35.1°（过激励破坏稳定性）。
                             # v43 结果：1.0，rotation=338.9°，success=100%，liftoff=10.18r/s。
                             # v44/v45 结论：策略固化，多轮实验 rotation 停在 332-344°，无法突破。
                             # v46 新增 W_ROT_COMPLETENESS，激励旋转到360°，而非停在286°成功门槛。
W_ROT_COMPLETENESS = 3000.0 # v48 提升：3000（v46=1000→v47=2000→v48=3000）+ 平方公式。
                             # v49 策略：公式基线从286°提升到350°（配合ROTATION_COMPLETE=350°门槛）。
                             # v57 调整：基线从350°→335°（扩大梯度覆盖区间，让342-350°区间有梯度）。
                             #   v56教训：基线350°时，rot@land=342-347°时completeness=0，梯度=0，agent无法感知方向。
                             #   新基线335°覆盖到当前rot@land位置：
                             #     342° 落地: (342-335)/25=0.28, completeness=0.078, 额外+234
                             #     347° 落地: (347-335)/25=0.48, completeness=0.230, 额外+691
                             #     355° 落地: (355-335)/25=0.80, completeness=0.640, 额外+1920
                             #     360° 落地: (360-335)/25=1.0, completeness=1.0, 额外+3000
                             # 防 gaming：超360°受 W_OVERROT=500 每步惩罚
W_LAND_TIMING = 2000.0      # v57 新增：落地时机一次性奖励（在just_landed时触发，独立于W_ROT_COMPLETENESS）。
                             # 直接给落地角度梯度，基线330°→360°（平方公式）。
                             # 激励结构（配合W_ROT_COMPLETENESS，总计强梯度）：
                             #   342°落地: 2000×((342-330)/30)²=2000×0.16=320 + W_ROT_COMPLETENESS=234 = 554
                             #   347°落地: 2000×((347-330)/30)²=2000×0.32=642 + W_ROT_COMPLETENESS=691 = 1333
                             #   355°落地: 2000×((355-330)/30)²=2000×0.694=1389 + W_ROT_COMPLETENESS=1920 = 3309
                             #   360°落地: 2000×1.0=2000 + W_ROT_COMPLETENESS=3000 = 5000
                             # 347°→360° 差价: 5000-1333=3667分，极强梯度，直接推动更晚落地
W_SUCCESS     = 1000.0 # 完成后空翻额外奖励（v30: 500→1000）
                       # v26 设为 500：首次 5% 成功。v27：pct>270°=80% 但仍只 5% 成功。
                       # 80% 能到 270°，但最后 16°（270°→286°）无法突破。
                       # 提前落地（237°）稳拿 W_LANDING*80=800，冒险冲 286° 才多 500。
                       # v30 修复：W_SUCCESS=1000，完整后空翻后 1000+800=1800 >> 提前着地=800。
                       # 差价 1000，应能显著推动 agent 主动冲击最后 16°。
W_LATERAL_FALL = 5.0  # 侧倒惩罚
W_POSE_GUIDE  = 4.0   # 关键帧姿态引导权重（v12: 6.0→4.0，恢复 v10 水平）
                      # v11 实验：6.0 仅改善 1.3°（61.7→63°），代价是限制探索自由度。
                      # v12 重点是 MOTOR_VELOCITY 翻倍，W_POSE_GUIDE 回归 4.0 避免过度约束。
W_NO_SPIN     = 0.5   # 腾空不旋转惩罚（每步）：阻止 agent 在空中"呆站"
                      # v1 问题：agent 腾空后原地僵直，等重力把自己拉下来
W_MILESTONE   = 3.0   # 旋转里程碑一次性奖励系数：v3 新增。
                       # 奖励真实累积旋转量（而非瞬时角速度），防振荡 gaming。
W_ROT_STEP    = 8.0   # 增量旋转奖励（v5 新增，v9: 2→8）：腾空时每新增一弧度旋转量给 W_ROT_STEP 奖励。
                       # 关键：只奖励历史最大旋转量的新增部分（单调），振荡无法 gaming。
                       # v9 大幅提升原因：agent 卡在 70-87°，第一个里程碑在 90°（π/2 ≈ 1.57 rad）。
                       # 旧值 2.0 时，从 80°→90° 增量奖励仅 2×0.17=0.34 分，梯度太弱。
                       # 新值 8.0 时，同样增量给 8×0.17=1.36 分，明显推动 agent 继续旋转。
                       # W_SPIN_AT_LAUNCH_ONCE（起飞一次性角速度奖励）已在 v9 完全删除：
                       # 8 轮实验证明任何线性正比于起飞角速度的奖励都会被 gaming，无解。
W_ANTI_ROLL   = 2.0   # 腾空防侧倒惩罚（v14 新增）：|roll| > 30° 时按角度大小惩罚。
                       # 动机：V13 ep_len 仅 35 步（最大 120），大量 episode 提前因侧倒终止。
                       # 侧倒原因：liftoff ω=15 r/s 的快速旋转产生陀螺进动 → 机体横向翻转。
                       # 惩罚公式：max(0, |roll|-0.52) × W_ANTI_ROLL（0.52 rad = 30° 容限）
W_OVERROT     = 500.0 # v37 新增：超旋转惩罚（每步 × 超过360°的弧度量）。
                       # v36 教训：W_OVERSPIN=5（阈值8r/s）物理上不可行！
                       #   物理约束：完成286°需要≥11.5 r/s（0.235m跳高+腾空时间）。
                       #   惩罚8r/s以上 → agent只能达到10r/s → 270.5°旋转 < 286° → 永远失败。
                       # v37 策略转变：不惩罚速度，惩罚超过360°的旋转量（每步）。
                       #   根因分析：V33-V35 gaming的本质不是"速度高"，而是"旋转过多"（412-432°=1.1-1.2圈）。
                       #   真正的后空翻只需旋转约360°，超出的是gaming。
                       # 激励结构：gaming(432°) → 超出72°=1.26rad，约2步×500=1260惩罚 vs W_SUCCESS=1000
                       #           净收益=-260（亏本！）→ 强烈阻止超旋转。
                       #   合理(300°) → 未超360°，惩罚=0，净收益=1000。
                       # 注意：agent必须到达286°才能成功，低于360°时无惩罚，完美激励区间。
ROTATION_TARGET = 2 * math.pi  # 360°，后空翻理想旋转目标（恰好一圈）

# ── 旋转里程碑（弧度）────────────────────────────────────────────────────
# 腾空阶段每越过一个角度门槛，给一次性奖励。
# 一次性 = 同一 episode 内不能重复拿（即使角度倒回再超过也无效），
# 彻底阻断"正转→反转→正转"振荡 gaming 路径。
MILESTONES_RAD = [math.pi / 2, math.pi, 3 * math.pi / 2]  # 90°, 180°, 270°
# v31 教训：增加 280° 里程碑导致 agent 重新探索，pct>270° 从 85% 崩溃到 0%。
# 280° 里程碑改变了奖励梯度，agent 找到新的局部最优（在 270° 前落地），放弃了 V30 建立的策略。
# v32 恢复到 V30 的里程碑设置，用降低学习率来精微调整。

# ── 状态阈值 ─────────────────────────────────────────────────────────────
HEIGHT_AIRBORNE    = 0.12   # m，高于此高度认为腾空
HEIGHT_LANDED      = 0.10   # m，低于此高度且曾腾空 = 落地
VZ_LAUNCH_THRESH   = 0.2    # m/s，向上速度超过此值认为已起跳
ROTATION_COMPLETE  = 5.0    # rad ≈ 286°（v51 恢复：350°→286°）。
                             # v50 失败复盘：ROTATION_COMPLETE=350°+V46热启(344.4°)→success=0%（落地323.8°<350°）。
                             # 根因：350°门槛过高，探索阶段agent在空中达到354.6°但落地时只有323.8°，永远不触发成功。
                             # v51策略：恢复宽松286°门槛，让V49的356.1°策略始终能成功（安全）；
                             #   靠W_ROT_COMPLETENESS公式(350°→360°, 3000分差价)提供向360°的梯度；
                             #   靠高ent_coef=0.01打破固化，探索突破356.1°。
MIN_BACKFLIP_STEPS = 60     # v39 教训：v38 将此设为0导致新gaming路径——agent在258.8°（<286°）落地，
                             # 收集 W_LANDING×80步持续奖励直到超时，比主动完成286°更划算。
                             # v38 失败：success=0%，rotation=270.7°，ep_len=120（全超时）。
                             # v39 恢复60步门槛（与v35/v37相同），同时保留 W_OVERROT=500 的旋转质量保障。
                             # v37 结果验证：318°旋转（<360°！），100%成功，ep_len=61=MIN+1（正常物理时序）。
                             # 物理验证：真实后空翻约需60步（10蓄力+21腾空+29着地稳定），门槛60恰好合理。
ROTATION_GATE      = math.radians(352)  # rad ≈ 352°，落地奖励的最低旋转要求（v55: 300°→352°）
                                        # v55 动机：V54 rot@land=347°（落地时只转了347°），
                                        # 落地后还要在地面转13°才到360.3°——说明机器人"转着就落了"，不够精准。
                                        # 提高到352°（高于347°落地点）：agent必须等转到352°才有落地奖励，
                                        # 自然推迟落地时机，让rot@land从347°→355°+。
                                        # 安全性：V54策略360.3°远高于352°门槛，热启初期落地奖励仍可触发。
                                        # 风险：若探索期退步到<352°落地，W_LANDING暂时消失（仅2分/步，影响小）。


class BittleBackflipEnv(gym.Env):
    """BittleX 后空翻专用 Gym 环境。

    核心设计思路：
      1. 允许完整翻转（不因 pitch > 90° 终止，只因侧倒 roll > 90° 终止）
      2. 密集奖励（每步都有信号），帮助 PPO 探索
      3. 状态机追踪翻转进度，给对应阶段的奖励
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, training_phase="full", disable_rsi=False):
        """
        training_phase 控制哪些奖励项生效（课程学习用）：
          "jump"   : 只奖励起跳高度（第一阶段）
          "rotate" : 奖励起跳 + 旋转（第二阶段）
          "land"   : 奖励起跳 + 旋转 + 落地（第三阶段）
          "full"   : 所有奖励 + 成功 bonus（第四阶段/最终）

        disable_rsi: True = 关闭随机空中初始化（评估时用，确保从地面起跳）
        """
        super().__init__()

        self.render_mode = render_mode
        self.training_phase = training_phase
        self.disable_rsi = disable_rsi

        obs_dim = 4 + 3 + 3 + 1 + 8 + 4  # = 23
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.robot_id       = None
        self.physics_client = None
        self.joint_id       = []
        self._bound_ang     = np.deg2rad(JOINT_LIMIT)

        # 每 episode 的动态状态（在 reset() 中清零）
        self.step_counter          = 0
        self._launched             = False   # 是否已起跳（vz 超过阈值）
        self._airborne             = False   # 是否腾空（高度超过阈值）
        self._max_height           = 0.0     # 本 episode 最大高度（评估起跳质量）
        self._pitch_accumulated    = 0.0     # 累计俯仰旋转量（rad），连续累加不截断
        self._prev_pitch           = 0.0     # RSI 初始化用：episode 开始时的俯仰角（v15: step() 不再使用，仅 reset()）
        self._landed_after_launch  = False   # 起跳后是否已着地
        self._success              = False   # 是否完成完整后空翻
        self._milestones_hit          = set()   # 本 episode 已触发的旋转里程碑集合（防重复奖励）
        self._prev_max_rot_airborne   = 0.0    # 腾空后已达到的最大累积旋转量（rad，单调递增）
        self._rsi_actual_pitch        = 0.0    # RSI 真实俯仰角（rad），v18：超过 ±π 时 Euler 回读错误，用存储值

        self.urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "bittle_esp32.urdf"
        )

    # ── 重置 ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 首次调用时建立 PyBullet 连接（之后复用）
        if self.physics_client is None:
            mode = p.GUI if (self.render_mode == "human") else p.DIRECT
            self.physics_client = p.connect(mode)

        # 完全重建物理场景（与 env.py 一致，避免残留碰撞状态）
        p.resetSimulation(physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                                   physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        self.robot_id = p.loadURDF(
            self.urdf_path, [0, 0, 0.08],
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

        # 动态发现 REVOLUTE 关节（不硬编码索引，参见血泪教训 #1）
        self.joint_id = []
        for j in range(p.getNumJoints(self.robot_id,
                                       physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, j,
                                  physicsClientId=self.physics_client)
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_id.append(j)
                p.changeDynamics(
                    self.robot_id, j,
                    maxJointVelocity=MOTOR_VELOCITY,
                    physicsClientId=self.physics_client,
                )

        # RSI（Random State Initialization，随机空中初始化，v6 新增）：
        # 原理：每次从地面起跳，agent 很难"见到"旋转中段（60-150° 这段最难区间），
        # RSI 让 agent 直接从空中随机状态开始练习，极大提升探索效率。
        # v9 改动：扩展到 full 阶段（v6-v8 只在 rotate，导致 full 策略完全固化 ±0°）。
        # v18 改动：RSI 概率 50%→25%，防止 final.zip 过拟合 RSI 起始状态。
        # v17 教训：50% RSI 导致 agent 学会在 RSI 起步时策略完美（rew=1096，len=120），
        # 地面起步反而 1 步倒下——策略为 RSI 分布过度优化，完全遗忘地面起步行为。
        _use_rsi = (not self.disable_rsi and self.training_phase in ("rotate", "full") and np.random.random() < 0.25)

        if _use_rsi:
            self._rsi_init()
        else:
            # 标准站立初始化
            init_rad = np.deg2rad([50, 0, 50, 0, 50, 0, 50, 0])
            for i, jid in enumerate(self.joint_id[:8]):
                p.resetJointState(self.robot_id, jid, init_rad[i],
                                  physicsClientId=self.physics_client)
            # 让机器人在初始姿态下稳定（避免起步时有抖动速度）
            for _ in range(10):
                p.stepSimulation(physicsClientId=self.physics_client)

        # 重置 episode 状态变量
        self.step_counter         = 0
        self._launched            = False
        self._airborne            = False
        self._max_height          = 0.0
        self._landed_after_launch = False
        self._success             = False
        self._milestones_hit         = set()
        self._prev_max_rot_airborne  = 0.0

        _, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        euler = p.getEulerFromQuaternion(orn)
        self._prev_pitch        = euler[1]
        self._pitch_accumulated = 0.0

        # RSI 后处理：将状态变量覆盖为"已在空中旋转中"
        if _use_rsi:
            pos, _ = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.physics_client
            )
            self._launched            = True
            self._airborne            = True
            self._max_height          = pos[2]
            self._pitch_accumulated   = self._rsi_actual_pitch  # v18: 用 _rsi_init() 存储的真实值
                                                                  # 不再从 Euler 回读：pitch > ±π 时 getEulerFromQuaternion
                                                                  # 返回的是等效角而非实际累积旋转量（如 -240°→+120°），
                                                                  # 会把 240° 倒转标记成 0° 倒转，完全破坏里程碑初始化。
            # v13: 只追踪向后旋转量（pitch 为负 = 向后翻转）
            backward_rot_init = max(0.0, -self._pitch_accumulated)
            self._prev_max_rot_airborne = backward_rot_init
            for m in MILESTONES_RAD:
                if backward_rot_init >= m:
                    self._milestones_hit.add(m)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                   physicsClientId=self.physics_client)
        return self._get_obs(), {}

    # ── 步进 ──────────────────────────────────────────────────────────────

    def step(self, action):
        was_airborne_before = self._airborne          # 腾空边沿检测（v6 腾空第一帧奖励用）
        was_landed_before   = self._landed_after_launch  # 落地边沿检测（eval 指标用）
        # 动作 → 关节目标角度（直接位置控制，非增量）
        # action ∈ [-1, 1] → target ∈ [-110°, +110°]
        target_angs = np.clip(action, -1.0, 1.0) * self._bound_ang

        # 高力矩位置控制（后空翻需要爆发力）
        # v14 实验：腾空期间 force=0（零力矩）—— 未改善旋转量（65°→63°）。
        # v15 恢复：全程 MOTOR_FORCE，让 agent 可在空中主动 tuck 腿（缩短转动惯量，加速旋转）。
        p.setJointMotorControlArray(
            self.robot_id, self.joint_id[:8],
            p.POSITION_CONTROL,
            targetPositions=target_angs,
            forces=np.full(8, MOTOR_FORCE),
            physicsClientId=self.physics_client,
        )

        # 运行 PHYSICS_STEPS 次物理仿真（比步态 env 更多，积分更精确）
        for _ in range(PHYSICS_STEPS):
            p.stepSimulation(physicsClientId=self.physics_client)

        # 读取机体状态
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )
        euler  = p.getEulerFromQuaternion(orn)
        height = pos[2]
        pitch  = euler[1]   # 绕 Y 轴旋转 = 俯仰（后空翻的主旋转轴）
        roll   = euler[0]   # 绕 X 轴旋转 = 横滚（侧倒）

        # ── 追踪俯仰积累量（v15: 角速度积分，替代 Euler 角差分）─────────
        # 旧方法（v1-v14）的致命缺陷：Euler 角在 pitch=±90° 处发生万向锁（Gimbal Lock）。
        # ZYX Euler 分解中，pitch=90° 时 roll/yaw 退化，d_pitch = pitch - prev_pitch
        # 在后空翻的第一个四分之一转（0→90°）末尾产生错误跳变，破坏奖励信号，
        # 导致 agent 永远无法在 60-70° 附近获得清晰梯度 → 旋转"卡墙"。
        #
        # 新方法：对世界坐标系 Y 轴角速度（ang_vel[1]）做积分。
        # ang_vel 来自 p.getBaseVelocity()，在世界坐标系下表达，不受 Euler 奇异性影响。
        # 向后翻转 = ang_vel[1] < 0（世界 Y 轴负方向旋转）→ 积累负值，符合原有约定。
        self._pitch_accumulated += ang_vel[1] * _DT

        # ── 更新阶段状态机 ────────────────────────────────────────────────
        vz = lin_vel[2]  # 垂直速度（正 = 向上）
        if height > HEIGHT_AIRBORNE:
            self._airborne  = True
            self._max_height = max(self._max_height, height)
        if vz > VZ_LAUNCH_THRESH and not self._airborne:
            self._launched  = True
        if self._launched and self._airborne and height < HEIGHT_LANDED:
            self._landed_after_launch = True

        # 判断成功：向后旋转量足够 + 落地 + 至少经历 MIN_BACKFLIP_STEPS 步
        # v34 新增 step_counter 门槛：防止 22步超速旋转 gaming（v33 问题）。
        # 真实后空翻（蹲伏蓄力→蹬腿→旋转→落地）至少需要 40 步（约 1 秒）。
        if (max(0.0, -self._pitch_accumulated) >= ROTATION_COMPLETE
                and self._landed_after_launch
                and self.step_counter >= MIN_BACKFLIP_STEPS):
            self._success = True

        # ── 奖励计算 ──────────────────────────────────────────────────────
        reward = 0.0
        contact_feet = self._get_foot_contacts()
        n_feet = sum(contact_feet)

        # 读取当前关节角（用于姿态引导奖励）
        joint_states = p.getJointStates(
            self.robot_id, self.joint_id[:8], physicsClientId=self.physics_client
        )
        current_joints = np.array([s[0] for s in joint_states])

        # 0. 关键帧姿态引导奖励（半 DeepMimic，只看关节角，不管机体轨迹）
        #
        #    原理：根据当前飞行阶段，指定 agent 应当模仿哪一帧的关节角度。
        #    为什么不做完整 DeepMimic：完整方案需要干净的机体轨迹，但 kinematic
        #    回放有侧倾（机体轨迹不可信）。只模仿关节角则规避了这个问题：
        #      → 告诉 agent "腾空时腿要折叠成帧2的样子"
        #      → 机体如何旋转由物理自行推导（不强制规定轨迹）
        #
        #    阶段映射：
        #      地面、未起跳         → 帧0（蹲伏蓄力）
        #      地面、已起跳         → 帧1（起跳爆发）
        #      腾空、旋转 < 180°   → 帧2（腾空收腿）
        #      腾空、旋转 > 180°   → 帧3（伸腿准备落地）
        #      已落地              → 帧4（站稳）
        if self.training_phase in ("jump", "rotate", "land", "full"):
            if self._landed_after_launch:
                ref_frame = BF_FRAMES_RAD[4]   # 站稳
            elif self._airborne:
                # v19 关键修复：收腿（帧2）保持到 270°，不再在 180° 时展腿。
                # v29 教训：延迟到 286° 反而让落地更难（0% 成功 vs 270° 的 5%），已回退。
                # 270° 是最优平衡点：保持足够旋转动量，同时给落地留出展腿时间。
                if abs(self._pitch_accumulated) < 3 * math.pi / 2:
                    ref_frame = BF_FRAMES_RAD[2]   # 腾空收腿（直到 270°）
                else:
                    ref_frame = BF_FRAMES_RAD[3]   # 伸腿准备落地（270° 之后）
            elif self._launched:
                ref_frame = BF_FRAMES_RAD[1]   # 起跳爆发
            else:
                ref_frame = BF_FRAMES_RAD[0]   # 蹲伏蓄力

            # 高斯形状奖励：关节角完全匹配时=1，误差越大越小
            # exp(-k * ||Δθ||²)：k=0.5 时，8 个关节各差 20° 时奖励 ≈ 0.4
            pose_error = float(np.sum((current_joints - ref_frame) ** 2))
            reward += W_POSE_GUIDE * math.exp(-0.5 * pose_error)

        # 1. 起跳奖励：向上速度越大越好（鼓励爆发起跳）
        #    条件：正在上升 且 还未腾空（地面起跳阶段）
        if self.training_phase in ("jump", "rotate", "land", "full"):
            if vz > 0 and not self._airborne:
                reward += W_JUMP * vz
                # 起跳时角动量奖励（rotate/land/full 阶段）：v3 新增。
                # 物理原理：后空翻的角动量 L = I×ω 必须在离地前由地面反力产生。
                # 腾空后处于自由旋转状态，总角动量守恒，无法凭空增加。
                # 因此在地面蹬腿阶段就鼓励向后倾斜（产生负方向俯仰角速度），
                # 让 agent 学会"向后翻转跳"而非"垂直直跳"。
                if self.training_phase in ("rotate", "land", "full"):
                    backward_ang = max(0.0, -ang_vel[1])
                    reward += W_SPIN_AT_LAUNCH * backward_ang

        # 2. 旋转奖励（腾空阶段）
        #    v3 改动：移除瞬时角速度奖励（backward_spin * 0.5），改为里程碑一次性奖励。
        #    v2 问题：瞬时角速度奖励可被 gaming —— agent 在空中正转+反转振荡，
        #    瞬时角速度峰值高但净旋转只有 60-70°，照样拿大量奖励（实测 1772-3653）。
        #    v3 方案：用绝对累积旋转量触发一次性里程碑奖励，振荡无法重复触发里程碑。
        if self.training_phase in ("rotate", "land", "full"):
            # 注：v9 已删除腾空第一帧一次性角速度奖励（W_SPIN_AT_LAUNCH_ONCE）。
            # 8 轮实验证明：任何线性正比于起飞角速度的奖励都会被 gaming，
            # agent 会把角速度飙到 15-29 rad/s 骗分，而不是真正旋转。
            # 现在完全依赖 W_ROT_STEP（增量旋转）+ 里程碑奖励来驱动旋转行为。

            if self._airborne:
                # v13 关键修复：只奖励向后旋转（负 pitch 方向）。
                # v12 bug：abs(_pitch_accumulated) 对前/后空翻一视同仁，
                # agent 在 MOTOR_VELOCITY=20π 时找到前空翻捷径（liftoff ω 变负 -8.97 r/s），
                # 导致完全学错方向。max(0, -pitch) 仅在向后旋转时为正，强制后空翻方向。
                backward_rot = max(0.0, -self._pitch_accumulated)

                # 累计旋转量奖励：向后旋转越多越好（连续、平滑的正向信号）
                rotation_progress = min(1.0, backward_rot / ROTATION_COMPLETE)
                reward += W_ROTATION * rotation_progress

                # 增量旋转奖励：只奖励向后旋转的历史最大值新增部分
                rot_now = backward_rot
                delta = rot_now - self._prev_max_rot_airborne
                if delta > 0:
                    self._prev_max_rot_airborne = rot_now
                    reward += W_ROT_STEP * delta

                # 旋转里程碑一次性奖励：每越过 90°/180°/270° 给一次 W_MILESTONE×W_ROTATION。
                for m in MILESTONES_RAD:
                    if backward_rot >= m and m not in self._milestones_hit:
                        self._milestones_hit.add(m)
                        reward += W_MILESTONE * W_ROTATION

                # 腾空不旋转惩罚：向后旋转不足（含前空翻情形）均惩罚
                if backward_rot < 0.1:
                    reward -= W_NO_SPIN

                # v37 新增：超旋转惩罚（每步，腾空时旋转超过360°则惩罚）
                # 目标：阻止 agent 超旋转（412-432°=1.1-1.2圈），推向干净的360°翻转。
                # v36 教训：惩罚角速度是错的（物理上必须高速才能完成翻转）。
                # 惩罚旋转量本身：只要不超过360°（ROTATION_TARGET），完全没有惩罚。
                if backward_rot > ROTATION_TARGET:
                    overrot_excess = backward_rot - ROTATION_TARGET
                    reward -= W_OVERROT * overrot_excess

                # 腾空防侧倒惩罚（v14 新增）：横滚 > 30° 时线性惩罚
                # 目的：减少 ep_len 过短（V13 平均 35 步）的侧倒提前终止
                # 0.52 rad ≈ 30° 容限，允许正常后空翻轻微横滚
                roll_excess = abs(roll) - 0.52
                if roll_excess > 0:
                    reward -= W_ANTI_ROLL * roll_excess

        # 3. 落地奖励：起跳后脚着地（需先旋转 ≥ 90°，才给落地奖励）
        #    v1 bug：无旋转门槛，agent 发现"跳不翻直接落"比"翻转落"奖励更高
        #    v2 修复：用 ROTATION_GATE（π/2 ≈ 90°）做门槛，断掉 gaming 路径
        if self.training_phase in ("land", "full"):
            if (self._launched and not self._airborne and n_feet >= 2
                    and max(0.0, -self._pitch_accumulated) >= ROTATION_GATE):
                reward += W_LANDING * (n_feet / 4)

        # 4. 站稳奖励（v41 改为绑定成功瞬间，不再每步给）
        #    W_STABILITY=0.0（常量），不再产生每步站稳奖励。
        #    v40 教训：W_STABILITY=2/步 → gaming: 落地259°站稳60步=120分持续收入，
        #    agent宁愿停在success门槛以下也不冒险推到286°。
        #    v41 修复：站稳奖励绑定到成功时刻（见下方 bonus 5），消除持续收入动机。
        if self.training_phase == "full":
            if self._landed_after_launch:
                uprightness = math.exp(-2.0 * (pitch ** 2 + roll ** 2))
                reward += W_STABILITY * uprightness  # W_STABILITY=0.0，等效不执行

        # 4b. 落地时机奖励（v57 新增）：落地瞬间（just_landed）一次性奖励，直接激励高rot@land。
        #    v56 教训：W_ROT_COMPLETENESS基线=350°，rot@land=342-347°时completeness=0，完全没有梯度！
        #    agent无法感知"多转几度落地"有多少收益→策略固化在347°。
        #    本奖励：基线330°→360°，在342°就有梯度，推动机器人延迟"伸腿减速"动作。
        #    设计要点：一次性（只在just_landed触发，不是per-step），不存在gaming的持续收入问题。
        if self.training_phase == "full":
            if self._landed_after_launch and not was_landed_before:
                rot_at_land = math.degrees(max(0.0, -self._pitch_accumulated))
                land_timing_ratio = min(1.0, max(0.0, (rot_at_land - 330.0) / (360.0 - 330.0)))
                reward += W_LAND_TIMING * land_timing_ratio ** 2

        # 5. 完成 bonus（稀疏大奖，只在完整后空翻成功时给一次）
        #    v41 新增：成功时附加直立姿态加成（W_UPRIGHTNESS_BONUS × uprightness × W_SUCCESS）。
        #    完美直立落地 → 总奖励 = W_SUCCESS × (1 + 0.5 × 1.0) = 1500
        #    翻倒落地     → 总奖励 = W_SUCCESS × (1 + 0.5 × ~0) ≈ 1000
        #    差价 500 → agent 学会翻完后站稳，而非gaming低分收入流。
        if self.training_phase == "full" and self._success:
            uprightness = math.exp(-2.0 * (pitch ** 2 + roll ** 2))
            reward += W_SUCCESS * (1.0 + W_UPRIGHTNESS_BONUS * uprightness)
            # v46 新增 / v48 升级：旋转完整度奖励（平方公式，让最后几度梯度更陡峭）。
            # v57 升级：基线从350°→335°，让342-360°区间全程有梯度（v55-v56在此区间梯度=0）。
            #   335°=梯度起点（completeness=0，额外+0）
            #   347°≈V55水平: (12/25)²=0.230，额外+691
            #   360°=满分（completeness=1，额外+3000）
            rot_deg = math.degrees(max(0.0, -self._pitch_accumulated))
            rot_ratio = min(1.0, max(0.0, (rot_deg - 335.0) / (360.0 - 335.0)))
            rot_completeness = rot_ratio ** 2  # 平方：越接近360°梯度越陡
            reward += W_ROT_COMPLETENESS * rot_completeness

        # ── 终止判断 ──────────────────────────────────────────────────────
        self.step_counter += 1
        terminated = truncated = False

        # 仅因侧倒终止（roll > 90°）
        # 注意：不因 pitch 终止！这是后空翻的核心设计，让机器人完整翻转
        if abs(roll) > math.pi / 2:
            terminated = True
            reward -= W_LATERAL_FALL   # 侧倒惩罚

        if self.step_counter >= EPISODE_LENGTH:
            truncated = True

        if self._success:
            terminated = True   # 成功完成后空翻，结束 episode

        info = {
            "success":      self._success,
            "max_height_m": self._max_height,
            "rotation_deg": math.degrees(abs(self._pitch_accumulated)),
            "launched":     self._launched,
            "airborne":     self._airborne,
            "landed":       self._landed_after_launch,
            "n_feet":       n_feet,
            # 腾空第一帧的原始角速度（rad/s，负值=向后）：用于 gaming 检测
            # None 表示本步不是腾空帧，eval 脚本只在非 None 时采样
            "liftoff_ang_vel_y": ang_vel[1] if (self._airborne and not was_airborne_before) else None,
            # 落地边沿：True 表示本步刚刚落地（用于记录落地时旋转角）
            "just_landed": self._landed_after_launch and not was_landed_before,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ── 辅助方法 ──────────────────────────────────────────────────────────

    def _rsi_init(self):
        """Random State Initialization：将机器人初始化为腾空旋转中的随机状态。

        仅设置 PyBullet 物理状态（位置、速度、关节角）；
        Python 状态变量（_launched、_airborne 等）由 reset() 统一处理。

        随机范围：
          height  : 0.10–0.35 m（跳跃弧线全程，包括下降段）
          pitch   : v18 双段分布：
                    70% → -60° ~ -150°（覆盖已验证有效区间）
                    30% → -150° ~ -240°（v18 新增：覆盖 agent 从未接触的 150°-240° 区间）
                    v18 动机：v17 RSI 仅覆盖 60-150°，agent 到 150° 后不知道如何继续翻，
                    扩展到 -240° 让 agent 练习"越过 180°、推进到 240°"的行为。
                    注：超过 ±π 时 Euler 回读有歧义（-240° 被读成 +120°），
                    通过 self._rsi_actual_pitch 绕过此问题。
          ang_vel : -3 ~ -8 rad/s 向后旋转（配合更大旋转角，角速度范围也扩大）
          vz      : 随高度线性下降（低处仍在上升，高处已在下落）
        """
        height    = float(np.random.uniform(0.10, 0.35))
        # v25: RSI 分布恢复平衡三段（50%/30%/20%），与 V19 相同。
        #   50% → -60° ~ -150°（基础区间）
        #   30% → -150° ~ -240°（中段）
        #   20% → -240° ~ -270°（冲刺段，安全上限 < ROTATION_COMPLETE）
        # v25 从零走完整课程（jump→rotate→land→full），需要均匀覆盖各旋转区间。
        # v25: RSI 分布恢复 V19 平衡版本（三段 50%/30%/20%）
        # v24 教训：80% 集中在 200°-270° 导致训练分布错位，
        # 策略在 200°-270° RSI 起步下优化，但从地面起步到 200° 的行为退化，反而更差。
        # v25 从零开始用 4.5N·m，需要平衡覆盖所有旋转区间，让课程学习自然过渡。
        rnd = np.random.random()
        if rnd < 0.50:
            pitch = float(np.random.uniform(-math.pi / 3, -5 * math.pi / 6))           # -60° ~ -150°（50%，基础区间）
        elif rnd < 0.80:
            pitch = float(np.random.uniform(-5 * math.pi / 6, -4 * math.pi / 3))       # -150° ~ -240°（30%，中段）
        else:
            pitch = float(np.random.uniform(-4 * math.pi / 3, -3 * math.pi / 2))       # -240° ~ -270°（20%，冲刺段）
            # v20 教训：超过 ROTATION_COMPLETE=5.0rad 会触发 gaming（RSI 起步直接算成功）。
            # v21 修复：上限严格 ≤ -270°=-3π/2=4.71rad < 5.0rad，安全。
        self._rsi_actual_pitch = pitch  # 存储真实值（超 ±π 时 Euler 回读会给出错误等效角）
        ang_vel_y = float(np.random.uniform(-8.0, -3.0))
        vz = float(np.clip(
            2.0 - 3.0 * (height - 0.15) + np.random.uniform(-0.5, 0.5),
            -2.0, 2.0
        ))

        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, height],
            p.getQuaternionFromEuler([0, pitch, 0]),
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[0, 0, vz],
            angularVelocity=[0, ang_vel_y, 0],
            physicsClientId=self.physics_client,
        )
        # 腾空收腿姿态（帧2：减小转动惯量，角动量守恒 → ω↑）
        for i, jid in enumerate(self.joint_id[:8]):
            p.resetJointState(self.robot_id, jid, BF_FRAMES_RAD[2][i],
                              physicsClientId=self.physics_client)

    def _get_obs(self) -> np.ndarray:
        """构建 23 维观测向量。"""
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )
        joint_states = p.getJointStates(
            self.robot_id, self.joint_id[:8],
            physicsClientId=self.physics_client,
        )
        joint_angs_norm = np.clip(
            np.array([s[0] for s in joint_states]) / self._bound_ang,
            -1.0, 1.0,
        )
        contact_feet = np.array(self._get_foot_contacts(), dtype=np.float32)

        return np.concatenate([
            np.array(orn,     dtype=np.float32),      # [0:4]  四元数
            np.clip(ang_vel,  -30.0, 30.0).astype(np.float32),  # [4:7]  角速度
            np.clip(lin_vel,  -5.0,  5.0).astype(np.float32),   # [7:10] 线速度
            np.array([pos[2]], dtype=np.float32),     # [10]   高度
            joint_angs_norm.astype(np.float32),       # [11:19] 关节角
            contact_feet,                              # [19:23] 触地状态
        ])

    def _discover_paw_links(self) -> list[int]:
        """动态发现 4 个爪子（*_paw）的 PyBullet link 索引。

        BittleX URDF 中，每条腿末端有一个 FIXED 关节，子链接名含 "paw"。
        这些 FIXED 关节不在 self.joint_id（仅含 REVOLUTE），
        需要单独扫描找到。
        """
        paw_links = []
        for j in range(p.getNumJoints(self.robot_id,
                                       physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, j,
                                  physicsClientId=self.physics_client)
            child_name = info[12].decode()   # 子链接名
            if "paw" in child_name.lower():
                paw_links.append(j)          # link index = joint index in PyBullet
        return paw_links

    def _get_foot_contacts(self) -> list[bool]:
        """检测 4 只爪子的触地状态。

        BittleX URDF 足端结构（通过 getContactPoints 实测）：
          link 3  = left_front_paw
          link 6  = right_front_paw
          link 9  = right_back_paw
          link 12 = left_back_paw
        动态发现，不硬编码（遵循血泪教训 #1）。
        """
        if not hasattr(self, "_paw_links") or not self._paw_links:
            self._paw_links = self._discover_paw_links()
        contacts = []
        for link_id in self._paw_links:
            pts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=link_id,
                                     physicsClientId=self.physics_client)
            contacts.append(bool(pts))
        # 确保始终返回 4 个值
        while len(contacts) < 4:
            contacts.append(False)
        return contacts[:4]

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[pos[0], pos[1], 0.1],
            distance=0.5, yaw=0, pitch=-20, roll=0, upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=320 / 240, nearVal=0.01, farVal=10
        )
        _, _, rgba, _, _ = p.getCameraImage(
            320, 240, view, proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client,
        )
        return np.array(rgba, dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
