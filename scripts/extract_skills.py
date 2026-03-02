#!/usr/bin/env python3
"""一次性脚本：解析 InstinctBittle.h 固件数据，生成 skills_keyframes.json。

固件帧格式：
  正 num_frames：简单步态/姿态
    header[4] = {num_frames, delay_ms, p3, p4}
    之后 num_frames × 8 字节（hw8-15，8个腿部关节）
  负 num_frames：复杂行为（back flip 等）
    header[4] = {-n, delay_ms, p3, p4}
    额外 3 字节 padding
    之后 n × 20 字节：前8=hw0-7（头/尾），中8=hw8-15（腿），后4=时序

Usage:
    cd /Users/mlamp/Workspace/JPRobot
    python scripts/extract_skills.py
"""

import re
import json
from pathlib import Path

SRC = Path("/Users/mlamp/Workspace/OpenCat-Quadruped-Robot/src/InstinctBittle.h")

# 固件腿部关节顺序（hw8-15）→ 仿真关节顺序的映射
# fw: [lf_shoulder, rf_shoulder, rb_hip, lb_hip, lf_knee, rf_knee, rb_knee, lb_knee]
# sim: [shoulder_left, elbow_left, shoulder_right, elbow_right,
#        hip_right,    knee_right,  hip_left,       knee_left]
# sim[i] = fw[REORDER[i]]
REORDER = [0, 4, 1, 5, 2, 6, 3, 7]

src_text = SRC.read_text()
pattern = re.compile(
    r'const int8_t (\w+)\[\] PROGMEM = \{(.*?)\};',
    re.DOTALL
)

skills = {}
for m in pattern.finditer(src_text):
    name = m.group(1)
    values = [int(x) for x in re.findall(r'-?\d+', m.group(2))]

    if len(values) < 4:
        print(f"  跳过 {name}：数据太短 ({len(values)} 值)")
        continue

    num_frames_raw = values[0]
    delay_ms = values[1]
    is_behavior = num_frames_raw < 0
    num_frames = abs(num_frames_raw)

    if is_behavior:
        # 行为格式：header(4) + padding(3) + n×20
        offset = 4 + 3   # 跳过 header 和 3 字节 padding
        frame_size = 20
        leg_start = 8    # 每帧的腿部关节从第 8 个值开始
        leg_end = 16
    else:
        # 标准格式：header(4) + n×8
        offset = 4
        frame_size = 8
        leg_start = 0
        leg_end = 8

    available = len(values) - offset
    max_frames = available // frame_size
    if num_frames > max_frames:
        print(f"  警告 {name}：声明 {num_frames} 帧，实际只有 {max_frames} 帧数据")
        num_frames = max_frames

    frames = []
    for i in range(num_frames):
        start = offset + i * frame_size
        segment = values[start:start + frame_size]
        fw = segment[leg_start:leg_end]
        if len(fw) < 8:
            break
        sim = [fw[REORDER[j]] for j in range(8)]
        frames.append(sim)

    skills[name] = {
        "num_frames": len(frames),
        "delay_ms": delay_ms,
        "is_behavior": is_behavior,
        "frames": frames,
    }

out = Path(__file__).parent.parent / "jprobot" / "data" / "skills_keyframes.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(skills, ensure_ascii=False, indent=2))
print(f"提取完成：{len(skills)} 个技能 → {out}")

# 打印统计
pos = sum(1 for v in skills.values() if not v["is_behavior"])
beh = sum(1 for v in skills.values() if v["is_behavior"])
print(f"  标准格式（步态/姿态）：{pos} 个")
print(f"  行为格式（复杂动作）：{beh} 个")
