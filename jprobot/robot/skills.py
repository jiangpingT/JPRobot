"""Skill registry for Petoi BittleX.

All 56 built-in skills from OpenCat InstinctBittle.h, organized by category.
Each skill has a short code (sent via serial) and metadata for LLM understanding.
"""

from dataclasses import dataclass
from enum import Enum


class SkillType(Enum):
    GAIT = "gait"           # Periodic locomotion (walking, trotting)
    POSTURE = "posture"     # Static pose (sit, stand)
    BEHAVIOR = "behavior"   # Finite-length action (flip, handshake)


@dataclass
class Skill:
    code: str               # Serial command code (e.g., "sit", "wkF")
    name: str               # Human-readable name
    name_cn: str            # Chinese name
    skill_type: SkillType
    description: str        # Description for LLM context


class SkillRegistry:
    """Registry of all BittleX skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register all 56 built-in skills from OpenCat."""

        # === Gaits (periodic locomotion) ===
        gaits = [
            ("wkF",  "Walk Forward",        "向前走",     "Walk forward at normal speed"),
            ("wkL",  "Walk Left",           "向左走",     "Walk turning left"),
            ("bk",   "Walk Backward",       "后退",       "Walk backward"),
            ("bkL",  "Walk Backward Left",  "向左后退",   "Walk backward turning left"),
            ("trF",  "Trot Forward",        "小跑前进",   "Trot forward (faster than walk)"),
            ("trL",  "Trot Left",           "小跑左转",   "Trot turning left"),
            ("crF",  "Crawl Forward",       "匍匐前进",   "Crawl forward slowly"),
            ("crL",  "Crawl Left",          "匍匐左转",   "Crawl turning left"),
            ("vtF",  "Step Forward",        "踏步前进",   "Step in place moving forward"),
            ("vtL",  "Step Left",           "踏步左转",   "Step in place turning left"),
            ("phF",  "Push Forward",        "推进",       "Push forward gait"),
            ("phL",  "Push Left",           "推进左转",   "Push forward turning left"),
            ("bdF",  "Bound Forward",       "跳跃前进",   "Bound forward (jumping gait)"),
            ("jpF",  "Jump Forward",        "跳步前进",   "Jump forward gait"),
        ]
        for code, name, name_cn, desc in gaits:
            self._register(code, name, name_cn, SkillType.GAIT, desc)

        # === Postures (static poses) ===
        postures = [
            ("balance", "Balance",      "站立平衡",   "Stand and balance"),
            ("buttUp",  "Butt Up",      "翘屁股",     "Raise butt up"),
            ("calib",   "Calibrate",    "校准姿态",   "Calibration pose"),
            ("dropped", "Dropped",      "跌落恢复",   "Recovery from being dropped"),
            ("lifted",  "Lifted",       "被抬起",     "Response to being lifted"),
            ("lnd",     "Landing",      "着陆",       "Landing pose"),
            ("rest",    "Rest",         "休息",       "Rest and shut down servos"),
            ("sit",     "Sit",          "坐下",       "Sit down"),
            ("str",     "Stretch",      "伸展",       "Stretch body"),
            ("up",      "Stand Up",     "站起来",     "Stand up tall"),
            ("zero",    "Zero",         "归零",       "All joints to zero position"),
        ]
        for code, name, name_cn, desc in postures:
            self._register(code, name, name_cn, SkillType.POSTURE, desc)

        # === Behaviors (finite-length actions) ===
        behaviors = [
            ("ang",  "Angry",          "生气",       "Show angry expression"),
            ("bf",   "Back Flip",      "后空翻",     "Perform a back flip"),
            ("bx",   "Boxing",         "拳击",       "Boxing punch motion"),
            ("chr",  "Cheers",         "干杯",       "Cheers celebration"),
            ("ck",   "Check",          "检查",       "Look around checking"),
            ("cmh",  "Come Here",      "过来",       "Beckoning come here gesture"),
            ("dg",   "Dig",            "挖掘",       "Dig at ground"),
            ("ff",   "Front Flip",     "前空翻",     "Perform a front flip"),
            ("fiv",  "High Five",      "击掌",       "High five gesture"),
            ("gdb",  "Good Boy",       "乖狗狗",     "Good boy happy response"),
            ("hds",  "Hand Stand",     "倒立",       "Perform a handstand"),
            ("hg",   "Hug",            "拥抱",       "Hug gesture"),
            ("hi",   "Say Hi",         "打招呼",     "Wave and say hello"),
            ("hsk",  "Hand Shake",     "握手",       "Shake hands"),
            ("hu",   "Hands Up",       "举手",       "Raise both hands up"),
            ("jmp",  "Jump",           "跳跃",       "Jump in place"),
            ("kc",   "Kick",           "踢",         "Kick with back leg"),
            ("mw",   "Moon Walk",      "月球漫步",   "Perform moon walk dance"),
            ("nd",   "Nod",            "点头",       "Nod head up and down"),
            ("pd",   "Play Dead",      "装死",       "Play dead (fall over)"),
            ("pee",  "Pee",            "撒尿",       "Lift leg to pee"),
            ("pu",   "Push Up",        "俯卧撑",     "Do push ups"),
            ("pu1",  "Push Up Single Arm", "单臂俯卧撑", "Push up with a single arm"),
            ("rc",   "Recover",        "恢复",       "Recover from fallen state"),
            ("rl",   "Roll",           "翻滚",       "Roll over"),
            ("scrh", "Scratch",        "抓痒",       "Scratch self"),
            ("snf",  "Sniff",          "嗅",         "Sniff the ground"),
            ("tbl",  "Table",          "桌子",       "Form a table shape"),
            ("ts",   "Test",           "测试",       "Test servo motion"),
            ("wh",   "Wave Head",      "摇头",       "Wave head side to side"),
            ("zz",   "Sleep",          "睡觉",       "Fall asleep"),
        ]
        for code, name, name_cn, desc in behaviors:
            self._register(code, name, name_cn, SkillType.BEHAVIOR, desc)

    def _register(self, code: str, name: str, name_cn: str,
                  skill_type: SkillType, description: str):
        self._skills[code] = Skill(code, name, name_cn, skill_type, description)

    def get(self, code: str) -> Skill | None:
        return self._skills.get(code)

    def search(self, keyword: str) -> list[Skill]:
        """Search skills by keyword in name, code, or description."""
        keyword = keyword.lower()
        return [
            s for s in self._skills.values()
            if keyword in s.name.lower()
            or keyword in s.code.lower()
            or keyword in s.description.lower()
            or keyword in s.name_cn
        ]

    def by_type(self, skill_type: SkillType) -> list[Skill]:
        return [s for s in self._skills.values() if s.skill_type == skill_type]

    @property
    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def to_prompt_text(self) -> str:
        """Generate skill list text for LLM system prompt."""
        lines = ["Available robot skills:"]
        for stype in SkillType:
            skills = self.by_type(stype)
            lines.append(f"\n## {stype.value.title()}s:")
            for s in skills:
                lines.append(f"  - {s.code}: {s.name} ({s.name_cn}) - {s.description}")
        return "\n".join(lines)
