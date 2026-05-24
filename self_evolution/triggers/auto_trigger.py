"""
自动触发 + 用户审批系统

状态机：
  MONITOR → PENDING → APPROVED → EXECUTING → DONE
                      ↘ REJECTED → MONITOR (冷却后重新计数)

数据源：直接读取 Hermes 内置的 ~/.hermes/skills/.usage.json
        use_count / view_count / last_used_at 来自 skill_view() 的实际调用
持久化：~/.hermes/self_evolution_state.json
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ApprovalStatus(Enum):
    MONITOR = "monitor"         # 监控中，计数累积
    PENDING = "pending"         # 待审批，已发送请求
    APPROVED = "approved"       # 用户批准，待执行
    REJECTED = "rejected"       # 用户拒绝，进入冷却
    EXECUTING = "executing"     # 正在执行进化
    DONE = "done"               # 已完成


@dataclass
class SkillEvoState:
    """单个 skill 的进化状态"""
    skill_name: str
    usage_count: int = 0
    first_seen: float = 0.0       # 首次记录时间
    last_used: float = 0.0        # 最后使用时间
    last_evolved: float = 0.0     # 上次进化完成时间
    status: str = "monitor"       # ApprovalStatus 值
    pending_since: float = 0.0    # 审批请求发送时间
    trigger_reason: str = ""      # 触发原因
    evolved_score: float = 0.0    # 上次进化后分数
    evolution_count: int = 0      # 历史进化次数
    reject_count: int = 0         # 连续拒绝次数（用于自动降阈值）


class AutoEvolutionManager:
    """
    自动进化管理器 — 状态机 + 持久化 + 审批

    工作流：
      1. 每次 skill 使用 → record_usage()
      2. 定时扫描 → scan_candidates() 生成审批请求
      3. 用户审批 → approve() / reject()
      4. 执行进化 → execute_approved()
    """

    STATE_FILE = Path.home() / ".hermes" / "self_evolution_state.json"

    # 默认阈值
    DEFAULT_THRESHOLD = 10          # 使用 N 次后触发
    DEFAULT_COOLDOWN_HOURS = 72     # 冷却时间（上次进化后）
    DEFAULT_PENDING_TTL_HOURS = 168 # 审批请求 7 天过期

    def __init__(self):
        self._states: dict[str, SkillEvoState] = {}
        self._load()

    # ── 计数 ─────────────────────────────────────────────────

    def record_usage(self, skill_name: str) -> None:
        """记录一次 skill 使用。"""
        now = time.time()
        state = self._get_or_create(skill_name)
        state.usage_count += 1
        state.last_used = now
        if state.usage_count == 1:
            state.first_seen = now
        self._save()

    # ── 扫描 ─────────────────────────────────────────────────

    def scan_candidates(
        self,
        threshold: int = None,
        cooldown_hours: int = None,
    ) -> list[SkillEvoState]:
        """
        扫描所有 skill，返回应该触发进化的候选项。

        数据来源：~/.hermes/skills/.usage.json（Hermes 内置使用追踪）
          - use_count: skill 被加载到 prompt 路径的次数
          - view_count: skill_view() 调用次数
          - last_used_at: 最后使用时间

        触发条件：
          1. (use_count + view_count) ≥ threshold
          2. 距上次进化 ≥ cooldown_hours
          3. 不在 REJECTED / EXECUTING 状态
        """
        threshold = threshold or self.DEFAULT_THRESHOLD
        cooldown_hours = cooldown_hours or self.DEFAULT_COOLDOWN_HOURS
        now = time.time()
        candidates = []

        # 从 Hermes 内置 .usage.json 读取真实使用数据
        usage_data = self._read_hermes_usage()

        for skill_name, usage in usage_data.items():
            # 跳过没有足够使用数据的
            activity = usage.get("use_count", 0) + usage.get("view_count", 0)
            if activity == 0:
                continue

            # 获取或创建自进化状态
            state = self._get_or_create(skill_name)

            # 同步使用数据
            state.usage_count = activity
            last_used_str = usage.get("last_used_at") or usage.get("last_viewed_at")
            if last_used_str:
                try:
                    state.last_used = self._parse_iso(last_used_str)
                except Exception:
                    pass

            # 跳过正在执行或已拒绝的
            if state.status in (ApprovalStatus.EXECUTING.value,):
                continue
            if state.status == ApprovalStatus.REJECTED.value:
                if now - state.pending_since > cooldown_hours * 3600:
                    state.status = ApprovalStatus.MONITOR.value
                    state.usage_count = 0
                    self._save()
                continue

            # 检查触发条件
            if state.usage_count < threshold:
                continue
            if state.last_evolved > 0:
                hours_since_evolve = (now - state.last_evolved) / 3600
                if hours_since_evolve < cooldown_hours:
                    continue

            # 生成触发原因
            state.trigger_reason = self._build_reason(state)
            state.status = ApprovalStatus.PENDING.value
            state.pending_since = now
            candidates.append(state)

        if candidates:
            self._save()

        return candidates

    # ── 审批 ─────────────────────────────────────────────────

    def approve(self, skill_name: str) -> bool:
        """批准进化请求。"""
        state = self._states.get(skill_name)
        if not state or state.status != ApprovalStatus.PENDING.value:
            return False
        state.status = ApprovalStatus.APPROVED.value
        self._save()
        return True

    def reject(self, skill_name: str) -> bool:
        """拒绝进化请求（进入冷却）。"""
        state = self._states.get(skill_name)
        if not state or state.status != ApprovalStatus.PENDING.value:
            return False
        state.status = ApprovalStatus.REJECTED.value
        state.pending_since = time.time()
        state.reject_count += 1
        self._save()
        return True

    def mark_rejected(self, skill_name: str) -> bool:
        """标记执行后被门控拒绝（不减冷却，仅计数）。"""
        state = self._states.get(skill_name)
        if not state:
            return False
        state.reject_count += 1
        # 连续3次被门控拒绝，自动降低触发阈值
        if state.reject_count >= 3:
            old_reason = state.trigger_reason
            state.trigger_reason = f"连续{state.reject_count}次被门控拒绝，自动降低触发要求"
            state.usage_count = max(1, state.usage_count // 2)  # 阈值减半
            logger.warning(
                f"[auto-evo] {skill_name}: {state.reject_count} consecutive rejects, "
                f"threshold halved to {state.usage_count}"
            )
        state.status = ApprovalStatus.MONITOR.value
        self._save()
        return True

    def mark_executing(self, skill_name: str) -> bool:
        """标记为正在执行。"""
        state = self._states.get(skill_name)
        if not state or state.status != ApprovalStatus.APPROVED.value:
            return False
        state.status = ApprovalStatus.EXECUTING.value
        self._save()
        return True

    def mark_done(self, skill_name: str, score: float = 0.0) -> bool:
        """标记进化完成。"""
        state = self._states.get(skill_name)
        if not state:
            return False
        state.status = ApprovalStatus.DONE.value
        state.last_evolved = time.time()
        state.evolved_score = score
        state.evolution_count += 1
        state.usage_count = 0        # 重置计数
        state.reject_count = 0       # 重置拒绝计数
        self._save()
        return True

    # ── 查询 ─────────────────────────────────────────────────

    def get_pending(self) -> list[SkillEvoState]:
        """获取所有待审批项。"""
        return [s for s in self._states.values()
                if s.status == ApprovalStatus.PENDING.value]

    def get_approved(self) -> list[SkillEvoState]:
        """获取所有已批准待执行项。"""
        return [s for s in self._states.values()
                if s.status == ApprovalStatus.APPROVED.value]

    def get_state(self, skill_name: str) -> Optional[SkillEvoState]:
        return self._states.get(skill_name)

    def list_all(self) -> list[SkillEvoState]:
        return list(self._states.values())

    # ── 审批请求格式化 ──────────────────────────────────────

    def format_approval_request(self, state: SkillEvoState) -> str:
        """格式化审批请求消息（用于微信等渠道）。"""
        days_since_evolve = ""
        if state.last_evolved > 0:
            days = (time.time() - state.last_evolved) / 86400
            days_since_evolve = f"\n- 距上次进化: {days:.0f} 天"

        return f"""🔧 **自进化审批请求**

**Skill:** `{state.skill_name}`
**触发原因:** {state.trigger_reason}
**统计:**
- 累计使用: {state.usage_count} 次{days_since_evolve}
- 历史进化: {state.evolution_count} 次
- 上次分数: {state.evolved_score:.3f}

**预计成本:** ~$0.05-0.20（数据集生成 + 5轮优化）

回复 **批准 {state.skill_name}** 或 **拒绝 {state.skill_name}**"""

    # ── 内部 ─────────────────────────────────────────────────

    def _read_hermes_usage(self) -> dict:
        """读取 Hermes 内置的 ~/.hermes/skills/.usage.json。"""
        usage_path = Path.home() / ".hermes" / "skills" / ".usage.json"
        if not usage_path.exists():
            return {}
        try:
            with open(usage_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    @staticmethod
    def _parse_iso(value: str) -> float:
        """解析 ISO 时间戳为 epoch 秒。"""
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            return time.time()

    def _get_or_create(self, skill_name: str) -> SkillEvoState:
        if skill_name not in self._states:
            self._states[skill_name] = SkillEvoState(skill_name=skill_name)
        return self._states[skill_name]

    def _build_reason(self, state: SkillEvoState) -> str:
        """自动生成触发原因。"""
        parts = [f"已使用 {state.usage_count} 次"]

        if state.last_evolved > 0:
            days = (time.time() - state.last_evolved) / 86400
            parts.append(f"距上次进化 {days:.0f} 天")
            if state.evolved_score < 0.5:
                parts.append("上次评分偏低，需进一步优化")
        else:
            parts.append("从未进化过，初次优化")

        if state.evolution_count >= 2:
            parts.append(f"已进化 {state.evolution_count} 次，持续迭代")

        return "；".join(parts)

    # ── 持久化 ───────────────────────────────────────────────

    def _save(self) -> None:
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: {
                "skill_name": s.skill_name,
                "usage_count": s.usage_count,
                "first_seen": s.first_seen,
                "last_used": s.last_used,
                "last_evolved": s.last_evolved,
                "status": s.status,
                "pending_since": s.pending_since,
                "trigger_reason": s.trigger_reason,
                "evolved_score": s.evolved_score,
                "evolution_count": s.evolution_count,
                "reject_count": s.reject_count,
            }
            for name, s in self._states.items()
        }
        with open(self.STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not self.STATE_FILE.exists():
            return
        try:
            with open(self.STATE_FILE) as f:
                data = json.load(f)
            for name, d in data.items():
                self._states[name] = SkillEvoState(**d)
        except (json.JSONDecodeError, TypeError):
            pass
