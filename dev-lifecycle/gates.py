from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class GateResult:
    passed: bool
    checks: List[dict] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class GateCheck:
    name: str
    check_fn: Callable[[dict], Tuple[bool, str]]
    description: str = ""


def _check_prd_exists(context: dict) -> Tuple[bool, str]:
    if context.get("prd_path"):
        return True, "PRD 文档已存在"
    return False, "PRD 文档缺失，请先完成 PRD 编写"


def _check_issues_split(context: dict) -> Tuple[bool, str]:
    count = context.get("issues_count", 0)
    if count > 0:
        return True, f"Issue 已拆解，共 {count} 个"
    return False, "Issue 尚未拆解，请将需求拆分为可执行的 Issue"


def _check_plan_written(context: dict) -> Tuple[bool, str]:
    if context.get("plan_path"):
        return True, "实现计划已编写"
    return False, "实现计划缺失，请先编写实现计划"


def _check_tests_passed(context: dict) -> Tuple[bool, str]:
    if context.get("tests_passed"):
        return True, "测试已通过"
    return False, "测试未通过，请修复失败的测试用例"


def _check_review_completed(context: dict) -> Tuple[bool, str]:
    if context.get("review_completed"):
        return True, "代码审查已完成"
    return False, "代码审查未完成，请提交代码审查"


def _check_no_blocking_bugs(context: dict) -> Tuple[bool, str]:
    count = context.get("blocking_bugs", 0)
    if count == 0:
        return True, "无阻塞 Bug"
    return False, f"存在 {count} 个阻塞 Bug，请优先修复"


class QualityGateManager:
    def __init__(self):
        self._gates: Dict[str, Dict[str, List[GateCheck]]] = {}
        self._register_builtin_gates()

    def _register_builtin_gates(self):
        ideate_to_build = [
            GateCheck(name="prd_exists", check_fn=_check_prd_exists, description="检查 PRD 文档是否存在"),
            GateCheck(name="issues_split", check_fn=_check_issues_split, description="检查 Issue 是否已拆解"),
            GateCheck(name="plan_written", check_fn=_check_plan_written, description="检查实现计划是否已编写"),
        ]
        for gate in ideate_to_build:
            self.register_gate("ideate", "build", gate)

        build_to_deliver = [
            GateCheck(name="tests_passed", check_fn=_check_tests_passed, description="检查测试是否通过"),
            GateCheck(name="review_completed", check_fn=_check_review_completed, description="检查代码审查是否完成"),
            GateCheck(name="no_blocking_bugs", check_fn=_check_no_blocking_bugs, description="检查是否存在阻塞 Bug"),
        ]
        for gate in build_to_deliver:
            self.register_gate("build", "deliver", gate)

    def register_gate(self, from_stage: str, to_stage: str, gate_check: GateCheck):
        key = f"{from_stage}->{to_stage}"
        if key not in self._gates:
            self._gates[key] = []
        self._gates[key].append(gate_check)

    def check(self, from_stage: str, to_stage: str, context: dict) -> GateResult:
        key = f"{from_stage}->{to_stage}"
        checks = self._gates.get(key, [])

        if not checks:
            return GateResult(passed=True)

        results: List[dict] = []
        failures: List[str] = []
        suggestions: List[str] = []

        for gate in checks:
            passed, message = gate.check_fn(context)
            results.append({
                "name": gate.name,
                "status": "passed" if passed else "failed",
                "message": message,
            })
            if not passed:
                failures.append(f"[{gate.name}] {message}")
                suggestions.append(message)

        return GateResult(
            passed=len(failures) == 0,
            checks=results,
            failures=failures,
            suggestions=suggestions,
        )
