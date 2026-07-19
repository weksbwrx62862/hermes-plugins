#!/usr/bin/env python3
"""
MUSE 启发的 5 阶段 Skill 生命周期流水线。

五阶段流程:
  Phase 1: CREATE  — 生成/提取 Skill 包 (SKILL.md + scripts/ + tests/ + resources/)
  Phase 2: TEST    — 跑 tests/ 目录中的所有测试
  Phase 3: REGISTER— 测试通过后注册入 Skill Bank
  Phase 4: MEMORY  — 追加 Skill 级记忆 (.memory.md)
  Phase 5: PRUNE   — 裁剪低质量/未使用的 Skill

设计原则（来自 MUSE-Autoskill）:
  - 测试驱动质量守门：只有通过测试的 Skill 才能注册
  - Skill 级记忆：每 Skill 一本 .memory.md 使用笔记
  - 渐进式检索：目录先于正文，100 Skill 仅 5-10K token
  - 闭环自愈：评估发现问题 -> 精炼执行修复
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("hermes_plugins.skill_pool.pipeline")

SKILLS_DIR = Path.home() / ".hermes" / "skills"


class SkillPipeline:
    """5 阶段 Skill 生命周期流水线。"""

    def __init__(self, skill_pool=None):
        self._pool = skill_pool
        self._stats = {
            "created": 0,
            "tested": 0,
            "registered": 0,
            "memory_appended": 0,
            "pruned": 0,
            "failed_creation": 0,
            "failed_tests": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _find_skill_dir(self, name: str) -> Optional[Path]:
        """查找 skill 目录。"""
        for sf in SKILLS_DIR.rglob("SKILL.md"):
            try:
                content = sf.read_text(encoding="utf-8")
                for line in content.split("\n")[:10]:
                    if f"name: {name}" in line:
                        return sf.parent
            except Exception:
                continue
        return None

    # Phase 1: CREATE
    def create_skill_skeleton(self, name: str, description: str,
                              category: str = "") -> dict:
        """生成 Skill 骨架目录。"""
        skill_dir = SKILLS_DIR / category / name if category else SKILLS_DIR / name
        if skill_dir.exists():
            return {"status": "failed", "reason": f"Already exists: {skill_dir}"}
        try:
            (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
            (skill_dir / "tests").mkdir(parents=True, exist_ok=True)
            (skill_dir / "resources").mkdir(parents=True, exist_ok=True)

            fm = f"---\nname: {name}\ndescription: {description}\n"
            if category:
                fm += f"category: {category}\n"
            fm += "---\n\n"
            fm += f"# {name}\n\n{description}\n\n## Usage\n\nTODO\n"
            (skill_dir / "SKILL.md").write_text(fm, encoding="utf-8")

            test_content = f'''#!/usr/bin/env python3
"""{name} 基础测试 - 验证目录结构完整性。"""
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent

def test_structure():
    assert (SKILL_DIR / "SKILL.md").exists(), "SKILL.md missing"
    assert (SKILL_DIR / "scripts").is_dir(), "scripts/ missing"
    assert (SKILL_DIR / "tests").is_dir(), "tests/ missing"
    print("OK: 目录结构完整")

if __name__ == "__main__":
    test_structure()
'''
            (skill_dir / "tests" / "test_basic.py").write_text(test_content, encoding="utf-8")
            (skill_dir / "tests" / "test_basic.py").chmod(0o755)
            self._stats["created"] += 1
            return {"status": "ok", "path": str(skill_dir),
                    "reason": f"Created: {skill_dir}"}
        except Exception as e:
            self._stats["failed_creation"] += 1
            return {"status": "failed", "reason": str(e)}

    # Phase 2: TEST
    def test_skill(self, name: str) -> dict:
        """跑 Skill 的 tests/ 目录中的所有测试。"""
        if self._pool:
            result = self._pool.evaluate_skill(name)
        else:
            result = self._standalone_test(name)
        self._stats["tested"] += 1
        if not result["passed"]:
            self._stats["failed_tests"] += 1
        return result

    def _standalone_test(self, name: str) -> dict:
        """独立测试（不依赖 SkillPool）。"""
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            return {"passed": False, "total": 0, "passed_count": 0,
                    "failures": [f"Not found: {name}"], "output": ""}
        test_dir = skill_dir / "tests"
        if not test_dir.exists():
            return {"passed": True, "total": 0, "passed_count": 0,
                    "failures": [], "output": "No tests dir - gate passes"}
        results = {"total": 0, "passed_count": 0, "failures": [], "output": ""}
        outputs = []
        for test_file in sorted(test_dir.iterdir()):
            if test_file.suffix not in (".py", ".sh"):
                continue
            results["total"] += 1
            try:
                cmd = ["python3", str(test_file)] if test_file.suffix == ".py" else ["bash", str(test_file)]
                proc = subprocess.run(cmd, cwd=str(skill_dir),
                                      capture_output=True, text=True, timeout=30)
                outputs.append(f"--- {test_file.name} ---\n{proc.stdout}")
                if proc.returncode == 0:
                    results["passed_count"] += 1
                else:
                    results["failures"].append(
                        f"{test_file.name}: exit={proc.returncode}\n{proc.stderr[:500]}")
            except subprocess.TimeoutExpired:
                results["failures"].append(f"{test_file.name}: timeout")
            except Exception as e:
                results["failures"].append(f"{test_file.name}: {e}")
        results["passed"] = len(results["failures"]) == 0
        results["output"] = "\n".join(outputs)
        return results

    # Phase 3: REGISTER
    def register_skill(self, name: str, test_result: dict,
                       force: bool = False) -> dict:
        """注册 Skill 入 Bank（测试守门）。"""
        if not force and not test_result.get("passed", False):
            return {"status": "failed",
                    "reason": f"Tests failed: {test_result.get('passed_count',0)}/{test_result.get('total',0)}"}
        if self._pool:
            result = self._pool.register_with_gate(name, check_tests=not force)
        else:
            result = {"status": "ok", "reason": "Registered (standalone)"}
        if result["status"] == "ok":
            self._stats["registered"] += 1
        return result

    # Phase 4: MEMORY
    def append_skill_memory(self, name: str, note: str,
                            test_result: Optional[dict] = None) -> dict:
        """追加 Skill 级记忆 (.memory.md)。"""
        memory_entry = note
        if test_result:
            info = f"Tests: {test_result.get('passed_count',0)}/{test_result.get('total',0)} passed"
            if test_result.get("failures"):
                memory_entry = f"{info}\nFailures: {'; '.join(test_result['failures'][:3])}\n{note}"
            else:
                memory_entry = f"{info} - all passed\n{note}"

        if self._pool:
            ok = self._pool.append_memory(name, memory_entry)
        else:
            skill_dir = self._find_skill_dir(name)
            if not skill_dir:
                return {"status": "failed", "name": name, "reason": "Not found"}
            memory_path = skill_dir / ".memory.md"
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(memory_path, "a", encoding="utf-8") as f:
                f.write(f"\n## [{ts}]\n{memory_entry}\n")
            ok = True

        if ok:
            self._stats["memory_appended"] += 1
        return {"status": "ok" if ok else "failed", "name": name}

    # Phase 5: PRUNE
    def prune_skills(self, min_usage: int = 0, max_fail_rate: float = 0.5,
                     dry_run: bool = True) -> dict:
        """裁剪低质量/未使用 Skill。"""
        if self._pool:
            result = self._pool.prune_skills(min_usage=min_usage,
                                              max_fail_rate=max_fail_rate,
                                              dry_run=dry_run)
        else:
            result = {"pruned": 0, "candidates": [], "dry_run": dry_run,
                      "reason": "No pool backend"}
        if not dry_run:
            self._stats["pruned"] += result.get("pruned", 0)
        return result

    # Full pipeline
    def run_full_pipeline(self, names: list[str]) -> dict:
        """运行完整 5 阶段流水线（批量）。"""
        results = []
        for name in names:
            stages = {}
            stages["test"] = self.test_skill(name)
            stages["register"] = self.register_skill(name, stages["test"])
            stages["memory"] = self.append_skill_memory(
                name,
                "Pipeline run: passed" if stages["test"]["passed"] else "Pipeline run: failed - needs refinement",
                test_result=stages["test"]
            )
            results.append({"name": name, "stages": stages})

        prune_result = self.prune_skills(dry_run=True)

        return {
            "summary": {
                "total": len(names),
                "passed": sum(1 for r in results if r["stages"]["test"]["passed"]),
                "failed": sum(1 for r in results if not r["stages"]["test"]["passed"]),
                "registered": sum(1 for r in results if r["stages"]["register"]["status"] == "ok"),
                "prune_candidates": len(prune_result.get("candidates", [])),
            },
            "results": results,
            "prune": prune_result,
        }


# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="MUSE 5-stage Skill Pipeline")
    parser.add_argument("--create", help="Create: name:description[:category]")
    parser.add_argument("--test", help="Test skill by name")
    parser.add_argument("--pipeline", help="Run full pipeline: name1,name2")
    parser.add_argument("--prune", action="store_true", help="Run prune")
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    pipeline = SkillPipeline()
    result = None

    if args.create:
        parts = args.create.split(":", 2)
        result = pipeline.create_skill_skeleton(
            parts[0], parts[1] if len(parts) > 1 else "",
            parts[2] if len(parts) > 2 else "")
    elif args.test:
        result = pipeline.test_skill(args.test)
    elif args.pipeline:
        names = [n.strip() for n in args.pipeline.split(",") if n.strip()]
        result = pipeline.run_full_pipeline(names)
    elif args.prune:
        result = pipeline.prune_skills(dry_run=not args.no_dry_run)

    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
