"""
联动编排器 — 串起自进化 + 技能池 + 常驻调优

三个 cron 变成一条流水线：
  自进化完成 → 重建索引 → 更新常驻 → 发起下一轮扫描
"""

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EvolutionOrchestrator:
    """
    联动编排器。

    一条命令串联：
      1. 执行已批准的进化
      2. 进化完成后自动重建技能池索引
      3. 自动调优常驻列表
      4. 发起新一轮扫描
    """

    def __init__(self, target_dir: str = None, api_key: str = None, base_url: str = None, model: str = "gpt-4o-mini"):
        self.target_dir = target_dir or str(Path.home() / ".hermes")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._evo_manager = None
        self._skill_pool = None
        self._approval_mgr = None

    # ── 主入口：全联动 ────────────────────────────────────

    def sync_all(self, fast: bool = True, iterations: int = 3, k_candidates: int = 4) -> dict:
        """
        一键全联动。

        Returns: {evolved: [...], index_rebuilt: bool, core_tuned: int, 
                  audit_report: dict, snapshot_written: bool, next_candidates: int}
        """
        report = {
            "evolved": [],
            "index_rebuilt": False,
            "core_tuned": 0,
            "snapshot_written": False,
            "next_candidates": 0,
            "audit_report": {},
            "errors": [],
            "duration_s": 0,
        }
        t0 = time.time()

        try:
            # 步骤 1: 执行已批准的进化
            result = self._execute_approved(
                fast=fast, iterations=iterations, k_candidates=k_candidates
            )
            report["evolved"] = result

            if result:
                # 步骤 2: 重建技能池索引
                report["index_rebuilt"] = self._rebuild_pool_index()

                # 步骤 2.5: 跨 skill 冲突回归检测
                if report["index_rebuilt"]:
                    report["regression"] = self._cross_skill_regression(result)

            # 步骤 3: 自动调优常驻列表
            report["core_tuned"] = self._auto_tune()

            # 步骤 4: 写 Hermes 快照（只含 core 技能，零代码侵入）
            report["snapshot_written"] = self._write_snapshot()

            # 步骤 5: 发起新一轮自进化扫描
            report["next_candidates"] = self._scan_next()

        except Exception as exc:
            report["errors"].append(str(exc))
            logger.error(f"[orchestrator] sync error: {exc}")

        report["duration_s"] = round(time.time() - t0, 1)
        return report

    # ── 单步操作 ───────────────────────────────────────────

    def _execute_approved(self, fast: bool, iterations: int, k_candidates: int = 4) -> list[dict]:
        """执行已批准的进化任务。"""
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        from self_evolution.core.evolution_provider import EvolutionPhase
        from self_evolution import create_manager

        aem = AutoEvolutionManager()
        approved = aem.get_approved()
        if not approved:
            return []

        # 初始化进化引擎
        manager = create_manager()
        manager.initialize_all(
            self.target_dir,
            model=self.model,
            iterations=iterations,
            use_llm_eval=not fast,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        results = []
        for state in approved:
            logger.info(f"[orchestrator] executing approved: {state.skill_name}")
            aem.mark_executing(state.skill_name)

            result = manager.evolve_skill(
                state.skill_name,
                phase=EvolutionPhase.SKILL,
                iterations=iterations,
                auto_deploy=False,  # 不自动部署，等审批
            )

            if result.error:
                # 判断是否为门控拒绝
                is_constraint_reject = "constraint" in result.error.lower() or "holdout" in result.error.lower()
                entry = {
                    "skill": state.skill_name, "status": "rejected" if is_constraint_reject else "error",
                    "error": result.error, "rejected": is_constraint_reject,
                }
                if is_constraint_reject:
                    entry["failures"] = result.constraint_details.failures
                results.append(entry)
                # 记录拒绝计数
                aem.mark_rejected(state.skill_name)
            else:
                results.append({
                    "skill": state.skill_name,
                    "status": "done",
                    "improvement": round(result.improvement, 3),
                    "constraint_passed": result.constraint_passed,
                    "audit_report": getattr(result, 'audit_report', {}),
                })
                aem.mark_done(state.skill_name, result.evolved_score)
                logger.info(f"[orchestrator] {state.skill_name}: "
                           f"{result.baseline_score:.3f} → {result.evolved_score:.3f} "
                           f"({result.improvement:+.3f})")

        aem._save()
        return results

    def _rebuild_pool_index(self) -> bool:
        """重建技能池索引（进化后 skill 文本变了）。"""
        try:
            from self_evolution.skill_pool.pool import SkillPool
            pool = SkillPool()
            pool.build_index()
            logger.info(f"[orchestrator] pool index rebuilt: {len(pool._entries)} skills")
            return True
        except Exception as exc:
            logger.error(f"[orchestrator] index rebuild failed: {exc}")
            return False

    def _auto_tune(self) -> int:
        """自动调优常驻列表。"""
        try:
            from self_evolution.skill_pool.pool import SkillPool
            pool = SkillPool()
            pool.load_index()
            count = pool.auto_tune_core(top_n=8, min_usage=5)
            logger.info(f"[orchestrator] core tuned: {count} skills")
            return count
        except Exception as exc:
            logger.error(f"[orchestrator] auto-tune failed: {exc}")
            return 0

    def _write_snapshot(self) -> bool:
        """写入 Hermes 快照（只含 core 技能）。"""
        try:
            from self_evolution.skill_pool.pool import SkillPool
            pool = SkillPool()
            pool.load_index()
            count = pool.write_hermes_snapshot()
            logger.info(f"[orchestrator] snapshot: {count} core skills written, "
                       f"next session will see only these")
            return count > 0
        except Exception as exc:
            logger.error(f"[orchestrator] snapshot write failed: {exc}")
            return False

    def _scan_next(self) -> int:
        """发起新一轮自进化扫描。"""
        try:
            from self_evolution.triggers.auto_trigger import AutoEvolutionManager
            aem = AutoEvolutionManager()
            candidates = aem.scan_candidates()
            logger.info(f"[orchestrator] next scan: {len(candidates)} candidates")
            return len(candidates)
        except Exception as exc:
            logger.error(f"[orchestrator] scan failed: {exc}")
            return 0

    # ── 报告 ───────────────────────────────────────────────

    def format_report(self, report: dict) -> str:
        """格式化联动报告。"""
        lines = ["🔄 **自进化联动报告**\n"]

        # 进化结果
        evolved = report.get("evolved", [])
        if evolved:
            done = [r for r in evolved if r["status"] == "done"]
            rejected = [r for r in evolved if r.get("rejected")]
            errors = [r for r in evolved if r["status"] == "error" and not r.get("rejected")]

            if done:
                lines.append(f"### 已执行进化 ({len(done)})")
                for r in done:
                    imp = f" ({r.get('improvement', 0):+.3f})"
                    lines.append(f"  ✅ `{r['skill']}`{imp}")
                lines.append("")

            if rejected:
                lines.append(f"### 门控拒绝 ({len(rejected)})")
                for r in rejected:
                    lines.append(f"  🚫 `{r['skill']}`: {r.get('error', 'unknown')}")
                    if r.get("failures"):
                        lines.append(f"     维度: {', '.join(r['failures'])}")
                lines.append("")
                lines.append(f"  → 已归档到 `~/.hermes/.evolution_rejects/`")
                lines.append(f"  → 连续拒绝3次将自动降低触发阈值")
                lines.append("")

            if errors:
                lines.append(f"### 执行错误 ({len(errors)})")
                for r in errors:
                    lines.append(f"  ❌ `{r['skill']}`: {r.get('error', 'unknown')}")
                lines.append("")

        # 索引重建
        lines.append(f"### 技能池索引")
        lines.append(f"  {'✅ 已重建' if report.get('index_rebuilt') else '❌ 未重建'}")
        lines.append("")

        # 常驻调优
        core = report.get("core_tuned", 0)
        lines.append(f"### 常驻技能")
        lines.append(f"  当前 {core} 个 (自动调优)")
        lines.append("")

        # 下一轮
        next_n = report.get("next_candidates", 0)
        lines.append(f"### 下一轮候选")
        lines.append(f"  {next_n} 个 skill 待审批" if next_n else "  暂无")
        lines.append("")

        # 耗时 + 错误
        lines.append(f"⏱ 耗时: {report.get('duration_s', 0)}s")

        # 跨 skill 冲突回归
        regression = report.get("regression", {})
        if regression and regression.get("conflicts"):
            lines.append("")
            lines.append(f"### 跨 Skill 回归检测")
            for conflict in regression["conflicts"]:
                lines.append(f"  ⚠ `{conflict['evolved']}` 与 `{conflict['affected']}` 路由冲突: "
                           f"相似度 {conflict['similarity']:.3f}")

        if report.get("errors"):
            lines.append(f"⚠ 错误: {', '.join(report['errors'])}")

        return "\n".join(lines)

    # ── 跨 skill 冲突回归 ─────────────────────────────────

    def _cross_skill_regression(self, results: list[dict]) -> dict:
        """
        进化后检查被进化的 skill 是否影响了其他 skill 的路由。

        方法：用 SkillPool 搜索被进化 skill 的关键词，
        检查是否有其他 skill 的相似度突然升高（可能路由冲突）。
        """
        conflicts = []
        try:
            from self_evolution.skill_pool.pool import SkillPool
            pool = SkillPool()
            if not pool.load_index():
                return {"conflicts": [], "checked": 0}

            for r in results:
                if r.get("status") != "done":
                    continue
                skill_name = r["skill"]
                # 找到该 skill 的条目
                target_entry = None
                for e in pool._entries:
                    if e.name == skill_name:
                        target_entry = e
                        break
                if not target_entry:
                    continue

                # 用该 skill 的描述搜索
                similar = pool.search(target_entry.description, k=8, include_core=True)
                # 检查是否有其他 skill 的相似度 >= 0.25（可能冲突）
                for e in similar:
                    if e.name == skill_name:
                        continue
                    # 从 results 中找该 skill 是否也被进化
                    is_evolved = any(
                        done_r.get("status") == "done" and done_r["skill"] == e.name
                        for done_r in results
                    )
                    # 用 embedding 直接算 cosine
                    import numpy as np
                    idx_a = next((i for i, x in enumerate(pool._entries) if x.name == skill_name), None)
                    idx_b = next((i for i, x in enumerate(pool._entries) if x.name == e.name), None)
                    if idx_a is not None and idx_b is not None:
                        sim = np.dot(pool._embeddings[idx_a], pool._embeddings[idx_b]) / (
                            np.linalg.norm(pool._embeddings[idx_a]) * np.linalg.norm(pool._embeddings[idx_b]) + 1e-8
                        )
                        if float(sim) > 0.25:
                            conflicts.append({
                                "evolved": skill_name,
                                "affected": e.name,
                                "similarity": round(float(sim), 3),
                                "co_evolved": is_evolved,
                            })

            unique_conflicts = []
            seen_pairs = set()
            for c in sorted(conflicts, key=lambda x: -x["similarity"]):
                pair = tuple(sorted([c["evolved"], c["affected"]]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    unique_conflicts.append(c)

            return {"conflicts": unique_conflicts[:5], "checked": len(results)}

        except Exception as exc:
            logger.error(f"[orchestrator] cross-skill regression failed: {exc}")
            return {"conflicts": [], "checked": 0, "error": str(exc)}
