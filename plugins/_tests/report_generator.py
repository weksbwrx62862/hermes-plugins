"""报告生成器

将 ResultCollector 收集的测试结果生成为完整的 Markdown 报告, 包含 7 个章节:
1. 测试概览
2. 每个插件的详细测试结果
3. 资源占用汇总表
4. 异常行为与错误信息
5. 性能问题清单
6. 改进建议
7. 测试环境信息
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from test_framework import ResultCollector, TestResult, short_status


# 状态徽章
BADGE = {
    "PASS": "✅",
    "FAIL": "❌",
    "WARN": "⚠️",
    "TIMEOUT": "⏱️",
    "SKIP": "⏭️",
}

# 性能阈值
PERF_CPU_WARN_SECONDS = 5.0
PERF_MEM_WARN_BYTES = 100 * 1024 * 1024


class ReportGenerator:
    """Markdown 报告生成器。"""

    def __init__(
        self,
        result_collector: ResultCollector,
        env_info: Dict[str, Any],
        resource_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.collector = result_collector
        self.env_info = env_info
        # 每插件资源快照: plugin_name -> format_snapshot 输出
        self.resource_snapshots = resource_snapshots or {}

    # -- 章节生成 ----------------------------------------------------------

    def _section_overview(self) -> str:
        """章节 1: 测试概览。"""
        summary = self.collector.summary()
        total = self.collector.total()
        plugins = self.collector.plugins()
        # 排除全局测试条目(如 "(global)"), 仅统计真实插件数
        real_plugins = [p for p in plugins if not p.startswith("(")]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "## 1. 测试概览",
            "",
            f"- **测试时间**: {now}",
            f"- **插件总数**: {len(real_plugins)}",
            f"- **测试用例总数**: {total}",
            f"- **通过 (PASS)**: {summary.get('PASS', 0)}",
            f"- **失败 (FAIL)**: {summary.get('FAIL', 0)}",
            f"- **超时 (TIMEOUT)**: {summary.get('TIMEOUT', 0)}",
            f"- **警告 (WARN)**: {summary.get('WARN', 0)}",
            f"- **跳过 (SKIP)**: {summary.get('SKIP', 0)}",
            "",
        ]
        # 简要状态条
        if summary.get("FAIL", 0) == 0 and summary.get("TIMEOUT", 0) == 0:
            lines.append("> 🎉 无失败用例, 整体状态良好。")
        else:
            lines.append("> ⚠️ 存在失败或超时用例, 请关注下文详情。")
        lines.append("")
        return "\n".join(lines)

    def _section_plugin_details(self) -> str:
        """章节 2: 每个插件的详细测试结果。"""
        lines = ["## 2. 插件详细测试结果", ""]
        for plugin in self.collector.plugins():
            results = self.collector.by_plugin(plugin)
            # 该插件整体状态: 取最严重
            severity = {"PASS": 0, "WARN": 1, "SKIP": 1, "TIMEOUT": 2, "FAIL": 2}
            overall = max((r.status for r in results), key=lambda s: severity.get(s, 2), default="SKIP")
            badge = BADGE.get(overall, "❓")
            lines.append(f"### {badge} {plugin}")
            lines.append("")

            # 按测试类型分组
            by_test: Dict[str, List[TestResult]] = {}
            for r in results:
                by_test.setdefault(r.test_name, []).append(r)

            for test_name, group in by_test.items():
                lines.append(f"**测试类型**: `{test_name}`")
                lines.append("")
                lines.append("| 状态 | 用例 | 耗时(ms) | 说明 |")
                lines.append("|------|------|----------|------|")
                for r in group:
                    badge = BADGE.get(r.status, "❓")
                    msg = _safe_str(r.message).replace("|", "\\|")
                    if len(msg) > 200:
                        msg = msg[:200] + "..."
                    lines.append(f"| {badge} | {r.plugin_name} | {r.duration_ms:.1f} | {msg} |")
                lines.append("")

            # 显示异常 traceback(若有)
            for r in results:
                if r.exception and r.status in ("FAIL", "TIMEOUT"):
                    lines.append(f"<details><summary>📂 {r.test_name} 异常 traceback</summary>")
                    lines.append("")
                    lines.append("```")
                    tb = r.exception if len(r.exception) < 4000 else r.exception[:4000] + "\n... (截断)"
                    lines.append(tb)
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

        return "\n".join(lines)

    def _section_resource_summary(self) -> str:
        """章节 3: 资源占用汇总表。"""
        lines = ["## 3. 资源占用汇总", ""]
        if not self.resource_snapshots:
            lines.append("无资源采样数据。")
            lines.append("")
            return "\n".join(lines)

        lines.append("| 插件 | CPU(s) | 壁钟(s) | 内存峰值(MB) | 当前内存(MB) | FD | 子进程 | 异常 |")
        lines.append("|------|--------|---------|-------------|-------------|----|--------|------|")
        for plugin, snap in self.resource_snapshots.items():
            cpu = snap.get("cpu_time", 0.0)
            wall = snap.get("wall_time", 0.0)
            mem_peak = snap.get("mem_peak_mb", 0.0)
            mem_cur = snap.get("mem_current_mb", 0.0)
            fd = snap.get("fd_count", 0)
            sub = snap.get("subprocess_count", 0)
            anomalies = snap.get("anomalies", [])
            anomaly_mark = " ⚠️ " + "; ".join(anomalies) if anomalies else "✅"
            lines.append(
                f"| {plugin} | {cpu:.3f} | {wall:.3f} | {mem_peak:.2f} | {mem_cur:.2f} | {fd} | {sub} | {anomaly_mark} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _section_anomalies(self) -> str:
        """章节 4: 异常行为与错误信息(按严重程度排序)。"""
        lines = ["## 4. 异常行为与错误信息", ""]
        # 按严重程度排序: FAIL/TIMEOUT > WARN
        order = {"FAIL": 0, "TIMEOUT": 1, "WARN": 2, "SKIP": 3, "PASS": 4}
        all_results = sorted(
            self.collector.results,
            key=lambda r: (order.get(r.status, 9), r.plugin_name, r.test_name),
        )
        abnormal = [r for r in all_results if r.status in ("FAIL", "TIMEOUT", "WARN")]
        if not abnormal:
            lines.append("无异常行为或错误。🎉")
            lines.append("")
            return "\n".join(lines)

        for r in abnormal:
            badge = BADGE.get(r.status, "❓")
            lines.append(f"- {badge} **[{r.status}]** `{r.plugin_name}` / `{r.test_name}`: {_safe_str(r.message)}")
        lines.append("")
        return "\n".join(lines)

    def _section_performance(self) -> str:
        """章节 5: 性能问题清单。"""
        lines = ["## 5. 性能问题清单", ""]
        perf_issues: List[str] = []

        # 从资源快照中筛选性能问题
        for plugin, snap in self.resource_snapshots.items():
            cpu = snap.get("cpu_time", 0.0)
            mem_peak = snap.get("mem_peak", 0.0)
            if cpu > PERF_CPU_WARN_SECONDS:
                perf_issues.append(f"插件 `{plugin}` register 阶段 CPU 时间 {cpu:.2f}s 超过 {PERF_CPU_WARN_SECONDS}s")
            if mem_peak > PERF_MEM_WARN_BYTES:
                perf_issues.append(f"插件 `{plugin}` 内存峰值 {mem_peak / 1024 / 1024:.1f}MB 超过 100MB")

        # 从 register 测试结果中筛选耗时>5s
        for r in self.collector.by_test("register"):
            if r.duration_ms > PERF_CPU_WARN_SECONDS * 1000:
                perf_issues.append(f"插件 `{r.plugin_name}` register 测试耗时 {r.duration_ms:.0f}ms 超过 5s")

        if not perf_issues:
            lines.append("未检出性能问题。✅")
            lines.append("")
            return "\n".join(lines)

        for issue in perf_issues:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")
        return "\n".join(lines)

    def _section_suggestions(self) -> str:
        """章节 6: 改进建议(基于测试发现自动生成)。"""
        lines = ["## 6. 改进建议", ""]
        suggestions: List[str] = []

        for plugin in self.collector.plugins():
            results = self.collector.by_plugin(plugin)
            has_fail = any(r.status in ("FAIL", "TIMEOUT") for r in results)
            has_warn = any(r.status == "WARN" for r in results)

            # 错误处理建议
            eh = next((r for r in results if r.test_name == "error_handling"), None)
            if eh and eh.details.get("try_except_count", 0) == 0 and not has_fail:
                suggestions.append(f"`{plugin}` 代码中未检出 try/except 块, 建议在关键路径(register / 工具 handler)增加错误处理。")

            # register 抛异常建议
            reg = next((r for r in results if r.test_name == "register"), None)
            if reg and reg.status in ("FAIL", "TIMEOUT"):
                suggestions.append(f"`{plugin}` register 调用失败或超时, 建议排查初始化逻辑, 避免阻塞或依赖缺失。")

            # manifest 不一致建议
            man = next((r for r in results if r.test_name == "manifest"), None)
            if man and man.status == "FAIL":
                suggestions.append(f"`{plugin}` plugin.yaml 清单校验未通过, 建议修正: {_safe_str(man.message)}。")

            # 工具未注册建议
            if reg and reg.details.get("missing_tools"):
                suggestions.append(f"`{plugin}` 声明的工具未全部注册: {reg.details['missing_tools']}, 建议核对 register 实现。")

            # 资源异常建议
            snap = self.resource_snapshots.get(plugin)
            if snap and snap.get("has_anomaly"):
                suggestions.append(f"`{plugin}` 存在资源占用异常({'; '.join(snap.get('anomalies', []))}), 建议优化初始化与内存使用。")

            # 幂等性建议
            if reg and reg.details.get("idempotent") == "WARN":
                suggestions.append(f"`{plugin}` 重复调用 register 会抛异常, 建议使其幂等(避免重复注册副作用)。")

            # 冒烟测试失败
            sm = next((r for r in results if r.test_name == "smoke"), None)
            if sm and sm.status == "FAIL":
                suggestions.append(f"`{plugin}` 冒烟测试中工具调用失败, 建议检查工具 handler 对空参数的容错。")

        if not suggestions:
            lines.append("未发现需要特别改进的项目。🎉")
            lines.append("")
            return "\n".join(lines)

        for s in suggestions:
            lines.append(f"- 💡 {s}")
        lines.append("")
        return "\n".join(lines)

    def _section_environment(self) -> str:
        """章节 7: 测试环境信息。"""
        lines = ["## 7. 测试环境信息", ""]
        env = self.env_info
        lines.append(f"- **Python 版本**: {env.get('python_version', '未知')}")
        lines.append(f"- **Hermes 版本**: {env.get('hermes_version', '未知')}")
        lines.append(f"- **操作系统**: {env.get('os', '未知')}")
        lines.append(f"- **测试时间**: {env.get('test_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        lines.append(f"- **插件目录**: {env.get('plugins_dir', '未知')}")
        lines.append(f"- **测试执行器**: Hermes 插件测试框架")
        lines.append("")
        return "\n".join(lines)

    # -- 主入口 ------------------------------------------------------------

    def generate(self) -> str:
        """生成完整 Markdown 报告。"""
        summary = self.collector.summary()
        header = [
            "# Hermes 插件功能测试报告",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 通过 {summary.get('PASS', 0)} / 失败 {summary.get('FAIL', 0)} / "
            f"警告 {summary.get('WARN', 0)} / 超时 {summary.get('TIMEOUT', 0)} / "
            f"跳过 {summary.get('SKIP', 0)}",
            "",
            "---",
            "",
        ]
        sections = [
            self._section_overview(),
            self._section_plugin_details(),
            self._section_resource_summary(),
            self._section_anomalies(),
            self._section_performance(),
            self._section_suggestions(),
            self._section_environment(),
        ]
        return "\n".join(header) + "\n---\n\n".join(sections)

    def save(self, report_str: str, path: Path) -> None:
        """保存报告到文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_str, encoding="utf-8")


def _safe_str(value: Any) -> str:
    """安全转字符串, 避免非字符串对象导致格式化异常。"""
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return repr(value)
