"""Hermes 插件测试统一入口

用法:
    python3 run_all.py --dry-run   # 仅列出待测试插件, 不执行测试
    python3 run_all.py             # 执行完整测试并生成报告

完整模式流程:
1. 收集测试环境信息
2. 发现所有插件
3. 依次执行: manifest -> import -> register(含资源采样) -> smoke -> error_handling
4. 所有插件测完后执行 main_interaction(全局验证)
5. 生成 Markdown 报告并保存
6. 打印报告路径与简要统计
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保本目录在 sys.path 上, 以便 import 同级模块
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import yaml  # type: ignore

from test_framework import (
    PluginDiscovery,
    PluginInfo,
    ResultCollector,
    TestResult,
    format_traceback,
)
from resource_sampler import ResourceSampler
from report_generator import ReportGenerator
from test_manifest import test_manifest
from test_import import test_import
from test_register import test_register
from test_smoke import test_smoke
from test_error_handling import test_error_handling
from test_main_interaction import test_main_interaction


# 测试产物目录
TESTS_DIR = Path.home() / ".hermes" / "plugins" / "_tests"


# ---------------------------------------------------------------------------
# 环境信息收集
# ---------------------------------------------------------------------------

def _collect_env_info(plugins: List[PluginInfo]) -> Dict[str, Any]:
    """收集测试环境信息。"""
    env: Dict[str, Any] = {}
    env["python_version"] = sys.version.split()[0]
    env["test_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env["plugins_dir"] = str(Path.home() / ".hermes" / "plugins")

    # 操作系统信息: 优先用 uname -a, 失败则降级到 platform
    try:
        out = subprocess.run(
            ["uname", "-a"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            env["os"] = out.stdout.strip()
        else:
            env["os"] = platform.platform()
    except Exception:
        env["os"] = platform.platform()

    # Hermes 版本: 多种途径尝试
    env["hermes_version"] = _detect_hermes_version()

    return env


def _detect_hermes_version() -> str:
    """尝试从多个来源探测 Hermes 版本。"""
    # 1. pyproject.toml
    pyproject = Path.home() / ".hermes" / "hermes-agent" / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("version") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass

    # 2. importlib.metadata
    try:
        import importlib.metadata as md
        return md.version("hermes-agent")
    except Exception:
        pass

    # 3. config.yaml 中的 version 字段
    config_path = Path.home() / ".hermes" / "config.yaml"
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                v = data.get("version")
                if v:
                    return str(v)
        except Exception:
            pass

    return "未知"


# ---------------------------------------------------------------------------
# 单插件测试编排
# ---------------------------------------------------------------------------

def _load_manifest_dict(plugin_info: PluginInfo) -> Dict[str, Any]:
    """加载 plugin.yaml 为字典, 失败返回空字典。"""
    try:
        text = Path(plugin_info.manifest_path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_add(
    collector: ResultCollector,
    plugin_name: str,
    test_name: str,
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """安全执行单个测试函数, 捕获未预期异常并记为 FAIL。

    成功时将返回的 TestResult 加入 collector。test_register 返回 (TestResult, ctx)
    元组, 此时只将 TestResult 部分加入 collector, 并将整个元组返回给调用方。
    """
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        # 将返回的 TestResult 加入 collector
        if isinstance(result, TestResult):
            collector.add(result)
        elif isinstance(result, tuple) and result and isinstance(result[0], TestResult):
            # test_register 返回 (TestResult, ctx)
            collector.add(result[0])
        return result
    except Exception as exc:
        tb = format_traceback(exc)
        collector.add(TestResult(
            plugin_name=plugin_name,
            test_name=test_name,
            status="FAIL",
            message=f"测试执行抛出未捕获异常: {exc}",
            details={"exception_type": type(exc).__name__},
            duration_ms=(time.perf_counter() - start) * 1000,
            exception=tb,
        ))
        return None


def _test_single_plugin(
    plugin_info: PluginInfo,
    collector: ResultCollector,
    resource_snapshots: Dict[str, Dict[str, Any]],
) -> None:
    """对单个插件执行全部测试用例。"""
    manifest = _load_manifest_dict(plugin_info)

    # 1. manifest 静态校验
    _safe_add(collector, plugin_info.name, "manifest", test_manifest, plugin_info)

    # 2. 模块导入测试
    import_result = _safe_add(
        collector, plugin_info.name, "import", test_import, plugin_info,
    )
    if import_result is None or import_result.status == "FAIL":
        # 导入失败, 后续测试无法进行
        return
    module = import_result.details.get("_module")
    if module is None:
        collector.add(TestResult(
            plugin_name=plugin_info.name,
            test_name="register",
            status="SKIP",
            message="模块加载成功但未取到模块引用, 跳过后续测试",
        ))
        return

    # 3. register 测试, 嵌入资源采样
    sampler = ResourceSampler()
    sampler.start()
    reg_outcome = _safe_add(
        collector, plugin_info.name, "register",
        test_register, plugin_info, module, manifest,
    )
    snap = sampler.stop()
    resource_snapshots[plugin_info.name] = ResourceSampler.format_snapshot(snap)

    # 取得 ctx 供 smoke 使用
    ctx = None
    if isinstance(reg_outcome, tuple) and len(reg_outcome) == 2:
        ctx = reg_outcome[1]

    # 4. smoke 冒烟测试(需要 ctx)
    if ctx is not None:
        _safe_add(
            collector, plugin_info.name, "smoke",
            test_smoke, plugin_info, ctx, manifest,
        )
    else:
        collector.add(TestResult(
            plugin_name=plugin_info.name,
            test_name="smoke",
            status="SKIP",
            message="register 未返回 ctx, 跳过冒烟测试",
        ))

    # 5. 错误处理测试
    _safe_add(
        collector, plugin_info.name, "error_handling",
        test_error_handling, plugin_info, module,
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes 插件全面功能测试框架")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅列出待测试插件, 不执行测试",
    )
    args = parser.parse_args()

    # 发现插件
    discovery = PluginDiscovery()
    plugins = discovery.discover()

    if args.dry_run:
        print(f"待测试插件共 {len(plugins)} 个:")
        for i, pi in enumerate(plugins, 1):
            print(f"  [{i:>2}/{len(plugins)}] {pi.name}")
            print(f"          路径: {pi.path}")
        print(f"\n共 {len(plugins)} 个插件待测试(dry-run 模式, 未执行测试)。")
        return 0

    print("=" * 60)
    print("Hermes 插件全面功能测试")
    print("=" * 60)
    print(f"发现 {len(plugins)} 个待测试插件。")
    print()

    # 收集环境信息
    env_info = _collect_env_info(plugins)
    print(f"Python: {env_info['python_version']}")
    print(f"Hermes: {env_info['hermes_version']}")
    print(f"OS: {env_info['os'][:80]}")
    print()

    collector = ResultCollector()
    resource_snapshots: Dict[str, Dict[str, Any]] = {}
    total = len(plugins)

    # 逐插件测试
    for i, pi in enumerate(plugins, 1):
        print(f"[{i}/{total}] 测试 {pi.name}...")
        t0 = time.perf_counter()
        try:
            _test_single_plugin(pi, collector, resource_snapshots)
        except Exception as exc:
            # 兜底: 任何未捕获异常都记为 FAIL, 不中断后续插件
            collector.add(TestResult(
                plugin_name=pi.name,
                test_name="(orchestration)",
                status="FAIL",
                message=f"测试编排抛出未捕获异常: {exc}",
                exception=format_traceback(exc),
            ))
        dt = time.perf_counter() - t0
        plugin_results = collector.by_plugin(pi.name)
        statuses = [r.status for r in plugin_results]
        print(f"      完成({dt:.1f}s) -> {statuses}")
        sys.stdout.flush()

    print()
    print(f"[{total}/{total}] 执行全局交互验证 main_interaction...")
    try:
        global_result = test_main_interaction(plugins)
        collector.add(global_result)
        print(f"      完成 -> {global_result.status}: {global_result.message}")
    except Exception as exc:
        collector.add(TestResult(
            plugin_name="(global)",
            test_name="main_interaction",
            status="FAIL",
            message=f"全局交互验证抛出未捕获异常: {exc}",
            exception=format_traceback(exc),
        ))
        print(f"      失败: {exc}")

    # 生成报告
    print()
    print("生成测试报告...")
    report_gen = ReportGenerator(collector, env_info, resource_snapshots)
    report_str = report_gen.generate()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = TESTS_DIR / f"report-{timestamp}.md"
    report_gen.save(report_str, report_path)

    # 打印简要统计
    summary = collector.summary()
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"插件总数: {len(plugins)}")
    print(f"测试用例总数: {collector.total()}")
    print(f"  ✅ PASS:    {summary.get('PASS', 0)}")
    print(f"  ❌ FAIL:    {summary.get('FAIL', 0)}")
    print(f"  ⏱️ TIMEOUT: {summary.get('TIMEOUT', 0)}")
    print(f"  ⚠️  WARN:    {summary.get('WARN', 0)}")
    print(f"  ⏭️ SKIP:    {summary.get('SKIP', 0)}")
    print()
    print(f"报告已保存: {report_path}")
    return 0 if summary.get("FAIL", 0) == 0 and summary.get("TIMEOUT", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
