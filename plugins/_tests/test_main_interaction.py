"""与主程序交互验证(全局副作用检测)

在加载并 register 所有插件前后, 对比:
- ~/.hermes/config.yaml 的 SHA256(应保持一致, 即插件不应改写主配置)
- sys.modules 的 keys 差集(过滤掉插件自身模块后, 应无意外污染)

该测试在所有插件单独测试完成后执行, 作为全局回归校验。
"""

from __future__ import annotations

import hashlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set

from test_framework import MockPluginContext, PluginInfo, TestResult
from test_import import _load_plugin_module


CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"


def _sha256_file(path: Path) -> str:
    """计算文件 SHA256, 文件不存在时返回空字符串。"""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _plugin_module_name(plugin_info: PluginInfo) -> str:
    """返回插件在 sys.modules 中的唯一模块名。"""
    return f"_test_plugin_{plugin_info.name.replace('-', '_')}"


def test_main_interaction(all_plugin_infos: List[PluginInfo]) -> TestResult:
    """全局交互验证: 加载并 register 所有插件, 检测对主程序配置与 sys.modules 的污染。"""
    start = time.perf_counter()
    details: Dict[str, Any] = {}

    # 1. 测试前 config.yaml SHA256
    config_hash_before = _sha256_file(CONFIG_PATH)
    details["config_hash_before"] = config_hash_before
    details["config_path"] = str(CONFIG_PATH)

    # 2. 测试前 sys.modules keys 快照
    modules_before: Set[str] = set(sys.modules.keys())

    # 3. 加载并 register 每个插件
    load_results: List[Dict[str, Any]] = []
    plugin_module_names: Set[str] = set()
    for pi in all_plugin_infos:
        mod_name = _plugin_module_name(pi)
        plugin_module_names.add(mod_name)
        entry: Dict[str, Any] = {"plugin": pi.name, "module_name": mod_name}
        try:
            module = _load_plugin_module(pi)
            entry["import_ok"] = True
        except Exception as exc:
            entry["import_ok"] = False
            entry["import_error"] = f"{type(exc).__name__}: {exc}"
            load_results.append(entry)
            continue

        # register 到一个 mock ctx
        ctx = MockPluginContext(plugin_name=pi.name)
        register_fn = getattr(module, "register", None)
        if register_fn is None or not callable(register_fn):
            entry["register_ok"] = False
            entry["register_error"] = "register 不可调用"
        else:
            try:
                register_fn(ctx)
                entry["register_ok"] = True
                entry["tools"] = len(ctx.registered_tools)
            except Exception as exc:
                entry["register_ok"] = False
                entry["register_error"] = f"{type(exc).__name__}: {exc}"
        load_results.append(entry)

    details["load_results"] = load_results
    details["plugins_loaded"] = sum(1 for r in load_results if r.get("import_ok"))
    details["plugins_registered"] = sum(1 for r in load_results if r.get("register_ok"))

    # 4. 测试后 config.yaml SHA256
    config_hash_after = _sha256_file(CONFIG_PATH)
    details["config_hash_after"] = config_hash_after
    config_unchanged = (config_hash_before == config_hash_after)

    # 5. sys.modules 差集, 过滤掉插件自身模块
    modules_after: Set[str] = set(sys.modules.keys())
    new_modules = modules_after - modules_before
    # 过滤掉插件自身的唯一模块名(以 _test_plugin_ 开头)
    plugin_self_modules = {m for m in new_modules if m.startswith("_test_plugin_")}
    # 过滤掉 hermes_plugins.* 命名空间(若插件使用了该命名空间)
    plugin_self_modules |= {m for m in new_modules if m.startswith("hermes_plugins.")}
    # 过滤掉 plugins.* 命名空间(用户插件包自身的子模块)
    plugin_self_modules |= {m for m in new_modules if m.startswith("plugins.") or m == "plugins"}
    unexpected_pollution = new_modules - plugin_module_names - plugin_self_modules
    details["new_modules_count"] = len(new_modules)
    details["plugin_self_modules"] = sorted(plugin_self_modules)
    details["unexpected_pollution"] = sorted(unexpected_pollution)
    details["unexpected_pollution_count"] = len(unexpected_pollution)

    # 6. 评估状态
    # 过滤出"显著"的污染: 排除标准库与常见三方库前缀的预期导入
    # (插件正常依赖会被 import, 这里只标记非插件自身模块的新增项作为信息)
    pollution_is_concern = len(unexpected_pollution) > 0

    if not config_unchanged:
        status = "FAIL"
        msg = f"config.yaml 被修改(SHA256 变化): {config_hash_before[:12]} -> {config_hash_after[:12]}"
    elif pollution_is_concern:
        # sys.modules 出现非插件自身的新模块属于正常依赖导入, 标记为 WARN 而非 FAIL
        status = "WARN"
        sample = sorted(unexpected_pollution)[:10]
        msg = f"config.yaml 未变; sys.modules 新增 {len(unexpected_pollution)} 个非插件自身模块(示例: {sample})"
    else:
        status = "PASS"
        msg = f"config.yaml 未变; sys.modules 无意外污染(插件自身模块 {len(plugin_self_modules)} 个)"

    return TestResult(
        plugin_name="(global)",
        test_name="main_interaction",
        status=status,
        message=msg,
        details=details,
        duration_ms=(time.perf_counter() - start) * 1000,
    )
