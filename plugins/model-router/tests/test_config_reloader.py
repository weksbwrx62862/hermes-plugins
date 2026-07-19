"""ConfigReloader 单元测试。

覆盖：
  - 文件修改检测
  - 配置变更后模型池更新
  - 路由矩阵重新加载
  - 缓存失效触发
  - 热重载期间并发读取不中断
  - 平台加分与策略变更摘要
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

HERMES_HOME = Path.home() / ".hermes"
if str(HERMES_HOME) not in sys.path:
    sys.path.insert(0, str(HERMES_HOME))

# 加载 cache.py
_cache_path = Path(__file__).resolve().parent.parent / "cache.py"
_cache_spec = importlib.util.spec_from_file_location("route_cache", str(_cache_path))
route_cache = importlib.util.module_from_spec(_cache_spec)
sys.modules["route_cache"] = route_cache
_cache_spec.loader.exec_module(route_cache)
RouteCache = route_cache.RouteCache

# 加载 config_reloader.py（使用包含连字符的包名）
_reloader_path = Path(__file__).resolve().parent.parent / "config_reloader.py"
_reloader_spec = importlib.util.spec_from_file_location(
    "plugins.model-router.config_reloader", str(_reloader_path)
)
config_reloader_mod = importlib.util.module_from_spec(_reloader_spec)
# 确保父包已注册
_imported_parent = importlib.import_module("plugins.model-router")
sys.modules["plugins.model-router.config_reloader"] = config_reloader_mod
_reloader_spec.loader.exec_module(config_reloader_mod)

ConfigReloader = config_reloader_mod.ConfigReloader

# 加载 model-router 主模块，用于 monkeypatch 配置加载
model_router = importlib.import_module("plugins.model-router")


def _sample_value():
    """构造一个符合缓存内容约定的示例值。"""
    return {
        "complexity": 3,
        "task_type": "code",
        "strategy": "auto",
        "model_scores_sorted": [("model-a", 95.0)],
    }


@pytest.fixture
def config_and_matrix(tmp_path, monkeypatch):
    """创建临时配置文件与路由矩阵，并劫持 model_router 的配置加载。"""
    config_path = tmp_path / "config.yaml"
    matrix_path = tmp_path / "routing_matrix.json"

    cfg = {
        "providers": {
            "openai": {
                "api_key": "sk-openai",
                "base_url": "https://api.deepseek.com/v1",
                "models": ["deepseek-v4-flash"],
            },
            "stepfun": {
                "api_key": "sk-stepfun",
                "base_url": "https://api.stepfun.com/v1",
                "models": ["step-3.5-flash"],
            },
        },
        "plugins": {
            "model-router": {
                "strategy": "auto",
                "scoring": {"stepfun_priority_bonus": 20},
                "allowed_providers": ["openai", "stepfun"],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    matrix = {
        "('simple_qa', 'light')": "cheapest",
        "('code', 'medium')": "balanced",
        "('long_doc', 'heavy')": "smartest",
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    def _load_temp_config():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(model_router, "_load_config", _load_temp_config)
    monkeypatch.setattr(model_router, "_CONFIG_CACHE", None)
    monkeypatch.setattr(model_router, "_CONFIG_CACHE_TIME", 0.0)
    monkeypatch.setattr(model_router, "_POOL_CACHE", None)
    monkeypatch.setattr(model_router, "_POOL_CACHE_TIME", 0.0)

    return config_path, matrix_path


def _make_reloader(config_path, matrix_path):
    """构造带临时路径的 ConfigReloader。"""
    cache = RouteCache()
    return (
        ConfigReloader(
            config_path=str(config_path),
            routing_matrix_path=str(matrix_path),
            poll_interval=0.1,
            route_cache=cache,
        ),
        cache,
    )


class TestConfigReloaderBasics:
    """基础加载与文件检测测试。"""

    def test_initial_load_builds_pool(self, config_and_matrix):
        """首次强制加载应正确构建模型池。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)

        assert reloader.force_reload() is True
        pool = reloader.get_pool()
        names = {m["name"] for m in pool}

        assert names == {"deepseek-v4-flash", "step-3.5-flash"}
        snapshot = reloader.get_snapshot()
        assert snapshot["strategy"] == "auto"
        assert snapshot["model_router_config"]["scoring"]["stepfun_priority_bonus"] == 20

    def test_file_change_detection(self, config_and_matrix):
        """_file_changed 应能检测到文件内容变更。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        # 重置元信息，模拟首次检测
        reloader._last_mtime = 0.0
        reloader._last_size = 0
        reloader._last_hash = ""

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["plugins"]["model-router"]["strategy"] = "smartest"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        assert reloader._file_changed() is True

    def test_no_change_returns_false(self, config_and_matrix):
        """文件未变更时 _file_changed 返回 False。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        assert reloader._file_changed() is False


class TestConfigReloaderUpdates:
    """配置变更后的更新与摘要测试。"""

    def test_reload_updates_model_pool(self, config_and_matrix):
        """修改配置后重载应更新模型池并生成变更摘要。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["providers"]["openai"]["models"] = ["deepseek-v4-pro"]
        del cfg["providers"]["stepfun"]
        cfg["providers"]["dashscope"] = {
            "api_key": "sk-dashscope",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": ["qwen-plus"],
        }
        cfg["plugins"]["model-router"]["allowed_providers"] = ["openai", "dashscope"]
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        assert reloader.force_reload() is True

        pool = reloader.get_pool()
        names = {m["name"] for m in pool}
        assert "deepseek-v4-pro" in names
        assert "qwen-plus" in names
        assert "step-3.5-flash" not in names

        summary = reloader.get_last_summary()
        assert "step-3.5-flash" in summary["removed_models"]
        assert "qwen-plus" in summary["added_models"]
        assert summary["model_count"] == 2
        assert "stepfun" in summary["allowed_provider_changes"]["removed"]
        assert "dashscope" in summary["allowed_provider_changes"]["added"]

    def test_reload_detects_modified_model_attrs(self, config_and_matrix):
        """模型属性变化应被识别为修改。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["plugins"]["model-router"]["model_attrs"] = {
            "deepseek-v4-flash": {"cost": 5, "speed": 1, "quality": 6}
        }
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        assert reloader.force_reload() is True
        summary = reloader.get_last_summary()
        modified_names = {m["name"] for m in summary["modified_models"]}
        assert "deepseek-v4-flash" in modified_names

    def test_scoring_and_strategy_summary(self, config_and_matrix):
        """平台加分与策略变更应体现在摘要中。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["plugins"]["model-router"]["strategy"] = "cheapest"
        cfg["plugins"]["model-router"]["scoring"]["stepfun_priority_bonus"] = 99
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        assert reloader.force_reload() is True

        summary = reloader.get_last_summary()
        assert summary["strategy_change"] == {"old": "auto", "new": "cheapest"}
        assert "stepfun_priority_bonus" in summary["scoring_changes"]
        assert summary["scoring_changes"]["stepfun_priority_bonus"] == {"old": 20, "new": 99}

        snapshot = reloader.get_snapshot()
        assert snapshot["strategy"] == "cheapest"
        assert snapshot["scoring"]["stepfun_priority_bonus"] == 99


class TestRoutingMatrixReload:
    """路由矩阵热重载测试。"""

    def test_routing_matrix_reloaded(self, config_and_matrix):
        """配置变更后路由矩阵应被重新加载。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        # 第一次重载会创建 RoutingMatrix 实例并记录 _cache_time
        first_time = reloader.get_snapshot()["routing_matrix"]._cache_time

        # 修改矩阵并再次重载
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        matrix["('simple_qa', 'light')"] = "smartest"
        matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

        assert reloader.force_reload() is True
        matrix_obj = reloader.get_snapshot()["routing_matrix"]
        assert matrix_obj.get_model("simple_qa", "light") == "smartest"
        assert matrix_obj._cache_time > first_time


class TestCacheInvalidation:
    """缓存失效测试。"""

    def test_reload_invalidates_route_cache(self, config_and_matrix):
        """热重载应触发路由缓存失效。"""
        config_path, matrix_path = config_and_matrix
        reloader, cache = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        cache.set("q1", _sample_value())
        cache.set("q2", _sample_value())
        assert cache.stats["size"] == 2

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["plugins"]["model-router"]["strategy"] = "fastest"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        assert reloader.force_reload() is True
        assert cache.stats["size"] == 0

        summary = reloader.get_last_summary()
        assert summary["cache_invalidated"] == 2


class TestConcurrency:
    """并发安全测试。"""

    def test_concurrent_readers_during_reload(self, config_and_matrix):
        """热重载期间并发读取模型池不应抛异常，且最终状态正确。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)
        reloader.force_reload()

        errors = []
        stop_event = threading.Event()
        read_count = {"value": 0}

        def reader():
            try:
                while not stop_event.is_set():
                    pool = reloader.get_pool()
                    assert isinstance(pool, list)
                    read_count["value"] += 1
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()

        # 在读者运行时多次修改配置并重载
        for i in range(6):
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            cfg["providers"]["openai"]["models"] = [
                "deepseek-v4-pro" if i % 2 else "deepseek-v4-flash"
            ]
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            assert reloader.force_reload() is True
            time.sleep(0.02)

        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

        assert not errors, f"并发读取出现异常: {errors}"
        assert read_count["value"] > 0

        pool = reloader.get_pool()
        names = {m["name"] for m in pool}
        # 最后一次重载把模型改成了 deepseek-v4-pro
        assert "deepseek-v4-pro" in names


class TestBackgroundWatcher:
    """后台轮询线程测试。"""

    def test_start_stop_watcher(self, config_and_matrix):
        """启动/停止后台监听线程应正常工作。"""
        config_path, matrix_path = config_and_matrix
        reloader, _ = _make_reloader(config_path, matrix_path)

        reloader.start()
        assert reloader.is_alive() is True

        # 修改文件，等待后台线程检测到并自动重载
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["plugins"]["model-router"]["strategy"] = "smartest"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if reloader.get_snapshot().get("strategy") == "smartest":
                break
            time.sleep(0.05)

        assert reloader.get_snapshot().get("strategy") == "smartest"

        reloader.stop()
        assert reloader.is_alive() is False
