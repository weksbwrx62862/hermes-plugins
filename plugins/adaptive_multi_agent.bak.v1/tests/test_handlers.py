"""handlers.py 集成测试：使用 mock PluginContext 与临时数据库验证工具函数。"""

import ast
import json

import pytest

from adaptive_multi_agent.checkpoint import AMACheckpoint
from adaptive_multi_agent.subagent import SubagentResult, SubagentStatus


class MockPluginContext:
    """模拟 Hermes PluginContext，提供 register_tool / register_hook / dispatch_tool。"""

    def __init__(self, responses=None):
        self.registered_tools = {}
        self.hooks = {}
        self.responses = responses or {}
        self.calls = []

    def register_tool(self, name, **kwargs):
        self.registered_tools[name] = kwargs

    def register_hook(self, event, callback):
        self.hooks.setdefault(event, []).append(callback)

    def dispatch_tool(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name in self.responses:
            return self.responses[name]
        if name == "delegate_task":
            return json.dumps({"results": [{"result": "mock result", "status": "completed", "tokens": {"total": 10}}]})
        return None


@pytest.fixture
def ama_test_env(tmp_path, monkeypatch):
    """将 AMA 持久化与技能注册表指向临时目录，并重置 handlers 全局单例。"""
    db_path = tmp_path / "ama_state.db"
    skill_path = tmp_path / "skill_registry.json"
    monkeypatch.setattr("adaptive_multi_agent.persistence._DB_PATH", db_path)
    monkeypatch.setattr("adaptive_multi_agent.persistence._persistence", None)
    monkeypatch.setattr("adaptive_multi_agent.persistence.AMAPersistence._instance", None)
    monkeypatch.setattr("adaptive_multi_agent.trajectory._recorder", None)
    monkeypatch.setattr("adaptive_multi_agent.skill_registry._skill_registry", None)
    monkeypatch.setattr("adaptive_multi_agent.skill_registry.SkillRegistry.PERSIST_PATH", skill_path)

    from adaptive_multi_agent import handlers as h

    h._reset_engine()
    h._reset_clarifier()
    h._plugin_ctx = None

    yield db_path

    h._reset_engine()
    h._reset_clarifier()
    h._plugin_ctx = None


@pytest.fixture
def mock_ctx():
    return MockPluginContext()


@pytest.fixture
def ctx(ama_test_env, mock_ctx):
    """将 mock_ctx 注入 handlers 模块并返回。"""
    from adaptive_multi_agent import handlers as h

    h._plugin_ctx = mock_ctx
    return mock_ctx


def _parse_response(response: str):
    """解析 tool_result 返回的字符串；若是 tool_error 则抛出异常。"""
    if response.startswith("ERROR:"):
        raise AssertionError(f"工具返回错误: {response}")
    return ast.literal_eval(response)


def _insert_execution(pers, created_at_expr=None, **kwargs):
    """向临时 ama_executions 表插入记录，created_at 支持 SQL 表达式。"""
    defaults = {
        "session_id": "s1",
        "task": "测试任务",
        "task_type": "code_generation",
        "complexity_score": 3.0,
        "mode_used": "generator_verifier",
        "original_mode": "generator_verifier",
        "success": 1,
        "token_usage": 100,
        "time_taken": 1.0,
        "switched_modes": 0,
    }
    defaults.update(kwargs)
    defaults.pop("created_at", None)
    columns = list(defaults.keys())
    placeholders = ["?"] * len(defaults)
    values = tuple(defaults.values())
    if created_at_expr:
        columns.append("created_at")
        placeholders.append(created_at_expr)
    sql = f"INSERT INTO ama_executions ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
    with pers.transaction() as conn:
        conn.execute(sql, values)


class TestHandleAmaAssess:
    """测试 handle_ama_assess：返回字段与 clarify 路径。"""

    def test_handle_ama_assess_returns_required_fields(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_assess({"task": "写一个 hello world 函数"})
        data = _parse_response(result)
        assert "complexity_score" in data
        assert "recommended_mode" in data
        assert "task_type" in data
        assert "diagnosis" in data
        assert "summary" in data["diagnosis"]

    def test_handle_ama_assess_with_clarify(self, ctx):
        from adaptive_multi_agent import handlers as h

        def _dispatch(name, args, **kwargs):
            ctx.calls.append((name, args, kwargs))
            if name == "send_message":
                return None
            if name != "delegate_task":
                return None
            goal = args.get("goal", "")
            if "需求分析专家" in goal or "needs_clarification" in goal:
                return json.dumps({
                    "needs_clarification": False,
                    "questions": [],
                    "extracted_features": {"has_explicit_verification": True},
                    "clarified_task": "写一个带单元测试的加法函数",
                    "reasoning": "需求已明确",
                })
            return json.dumps({
                "rubric": {"steps": 2, "domain": 2, "verification": 3, "collaboration": 1, "uncertainty": 1},
                "complexity_score": 3.5,
                "task_type": "code_generation",
                "features": {"has_explicit_verification": True},
                "recommended_mode": "generator_verifier",
                "reasoning": "简单任务",
            })

        ctx.dispatch_tool = _dispatch
        result = h.handle_ama_assess({"task": "写个加法函数", "clarify": True})
        data = _parse_response(result)
        assert data["complexity_score"] == 3.5
        assert data["recommended_mode"] == "generator_verifier"
        assert data["task_type"] == "code_generation"
        assert data["diagnosis"]["task_type_cn"] == "编码任务"


class TestHandleAmaStats:
    """测试 handle_ama_stats：字段与 period 过滤。"""

    def test_handle_ama_stats_returns_required_fields(self, ctx):
        from adaptive_multi_agent import handlers as h
        from adaptive_multi_agent import persistence as p

        pers = p.get_persistence()
        _insert_execution(pers)
        result = h.handle_ama_stats({})
        data = _parse_response(result)
        assert data["total_executions"] == 1
        assert "mode_usage" in data
        assert "success_rates" in data
        assert "summary" in data
        assert data["success_rates"]["generator_verifier"]["rate"] == 1.0

    def test_handle_ama_stats_period_filter(self, ctx):
        from adaptive_multi_agent import handlers as h
        from adaptive_multi_agent import persistence as p

        pers = p.get_persistence()
        _insert_execution(pers, mode_used="generator_verifier", created_at_expr="datetime('now', '-2 days')")
        _insert_execution(pers, mode_used="orchestrator_subagent", created_at_expr="datetime('now')")

        all_data = _parse_response(h.handle_ama_stats({"period": "all"}))
        assert all_data["total_executions"] == 2

        day_data = _parse_response(h.handle_ama_stats({"period": "day"}))
        assert day_data["total_executions"] == 1
        assert "orchestrator_subagent" in day_data["mode_usage"]


class TestHandleAmaWorkflow:
    """测试 handle_ama_workflow：list / info / 无效 workflow_id。"""

    def test_handle_ama_workflow_list(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_workflow({"action": "list"})
        data = _parse_response(result)
        assert "workflows" in data
        assert data["total"] > 0
        assert any(wf["id"] == "software_dev" for wf in data["workflows"])

    def test_handle_ama_workflow_info_valid(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_workflow({"action": "info", "workflow_id": "software_dev"})
        data = _parse_response(result)
        assert data["id"] == "software_dev"
        assert data["name"] == "软件开发"
        assert len(data["stages"]) > 0
        assert "default_mode" in data

    def test_handle_ama_workflow_info_invalid(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_workflow({"action": "info", "workflow_id": "not_exist"})
        assert result.startswith("ERROR:")


class TestHandleAmaCancel:
    """测试 handle_ama_cancel：存在/不存在的 task_id。"""

    def test_handle_ama_cancel_existing(self, ctx):
        from adaptive_multi_agent import handlers as h

        engine = h._get_engine()
        sr = SubagentResult(task_id="task-123", status=SubagentStatus.RUNNING)
        engine.result_store.put("task-123", sr)
        result = h.handle_ama_cancel({"task_id": "task-123"})
        data = _parse_response(result)
        assert data["success"] is True

    def test_handle_ama_cancel_missing(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_cancel({"task_id": "missing-id"})
        data = _parse_response(result)
        assert data["success"] is False


class TestHandleAmaDiagnose:
    """测试 handle_ama_diagnose：字段与 include_ts_params 过滤。"""

    def test_handle_ama_diagnose_returns_required_fields(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_diagnose({})
        data = _parse_response(result)
        assert "summary" in data
        assert "ts_params" in data
        assert "circuit_breakers" in data
        assert "recent_errors" in data
        assert "performance" in data

    def test_handle_ama_diagnose_exclude_ts_params(self, ctx):
        from adaptive_multi_agent import handlers as h

        result = h.handle_ama_diagnose({"include_ts_params": False})
        data = _parse_response(result)
        assert "ts_params" not in data
        assert "circuit_breakers" in data

    def test_handle_ama_diagnose_recent_errors(self, ctx):
        from adaptive_multi_agent import handlers as h
        from adaptive_multi_agent import persistence as p

        pers = p.get_persistence()
        _insert_execution(pers, success=0, error_category="timeout")
        result = h.handle_ama_diagnose({})
        data = _parse_response(result)
        assert any(err["category"] == "timeout" for err in data["recent_errors"])


class TestHandleAmaExecute:
    """测试 handle_ama_execute：完整成功流程与 force_mode 强制指定。"""

    def _make_gv_dispatch(self, ctx, gen_result="print('hello')"):
        def _dispatch(name, args, **kwargs):
            ctx.calls.append((name, args, kwargs))
            if name == "send_message":
                return None
            if name != "delegate_task":
                return None
            goal = args.get("goal", "")
            if "【生成任务】" in goal:
                return json.dumps({"results": [{"result": gen_result, "status": "completed", "tokens": {"total": 50}}]})
            if "【审核任务】" in goal:
                return json.dumps({"results": [{"result": json.dumps({
                    "passed": True,
                    "issues": [],
                    "scores": {"completeness": 90, "correctness": 90, "clarity": 90, "relevance": 90},
                }), "status": "completed", "tokens": {"total": 30}}]})
            return json.dumps({"results": [{"result": "ok", "status": "completed"}]})

        ctx.dispatch_tool = _dispatch

    def test_handle_ama_execute_success(self, ctx):
        from adaptive_multi_agent import handlers as h

        self._make_gv_dispatch(ctx)
        result = h.handle_ama_execute({"task": "写一个 hello world 函数", "force_mode": "generator_verifier"})
        data = _parse_response(result)
        assert data["success"] is True
        assert data["mode_used"] == "generator_verifier"
        assert data["result"] == "print('hello')"
        assert any(c[0] == "delegate_task" for c in ctx.calls)

    def test_handle_ama_execute_force_mode_overrides_selection(self, ctx):
        from adaptive_multi_agent import handlers as h

        self._make_gv_dispatch(ctx)
        complex_task = (
            "设计一个完整微服务电商平台，包含用户认证、订单、支付、库存服务，"
            "需要多角色团队协作，并行开发多个模块，并编写详细的架构设计文档"
        )
        result = h.handle_ama_execute({"task": complex_task, "force_mode": "generator_verifier"})
        data = _parse_response(result)
        assert data["success"] is True
        assert data["mode_used"] == "generator_verifier"


class TestHandleAmaResume:
    """测试 handle_ama_resume：list / resume。"""

    def test_handle_ama_resume_list(self, ctx):
        from adaptive_multi_agent import handlers as h
        from adaptive_multi_agent import persistence as p

        pers = p.get_persistence()
        trace_id = "trace-list-001"
        _insert_execution(
            pers,
            task="中断任务",
            success=0,
            trace_id=trace_id,
            status="interrupted",
            created_at_expr="datetime('now')",
        )
        AMACheckpoint.save(trace_id, 0, "中断任务", mode="generator_verifier", task_type="code_generation", complexity_score=3.0)

        result = h.handle_ama_resume({"action": "list"})
        data = _parse_response(result)
        assert len(data["traces"]) >= 1
        assert any(t["trace_id"] == trace_id for t in data["traces"])

    def test_handle_ama_resume_resume(self, ctx):
        from adaptive_multi_agent import handlers as h

        trace_id = "trace-resume-001"
        AMACheckpoint.save(trace_id, 0, "恢复任务", mode="generator_verifier", task_type="code_generation", complexity_score=3.0)

        def _dispatch(name, args, **kwargs):
            ctx.calls.append((name, args, kwargs))
            if name == "send_message":
                return None
            if name != "delegate_task":
                return None
            goal = args.get("goal", "")
            if "【生成任务】" in goal:
                return json.dumps({"results": [{"result": "已恢复结果", "status": "completed", "tokens": {"total": 20}}]})
            if "【审核任务】" in goal:
                return json.dumps({"results": [{"result": json.dumps({
                    "passed": True,
                    "issues": [],
                    "scores": {"completeness": 95, "correctness": 95, "clarity": 95, "relevance": 95},
                }), "status": "completed", "tokens": {"total": 20}}]})
            return json.dumps({"results": [{"result": "ok", "status": "completed"}]})

        ctx.dispatch_tool = _dispatch
        result = h.handle_ama_resume({"action": "resume", "trace_id": trace_id})
        data = _parse_response(result)
        assert data["resumed"] is True
        assert data["result"]["success"] is True
