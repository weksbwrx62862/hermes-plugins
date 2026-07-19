"""DAGOrchestrator 单元测试。"""

import pytest

from adaptive_multi_agent.dag_orchestrator import DAGOrchestrator


def _make_dag():
    """构造一个菱形 DAG：a -> b/c -> d。"""
    dag = DAGOrchestrator(max_workers=2)
    dag.add_node("a", lambda ctx, **kw: "A")
    dag.add_node("b", lambda ctx, **kw: f"B:{kw['dependency_results']['a']}", dependencies=["a"])
    dag.add_node("c", lambda ctx, **kw: f"C:{kw['dependency_results']['a']}", dependencies=["a"])
    dag.add_node("d", lambda ctx, **kw: f"D:{kw['dependency_results']['b']}+{kw['dependency_results']['c']}", dependencies=["b", "c"])
    return dag


class TestTopologicalSort:
    """测试拓扑排序与并行组正确性。"""

    def test_linear_groups(self):
        dag = DAGOrchestrator()
        dag.add_node("step1", lambda ctx, **kw: 1)
        dag.add_node("step2", lambda ctx, **kw: 2, dependencies=["step1"])
        dag.add_node("step3", lambda ctx, **kw: 3, dependencies=["step2"])
        groups = dag._topological_sort()
        assert groups == [["step1"], ["step2"], ["step3"]]

    def test_diamond_groups(self):
        dag = _make_dag()
        groups = dag._topological_sort()
        assert groups[0] == ["a"]
        assert set(groups[1]) == {"b", "c"}
        assert groups[2] == ["d"]


class TestValidation:
    """测试 DAG 结构验证。"""

    def test_cycle_detection_raises(self):
        dag = DAGOrchestrator()
        dag.add_node("a", lambda ctx, **kw: 1, dependencies=["b"])
        dag.add_node("b", lambda ctx, **kw: 2, dependencies=["a"])
        with pytest.raises(ValueError, match="环"):
            dag._validate()

    def test_missing_dependency_raises(self):
        dag = DAGOrchestrator()
        dag.add_node("a", lambda ctx, **kw: 1, dependencies=["missing"])
        with pytest.raises(ValueError, match="不存在"):
            dag._validate()


class TestExecution:
    """测试执行成功/失败路径。"""

    def test_success_path(self):
        dag = _make_dag()
        result = dag.execute(ctx=None, task="菱形任务")
        assert result.success is True
        assert result.results["a"] == "A"
        assert result.results["d"].startswith("D:")
        assert result.parallel_groups[0] == ["a"]

    def test_failure_path(self):
        dag = DAGOrchestrator()
        dag.add_node("ok", lambda ctx, **kw: "ok")
        dag.add_node("fail", lambda ctx, **kw: (_ for _ in ()).throw(RuntimeError("故意失败")))

        result = dag.execute(ctx=None, task="失败任务")
        assert result.success is False
        assert "fail" in result.errors
        assert "ok" in result.results

    def test_validation_error_returned(self):
        dag = DAGOrchestrator()
        dag.add_node("a", lambda ctx, **kw: 1, dependencies=["b"])
        dag.add_node("b", lambda ctx, **kw: 2, dependencies=["a"])
        result = dag.execute(ctx=None, task="环任务")
        assert result.success is False
        assert "validation" in result.errors
