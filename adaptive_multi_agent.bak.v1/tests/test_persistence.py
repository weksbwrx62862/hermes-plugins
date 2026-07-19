"""AMAPersistence 单元测试。"""

import sqlite3

import pytest

from adaptive_multi_agent.persistence import migrate_legacy_trajectories


class TestDatabaseLifecycle:
    """测试 AMAPersistence 在临时目录中创建数据库和表。"""

    def test_database_file_created(self, ama_persistence):
        assert ama_persistence.db_path.exists()

    def test_tables_created(self, ama_persistence):
        conn = sqlite3.connect(str(ama_persistence.db_path))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "ama_performance" in tables
            assert "ama_executions" in tables
            assert "ama_trajectories" in tables
            assert "ama_state_snapshots" in tables
            assert "ama_memory" in tables
        finally:
            conn.close()


class TestPerformanceRoundTrip:
    """测试 save_performance / load_performance 往返。"""

    def test_save_and_load_performance(self, ama_persistence):
        stats = {
            "trials": 10,
            "successes": 7,
            "avg_tokens": 1500.5,
            "avg_time": 12.3,
        }
        ama_persistence.save_performance("code_generation", "generator_verifier", stats)
        loaded = ama_persistence.load_performance()
        assert loaded["code_generation"]["generator_verifier"] == stats


class TestExecutionRoundTrip:
    """测试 record_execution / get_stats 往返。"""

    def test_record_and_get_stats(self, ama_persistence):
        ama_persistence.record_execution(
            session_id="session-1",
            task="测试任务",
            task_type="code_generation",
            complexity_score=4.5,
            mode_used="orchestrator_subagent",
            original_mode="generator_verifier",
            success=True,
            token_usage=2000,
            time_taken=5.5,
            switched_modes=True,
            switch_reason="成本感知降级",
        )
        stats = ama_persistence.get_stats()
        assert stats["total_executions"] == 1
        assert stats["mode_usage"]["orchestrator_subagent"] == 1
        assert stats["success_rates"]["orchestrator_subagent"]["rate"] == 1.0


class TestTrajectoryRoundTrip:
    """测试 trajectory 保存/查询/update_grade。"""

    @pytest.fixture
    def sample_trajectory(self):
        return {
            "trajectory_id": "traj-001",
            "task": "测试轨迹任务",
            "context": "测试上下文",
            "mode": "agent_teams",
            "complexity_score": 6.5,
            "task_type": "complex",
            "steps": [{"step": 1, "action": "plan"}],
            "final_result": "完成",
            "success": True,
            "error": None,
            "total_duration_ms": 1234.0,
            "started_at": 1700000000.0,
            "completed_at": 1700000100.0,
            "metadata": {"version": "test"},
            "grade_score": None,
            "grade_feedback": None,
        }

    def test_save_and_query_trajectory(self, ama_persistence, sample_trajectory):
        ama_persistence.save_trajectory(sample_trajectory)
        rows = ama_persistence.query_trajectories(limit=10)
        assert len(rows) == 1
        assert rows[0]["trajectory_id"] == "traj-001"
        assert rows[0]["mode"] == "agent_teams"

    def test_update_grade(self, ama_persistence, sample_trajectory):
        ama_persistence.save_trajectory(sample_trajectory)
        ama_persistence.update_grade("traj-001", score=8.5, feedback="表现良好")
        loaded = ama_persistence.get_trajectory("traj-001")
        assert loaded["grade_score"] == 8.5
        assert loaded["grade_feedback"] == "表现良好"


class TestLegacyMigration:
    """测试 migrate_legacy_trajectories 能从模拟旧库导入数据。"""

    def test_migrate_legacy_trajectories(self, ama_persistence, tmp_path):
        legacy_db = tmp_path / "legacy" / "trajectories.db"
        legacy_db.parent.mkdir(parents=True, exist_ok=True)
        marker = tmp_path / "legacy" / ".migrated"

        conn = sqlite3.connect(str(legacy_db))
        try:
            conn.execute(
                """
                CREATE TABLE trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    task TEXT,
                    context TEXT,
                    mode TEXT,
                    complexity_score REAL,
                    task_type TEXT,
                    steps_json TEXT,
                    metadata_json TEXT,
                    final_result TEXT,
                    success INTEGER,
                    error TEXT,
                    total_duration_ms REAL,
                    started_at REAL,
                    completed_at REAL,
                    grade_score REAL,
                    grade_feedback TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-001",
                    "旧库任务",
                    None,
                    "generator_verifier",
                    3.0,
                    "code_generation",
                    '[{"step": 1}]',
                    '{"src": "legacy"}',
                    "旧结果",
                    1,
                    None,
                    500.0,
                    1600000000.0,
                    1600000100.0,
                    None,
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        count = migrate_legacy_trajectories(legacy_db, marker, persistence=ama_persistence)
        assert count == 1
        assert marker.exists()

        loaded = ama_persistence.get_trajectory("legacy-001")
        assert loaded is not None
        assert loaded["task"] == "旧库任务"
        assert loaded["mode"] == "generator_verifier"
        assert loaded["metadata"] == {"src": "legacy"}
