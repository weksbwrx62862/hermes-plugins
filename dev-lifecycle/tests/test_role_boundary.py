"""角色边界约束单元测试。"""

import sys
import unittest
from pathlib import Path

# 添加插件根目录到 sys.path
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from gates import RoleBoundary, ROLE_BOUNDARIES, get_role_boundary
from schemas import enrich_skill_with_role_boundary


class TestRoleBoundary(unittest.TestCase):
    """角色边界约束测试套件。"""

    def test_requesting_code_review_boundary(self):
        """测试 requesting-code-review 角色边界。"""
        b = get_role_boundary("requesting-code-review")
        self.assertIsNotNone(b, "requesting-code-review 边界未定义")
        self.assertIsInstance(b, RoleBoundary)
        self.assertIn("修复发现的 bug", b.forbidden, "forbidden 列表缺少 '修复发现的 bug'")
        self.assertIn("直接修改被审查的代码", b.forbidden)
        self.assertTrue(b.rationale, "rationale 不能为空")
        self.assertTrue(b.allowed, "allowed 列表不能为空")
        print(f"✓ requesting-code-review: forbidden={b.forbidden}")

    def test_systematic_debugging_boundary(self):
        """测试 systematic-debugging 角色边界。"""
        b = get_role_boundary("systematic-debugging")
        self.assertIsNotNone(b, "systematic-debugging 边界未定义")
        self.assertIsInstance(b, RoleBoundary)
        self.assertIn("跳过假设验证步骤", b.forbidden)
        self.assertIn("盲目修改代码", b.forbidden)
        self.assertTrue(b.rationale)
        print(f"✓ systematic-debugging: forbidden={b.forbidden}")

    def test_tdd_boundary(self):
        """测试 test-driven-development 角色边界。"""
        b = get_role_boundary("test-driven-development")
        self.assertIsNotNone(b, "test-driven-development 边界未定义")
        self.assertIsInstance(b, RoleBoundary)
        self.assertIn("先写实现再补测试", b.forbidden)
        self.assertIn("跳过红绿重构循环", b.forbidden)
        self.assertTrue(b.rationale)
        print(f"✓ test-driven-development: forbidden={b.forbidden}")

    def test_unknown_skill_returns_none(self):
        """测试未定义边界的技能返回 None。"""
        b = get_role_boundary("unknown-skill")
        self.assertIsNone(b, "未定义边界的技能应返回 None")
        print("✓ unknown-skill 返回 None")

    def test_role_boundaries_registry(self):
        """测试 ROLE_BOUNDARIES 注册表。"""
        self.assertGreaterEqual(len(ROLE_BOUNDARIES), 3, "注册表应至少包含 3 个技能")
        self.assertIn("requesting-code-review", ROLE_BOUNDARIES)
        self.assertIn("systematic-debugging", ROLE_BOUNDARIES)
        self.assertIn("test-driven-development", ROLE_BOUNDARIES)
        # 验证每个注册表条目类型正确
        for name, boundary in ROLE_BOUNDARIES.items():
            self.assertIsInstance(boundary, RoleBoundary)
            self.assertEqual(boundary.skill_name, name)
        print(f"✓ 注册表包含 {len(ROLE_BOUNDARIES)} 个技能")

    def test_enrich_skill_with_role_boundary_hit(self):
        """测试 schemas.enrich_skill_with_role_boundary 命中场景。"""
        skill_info = {"name": "systematic-debugging", "summary": "测试摘要"}
        result = enrich_skill_with_role_boundary(skill_info)
        self.assertIn("role_boundary", result)
        self.assertIsNotNone(result["role_boundary"])
        self.assertIsInstance(result["role_boundary"], RoleBoundary)
        self.assertEqual(result["role_boundary"].skill_name, "systematic-debugging")
        print("✓ enrich_skill_with_role_boundary 命中")

    def test_enrich_skill_with_role_boundary_miss(self):
        """测试 schemas.enrich_skill_with_role_boundary 未命中场景。"""
        skill_info = {"name": "unknown-skill"}
        result = enrich_skill_with_role_boundary(skill_info)
        self.assertIn("role_boundary", result)
        self.assertIsNone(result["role_boundary"])
        print("✓ enrich_skill_with_role_boundary 未命中")

    def test_enrich_skill_with_role_boundary_skill_name_key(self):
        """测试使用 skill_name 键查询。"""
        skill_info = {"skill_name": "test-driven-development"}
        result = enrich_skill_with_role_boundary(skill_info)
        self.assertIsNotNone(result["role_boundary"])
        self.assertEqual(result["role_boundary"].skill_name, "test-driven-development")
        print("✓ enrich_skill_with_role_boundary 支持 skill_name 键")

    def test_enrich_skill_with_role_boundary_empty(self):
        """测试空字典场景，role_boundary 应为 None。"""
        skill_info = {}
        result = enrich_skill_with_role_boundary(skill_info)
        self.assertIn("role_boundary", result)
        self.assertIsNone(result["role_boundary"])
        print("✓ enrich_skill_with_role_boundary 空字典返回 None")


def test_requesting_code_review_boundary():
    """测试 requesting-code-review 角色边界（独立函数，便于直接运行）。"""
    b = get_role_boundary("requesting-code-review")
    assert b is not None, "requesting-code-review 边界未定义"
    assert "修复发现的 bug" in b.forbidden, "forbidden 列表缺少 '修复发现的 bug'"
    assert b.rationale != "", "rationale 不能为空"
    print(f"✓ requesting-code-review: forbidden={b.forbidden}")


def test_systematic_debugging_boundary():
    """测试 systematic-debugging 角色边界。"""
    b = get_role_boundary("systematic-debugging")
    assert b is not None, "systematic-debugging 边界未定义"
    assert "跳过假设验证步骤" in b.forbidden
    print(f"✓ systematic-debugging: forbidden={b.forbidden}")


def test_tdd_boundary():
    """测试 test-driven-development 角色边界。"""
    b = get_role_boundary("test-driven-development")
    assert b is not None, "test-driven-development 边界未定义"
    assert "先写实现再补测试" in b.forbidden
    print(f"✓ test-driven-development: forbidden={b.forbidden}")


def test_unknown_skill_returns_none():
    """测试未定义边界的技能返回 None。"""
    b = get_role_boundary("unknown-skill")
    assert b is None, "未定义边界的技能应返回 None"
    print("✓ unknown-skill 返回 None")


def test_role_boundaries_registry():
    """测试 ROLE_BOUNDARIES 注册表。"""
    assert len(ROLE_BOUNDARIES) >= 3, "注册表应至少包含 3 个技能"
    assert "requesting-code-review" in ROLE_BOUNDARIES
    assert "systematic-debugging" in ROLE_BOUNDARIES
    assert "test-driven-development" in ROLE_BOUNDARIES
    print(f"✓ 注册表包含 {len(ROLE_BOUNDARIES)} 个技能")


if __name__ == "__main__":
    # 支持直接 python3 运行
    test_requesting_code_review_boundary()
    test_systematic_debugging_boundary()
    test_tdd_boundary()
    test_unknown_skill_returns_none()
    test_role_boundaries_registry()
    print("\n所有测试通过 ✓")
