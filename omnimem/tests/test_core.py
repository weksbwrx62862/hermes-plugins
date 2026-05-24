"""L1 工作记忆核心模块测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnimem.core.attachment import CompactAttachment, build_attachments
from omnimem.core.block import CoreBlock
from omnimem.core.budget import BudgetManager
from omnimem.core.soul import SoulSystem


class TestCoreBlock(unittest.TestCase):
    """CoreBlock 测试。"""

    def test_default_values(self) -> None:
        cb = CoreBlock()
        self.assertEqual(cb.identity_block, "")
        self.assertEqual(cb.context_block, "")
        self.assertEqual(cb.plan_block, "")

    def test_to_prompt_text_all(self) -> None:
        cb = CoreBlock(identity_block="我是AI", context_block="当前任务", plan_block="步骤1")
        text = cb.to_prompt_text()
        self.assertIn("### Identity", text)
        self.assertIn("我是AI", text)
        self.assertIn("### Current Context", text)
        self.assertIn("当前任务", text)
        self.assertIn("### Plan", text)
        self.assertIn("步骤1", text)

    def test_to_prompt_text_partial(self) -> None:
        cb = CoreBlock(identity_block="AI")
        text = cb.to_prompt_text()
        self.assertIn("### Identity", text)
        self.assertNotIn("### Current Context", text)
        self.assertNotIn("### Plan", text)

    def test_update_context(self) -> None:
        cb = CoreBlock()
        cb.update_context("新上下文")
        self.assertEqual(cb.context_block, "新上下文")

    def test_update_plan(self) -> None:
        cb = CoreBlock()
        cb.update_plan("新计划")
        self.assertEqual(cb.plan_block, "新计划")


class TestCompactAttachment(unittest.TestCase):
    """CompactAttachment 测试。"""

    def test_to_text(self) -> None:
        att = CompactAttachment(kind="key_decisions", title="决策", body="选择了Python")
        text = att.to_text()
        self.assertIn("[key_decisions]", text)
        self.assertIn("决策", text)
        self.assertIn("选择了Python", text)

    def test_to_text_truncation(self) -> None:
        att = CompactAttachment(kind="task", title="T", body="X" * 1000)
        text = att.to_text(max_body_len=50)
        self.assertLessEqual(len(text), 100)

    def test_build_attachments_empty(self) -> None:
        result = build_attachments([])
        self.assertEqual(result, [])

    def test_build_attachments_with_decisions(self) -> None:
        msgs = [
            {"role": "user", "content": "我决定使用Python开发"},
            {"role": "assistant", "content": "好的，Python很适合"},
        ]
        result = build_attachments(msgs)
        kinds = [a.kind for a in result]
        self.assertIn("key_decisions", kinds)

    def test_build_attachments_with_preferences(self) -> None:
        msgs = [{"role": "user", "content": "我喜欢暗色主题"}]
        result = build_attachments(msgs)
        kinds = [a.kind for a in result]
        self.assertIn("user_preferences", kinds)

    def test_build_attachments_with_errors(self) -> None:
        msgs = [{"role": "assistant", "content": "遇到了一个错误"}]
        result = build_attachments(msgs)
        kinds = [a.kind for a in result]
        self.assertIn("error_patterns", kinds)


class TestSoulSystem(unittest.TestCase):
    """SoulSystem 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.soul = SoulSystem(Path(self.tmpdir))

    def test_load_identity_empty(self) -> None:
        result = self.soul.load_identity()
        self.assertEqual(result, "")

    def test_set_soul_and_load(self) -> None:
        self.soul.set_soul("我是AI助手")
        result = self.soul.load_identity()
        self.assertIn("我是AI助手", result)

    def test_set_soul_only_once(self) -> None:
        self.soul.set_soul("第一版")
        self.soul.set_soul("第二版")  # 不应覆盖
        result = self.soul.load_identity()
        self.assertIn("第一版", result)
        self.assertNotIn("第二版", result)

    def test_update_identity(self) -> None:
        self.soul.update_identity("开发者助手")
        result = self.soul.load_identity()
        self.assertIn("开发者助手", result)

    def test_update_user_profile_append(self) -> None:
        self.soul.update_user_profile("喜欢简洁")
        self.soul.update_user_profile("偏好中文")
        result = self.soul.load_identity()
        self.assertIn("喜欢简洁", result)
        self.assertIn("偏好中文", result)


class TestBudgetManager(unittest.TestCase):
    """BudgetManager 测试。"""

    def test_default_max_tokens(self) -> None:
        bm = BudgetManager()
        self.assertEqual(bm.max_tokens, 4000)

    def test_custom_max_tokens(self) -> None:
        bm = BudgetManager(max_tokens=2000)
        self.assertEqual(bm.max_tokens, 2000)

    def test_estimate_tokens(self) -> None:
        bm = BudgetManager()
        tokens = bm.estimate_tokens("Hello world")
        self.assertGreater(tokens, 0)

    def test_estimate_tokens_chinese(self) -> None:
        bm = BudgetManager()
        tokens = bm.estimate_tokens("你好世界")
        self.assertGreater(tokens, 0)

    def test_trim_to_budget(self) -> None:
        bm = BudgetManager(max_tokens=10)
        items = [
            {"content": "A" * 100},
            {"content": "B" * 100},
        ]
        result = bm.trim_to_budget(items)
        self.assertLessEqual(len(result), 2)

    def test_fits(self) -> None:
        bm = BudgetManager(max_tokens=1000)
        self.assertTrue(bm.fits("短文本"))
        self.assertFalse(bm.fits("X" * 10000))
