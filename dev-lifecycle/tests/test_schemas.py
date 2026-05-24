"""dev-lifecycle 插件 — DEV_WORKFLOW_SCHEMA 结构验证测试。"""

import sys
import unittest

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from schemas import DEV_WORKFLOW_SCHEMA


class TestSchemaStructure(unittest.TestCase):

    def test_schema_has_allOf(self):
        params = DEV_WORKFLOW_SCHEMA["parameters"]
        self.assertIn("allOf", params)
        self.assertIsInstance(params["allOf"], list)

    def test_schema_allOf_stage_requires_stage_name(self):
        all_of = DEV_WORKFLOW_SCHEMA["parameters"]["allOf"]
        stage_rule = None
        for item in all_of:
            if_const = item.get("if", {}).get("properties", {}).get("action", {}).get("const")
            if if_const == "stage":
                stage_rule = item
                break
        self.assertIsNotNone(stage_rule, "未找到 action=stage 的条件规则")
        then_required = stage_rule.get("then", {}).get("required", [])
        self.assertIn("stage_name", then_required)

    def test_schema_allOf_skill_requires_skill_name(self):
        all_of = DEV_WORKFLOW_SCHEMA["parameters"]["allOf"]
        skill_rule = None
        for item in all_of:
            if_const = item.get("if", {}).get("properties", {}).get("action", {}).get("const")
            if if_const == "skill":
                skill_rule = item
                break
        self.assertIsNotNone(skill_rule, "未找到 action=skill 的条件规则")
        then_required = skill_rule.get("then", {}).get("required", [])
        self.assertIn("skill_name", then_required)

    def test_schema_has_new_actions(self):
        action_enum = DEV_WORKFLOW_SCHEMA["parameters"]["properties"]["action"]["enum"]
        for a in ("start", "advance", "rollback", "resume", "report"):
            self.assertIn(a, action_enum)

    def test_schema_has_project_path(self):
        props = DEV_WORKFLOW_SCHEMA["parameters"]["properties"]
        self.assertIn("project_path", props)
        self.assertIn("to_stage", props)


if __name__ == "__main__":
    unittest.main()
