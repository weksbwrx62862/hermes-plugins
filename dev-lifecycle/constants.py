"""dev-lifecycle 插件 — 生命周期阶段和辅助技能常量定义。"""

from __future__ import annotations

from typing import Any, Dict


LIFECYCLE: Dict[str, Dict[str, Any]] = {
    "ideate": {
        "emoji": "💡",
        "name_cn": "构思阶段",
        "order": 1,
        "description": "从需求澄清到任务拆解——把模糊的想法变成可执行的计划",
        "flow": [
            ("grill-me", "无文档需求深挖"),
            ("grill-with-docs", "带 CONTEXT.md/ADR 的需求深挖"),
            ("to-prd", "对话转 PRD 文档"),
            ("to-issues", "PRD 拆解为独立 Issue"),
            ("plan", "写实现计划（.hermes/plans/）"),
            ("writing-plans", "编写可执行的分步计划"),
        ],
    },
    "build": {
        "emoji": "🔨",
        "name_cn": "构建阶段",
        "order": 2,
        "description": "原型验证 → 架构审查 → TDD 实现 → 子代理并行执行",
        "flow": [
            ("prototype", "抛弃式原型验证设计"),
            ("spike", "针对性技术实验"),
            ("improve-codebase-architecture", "深度模块分析 + 删除测试"),
            ("zoom-out", "获取代码库更广上下文"),
            ("test-driven-development", "RED-GREEN-REFACTOR 循环"),
            ("subagent-driven-development", "委派子代理并行执行计划"),
            ("user-auth-system", "用户认证系统设计模式"),
        ],
    },
    "deliver": {
        "emoji": "🚀",
        "name_cn": "交付阶段",
        "order": 3,
        "description": "调试 → 审查 → 分类 → 交接——把代码变成可交付的成果",
        "flow": [
            ("systematic-debugging", "四阶段根因调试"),
            ("python-debugpy", "Python pdb + debugpy 远程调试"),
            ("node-inspect-debugger", "Node.js Chrome DevTools 调试"),
            ("requesting-code-review", "提交前安全检查 + 质量门禁"),
            ("triage", "Issue 分类状态机"),
            ("handoff", "紧凑会话交接文档"),
        ],
    },
}

AUX_SKILLS = {
    "hermes-agent-skill-authoring": "编写 Hermes Agent 技能文件",
    "debugging-hermes-tui-commands": "调试 Hermes TUI 斜杠命令",
}
