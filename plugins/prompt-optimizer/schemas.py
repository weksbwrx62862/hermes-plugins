"""prompt-optimizer 插件 — 工具 schema 定义。"""

PROMPT_OPTIMIZE_SCHEMA = {
    "name": "prompt_optimize",
    "description": (
        "提示词优化工具。把模糊、随意的提示词，优化成结构化、可测试、可复用的版本。\n"
        "基于六维优化框架：角色(Role)、对象(Object)、结构(Structure)、风格(Style)、约束(Constraints)、输出目标(Output)。\n\n"
        "模式：\n"
        "  - user: 优化用户提示词（本次任务说明，如'帮我写一篇文章'）\n"
        "  - system: 优化系统提示词（长期角色设定，如'你是一个资深编辑'）\n\n"
        "操作：\n"
        "  - optimize: 优化提示词（需 prompt 参数）\n"
        "  - template: 获取优化模板/框架（可选 topic 参数指定场景）\n"
        "  - save: 优化并保存到 Prompt Garden（需 name 参数）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["optimize", "template", "save"],
                "description": "操作类型：optimize=优化, template=获取模板, save=优化并保存",
            },
            "prompt": {
                "type": "string",
                "description": "待优化的原始提示词（action=optimize/save 时必填）",
            },
            "mode": {
                "type": "string",
                "enum": ["user", "system"],
                "description": "提示词类型：user=用户提示词（任务指令）, system=系统提示词（角色设定）。默认 user",
                "default": "user",
            },
            "name": {
                "type": "string",
                "description": "保存到 Prompt Garden 的名称（action=save 时必填）",
            },
            "tags": {
                "type": "string",
                "description": "标签，逗号分隔（action=save 时可选）",
            },
            "topic": {
                "type": "string",
                "description": "场景主题，如'写作'、'代码'、'翻译'（action=template 时可选）",
            },
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "optimize"}}, "required": ["action"]},
                "then": {"required": ["prompt"]},
            },
            {
                "if": {"properties": {"action": {"const": "save"}}, "required": ["action"]},
                "then": {"required": ["prompt", "name"]},
            },
        ],
    },
}


PROMPT_ANALYZE_SCHEMA = {
    "name": "prompt_analyze",
    "description": (
        "提示词质量分析工具。从 6 个维度评分并给出改进建议。\n\n"
        "评分维度（每项 1-10 分）：\n"
        "  - clarity: 清晰度 — 意图是否明确，有无歧义\n"
        "  - specificity: 具体性 — 是否包含足够细节\n"
        "  - structure: 结构性 — 是否有层次和逻辑\n"
        "  - role_definition: 角色定义 — 是否明确了 AI 的角色/身份\n"
        "  - constraints: 约束性 — 是否有边界条件和限制\n"
        "  - reusability: 可复用性 — 是否可迁移到类似场景"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "待分析的提示词",
            },
            "mode": {
                "type": "string",
                "enum": ["user", "system"],
                "description": "提示词类型。默认 user",
                "default": "user",
            },
        },
        "required": ["prompt"],
    },
}


PROMPT_COMPARE_SCHEMA = {
    "name": "prompt_compare",
    "description": (
        "提示词 A/B 对比工具。将两个版本的提示词进行结构化对比。\n\n"
        "对比维度：\n"
        "  - 完整性：哪个覆盖了更多必要信息\n"
        "  - 清晰度：哪个意图更明确\n"
        "  - 结构性：哪个组织更好\n"
        "  - 可控性：哪个输出更可预测\n"
        "  - 可复用性：哪个更容易迁移"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt_a": {
                "type": "string",
                "description": "提示词版本 A（通常是原始版本）",
            },
            "prompt_b": {
                "type": "string",
                "description": "提示词版本 B（通常是优化版本）",
            },
            "task_context": {
                "type": "string",
                "description": "可选的任务上下文描述，帮助更准确对比",
            },
        },
        "required": ["prompt_a", "prompt_b"],
    },
}


PROMPT_GARDEN_SCHEMA = {
    "name": "prompt_garden",
    "description": (
        "Prompt Garden — 提示词资产管理。保存、检索、版本管理你的提示词库。\n\n"
        "操作：\n"
        "  - save: 保存提示词（需 name + prompt，可选 tags/description）\n"
        "  - list: 列出所有保存的提示词（可选 tag 过滤）\n"
        "  - get: 获取指定提示词详情（需 name）\n"
        "  - search: 搜索提示词（需 query）\n"
        "  - delete: 删除提示词（需 name）\n"
        "  - export: 导出全部为 JSON\n"
        "  - history: 查看某条提示词的版本历史（需 name）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "list", "get", "search", "delete", "export", "history"],
                "description": "操作类型",
            },
            "name": {
                "type": "string",
                "description": "提示词名称（save/get/delete/history 时必填）",
            },
            "prompt": {
                "type": "string",
                "description": "提示词内容（save 时必填）",
            },
            "description": {
                "type": "string",
                "description": "提示词描述/用途说明（save 时可选）",
            },
            "tags": {
                "type": "string",
                "description": "标签，逗号分隔（save 时可选，list 时用于过滤）",
            },
            "query": {
                "type": "string",
                "description": "搜索关键词（action=search 时必填）",
            },
            "mode": {
                "type": "string",
                "enum": ["user", "system"],
                "description": "提示词类型。默认 user",
                "default": "user",
            },
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "save"}}, "required": ["action"]},
                "then": {"required": ["name", "prompt"]},
            },
            {
                "if": {"properties": {"action": {"const": "get"}}, "required": ["action"]},
                "then": {"required": ["name"]},
            },
            {
                "if": {"properties": {"action": {"const": "delete"}}, "required": ["action"]},
                "then": {"required": ["name"]},
            },
            {
                "if": {"properties": {"action": {"const": "search"}}, "required": ["action"]},
                "then": {"required": ["query"]},
            },
            {
                "if": {"properties": {"action": {"const": "history"}}, "required": ["action"]},
                "then": {"required": ["name"]},
            },
        ],
    },
}
