from __future__ import annotations

AMA_TOOL_SCHEMAS = {
    "ama_execute": {
        "name": "ama_execute",
        "description": (
            "自适应多智能体执行入口。评估任务复杂度，自动选择最佳多智能体协作模式执行任务。"
            "支持五种模式：generator_verifier（生成-验证）、orchestrator_subagent（协调-子代理）、"
            "agent_teams（团队协作）、message_bus（事件驱动）、shared_state（共享状态）。"
            "适合需要多步骤、多角色协作的复杂任务。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "任务描述，越详细越好。包含目标、约束、期望输出格式等信息。",
                },
                "context": {
                    "type": "string",
                    "description": "额外上下文信息，如背景资料、相关文件内容、历史决策等。",
                },
                "force_mode": {
                    "type": "string",
                    "enum": [
                        "generator_verifier",
                        "orchestrator_subagent",
                        "agent_teams",
                        "message_bus",
                        "shared_state",
                    ],
                    "description": "强制指定执行模式。不指定则由系统自动选择。",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "子代理执行超时时间（秒）。不指定则使用模式默认超时。",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "指定子代理配置类型，覆盖默认的模式配置。不指定则由系统自动选择。",
                },
                "clarify": {
                    "type": "boolean",
                    "description": "是否先通过大模型澄清需求再执行。为true时，系统会先多轮提问明确需求，再基于澄清结果执行。默认false。",
                    "default": False,
                },
                "human_input_mode": {
                    "type": "string",
                    "enum": ["NEVER", "ON_ERROR", "ALWAYS"],
                    "description": "人工介入模式。NEVER=全自动（默认），ON_ERROR=仅错误时确认，ALWAYS=每步确认。",
                    "default": "NEVER",
                },
            },
            "required": ["task"],
        },
    },
    "ama_assess": {
        "name": "ama_assess",
        "description": (
            "评估任务复杂度并推荐最佳多智能体模式，但不执行任务。"
            "返回复杂度评分（1-10）、任务类型、推荐模式等信息。"
            "用于在执行前预判任务难度和资源需求。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "任务描述。",
                },
                "context": {
                    "type": "string",
                    "description": "额外上下文信息。",
                },
                "clarify": {
                    "type": "boolean",
                    "description": "是否先通过大模型澄清需求再评估。为true时，系统会先多轮提问明确需求，再基于澄清结果评估。默认false。",
                    "default": False,
                },
            },
            "required": ["task"],
        },
    },
    "ama_switch_mode": {
        "name": "ama_switch_mode",
        "description": (
            "手动切换当前会话的多智能体执行模式。"
            "切换后后续任务将使用新模式执行。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [
                        "generator_verifier",
                        "orchestrator_subagent",
                        "agent_teams",
                        "message_bus",
                        "shared_state",
                    ],
                    "description": "目标模式。",
                },
            },
            "required": ["mode"],
        },
    },
    "ama_stats": {
        "name": "ama_stats",
        "description": (
            "查询自适应多智能体调度插件的执行统计。"
            "返回模式使用频率、成功率、平均 token 消耗、平均耗时等信息。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "detail": {
                    "type": "boolean",
                    "description": "是否返回详细的按任务类型分类统计。默认 false。",
                },
                "period": {
                    "type": "string",
                    "enum": ["day", "week", "month", "all"],
                    "description": "统计时间范围。day=今天, week=最近7天, month=最近30天, all=全部（默认）。",
                    "default": "all",
                },
            },
        },
    },
    "ama_cancel": {
        "name": "ama_cancel",
        "description": (
            "取消正在执行的自适应多智能体任务。通过 task_id 指定要取消的任务。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要取消的任务 ID，由 ama_execute 返回的 task_id 字段获取。",
                },
            },
            "required": ["task_id"],
        },
    },
    "ama_clarify": {
        "name": "ama_clarify",
        "description": (
            "需求澄清与智能评分工具。通过大模型多轮提问帮助用户明确任务需求，"
            "最终由大模型依据评分标准直接完成复杂度评分。"
            "适用于用户任务描述模糊、需求不明确的场景。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "任务描述，可以是模糊的、不完整的需求描述。",
                },
                "context": {
                    "type": "string",
                    "description": "额外上下文信息。",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "最大澄清轮次，默认3轮。每轮LLM会生成提问帮助明确需求。",
                    "default": 3,
                },
            },
            "required": ["task"],
        },
    },
    "ama_diagnose": {
        "name": "ama_diagnose",
        "description": (
            "诊断 AMA 内部状态：查看 Thompson Sampling 参数（各模式 Beta 分布）、"
            "性能历史、熔断器状态、当前会话模式覆盖等。"
            "用于排查选型异常或理解引擎学习状态。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "include_ts_params": {
                    "type": "boolean",
                    "description": "是否包含 Thompson Sampling Beta 参数。默认 true。",
                    "default": True,
                },
                "include_circuit_breakers": {
                    "type": "boolean",
                    "description": "是否包含熔断器状态。默认 true。",
                    "default": True,
                },
                "trace_id": {
                    "type": "string",
                    "description": "按 trace_id 查询执行详情和工具调用链。",
                },
            },
        },
    },
}
