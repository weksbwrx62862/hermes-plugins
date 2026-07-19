from __future__ import annotations

AMA_TOOL_SCHEMAS = {
    "ama_execute": {
        "name": "ama_execute",
        "description": (
            "自适应多智能体执行入口。评估任务复杂度，自动选择最佳多智能体协作模式执行任务。"
            "支持六种模式：generator_verifier（生成-验证）、orchestrator_subagent（协调-子代理）、"
            "agent_teams（团队协作）、message_bus（事件驱动）、shared_state（共享状态）、"
            "parallel_fusion（并行融合）。"
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
                        "parallel_fusion",
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
                        "parallel_fusion",
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
    "ama_trajectories": {
        "name": "ama_trajectories",
        "description": (
            "查询 Agent 执行轨迹记录。返回最近的执行轨迹、统计信息。"
            "用于分析执行历史、调试失败任务、评估 Agent 性能。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "返回的轨迹数量上限，默认 20。",
                    "default": 20,
                },
                "success_only": {
                    "type": "boolean",
                    "description": "仅返回成功/失败的轨迹。不指定则返回所有。",
                },
                "mode": {
                    "type": "string",
                    "description": "按执行模式过滤。",
                    "enum": [
                        "generator_verifier",
                        "orchestrator_subagent",
                        "agent_teams",
                        "message_bus",
                        "shared_state",
                        "parallel_fusion",
                    ],
                },
            },
        },
    },
    "ama_grade": {
        "name": "ama_grade",
        "description": (
            "对指定执行轨迹进行 LLM-as-Judge 多维度评分。"
            "评分维度：完整性、正确性、效率、工具使用、错误处理。"
            "用于持续改进 Agent 执行质量。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trajectory_id": {
                    "type": "string",
                    "description": "要评分的轨迹 ID。",
                },
            },
            "required": ["trajectory_id"],
        },
    },
    "ama_skills": {
        "name": "ama_skills",
        "description": (
            "查询 AMA 子代理技能注册表统计。"
            "返回每个技能域的试验次数、成功率、平均耗时、最佳模式等信息。"
            "用于了解 Agent 在各种任务类型上的历史表现。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "技能域过滤（可选）：programming/data_analysis/research/writing/debugging/architecture/devops/security",
                },
            },
        },
    },
    "ama_workflow": {
        "name": "ama_workflow",
        "description": (
            "列出或检查 AMA 工作流模板。"
            "返回每个工作流的阶段、角色分配、默认模式等信息。"
            "用于在任务执行前选择最佳工作流模板。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "info"],
                    "description": "list=列出所有工作流, info=查询具体工作流详情",
                    "default": "list",
                },
                "workflow_id": {
                    "type": "string",
                    "description": "工作流 ID（action=info 时必填）",
                },
            },
        },
    },
    "ama_resume": {
        "name": "ama_resume",
        "description": (
            "恢复中断的 AMA 任务。列出可恢复的 trace 或指定 trace_id 从最近的 checkpoint 继续执行。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "resume"],
                    "description": "list=列出可恢复的 trace, resume=恢复执行",
                    "default": "list",
                },
                "trace_id": {
                    "type": "string",
                    "description": "要恢复的 trace ID（action=resume 时必填）",
                },
            },
        },
    },
}
