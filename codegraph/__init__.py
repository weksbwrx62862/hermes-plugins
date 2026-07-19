"""CodeGraph 代码知识图谱插件

提供代码库索引、符号搜索、依赖分析和 MCP 服务功能。
"""

from __future__ import annotations

try:
    from .provider import CodeGraphProvider
except Exception:
    CodeGraphProvider = None


def register(ctx) -> None:
    """注册 CodeGraph 提供者"""
    if ctx is None:
        import logging
        logging.getLogger(__name__).warning("codegraph: ctx 为 None，跳过注册")
        return

    try:
        from .provider import CodeGraphProvider as _CGP
        provider = _CGP()

        # 注册工具
        if hasattr(ctx, 'register_tool'):
            ctx.register_tool(
                name='codegraph_index',
                toolset='codegraph',
                schema={"type": "object", "properties": {"path": {"type": "string", "description": "代码库路径"}, "target_dir": {"type": "string", "description": "项目目录路径"}}},
                handler=provider.index,
                description="索引代码库",
            )
            ctx.register_tool(
                name='codegraph_query',
                toolset='codegraph',
                schema={"type": "object", "properties": {"query": {"type": "string", "description": "查询关键词"}, "target_dir": {"type": "string", "description": "项目目录路径"}}},
                handler=provider.query,
                description="查询代码符号",
            )
            ctx.register_tool(
                name='codegraph_files',
                toolset='codegraph',
                schema={"type": "object", "properties": {"pattern": {"type": "string", "description": "文件匹配模式"}, "target_dir": {"type": "string", "description": "项目目录路径"}}},
                handler=provider.files,
                description="列出代码库文件",
            )
            ctx.register_tool(
                name='codegraph_status',
                toolset='codegraph',
                schema={"type": "object", "properties": {"target_dir": {"type": "string", "description": "项目目录路径"}}},
                handler=provider.status,
                description="查看索引状态",
            )
            ctx.register_tool(
                name='codegraph_serve',
                toolset='codegraph',
                schema={"type": "object", "properties": {"target_dir": {"type": "string", "description": "项目目录路径"}}},
                handler=provider.serve,
                description="启动 MCP 服务",
            )

        # 注册提供者
        if hasattr(ctx, 'register_code_intelligence_provider'):
            ctx.register_code_intelligence_provider(provider)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("codegraph: register 失败: %s", e)


def create_provider() -> CodeGraphProvider:
    """创建 CodeGraph 提供者实例"""
    return CodeGraphProvider()
