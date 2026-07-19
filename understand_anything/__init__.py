"""Understand-Anything 代码理解仪表盘插件

生成交互式知识图谱和可视化仪表盘，帮助理解代码库。
"""

from __future__ import annotations

try:
    from .provider import UnderstandAnythingProvider
except Exception:
    UnderstandAnythingProvider = None


def register(ctx) -> None:
    """注册 Understand-Anything 提供者"""
    # ctx 校验：容忍 None / 非法类型，避免插件加载器崩溃
    if ctx is None:
        import logging
        logging.getLogger(__name__).warning("understand_anything: ctx 为 None，跳过注册")
        return

    try:
        provider = UnderstandAnythingProvider()

        # 注册工具
        if hasattr(ctx, 'register_tool'):
            ctx.register_tool(
                name='understand_analyze',
                toolset='understand_anything',
                schema={"type": "object", "properties": {
                    "path": {"type": "string", "description": "代码库路径"},
                    "target_dir": {"type": "string", "description": "项目目录路径"},
                }},
                handler=provider.analyze,
                description="分析代码库生成知识图谱",
            )
            ctx.register_tool(
                name='understand_dashboard',
                toolset='understand_anything',
                schema={"type": "object", "properties": {"target_dir": {"type": "string", "description": "项目目录路径"}}},
                handler=provider.dashboard,
                description="启动代码理解仪表盘",
            )
            ctx.register_tool(
                name='understand_search',
                toolset='understand_anything',
                schema={"type": "object", "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "target_dir": {"type": "string", "description": "项目目录路径"},
                }},
                handler=provider.search,
                description="搜索代码库中的符号",
            )
            ctx.register_tool(
                name='understand_explain',
                toolset='understand_anything',
                schema={"type": "object", "properties": {
                    "symbol": {"type": "string", "description": "符号名"},
                    "target_dir": {"type": "string", "description": "项目目录路径"},
                }},
                handler=provider.explain,
                description="解释代码符号",
            )

        # 注册提供者
        if hasattr(ctx, 'register_code_understanding_provider'):
            ctx.register_code_understanding_provider(provider)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("understand_anything: register 失败: %s", e)


def create_provider() -> UnderstandAnythingProvider:
    """创建 Understand-Anything 提供者实例"""
    return UnderstandAnythingProvider()
