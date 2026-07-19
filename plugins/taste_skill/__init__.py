"""Taste-Skill AI前端设计约束插件

提供专业的前端设计规范和约束，帮助生成高质量的前端界面。
"""

from __future__ import annotations

try:
    from .provider import TasteSkillProvider
except Exception:  # 顶部导入失败时降级，register 内部会再做延迟导入
    TasteSkillProvider = None


def register(ctx) -> None:
    """注册 Taste-Skill 提供者"""
    if ctx is None or not hasattr(ctx, "register_tool"):
        import logging
        logging.getLogger(__name__).warning("taste_skill: ctx 无效，跳过注册")
        return
    try:
        # 延迟导入：防止顶部导入失败导致整个模块不可用
        from .provider import TasteSkillProvider
        provider = TasteSkillProvider()

        # 注册工具
        ctx.register_tool(
            name='taste_design',
            toolset='taste_skill',
            schema={"type": "object", "properties": {
                "vibe": {"type": "string", "description": "设计风格"},
                "page_type": {"type": "string", "description": "页面类型"},
            }},
            handler=provider.design,
            description="生成前端设计方案",
        )
        ctx.register_tool(
            name='taste_redesign',
            toolset='taste_skill',
            schema={"type": "object", "properties": {
                "skill_name": {"type": "string", "description": "技能名"},
            }},
            handler=provider.redesign,
            description="重新设计已有技能",
        )
        ctx.register_tool(
            name='taste_audit',
            toolset='taste_skill',
            schema={"type": "object", "properties": {
                "skill_name": {"type": "string", "description": "技能名"},
            }},
            handler=provider.audit,
            description="审计技能设计质量",
        )
        ctx.register_tool(
            name='taste_reference',
            toolset='taste_skill',
            schema={"type": "object", "properties": {}},
            handler=provider.reference,
            description="获取设计参考",
        )

        # 注册提供者
        if hasattr(ctx, 'register_design_provider'):
            ctx.register_design_provider(provider)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("taste_skill: register 失败: %s", e)


def create_provider() -> TasteSkillProvider:
    """创建 Taste-Skill 提供者实例"""
    return TasteSkillProvider()
