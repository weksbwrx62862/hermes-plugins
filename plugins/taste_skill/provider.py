"""Taste-Skill AI前端设计约束 Provider

提供专业的前端设计规范和约束，帮助生成高质量的前端界面。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class TasteSkillProvider:
    """Taste-Skill 前端设计约束提供者"""
    
    def __init__(self, skills_path: Optional[str] = None):
        if skills_path is None:
            skills_path = str(Path.home() / ".hermes" / "skills")
        self.skills_path = Path(skills_path)
        self._skills = {}
        self._load_skills()
    
    def _load_skills(self) -> None:
        """加载所有 taste-skill 相关技能"""
        skill_dirs = [
            "taste-skill",
            "redesign-skill",
            "output-skill",
            "soft-skill",
            "minimalist-skill",
            "brutalist-skill"
        ]
        
        for skill_dir in skill_dirs:
            skill_path = self.skills_path / skill_dir / "SKILL.md"
            if skill_path.exists():
                with open(skill_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1].strip()
                        skill_content = parts[2].strip()
                        
                        name = skill_dir
                        description = ""
                        
                        for line in frontmatter.split('\n'):
                            if line.startswith('name:'):
                                name = line.split(':', 1)[1].strip()
                            elif line.startswith('description:'):
                                description = line.split(':', 1)[1].strip()
                        
                        self._skills[skill_dir] = {
                            "name": name,
                            "description": description,
                            "content": skill_content,
                            "path": str(skill_path)
                        }
    
    @property
    def name(self) -> str:
        return "taste-skill"
    
    def is_available(self) -> bool:
        """检查 taste-skill 是否可用"""
        return len(self._skills) > 0
    
    def design(self, args: dict = None, **kwargs) -> str:
        """获取前端设计约束"""
        try:
            args = args or {}
            page_type = args.get('page_type', 'landing') if isinstance(args, dict) else 'landing'
            vibe = args.get('vibe', 'minimalist') if isinstance(args, dict) else 'minimalist'

            skill_name = self._select_skill(page_type, vibe)

            if skill_name not in self._skills:
                return json.dumps({
                    "success": False,
                    "message": f"未找到适合的设计技能: {skill_name}"
                }, ensure_ascii=False)

            skill = self._skills[skill_name]

            return json.dumps({
                "success": True,
                "skill": skill["name"],
                "description": skill["description"],
                "content": skill["content"],
                "page_type": page_type,
                "vibe": vibe
            }, ensure_ascii=False)
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("taste.design 执行失败: %s", e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    def _select_skill(self, page_type: str, vibe: str) -> str:
        """根据页面类型和氛围选择技能"""
        page_type_mapping = {
            "landing": "taste-skill",
            "dashboard": "minimalist-skill",
            "form": "soft-skill",
            "editorial": "minimalist-skill",
            "portfolio": "brutalist-skill",
            "ecommerce": "soft-skill",
            "blog": "minimalist-skill",
            "app": "minimalist-skill",
        }

        page_skill = page_type_mapping.get(page_type)
        if page_skill and page_skill in self._skills:
            return page_skill

        vibe_mapping = {
            "minimalist": "minimalist-skill",
            "soft": "soft-skill",
            "brutalist": "brutalist-skill",
            "editorial": "minimalist-skill",
            "agency": "soft-skill",
            "tech": "brutalist-skill"
        }

        vibe_skill = vibe_mapping.get(vibe, "taste-skill")
        if vibe_skill in self._skills:
            return vibe_skill

        if "taste-skill" in self._skills:
            return "taste-skill"
        return next(iter(self._skills), "taste-skill")
    
    def redesign(self, args: dict = None, **kwargs) -> str:
        """获取重新设计约束"""
        if "redesign-skill" not in self._skills:
            return json.dumps({
                "success": False,
                "message": "未找到 redesign-skill"
            }, ensure_ascii=False)
        
        skill = self._skills["redesign-skill"]
        
        return json.dumps({
            "success": True,
            "skill": skill["name"],
            "description": skill["description"],
            "content": skill["content"]
        }, ensure_ascii=False)
    
    def audit(self, args: dict = None, **kwargs) -> str:
        """获取设计审计约束"""
        if "output-skill" not in self._skills:
            return json.dumps({
                "success": False,
                "message": "未找到 output-skill"
            }, ensure_ascii=False)
        
        skill = self._skills["output-skill"]
        
        return json.dumps({
            "success": True,
            "skill": skill["name"],
            "description": skill["description"],
            "content": skill["content"]
        }, ensure_ascii=False)
    
    def reference(self, args: dict = None, **kwargs) -> str:
        """获取设计参考"""
        try:
            args = args or {}
            skill_type = args.get('type', 'all') if isinstance(args, dict) else 'all'

            if skill_type == 'all':
                skills_list = []
                for name, skill in self._skills.items():
                    skills_list.append({
                        "name": name,
                        "description": skill["description"]
                    })

                return json.dumps({
                    "success": True,
                    "skills": skills_list
                }, ensure_ascii=False)

            if skill_type not in self._skills:
                return json.dumps({
                    "success": False,
                    "message": f"未找到技能: {skill_type}"
                }, ensure_ascii=False)

            skill = self._skills[skill_type]

            return json.dumps({
                "success": True,
                "skill": skill["name"],
                "description": skill["description"],
                "content": skill["content"]
            }, ensure_ascii=False)
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("taste.reference 执行失败: %s", e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    def shutdown(self) -> None:
        """关闭提供者"""
        pass
