#!/usr/bin/env python3
"""Taste-Skill CLI 包装器"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Taste-Skill AI前端设计约束工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # design 命令
    design_parser = subparsers.add_parser('design', help='获取设计约束')
    design_parser.add_argument('--page-type', default='landing', 
                              choices=['landing', 'portfolio', 'redesign', 'editorial'],
                              help='页面类型')
    design_parser.add_argument('--vibe', default='minimalist',
                              choices=['minimalist', 'soft', 'brutalist', 'editorial', 'agency', 'tech'],
                              help='设计氛围')
    
    # redesign 命令
    redesign_parser = subparsers.add_parser('redesign', help='获取重新设计约束')
    
    # audit 命令
    audit_parser = subparsers.add_parser('audit', help='获取设计审计约束')
    
    # reference 命令
    reference_parser = subparsers.add_parser('reference', help='获取设计参考')
    reference_parser.add_argument('--type', default='all',
                                 choices=['all', 'taste-skill', 'redesign-skill', 'output-skill', 
                                         'soft-skill', 'minimalist-skill', 'brutalist-skill'],
                                 help='技能类型')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出所有技能')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    skills_path = Path.home() / '.hermes' / 'skills'
    
    if args.command == 'list':
        skill_dirs = [
            'taste-skill',
            'redesign-skill', 
            'output-skill',
            'soft-skill',
            'minimalist-skill',
            'brutalist-skill'
        ]
        
        print("可用设计技能:")
        for skill_dir in skill_dirs:
            skill_path = skills_path / skill_dir / 'SKILL.md'
            if skill_path.exists():
                print(f"  ✓ {skill_dir}")
            else:
                print(f"  ✗ {skill_dir} (未安装)")
    
    elif args.command == 'design':
        # 选择技能
        vibe_mapping = {
            'minimalist': 'minimalist-skill',
            'soft': 'soft-skill',
            'brutalist': 'brutalist-skill',
            'editorial': 'minimalist-skill',
            'agency': 'soft-skill',
            'tech': 'brutalist-skill'
        }
        
        skill_name = vibe_mapping.get(args.vibe, 'taste-skill')
        skill_path = skills_path / skill_name / 'SKILL.md'
        
        if not skill_path.exists():
            print(f"错误: 技能 {skill_name} 未安装")
            return
        
        print(f"设计约束 ({args.page_type} 页面, {args.vibe} 氛围):")
        print(f"技能: {skill_name}")
        print()
        
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 显示前 50 行
        lines = content.split('\n')
        for line in lines[:50]:
            print(line)
        
        if len(lines) > 50:
            print(f"\n... (共 {len(lines)} 行)")
    
    elif args.command == 'redesign':
        skill_path = skills_path / 'redesign-skill' / 'SKILL.md'
        
        if not skill_path.exists():
            print("错误: redesign-skill 未安装")
            return
        
        print("重新设计约束:")
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line in lines[:50]:
            print(line)
        
        if len(lines) > 50:
            print(f"\n... (共 {len(lines)} 行)")
    
    elif args.command == 'audit':
        skill_path = skills_path / 'output-skill' / 'SKILL.md'
        
        if not skill_path.exists():
            print("错误: output-skill 未安装")
            return
        
        print("设计审计约束:")
        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line in lines[:50]:
            print(line)
        
        if len(lines) > 50:
            print(f"\n... (共 {len(lines)} 行)")
    
    elif args.command == 'reference':
        if args.type == 'all':
            skill_dirs = [
                'taste-skill',
                'redesign-skill',
                'output-skill',
                'soft-skill',
                'minimalist-skill',
                'brutalist-skill'
            ]
            
            print("设计参考技能:")
            for skill_dir in skill_dirs:
                skill_path = skills_path / skill_dir / 'SKILL.md'
                if skill_path.exists():
                    with open(skill_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取描述
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            frontmatter = parts[1].strip()
                            for line in frontmatter.split('\n'):
                                if line.startswith('description:'):
                                    desc = line.split(':', 1)[1].strip()
                                    print(f"\n{skill_dir}:")
                                    print(f"  {desc}")
                                    break
        else:
            skill_path = skills_path / args.type / 'SKILL.md'
            if not skill_path.exists():
                print(f"错误: 技能 {args.type} 未安装")
                return
            
            print(f"设计参考 ({args.type}):")
            with open(skill_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines[:50]:
                print(line)
            
            if len(lines) > 50:
                print(f"\n... (共 {len(lines)} 行)")


if __name__ == '__main__':
    main()
