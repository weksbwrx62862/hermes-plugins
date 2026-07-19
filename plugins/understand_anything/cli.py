#!/usr/bin/env python3
"""Understand-Anything CLI 包装器"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Understand-Anything 代码理解仪表盘工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析代码库')
    analyze_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    # dashboard 命令
    dashboard_parser = subparsers.add_parser('dashboard', help='启动仪表盘')
    dashboard_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='搜索知识图谱')
    search_parser.add_argument('query', help='搜索词')
    search_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    # build 命令
    build_parser = subparsers.add_parser('build', help='构建项目')
    build_parser.add_argument('path', nargs='?', default='/tmp/Understand-Anything', help='项目路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    install_path = Path('/tmp/Understand-Anything')
    
    if args.command == 'analyze':
        # 检查是否已分析
        understand_dir = Path(args.path) / '.understand-anything'
        if not understand_dir.exists():
            print("请先在 Claude Code 中运行 /understand 命令分析代码库")
            print(f"命令: cd {args.path} && # 在 Claude Code 中运行 /understand")
            return
        
        graph_file = understand_dir / 'knowledge-graph.json'
        if graph_file.exists():
            import json
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            print(f"知识图谱统计:")
            print(f"  节点数: {len(nodes)}")
            print(f"  边数: {len(edges)}")
            print(f"  文件: {graph_file}")
        else:
            print("知识图谱文件不存在")
    
    elif args.command == 'dashboard':
        dashboard_dir = install_path / 'understand-anything-plugin' / 'packages' / 'dashboard'
        if not dashboard_dir.exists():
            print("仪表盘未构建，请先运行: understand-anything build")
            return
        
        print("启动仪表盘开发服务器...")
        print(f"命令: cd {install_path} && pnpm dev:dashboard")
        print("仪表盘将在 http://localhost:5173 启动")
    
    elif args.command == 'search':
        understand_dir = Path(args.path) / '.understand-anything'
        if not understand_dir.exists():
            print("请先分析代码库")
            return
        
        graph_file = understand_dir / 'knowledge-graph.json'
        if not graph_file.exists():
            print("知识图谱文件不存在")
            return
        
        import json
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        nodes = graph_data.get('nodes', [])
        results = []
        query_lower = args.query.lower()
        
        for node in nodes:
            name = node.get('name', '').lower()
            description = node.get('description', '').lower()
            
            if query_lower in name or query_lower in description:
                results.append(node)
        
        print(f"搜索结果: {len(results)} 个节点")
        for node in results[:10]:
            print(f"  - {node.get('name', 'Unknown')}: {node.get('description', '')[:50]}")
    
    elif args.command == 'build':
        print(f"构建 Understand-Anything...")
        print(f"路径: {args.path}")
        
        # 运行构建命令
        try:
            subprocess.run(['pnpm', '--filter', '@understand-anything/core', 'build'], 
                         cwd=args.path, check=True)
            subprocess.run(['pnpm', '--filter', '@understand-anything/skill', 'build'], 
                         cwd=args.path, check=True)
            print("构建完成!")
        except subprocess.CalledProcessError as e:
            print(f"构建失败: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
