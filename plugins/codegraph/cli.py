#!/usr/bin/env python3
"""CodeGraph CLI 包装器"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='CodeGraph 代码知识图谱工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # init 命令
    init_parser = subparsers.add_parser('init', help='初始化项目索引')
    init_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    init_parser.add_argument('-i', '--interactive', action='store_true', help='交互式初始化')
    
    # query 命令
    query_parser = subparsers.add_parser('query', help='搜索符号')
    query_parser.add_argument('search', help='搜索词')
    query_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    # files 命令
    files_parser = subparsers.add_parser('files', help='获取文件结构')
    files_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    # status 命令
    status_parser = subparsers.add_parser('status', help='获取索引状态')
    status_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    # serve 命令
    serve_parser = subparsers.add_parser('serve', help='启动 MCP 服务')
    serve_parser.add_argument('path', nargs='?', default='.', help='项目路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 构建命令
    cmd = ['codegraph', args.command]
    
    if args.command == 'init':
        if args.interactive:
            cmd.append('-i')
        cmd.append(args.path)
    elif args.command == 'query':
        cmd.append(args.search)
        cmd.append(args.path)
    elif args.command == 'files':
        cmd.append(args.path)
    elif args.command == 'status':
        cmd.append(args.path)
    elif args.command == 'serve':
        cmd.append(args.path)
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("错误: codegraph 未安装")
        print("请运行: curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh")
        sys.exit(1)


if __name__ == '__main__':
    main()
