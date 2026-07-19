"""CodeGraph 代码知识图谱 Provider

提供代码库索引、符号搜索、依赖分析和 MCP 服务功能。
基于 tree-sitter 解析代码，使用 SQLite (FTS5) 存储符号/边/文件。
"""

from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path
from typing import Any, Optional


class CodeGraphProvider:
    """CodeGraph 代码知识图谱提供者"""
    
    @property
    def name(self) -> str:
        return "codegraph"
    
    def is_available(self) -> bool:
        """检查 codegraph CLI 是否可用"""
        try:
            result = subprocess.run(
                ["codegraph", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def initialize(self, args: dict, **kwargs) -> None:
        """初始化目标目录的 CodeGraph 索引"""
        target_dir = args.get("target_dir") if isinstance(args, dict) else args
        if not target_dir:
            raise ValueError("target_dir 参数缺失")
        if not self.is_available():
            raise RuntimeError("codegraph CLI 未安装")
        
        result = subprocess.run(
            ["codegraph", "init", "-i"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"初始化失败: {result.stderr}")
    
    def index(self, args: dict = None, **kwargs) -> str:
        """索引代码库"""
        target_dir = (args or {}).get("target_dir") if isinstance(args, dict) else args
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        result = subprocess.run(
            ["codegraph", "index"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }, ensure_ascii=False)
    
    def sync(self, args: dict = None, **kwargs) -> str:
        """同步变更"""
        target_dir = (args or {}).get("target_dir") if isinstance(args, dict) else args
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        result = subprocess.run(
            ["codegraph", "sync"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }, ensure_ascii=False)
    
    def status(self, args: dict = None, **kwargs) -> str:
        """获取索引状态"""
        target_dir = (args or {}).get("target_dir") if isinstance(args, dict) else args
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        result = subprocess.run(
            ["codegraph", "status"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }, ensure_ascii=False)
    
    def query(self, args: dict = None, **kwargs) -> str:
        """搜索符号"""
        args = args or {}
        target_dir = args.get("target_dir") if isinstance(args, dict) else args
        search = args.get("search", "") if isinstance(args, dict) else ""
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        result = subprocess.run(
            ["codegraph", "query", search],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }, ensure_ascii=False)
    
    def files(self, args: dict = None, **kwargs) -> str:
        """获取文件结构"""
        target_dir = (args or {}).get("target_dir") if isinstance(args, dict) else args
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        result = subprocess.run(
            ["codegraph", "files"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }, ensure_ascii=False)
    
    def serve(self, args: dict = None, **kwargs) -> str:
        """启动 MCP 服务作为后台进程"""
        target_dir = (args or {}).get("target_dir") if isinstance(args, dict) else args
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)

        try:
            process = subprocess.Popen(
                ["codegraph", "serve"],
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return json.dumps({
                "success": True,
                "pid": process.pid,
                "message": f"codegraph serve 已启动 (PID={process.pid})",
            }, ensure_ascii=False)
        except FileNotFoundError:
            return json.dumps({"success": False, "error": "codegraph CLI 未安装"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": f"启动失败: {e}"}, ensure_ascii=False)
    
    def shutdown(self) -> None:
        """关闭提供者"""
        pass
