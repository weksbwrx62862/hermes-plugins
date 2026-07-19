"""Understand-Anything 代码理解仪表盘 Provider

生成交互式知识图谱和可视化仪表盘，帮助理解代码库。
"""

from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path
from typing import Any, Optional


class UnderstandAnythingProvider:
    """Understand-Anything 代码理解提供者"""
    
    def __init__(self, install_path: str = "/tmp/Understand-Anything"):
        self.install_path = Path(install_path)
        self._built = False
    
    @property
    def name(self) -> str:
        return "understand-anything"
    
    def is_available(self) -> bool:
        """检查 Understand-Anything 是否可用"""
        core_dist = self.install_path / "understand-anything-plugin" / "packages" / "core" / "dist"
        skill_dist = self.install_path / "understand-anything-plugin" / "dist"
        
        return core_dist.exists() and skill_dist.exists()
    
    def _ensure_built(self) -> None:
        """确保项目已构建"""
        if self._built:
            return
        
        if not self.is_available():
            subprocess.run(
                ["pnpm", "--filter", "@understand-anything/core", "build"],
                cwd=self.install_path,
                capture_output=True,
                timeout=120
            )
            subprocess.run(
                ["pnpm", "--filter", "@understand-anything/skill", "build"],
                cwd=self.install_path,
                capture_output=True,
                timeout=120
            )
        
        self._built = True
    
    def _extract_target_dir(self, args) -> Optional[str]:
        """从 args dict 中提取 target_dir"""
        if isinstance(args, dict):
            return args.get("target_dir")
        return args if args else None
    
    def analyze(self, args: dict = None, **kwargs) -> str:
        """分析代码库并生成知识图谱"""
        target_dir = self._extract_target_dir(args)
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        self._ensure_built()
        
        understand_dir = Path(target_dir) / ".understand-anything"
        
        if not understand_dir.exists():
            return json.dumps({
                "success": False,
                "message": "请先在 Claude Code 中运行 /understand 命令分析代码库",
                "command": f"cd {target_dir} && # 在 Claude Code 中运行 /understand"
            }, ensure_ascii=False)
        
        graph_file = understand_dir / "knowledge-graph.json"
        if graph_file.exists():
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            return json.dumps({
                "success": True,
                "graph": graph_data,
                "stats": {
                    "nodes": len(graph_data.get("nodes", [])),
                    "edges": len(graph_data.get("edges", []))
                }
            }, ensure_ascii=False)
        
        return json.dumps({
            "success": False,
            "message": "知识图谱文件不存在"
        }, ensure_ascii=False)
    
    def dashboard(self, args: dict = None, **kwargs) -> str:
        """启动可视化仪表盘"""
        target_dir = self._extract_target_dir(args)
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        self._ensure_built()
        
        dashboard_dir = self.install_path / "understand-anything-plugin" / "packages" / "dashboard"
        
        if not dashboard_dir.exists():
            return json.dumps({
                "success": False,
                "message": "仪表盘未构建"
            }, ensure_ascii=False)
        
        return json.dumps({
            "success": True,
            "message": "请在终端运行仪表盘开发服务器",
            "command": f"cd {self.install_path} && pnpm dev:dashboard",
            "note": "仪表盘将在 http://localhost:5173 启动"
        }, ensure_ascii=False)
    
    def search(self, args: dict = None, **kwargs) -> str:
        """在知识图谱中搜索"""
        args = args or {}
        target_dir = args.get("target_dir") if isinstance(args, dict) else args
        query = args.get("query", "") if isinstance(args, dict) else ""
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        analysis_str = self.analyze(args)
        analysis = json.loads(analysis_str)
        
        if not analysis.get("success"):
            return analysis_str
        
        graph = analysis.get("graph", {})
        nodes = graph.get("nodes", [])
        
        results = []
        query_lower = query.lower()
        
        for node in nodes:
            name = node.get("name", "").lower()
            description = node.get("description", "").lower()
            
            if query_lower in name or query_lower in description:
                results.append(node)
        
        return json.dumps({
            "success": True,
            "query": query,
            "results": results[:20],
            "total": len(results)
        }, ensure_ascii=False)
    
    def explain(self, args: dict = None, **kwargs) -> str:
        """解释特定节点"""
        args = args or {}
        target_dir = args.get("target_dir") if isinstance(args, dict) else args
        node_id = args.get("node_id", "") if isinstance(args, dict) else ""
        if not target_dir:
            return json.dumps({"success": False, "error": "target_dir 参数缺失"}, ensure_ascii=False)
        analysis_str = self.analyze(args)
        analysis = json.loads(analysis_str)
        
        if not analysis.get("success"):
            return analysis_str
        
        graph = analysis.get("graph", {})
        nodes = graph.get("nodes", [])
        
        for node in nodes:
            if node.get("id") == node_id:
                return json.dumps({
                    "success": True,
                    "node": node,
                    "connections": self._get_connections(graph, node_id)
                }, ensure_ascii=False)
        
        return json.dumps({
            "success": False,
            "message": f"节点 {node_id} 未找到"
        }, ensure_ascii=False)
    
    def _get_connections(self, graph: dict, node_id: str) -> dict:
        """获取节点的连接关系"""
        edges = graph.get("edges", [])
        
        incoming = []
        outgoing = []
        
        for edge in edges:
            if edge.get("target") == node_id:
                incoming.append(edge)
            elif edge.get("source") == node_id:
                outgoing.append(edge)
        
        return {
            "incoming": incoming,
            "outgoing": outgoing
        }
    
    def shutdown(self) -> None:
        """关闭提供者"""
        pass
