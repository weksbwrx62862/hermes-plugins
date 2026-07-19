"""MermaidCanvas — Mermaid 符号化压缩模块。

参考 TencentDB Agent Memory 的符号化记忆方案：
- 工具日志卸载到外部文件（refs/目录）
- 提取任务状态流转，生成 Mermaid graph
- 每个 node 携带 node_id，可按需下钻恢复原文

优化点：
1. refs/ 目录过期清理机制（max_refs_age_days）
2. _is_tool_log 支持 config 可配 pattern
3. refs/_index.json 映射持久化
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MermaidCanvas:
    """Mermaid 符号画布 — 将工具调用日志压缩为轻量 Mermaid 图谱。

    参考 TencentDB Agent Memory 的符号化记忆方案：
    - 工具日志卸载到外部文件（refs/目录）
    - 提取任务状态流转，生成 Mermaid graph
    - 每个 node 携带 node_id，可按需下钻恢复原文
    """

    def __init__(self, data_dir: Path, config: Any = None):
        """初始化 MermaidCanvas。

        Args:
            data_dir: 数据目录（如 ~/.hermes/omnimem）
            config: 配置对象，用于读取 mermaid_tool_log_patterns 等配置
        """
        self._data_dir = data_dir
        self._refs_dir = data_dir / "refs"
        self._refs_dir.mkdir(parents=True, exist_ok=True)
        self._config = config

        # 优化1: refs/ 目录过期清理机制
        self._max_refs_age_days = config.get("max_refs_age_days", 30) if config else 30

        # 优化3: ref_path 映射持久化
        self._index_path = self._refs_dir / "_index.json"
        self._index: dict[str, str] = {}
        self._load_index()

    def _load_index(self) -> None:
        """加载 ref_path 映射索引。"""
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load ref index: %s", e)
                self._index = {}

    def _save_index(self) -> None:
        """保存 ref_path 映射索引。"""
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("Failed to save ref index: %s", e)

    def offload_and_compress(
        self,
        tool_logs: list[dict],
        session_key: str,
    ) -> tuple[str, dict[str, str]]:
        """卸载工具日志 + 生成 Mermaid 画布。

        Args:
            tool_logs: 工具调用日志列表，每个元素包含：
                - tool_name: 工具名称
                - args: 参数（可选）
                - result: 结果（可选）
                - status: 状态（success/failure/pending）
            session_key: 会话标识，用于生成唯一文件名

        Returns:
            (mermaid_text, node_id_to_ref_path)
            mermaid_text: 轻量 Mermaid 图谱（几百 Token）
            node_id_to_ref_path: node_id → refs 文件路径映射
        """
        # 优化1: 清理过期 refs 文件
        self._cleanup_stale_refs()

        if not tool_logs:
            return "", {}

        # 1. 为每个工具调用生成 node_id（格式：T{序号}）
        nodes = []
        node_id_to_ref_path = {}

        for i, log in enumerate(tool_logs):
            node_id = f"T{i+1}"
            nodes.append({
                "node_id": node_id,
                "tool_name": log.get("tool_name", "unknown"),
                "status": log.get("status", "pending"),
                "args": log.get("args"),
                "result": log.get("result"),
            })

            # 2. 将完整日志写入 refs/{session_key}_{node_id}.md
            ref_path = self._refs_dir / f"{session_key}_{node_id}.md"
            self._write_ref_file(ref_path, log)
            node_id_to_ref_path[node_id] = str(ref_path)

            # 优化3: 更新索引
            self._index[node_id] = str(ref_path)

        # 3. 提取状态流转关系（success/failure/pending）
        edges = self._extract_edges(tool_logs)

        # 4. 生成 Mermaid graph 语法
        mermaid_text = self._generate_mermaid(nodes, edges)

        # 优化3: 保存索引
        self._save_index()

        return mermaid_text, node_id_to_ref_path

    def recover_by_node_id(self, node_id: str) -> str | None:
        """按 node_id 恢复完整原文。

        Args:
            node_id: 节点标识（如 T1, T2）

        Returns:
            原文内容，如果不存在则返回 None
        """
        # 从索引中查找 ref_path
        ref_path_str = self._index.get(node_id)
        if not ref_path_str:
            logger.warning("Node %s not found in index", node_id)
            return None

        ref_path = Path(ref_path_str)
        if not ref_path.exists():
            logger.warning("Ref file not found: %s", ref_path)
            return None

        try:
            return ref_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read ref file %s: %s", ref_path, e)
            return None

    def _write_ref_file(self, ref_path: Path, log: dict) -> None:
        """将工具日志写入 ref 文件。"""
        content_parts = []

        if "tool_name" in log:
            content_parts.append(f"## Tool: {log['tool_name']}")

        if "args" in log and log["args"]:
            content_parts.append(f"### Args\n```json\n{json.dumps(log['args'], ensure_ascii=False, indent=2)}\n```")

        if "result" in log and log["result"]:
            result_str = log["result"] if isinstance(log["result"], str) else json.dumps(log["result"], ensure_ascii=False, indent=2)
            content_parts.append(f"### Result\n```\n{result_str}\n```")

        if "status" in log:
            content_parts.append(f"### Status: {log['status']}")

        # 添加其他字段
        for key, value in log.items():
            if key not in ["tool_name", "args", "result", "status"]:
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, ensure_ascii=False, indent=2)
                else:
                    value_str = str(value)
                content_parts.append(f"### {key}: {value_str}")

        content = "\n\n".join(content_parts) if content_parts else json.dumps(log, ensure_ascii=False, indent=2)

        try:
            ref_path.write_text(content, encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to write ref file %s: %s", ref_path, e)

    def _extract_edges(self, tool_logs: list[dict]) -> list[tuple[str, str, str]]:
        """从工具调用序列中提取状态边。

        Args:
            tool_logs: 工具调用日志列表

        Returns:
            边列表：[(from_node, to_node, label), ...]
        """
        edges = []

        for i in range(len(tool_logs) - 1):
            from_node = f"T{i+1}"
            to_node = f"T{i+2}"

            # 根据状态生成边标签
            from_status = tool_logs[i].get("status", "pending")
            to_status = tool_logs[i+1].get("status", "pending")

            if from_status == "success" and to_status == "success":
                label = "success"
            elif from_status == "failure":
                label = "error"
            elif from_status == "success" and to_status == "failure":
                label = "failed"
            else:
                label = from_status

            edges.append((from_node, to_node, label))

        return edges

    def _generate_mermaid(self, nodes: list[dict], edges: list[tuple[str, str, str]]) -> str:
        """生成 Mermaid graph 语法。

        Args:
            nodes: 节点列表，每个节点包含 node_id, tool_name, status
            edges: 边列表，每个边包含 from_node, to_node, label

        Returns:
            Mermaid graph 语法字符串
        """
        lines = ["graph TD"]

        # 生成节点
        for node in nodes:
            node_id = node["node_id"]
            tool_name = node["tool_name"]
            status = node["status"]

            # 根据状态选择样式
            if status == "success" or status == "failure":
                style = f'{node_id}["{tool_name}"]'
            else:
                style = f'{node_id}["{tool_name}"]'

            lines.append(f"    {style}")

        # 生成边
        for from_node, to_node, label in edges:
            lines.append(f"    {from_node} -->|{label}| {to_node}")

        return "\n".join(lines)

    def _cleanup_stale_refs(self) -> None:
        """优化1: 清理超过 max_refs_age_days 的 refs 文件。"""
        if self._max_refs_age_days < 0:
            return

        cutoff = time.time() - self._max_refs_age_days * 86400
        cleaned_count = 0

        for ref_file in self._refs_dir.glob("*.md"):
            if ref_file.name.startswith("_"):  # 跳过索引文件
                continue

            try:
                if ref_file.stat().st_mtime < cutoff:
                    ref_file.unlink()
                    cleaned_count += 1

                    # 从索引中移除
                    node_id = ref_file.stem.split("_")[-1] if "_" in ref_file.stem else None
                    if node_id and node_id in self._index:
                        del self._index[node_id]
            except OSError as e:
                logger.warning("Failed to cleanup ref file %s: %s", ref_file, e)

        if cleaned_count > 0:
            logger.info("Cleaned up %d stale ref files", cleaned_count)
            self._save_index()

    def is_tool_log(self, content: str) -> bool:
        """优化2: 检测内容是否为工具调用日志。

        支持从 config 读取可配 pattern 列表，硬编码正则作为 fallback。

        Args:
            content: 待检测内容

        Returns:
            是否为工具调用日志
        """
        # 从 config 读取用户自定义 pattern
        custom_patterns = []
        if self._config:
            custom_patterns = self._config.get("mermaid_tool_log_patterns", [])

        # 硬编码正则作为 fallback
        default_patterns = [
            r'(tool_call|function_call|result_ref|command:)',
            r'(调用工具|执行命令|工具结果|工具调用)',
            r'```json\s*\{.*"tool_name"',
        ]

        patterns = custom_patterns if custom_patterns else default_patterns
        head = content[:500]

        return any(re.search(p, head, re.IGNORECASE | re.DOTALL) for p in patterns)

    def parse_tool_logs(self, content: str) -> list[dict]:
        """从内容中解析工具调用序列。

        Args:
            content: 包含工具调用日志的内容

        Returns:
            解析后的工具调用列表
        """
        tool_logs = []

        # 简单的解析逻辑：按行扫描，提取 tool_name / args / result / status
        lines = content.split("\n")
        current_log: dict[str, Any] = {}

        for line in lines:
            line = line.strip()

            # 检测工具调用开始
            if re.search(r'(tool_call|function_call|调用工具)', line, re.IGNORECASE):
                if current_log:
                    tool_logs.append(current_log)
                current_log = {"status": "pending"}

                # 尝试提取工具名
                tool_match = re.search(r'"tool_name":\s*"([^"]+)"', line)
                if tool_match:
                    current_log["tool_name"] = tool_match.group(1)

            # 检测结果
            elif re.search(r'(result|结果|返回)', line, re.IGNORECASE):
                if current_log:
                    current_log["result"] = line
                    current_log["status"] = "success"

            # 检测错误
            elif re.search(r'(error|错误|失败)', line, re.IGNORECASE):
                if current_log:
                    current_log["status"] = "failure"
                    current_log["result"] = line

        # 添加最后一个日志
        if current_log:
            tool_logs.append(current_log)

        return tool_logs
