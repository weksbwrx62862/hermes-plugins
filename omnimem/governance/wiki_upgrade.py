"""WikiUpgradePipeline — 从 OmniMem 记忆升级到 LLM Wiki 页面。

流程：
1. 提取记忆上下文（通过 store.get）
2. LLM 判断页面类型（entity/concept/comparison）+ 生成内容
3. 写入 ~/wiki/{type}/
4. 更新 index.md + log.md
5. 标记 OmniMem 记忆为 upgraded_to_wiki
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# LLM 提示词模板
_UPGRADE_PROMPT = """你是一个知识整理专家。下面是一条来自 OmniMem 的记忆条目，请根据内容判断：
1. 最适合的 Wiki 页面类型：entity（实体/人物/工具/项目）、concept（概念/技术/方法论）、comparison（对比分析）
2. 生成一个完整的 Wiki 页面（Markdown 格式）

记忆内容：
---
{content}
---

现有 Wiki 页面列表（用于交叉引用）：
{existing_pages}

要求：
- 标题简洁明确，适合作为文件名
- 如果是 entity 页面：概述 + 关键事实 + 与其他实体的关系
- 如果是 concept 页面：定义 + 核心要点 + 相关概念
- 如果是 comparison 页面：对比维度 + 表格 + 结论
- 正文必须包含至少 2 个 [[wikilinks]] 链接到现有页面
- 使用 YAML frontmatter，包含：title, created, updated, type, tags, sources, omnimem_id, confidence: medium
- 内容精炼，适合 30 秒内扫读

请输出 JSON 格式：
{{
  "page_type": "entity|concept|comparison",
  "title": "页面标题",
  "filename": "文件名（小写，连字符分隔，无空格）",
  "content": "完整的 Markdown 内容（含 frontmatter）"
}}"""


def _slugify(text: str) -> str:
    """将中文/英文标题转为 URL-safe 的 slug。"""
    # 移除特殊字符，保留中英文和数字
    clean = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', text.lower())
    # 替换空格为连字符
    clean = re.sub(r'[\s]+', '-', clean)
    # 去除首尾连字符
    return clean.strip('-')[:80]


class WikiUpgradePipeline:
    """Wiki 升级管道：从 OmniMem 记忆生成 LLM Wiki 页面。"""

    def __init__(
        self,
        wiki_path: str | Path,
        store: Any = None,
        forgetting: Any = None,
        llm_call: Any = None,
    ):
        """
        Args:
            wiki_path: Wiki 根目录路径 (如 ~/wiki)
            store: OmniMem 的 DrawerClosetStore 实例
            forgetting: OmniMem 的 ForgettingCurve 实例
            llm_call: LLM 调用函数，签名 (prompt: str) -> str
        """
        self._wiki_path = Path(wiki_path).expanduser()
        self._store = store
        self._forgetting = forgetting
        self._llm_call = llm_call

    def _get_existing_pages(self) -> list[str]:
        """扫描 Wiki 目录，返回现有页面列表。"""
        pages = []
        for subdir in ("entities", "concepts", "comparisons", "queries"):
            dir_path = self._wiki_path / subdir
            if dir_path.exists():
                for f in dir_path.glob("*.md"):
                    pages.append(f"[{subdir}/{f.stem}]")
        return pages

    def _update_index(self, page_type: str, filename: str, title: str, summary: str) -> None:
        """在 index.md 中添加新条目。"""
        index_path = self._wiki_path / "index.md"
        if not index_path.exists():
            return

        content = index_path.read_text(encoding="utf-8")
        section_map = {
            "entity": "## Entities",
            "concept": "## Concepts",
            "comparison": "## Comparisons",
            "query": "## Queries",
        }
        section_header = section_map.get(page_type, "## Concepts")
        wikilink = f"- [[{filename}]] — {summary}"

        # 在对应 section 的末尾插入
        lines = content.split("\n")
        insert_idx = len(lines)
        in_section = False
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                in_section = True
                continue
            if in_section and line.startswith("## "):
                insert_idx = i
                break
            if in_section and line.strip() == "":
                # section 内的空行可能是分隔
                pass

        lines.insert(insert_idx, wikilink)

        # 更新 total pages count
        updated = "\n".join(lines)
        total_match = re.search(r'Total pages: (\d+)', updated)
        if total_match:
            new_total = int(total_match.group(1)) + 1
            updated = updated.replace(f'Total pages: {total_match.group(1)}', f'Total pages: {new_total}')

        # 更新 last updated
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        updated = re.sub(r'Last updated: \d{4}-\d{2}-\d{2}', f'Last updated: {today}', updated)

        index_path.write_text(updated, encoding="utf-8")

    def _update_log(self, action: str, subject: str, files: list[str]) -> None:
        """在 log.md 中追加条目。"""
        log_path = self._wiki_path / "log.md"
        if not log_path.exists():
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = f"\n## [{today}] {action} | {subject}\n"
        for f in files:
            entry += f"- {f}\n"

        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(entry)

    def upgrade_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "concept",
    ) -> dict[str, Any]:
        """将一条 OmniMem 记忆升级为 Wiki 页面。

        Args:
            memory_id: OmniMem 记忆 ID
            content: 记忆的完整内容
            memory_type: 记忆类型（用于判断升级路径）

        Returns:
            {"status": "ok", "page_type": str, "path": str} 或 {"status": "error", "reason": str}
        """
        if not self._wiki_path.exists():
            return {"status": "error", "reason": "Wiki path does not exist"}

        # 获取现有页面
        existing = self._get_existing_pages()
        existing_str = "\n".join(existing[:50]) if existing else "(暂无页面)"

        # 调用 LLM 生成页面
        if not self._llm_call:
            return {"status": "error", "reason": "No LLM call function provided"}

        prompt = _UPGRADE_PROMPT.format(
            content=content,
            existing_pages=existing_str,
        )

        try:
            llm_output = self._llm_call(prompt)
        except Exception as e:
            logger.warning("LLM call for wiki upgrade failed: %s", e)
            return {"status": "error", "reason": f"LLM call failed: {e}"}

        # 解析 LLM 输出
        try:
            result = json.loads(llm_output)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            json_match = re.search(r'\{[^{}]*"page_type"[^{}]*\}', llm_output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return {"status": "error", "reason": f"LLM output not valid JSON: {llm_output[:200]}"}

        page_type = result.get("page_type", "concept")
        title = result.get("title", "Untitled")
        filename = result.get("filename", _slugify(title))
        page_content = result.get("content", "")

        # 确保目录存在
        subdir = {
            "entity": "entities",
            "concept": "concepts",
            "comparison": "comparisons",
        }.get(page_type, "concepts")
        dir_path = self._wiki_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)

        # 写入页面
        page_path = dir_path / f"{filename}.md"
        page_path.write_text(page_content, encoding="utf-8")

        # 更新 index.md 和 log.md
        self._update_index(page_type, filename, title, content[:60])
        self._update_log("promote", title, [f"{subdir}/{filename}.md"])

        # 标记 OmniMem 记忆
        if self._forgetting:
            wiki_rel_path = f"{subdir}/{filename}.md"
            self._forgetting.mark_upgraded_to_wiki(memory_id, wiki_rel_path)

        return {
            "status": "ok",
            "page_type": page_type,
            "path": str(page_path),
            "title": title,
            "filename": filename,
        }

    def batch_upgrade(self, memory_ids: list[str], contents: dict[str, str]) -> list[dict[str, Any]]:
        """批量升级多条记忆到 Wiki。

        Args:
            memory_ids: 记忆 ID 列表
            contents: {memory_id: content} 映射

        Returns:
            升级结果列表
        """
        results = []
        for mid in memory_ids:
            content = contents.get(mid, "")
            if not content:
                results.append({"memory_id": mid, "status": "error", "reason": "Content not found"})
                continue
            result = self.upgrade_memory(mid, content)
            results.append({"memory_id": mid, **result})
        return results
