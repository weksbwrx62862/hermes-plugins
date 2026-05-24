"""CatalogRetriever — 目录递归检索通道。

内化 OpenViking find() 的目录定位+语义搜索原理：
- 阶段 1：从查询中提取结构化线索（类型/隐私/话题），映射到 Wing/Hall/Room 目录
- 阶段 2：在定位到的目录内做向量/BM25 搜索，而非全库扫描

与 OpenViking find() 的对应关系：
  OpenViking: target_uri 定位 → 语义搜索
  OmniMem:    Wing/Hall/Room 定位 → 向量/BM25 定向搜索
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CatalogRetriever:
    """目录递归检索通道 — 内化 OpenViking find() 的目录定位+语义搜索。
    
    检索流程（两阶段）：
    阶段 1 — 目录定位：
      从 query 中提取结构化线索（类型/隐私/话题），
      映射到 Wing/Hall/Room 目录，缩小搜索空间。
    阶段 2 — 定向搜索：
      在定位到的目录内做向量/BM25 搜索，
      而非全库扫描。
    """
    
    # 类型关键词 → Hall 映射
    _HALL_KEYWORDS: dict[str, list[str]] = {
        "preferences": ["偏好", "喜欢", "讨厌", "爱好", "兴趣", "prefer", "like", "dislike"],
        "facts": ["事实", "记住", "信息", "fact", "remember", "know"],
        "corrections": ["纠正", "修正", "更正", "correction", "fix", "wrong"],
        "skills": ["技能", "教程", "步骤", "skill", "tutorial", "how to"],
        "procedures": ["流程", "操作", "指南", "procedure", "guide", "steps"],
        "actions": ["行为", "工具", "调用", "action", "tool", "call"],
        "reasoning": ["推理", "教训", "经验", "reasoning", "lesson", "insight"],
    }
    
    # 隐私关键词 → Wing 映射
    _WING_KEYWORDS: dict[str, list[str]] = {
        "personal": ["我的", "私人", "秘密", "个人", "my ", "private", "secret"],
        "team": ["团队", "共享", "协作", "team", "shared", "collab"],
        "public": ["公开", "公共", "所有人", "public", "everyone"],
    }
    
    def __init__(
        self,
        index: Any,
        wing_room: Any,
        vector_retriever: Any,
        bm25_retriever: Any,
    ):
        """初始化目录递归检索通道。
        
        Args:
            index: ThreeLevelIndex 实例
            wing_room: WingRoomManager 实例
            vector_retriever: VectorRetriever 实例
            bm25_retriever: BM25Retriever 实例
        """
        self._index = index
        self._wing_room = wing_room
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        hint_wing: str = "",
        hint_hall: str = "",
    ) -> list[dict[str, Any]]:
        """目录递归检索。
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            hint_wing: 提示 Wing（如用户指定了隐私级别）
            hint_hall: 提示 Hall（如用户指定了记忆类型）
        
        Returns:
            检索结果列表，格式与其他通道一致
        """
        # 阶段 1：目录定位
        located_dirs = self._locate_directories(query, hint_wing, hint_hall)
        
        # 阶段 2：定向搜索
        if located_dirs:
            return self._search_in_directories(query, located_dirs, top_k)
        
        # 定位失败 → 回退到全库搜索
        logger.warning("Catalog: no directory located, falling back to full search")
        return []
    
    def _locate_directories(
        self,
        query: str,
        hint_wing: str,
        hint_hall: str,
    ) -> list[dict[str, str]]:
        """阶段 1：从查询中推断目标目录。
        
        推断策略：
        1. 显式 hint（用户指定 wing/hall）
        2. 类型关键词 → Hall 映射（"偏好" → preferences）
        3. 隐私关键词 → Wing 映射（"秘密" → personal）
        4. 话题关键词 → Room 匹配（从 search_l0 结果中匹配）
        """
        dirs = []
        
        # 策略 2：类型关键词 → Hall
        hall = hint_hall or self._infer_hall(query)
        # 策略 3：隐私关键词 → Wing
        wing = hint_wing or self._infer_wing(query)
        
        if wing or hall:
            # 从 L0 索引查询匹配的 Room
            try:
                rooms = self._index.search_l0(wing=wing, hall=hall)
            except Exception as e:
                logger.warning("Catalog: search_l0 failed: %s", e)
                rooms = []
            
            if rooms:
                # 策略 4：话题关键词匹配 Room
                matched_rooms = self._match_rooms(query, rooms)
                for room in matched_rooms[:3]:
                    dirs.append({"wing": wing or "", "hall": hall or "", "room": room})
            else:
                dirs.append({"wing": wing, "hall": hall, "room": ""})
        
        return dirs
    
    def _infer_hall(self, query: str) -> str:
        """从查询中推断 Hall（类型大厅）。"""
        q_lower = query.lower()
        for hall, keywords in self._HALL_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                return hall
        return ""
    
    def _infer_wing(self, query: str) -> str:
        """从查询中推断 Wing（隐私空间）。"""
        q_lower = query.lower()
        for wing, keywords in self._WING_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                return wing
        return ""
    
    def _match_rooms(self, query: str, rooms: list[str]) -> list[str]:
        """从 Room 列表中匹配与查询相关的 Room。"""
        q_lower = query.lower()
        scored = []
        for room in rooms:
            score = 0
            room_lower = room.lower()
            if room_lower in q_lower:
                score = 3  # 完全包含
            elif any(kw in room_lower for kw in q_lower.split() if len(kw) >= 2):
                score = 2  # 部分匹配
            elif any(c in room_lower for c in q_lower if c.isalpha()):
                score = 1  # 字符重叠
            if score > 0:
                scored.append((room, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]
    
    def _search_in_directories(
        self,
        query: str,
        dirs: list[dict[str, str]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """阶段 2：在定位到的目录内做定向搜索。"""
        all_results = []
        seen_ids = set()
        
        for d in dirs:
            wing, hall, room = d["wing"], d["hall"], d["room"]
            
            # 从 ThreeLevelIndex 获取该目录下的 memory_id 列表
            try:
                entries = self._index.search_by_directory(wing=wing, hall=hall, room=room)
            except Exception as e:
                logger.warning("Catalog: search_by_directory failed: %s", e)
                continue
            
            if not entries:
                continue
            
            dir_memory_ids = {e["memory_id"] for e in entries}
            
            # 在全库向量搜索结果中过滤出目录内的结果
            try:
                vec_results = self._vector.search(query, top_k=top_k * 3)
            except Exception as e:
                logger.warning("Catalog: vector search failed: %s", e)
                vec_results = []
            
            for r in vec_results:
                mid = r.get("memory_id", "")
                if mid in dir_memory_ids and mid not in seen_ids:
                    r["_source"] = "catalog"
                    r["catalog_dir"] = f"{wing}/{hall}/{room}"
                    all_results.append(r)
                    seen_ids.add(mid)
            
            # 补充：目录内 BM25 搜索
            try:
                bm25_results = self._bm25.search(query, top_k=top_k * 2)
            except Exception as e:
                logger.warning("Catalog: BM25 search failed: %s", e)
                bm25_results = []
            
            for r in bm25_results:
                mid = r.get("memory_id", "")
                if mid in dir_memory_ids and mid not in seen_ids:
                    r["_source"] = "catalog_bm25"
                    r["catalog_dir"] = f"{wing}/{hall}/{room}"
                    all_results.append(r)
                    seen_ids.add(mid)
        
        # 按分数排序，截取 top_k
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]
