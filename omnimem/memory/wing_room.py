"""WingRoomManager — 宫殿导航：Wing(人/项目) > Hall(类型) > Room(话题)。

参考 MemPalace 的 Wing/Room/Hall 三级结构，实现记忆的空间组织。
Wing: 顶层分类（按 privacy 值直接命名：personal/team/public/secret）
Hall: 中层分类（facts/events/preferences/skills/corrections）
Room: 底层话题（根据内容自动检测或手动指定）

改进：话题检测复用 KnowledgeGraph 的 extract_entities，提升精度。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Wing 映射（scope → wing，保留向后兼容）
_SCOPE_TO_WING = {
    "personal": "personal",
    "private": "personal",
    "project": "projects",
    "shared": "shared",
    "team": "team",
    "public": "public",
    "secret": "personal",
}

# ★ R25修复BUG-1：Privacy→Wing 单层直接映射
# privacy 值直接作为 wing 名称，保持语义一致性
# public→public, team→team, personal→personal, secret→personal(私密归入个人空间)
_PRIVACY_TO_WING = {
    "personal": "personal",
    "private": "personal",
    "secret": "personal",
    "team": "team",
    "public": "public",
}

# Hall 映射
_TYPE_TO_HALL = {
    "fact": "facts",
    "preference": "preferences",
    "correction": "corrections",
    "skill": "skills",
    "procedural": "procedures",
    "event": "events",
    "action": "actions",       # Agent 行为记忆
    "reasoning": "reasoning",  # 推理链条记忆
}

# 停用词表（扩展版）
_STOPWORDS = {
    # 中文停用词
    "这是",
    "那个",
    "这个",
    "什么",
    "怎么",
    "如何",
    "可以",
    "需要",
    "我们",
    "他们",
    "它们",
    "因为",
    "所以",
    "如果",
    "但是",
    "而且",
    "就是",
    "已经",
    "还是",
    "或者",
    "不是",
    "没有",
    "可能",
    "应该",
    "一下",
    "一些",
    "一种",
    "之后",
    "之前",
    "之间",
    "关于",
    "对于",
    "通过",
    "使用",
    "进行",
    "实现",
    "包括",
    "根据",
    "目前",
    "同时",
    # 英文停用词
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "this",
    "that",
    "with",
    "from",
    "have",
    "been",
    "does",
    "will",
    "would",
    "could",
    "should",
}


class WingRoomManager:
    """宫殿导航：Wing(人/项目) > Hall(类型) > Room(话题)。"""

    def __init__(self, palace_dir: Path):
        self._palace_dir = palace_dir
        self._palace_dir.mkdir(parents=True, exist_ok=True)
        # ★ 话题检测缓存：减少高频 add 时的重复解析
        self._topic_cache: dict[str, str] = {}
        self._topic_cache_max = 1000

    def resolve_wing(self, scope: str) -> str:
        """将 scope 映射为 Wing 名称。"""
        wing = _SCOPE_TO_WING.get(scope, scope)
        if wing == scope and scope not in _SCOPE_TO_WING:
            logger.warning("Unknown scope '%s', using as-is", scope)
        return wing

    def resolve_wing_from_privacy(self, privacy: str, memory_type: str = "") -> str:
        """直接从 privacy 值映射到 wing。

        映射规则：privacy 值直接作为 wing 名称
          - personal/private → personal
          - public → public
          - team → team
          - secret → personal（私密归入个人空间）

        Args:
            privacy: 隐私级别 (personal/private/secret/team/public)
            memory_type: 记忆类型（保留参数，当前不参与映射）

        Returns:
            wing 名称
        """
        wing = _PRIVACY_TO_WING.get(privacy, "personal")
        if privacy not in _PRIVACY_TO_WING:
            logger.warning("Unknown privacy '%s', defaulting to 'personal'", privacy)
        return wing

    def resolve_hall(self, memory_type: str) -> str:
        """将 memory_type 映射为 Hall 名称。"""
        return _TYPE_TO_HALL.get(memory_type, memory_type)

    def resolve_room(self, content: str, wing: str, memory_type: str = "") -> str:
        """根据内容自动分配 Room（话题检测），带缓存。

        检测策略：
          1. 尝试使用 KnowledgeGraph 的 extract_entities
          2. 回退到增强版正则提取
          3. 最终回退到内容哈希
        """
        # ★ 缓存检查：基于内容前 200 字的 hash
        cache_key = hashlib.md5(content[:200].encode()).hexdigest()[:16]
        cached = self._topic_cache.get(cache_key)
        if cached:
            return cached

        # 缓存满时清空（简单 LRU 策略）
        if len(self._topic_cache) >= self._topic_cache_max:
            self._topic_cache.clear()

        # 尝试提取关键实体
        room = self._detect_topic(content)
        if room:
            result = self._sanitize_name(room)
            self._topic_cache[cache_key] = result
            return result

        # 回退：使用类型 + 哈希
        if memory_type:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
            result = f"{memory_type}-{content_hash}"
            self._topic_cache[cache_key] = result
            return result

        # 最终回退
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        result = f"room-{content_hash}"
        self._topic_cache[cache_key] = result
        return result

    def get_room_path(self, wing: str, hall: str, room: str) -> Path:
        """获取 Room 的完整路径。"""
        path = self._palace_dir / wing / hall / room
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_wings(self) -> list[str]:
        """列出所有 Wing。"""
        if not self._palace_dir.exists():
            return []
        return [
            d.name for d in self._palace_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
        ]

    def list_halls(self, wing: str) -> list[str]:
        """列出指定 Wing 下的所有 Hall。"""
        wing_path = self._palace_dir / wing
        if not wing_path.exists():
            return []
        return [d.name for d in wing_path.iterdir() if d.is_dir() and not d.name.startswith("_")]

    def list_rooms(self, wing: str, hall: str) -> list[str]:
        """列出指定 Hall 下的所有 Room。"""
        hall_path = self._palace_dir / wing / hall
        if not hall_path.exists():
            return []
        return [d.name for d in hall_path.iterdir() if d.is_dir() and not d.name.startswith("_")]

    def tree(self, wing: str = "", hall: str = "") -> dict[str, Any]:
        """展示目录树 — 内化 OpenViking tree()。

        Returns:
            嵌套字典表示的目录树：
            {
                "personal": {
                    "facts": ["docker", "python", "猫咪"],
                    "preferences": ["称呼", "饮食"],
                },
                "team": {
                    "skills": ["部署"],
                },
            }
        """
        result: dict[str, Any] = {}
        wings = [wing] if wing else self.list_wings()
        for w in wings:
            halls = [hall] if hall else self.list_halls(w)
            result[w] = {}
            for h in halls:
                rooms = self.list_rooms(w, h)
                result[w][h] = rooms
        return result

    def grep_rooms(self, pattern: str) -> list[dict[str, str]]:
        """搜索 Room 名称 — 内化 OpenViking grep()。

        在所有 Wing/Hall 下搜索匹配 pattern 的 Room 名称。

        Args:
            pattern: 搜索模式（子串匹配，不区分大小写）

        Returns:
            [{"wing": str, "hall": str, "room": str}, ...]
        """
        pattern_lower = pattern.lower()
        results = []
        for wing in self.list_wings():
            for hall in self.list_halls(wing):
                for room in self.list_rooms(wing, hall):
                    if pattern_lower in room.lower():
                        results.append({"wing": wing, "hall": hall, "room": room})
        return results

    def count_memories(self, wing: str = "", hall: str = "", room: str = "") -> dict[str, int]:
        """统计各目录下的记忆数量。"""
        result = {}
        wings = [wing] if wing else self.list_wings()
        for w in wings:
            halls = [hall] if hall else self.list_halls(w)
            for h in halls:
                rooms = [room] if room else self.list_rooms(w, h)
                for r in rooms:
                    path = self.get_room_path(w, h, r)
                    drawer_path = path / "drawer"
                    if drawer_path.exists():
                        count = len(list(drawer_path.glob("*.md")))
                        key = f"{w}/{h}/{r}"
                        result[key] = count
        return result

    def _detect_topic(self, content: str) -> str | None:
        """从内容中检测话题。

        策略优先级：
          1. KnowledgeGraph.extract_entities（最精确）
          2. 英文小写技术关键词（react, docker, typescript 等）
          3. CamelCase / ALLCAPS 模式
          4. 中文名词短语（最宽泛，放最后）
        """
        # 策略 1: 尝试使用 KG 的 extract_entities
        try:
            from omnimem.deep.kg import extract_entities

            entities = extract_entities(content)
            # 过滤停用词和太短的实体
            valid = [e for e in entities if e.lower() not in _STOPWORDS and len(e) >= 2]
            if valid:
                # 优先纯英文实体（技术术语更精确）；同优先级时按在原文中出现顺序选择
                def _entity_priority(e: str) -> tuple[int, int, int]:
                    is_pure_en = bool(re.match(r"^[A-Za-z0-9_.-]+$", e))
                    is_mixed = bool(re.search(r"[A-Za-z]", e) and re.search(r"[\u4e00-\u9fff]", e))
                    # 位置越靠前优先级越高（数值越小）
                    pos = content.find(e)
                    if pos < 0:
                        pos = len(content)
                    return (0 if is_pure_en else (1 if not is_mixed else 2), pos, len(e))

                valid.sort(key=_entity_priority)
                best = valid[0]
                # 纯英文实体统一小写，便于路径规范
                if re.match(r"^[A-Za-z0-9_.-]+$", best):
                    return str(best.lower())
                return str(best)
        except ImportError:
            pass

        # 策略 2: 英文小写技术关键词（优先级高于中文正则）
        # 不用 \b，因为中英混合时 \b 在英文和中文之间无效
        # 用前后断言防止子串匹配（如 "pythonic" 不应匹配 "python"）
        tech_pattern = r"(?<![a-z])(python|java|go|rust|typescript|react|vue|docker|k8s|redis|mysql|postgresql|mongodb|neo4j|chromadb|sqlite|api|sql|rest|graphql|kubernetes|nginx|flask|django|fastapi|tensorflow|pytorch)(?![a-z])"
        tech_matches = re.findall(tech_pattern, content[:300].lower())
        if tech_matches:
            return str(tech_matches[0])

        # 策略 3: CamelCase / ALLCAPS 模式
        en_pattern = r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[A-Z]{2,}\b"
        en_matches = re.findall(en_pattern, content[:300])
        if en_matches:
            for m in en_matches:
                if len(m) >= 3:
                    return str(m.lower())

        # 策略 4: 中文名词短语（最宽泛，放最后）
        # ★ 优先从内容前 50 字提取，避免从长内容中间提取随机词（如"强调"）
        zh_pattern = r"[\u4e00-\u9fff]{2,6}"
        # 先尝试前 50 字
        zh_matches_head = re.findall(zh_pattern, content[:50])
        for m in zh_matches_head:
            if m.lower() not in _STOPWORDS and len(m) >= 2:
                return str(m)
        # 回退到前 200 字
        zh_matches = re.findall(zh_pattern, content[:200])
        for m in zh_matches:
            if m.lower() not in _STOPWORDS and len(m) >= 2:
                return str(m)

        return None

    @staticmethod
    def _sanitize_name(name: str, max_len: int = 20) -> str:
        """清理名称，使其可安全用于文件路径。最长 20 字符。"""
        sanitized = re.sub(r"[^\w\u4e00-\u9fff-]", "-", name)
        sanitized = sanitized.strip("-")
        if not sanitized:
            return "unnamed"
        # 优先截取首个有意义的片段
        if len(sanitized) > max_len:
            # 尝试在标点或空格处截断
            cut = sanitized[:max_len]
            last_dash = max(cut.rfind("-"), cut.rfind("_"))
            if last_dash > max_len // 2:
                cut = cut[:last_dash]
            return cut
        return sanitized
