"""PrivacyManager — 隐私分级管理。

4级隐私:
  - public: 所有人可见
  - team: 团队内可见
  - personal: 仅本人可见（默认）
  - secret: 加密存储，需认证访问

过滤规则:
  - 不同 session_id 可能对应不同用户
  - secret 级记忆在检索结果中被标记为加密，不直接过滤丢弃
  - team 级记忆在非团队 session 中被过滤

OPT-1 改进:
  - secret 级内容不再直接丢弃，而是保留并标记 _encrypted=True
  - 提供 encrypt_content / decrypt_content 接口供存储层调用
  - 依赖 MemoryEncryption 实现真正的加解密
"""

from __future__ import annotations

import logging
from typing import Any

from omnimem.governance.encryption import MemoryEncryption

logger = logging.getLogger(__name__)

# 隐私级别（从低到高）
_PRIVACY_LEVELS = ["public", "team", "personal", "secret"]
_PRIVACY_ORDER = {level: i for i, level in enumerate(_PRIVACY_LEVELS)}


class PrivacyManager:
    """隐私分级管理，支持 secret 级加密存储。"""

    def __init__(self, default_level: str = "personal", session_id: str = "",
                 kms_manager: Any = None):
        self._default_level = default_level
        self._overrides: dict[str, str] = {}
        self._store: Any | None = None  # ★ 延迟绑定存储层，用于回填
        # OPT-1: 初始化加密器，优先使用 KMS 密钥
        self._encryption = MemoryEncryption(session_seed=session_id, kms_manager=kms_manager)

    def bind_store(self, store: Any) -> None:
        """绑定存储层，用于从持久化数据回填隐私级别。"""
        self._store = store

    @property
    def encryption(self) -> MemoryEncryption:
        """返回加密器实例，供存储层调用。"""
        return self._encryption

    def set(self, memory_id: str, level: str, new_wing: str = "") -> None:
        """设置记忆的隐私级别。

        同步写入内存覆盖表 + 持久化到存储层，确保进程重启后不丢失。
        ★ R33修复Minor-4：支持 new_wing 参数，同步更新 wing。
        """
        if level not in _PRIVACY_LEVELS:
            return
        self._overrides[memory_id] = level
        # ★ 持久化到存储层
        if self._store is not None:
            try:
                self._store.update_privacy(memory_id, level, new_wing=new_wing or None)
            except Exception as e:
                logger.warning("Privacy persist failed for %s: %s", memory_id, e)

    def get(self, memory_id: str) -> str:
        """获取记忆的隐私级别。

        查找顺序：
        1. 内存覆盖表（_overrides，最近设置的）
        2. 存储层（_closet_index，持久化的）
        3. 默认值（personal）
        """
        if memory_id in self._overrides:
            return self._overrides[memory_id]
        # ★ 从存储层回填：确保进程重启后隐私级别不丢失
        if self._store is not None:
            try:
                entry = self._store.get(memory_id)
                if entry and "privacy" in entry:
                    privacy = entry["privacy"]
                    self._overrides[memory_id] = privacy  # 缓存
                    return str(privacy)
            except Exception as e:
                logger.warning("Privacy store lookback failed for %s: %s", memory_id, e)
        return self._default_level

    def encrypt_content(self, content: str) -> str:
        """OPT-1: 加密内容。如果加密不可用，返回带标记的明文。"""
        return self._encryption.encrypt(content)

    def encrypt_content_with_status(self, content: str) -> dict[str, str]:
        """OPT-1: 加密内容并返回加密状态。"""
        return self._encryption.encrypt_with_status(content)

    def decrypt_content(self, ciphertext: str) -> str:
        """OPT-1: 解密内容。处理加密不可用或解密失败的情况。"""
        return self._encryption.decrypt(ciphertext)

    def is_encrypted(self, text: str) -> bool:
        """OPT-1: 判断文本是否已被加密。"""
        return self._encryption.is_encrypted(text)

    def filter(
        self,
        results: list[dict[str, Any]],
        session_id: str = "",
        max_privacy: str = "personal",
    ) -> list[dict[str, Any]]:
        """按隐私级别过滤检索结果。

        OPT-1 改进: secret 级不再直接丢弃，而是保留并标记 _encrypted=True，
        由上层决定如何展示（如提示用户需要解锁）。

        Args:
            results: 检索结果列表
            session_id: 当前 session ID
            max_privacy: 最大允许的隐私级别

        Returns:
            过滤后的结果
        """
        max_order = _PRIVACY_ORDER.get(max_privacy, 2)
        filtered = []

        for r in results:
            privacy = r.get("privacy", self._default_level)
            privacy_order = _PRIVACY_ORDER.get(privacy, 2)

            # OPT-1: secret 级保留但标记加密，不再直接过滤
            if privacy == "secret":
                r["_encrypted"] = True
                r["content"] = "[加密记忆 — 使用 omni_detail 解锁]"
                filtered.append(r)
                continue

            # 超过最大允许级别的过滤
            if privacy_order > max_order:
                continue

            filtered.append(r)

        return filtered
