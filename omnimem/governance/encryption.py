"""Encryption utilities for secret-level memory content.

使用 Fernet（AES-128-CBC + HMAC）进行加密。当 cryptography 不可用或未配置密钥时，
直接抛出 EncryptionUnavailableError，不再降级为明文标记。
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 保留旧标记仅用于兼容已持久化的历史数据
_UNENCRYPTED_PREFIX = "[UNENCRYPTED]"
_DECRYPTION_FAILED = "[DECRYPTION_FAILED]"
# 带随机盐的新格式密文前缀
_V1_PREFIX = "OMNI_ENC_V1:"


class EncryptionUnavailableError(RuntimeError):
    """加密不可用时抛出的明确错误。"""

    def __init__(self, reason: str = ""):
        msg = "加密不可用：secret 级记忆无法存储"
        if reason:
            msg = f"{msg}（{reason}）"
        super().__init__(msg)


class MemoryEncryption:
    """Encrypt/decrypt secret-level memory content.

    Features:
      - Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256)
      - Key derived from session seed via PBKDF2
      - 加密不可用时抛出 EncryptionUnavailableError，不再降级存储
      - Thread-safe (Fernet instances are stateless)
    """

    def __init__(
        self,
        session_seed: str = "",
        kms_manager: Any = None,
        master_key: bytes | None = None,
    ):
        """Initialize with a session seed, master key, or KMS manager.

        Args:
            session_seed: A stable per-session string (e.g., session_id).
                          If empty, falls back to OMNIMEM_ENCRYPTION_KEY env var.
            kms_manager: Optional KMSManager instance for key management.
                         If provided, uses kms.get_encryption_key() for key retrieval.
            master_key: Optional 32-byte raw key or 44-byte base64-encoded Fernet key.
                        Takes precedence over session_seed but after kms_manager.
        """
        self._kms = kms_manager
        self._master_key: bytes | None = None
        self._seed: str = ""
        self._encryption_enabled = True
        if kms_manager is not None:
            self._key = kms_manager.get_encryption_key()
        elif master_key is not None:
            self._key = self._normalize_master_key(master_key)
            self._master_key = self._key
        else:
            if not session_seed:
                session_seed = os.environ.get("OMNIMEM_ENCRYPTION_KEY", "")
            if not session_seed:
                self._encryption_enabled = False
                self._key = self._derive_key("disabled")
                logger.warning(
                    "OmniMem encryption DISABLED: no session_seed or OMNIMEM_ENCRYPTION_KEY. "
                    "Secret-level content will be rejected."
                )
            else:
                self._seed = session_seed
                self._key = self._derive_key(session_seed)
        self._fernet: Any | None = None
        self._available: bool | None = None

    @staticmethod
    def _derive_key(seed: str, salt: bytes | None = None) -> bytes:
        """Derive a Fernet-compatible key from seed via PBKDF2.

        当 salt 为 None 时使用与 seed 强相关的兼容盐，保证已有 seed 配置仍可
        解密历史数据；传入随机盐可实现每次加密使用独立密钥。
        """
        if salt is None:
            salt = hashlib.sha256(seed.encode("utf-8")).digest()[:16]
        key = hashlib.pbkdf2_hmac(
            "sha256", seed.encode("utf-8"), salt, iterations=100_000, dklen=32
        )
        return base64.urlsafe_b64encode(key)

    @staticmethod
    def _normalize_master_key(master_key: bytes) -> bytes:
        """Normalize a 32-byte raw key or an already base64-encoded Fernet key."""
        if len(master_key) == 32:
            return base64.urlsafe_b64encode(master_key)
        if len(master_key) in (43, 44):
            return master_key
        raise ValueError(
            f"master_key 长度必须为 32 字节原始密钥或 44 字节 base64 编码密钥，"
            f"当前为 {len(master_key)} 字节"
        )

    def is_available(self) -> bool:
        """Check if encryption is actually available (cryptography installed + key configured)."""
        if not self._encryption_enabled:
            return False
        if self._available is not None:
            return self._available
        try:
            from cryptography.fernet import Fernet  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _make_fernet(self, key: bytes) -> Any | None:
        """Create a Fernet instance from the given key."""
        try:
            from cryptography.fernet import Fernet

            return Fernet(key)
        except ImportError:
            logger.warning("cryptography not installed — secret encryption unavailable")
            return None

    def _get_fernet(self) -> Any:
        """Lazy-init Fernet instance."""
        if self._fernet is not None:
            return self._fernet
        self._fernet = self._make_fernet(self._key)
        return self._fernet

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext. Returns base64-encoded ciphertext.

        Raises:
            EncryptionUnavailableError: 当 cryptography 未安装或未配置密钥时抛出。
        """
        return self.encrypt_with_status(plaintext)["content"]

    def encrypt_with_status(self, plaintext: str) -> dict[str, str]:
        """加密明文并同时返回加密状态。

        Raises:
            EncryptionUnavailableError: 当加密不可用时抛出。

        Returns:
            {"content": <密文>, "encryption_status": "enabled"}
        """
        if not plaintext:
            # 空内容无需加密，状态取决于加密功能是否可用
            if not self.is_available():
                raise EncryptionUnavailableError(self._disabled_reason())
            return {
                "content": plaintext,
                "encryption_status": "enabled",
            }

        if not self.is_available():
            raise EncryptionUnavailableError(self._disabled_reason())

        f = self._get_fernet()
        if f is None:
            raise EncryptionUnavailableError("cryptography not installed")

        try:
            # 使用 seed 模式时采用随机盐并随密文存储，增强密钥派生安全性
            if self._seed and self._kms is None and self._master_key is None:
                salt = os.urandom(16)
                key = self._derive_key(self._seed, salt)
                token = self._make_fernet(key).encrypt(plaintext.encode("utf-8"))
                salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8")
                content = f"{_V1_PREFIX}{salt_b64}:{token.decode('utf-8')}"
            else:
                token = f.encrypt(plaintext.encode("utf-8"))
                content = token.decode("utf-8")
            return {
                "content": content,
                "encryption_status": "enabled",
            }
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            raise EncryptionUnavailableError(f"encryption failed: {e}") from e

    def _disabled_reason(self) -> str:
        """返回加密不可用的原因，用于错误信息。"""
        if not self._encryption_enabled:
            return "no session_seed or OMNIMEM_ENCRYPTION_KEY configured"
        return "cryptography not installed"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext. Handles legacy fallback markers gracefully."""
        if not ciphertext:
            return ciphertext
        if ciphertext.startswith(_UNENCRYPTED_PREFIX):
            return ciphertext[len(_UNENCRYPTED_PREFIX) :]
        f = self._get_fernet()
        if f is None:
            logger.error("Cannot decrypt: cryptography not available")
            return _DECRYPTION_FAILED
        try:
            if ciphertext.startswith(_V1_PREFIX):
                # 新格式：盐随密文存储，使用相同 seed 重新派生密钥
                payload = ciphertext[len(_V1_PREFIX) :]
                salt_b64, token = payload.split(":", 1)
                salt = base64.urlsafe_b64decode(salt_b64)
                if self._seed:
                    key = self._derive_key(self._seed, salt)
                else:
                    key = self._key
                f = self._make_fernet(key)
                if f is None:
                    return _DECRYPTION_FAILED
                return f.decrypt(token.encode("utf-8")).decode("utf-8")
            return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            return _DECRYPTION_FAILED

    def is_encrypted(self, text: str) -> bool:
        """Heuristic: check if text appears to be encrypted by this class."""
        if not text:
            return False
        if text.startswith(_UNENCRYPTED_PREFIX):
            return True
        if text.startswith(_V1_PREFIX):
            return True
        # Fernet tokens start with "gAAAA" (base64 of version byte 0x80)
        return bool(text.startswith("gAAAA"))
