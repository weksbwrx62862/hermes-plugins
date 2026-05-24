"""Encryption utilities for secret-level memory content.

Lightweight encryption using Fernet (AES-128-CBC + HMAC) from cryptography.
Key is derived from a session-specific seed. Falls back to plaintext marking
if cryptography is unavailable.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Prefix markers for fallback mode (no cryptography)
_UNENCRYPTED_PREFIX = "[UNENCRYPTED]"
_DECRYPTION_FAILED = "[DECRYPTION_FAILED]"


class MemoryEncryption:
    """Encrypt/decrypt secret-level memory content.

    Features:
      - Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256)
      - Key derived from session seed via PBKDF2
      - Graceful fallback if cryptography not installed
      - Thread-safe (Fernet instances are stateless)
    """

    def __init__(self, session_seed: str = "", kms_manager: Any = None):
        """Initialize with a session seed for deterministic key derivation.

        Args:
            session_seed: A stable per-session string (e.g., session_id).
                          If empty, falls back to OMNIMEM_ENCRYPTION_KEY env var
                          or generates a per-process random key (non-recoverable).
            kms_manager: Optional KMSManager instance for key management.
                         If provided, uses kms.get_encryption_key() for key retrieval.
        """
        self._kms = kms_manager
        if kms_manager is not None:
            self._key = kms_manager.get_encryption_key()
        else:
            if not session_seed:
                session_seed = os.environ.get("OMNIMEM_ENCRYPTION_KEY", "")
            if not session_seed:
                session_seed = os.urandom(32).hex()
                logger.warning(
                    "No session_seed or OMNIMEM_ENCRYPTION_KEY provided — "
                    "using per-process random key. Encrypted data from previous "
                    "sessions will not be decryptable after restart. Set "
                    "OMNIMEM_ENCRYPTION_KEY for persistent encryption."
                )
            self._key = self._derive_key(session_seed)
        self._fernet: Any | None = None
        self._available: bool | None = None

    @staticmethod
    def _derive_key(seed: str) -> bytes:
        """Derive a Fernet-compatible 32-byte key from seed via PBKDF2."""
        salt = hashlib.sha256(seed.encode("utf-8")).digest()[:16]
        key = hashlib.pbkdf2_hmac(
            "sha256", seed.encode("utf-8"), salt, iterations=100_000, dklen=32
        )
        return base64.urlsafe_b64encode(key)

    def is_available(self) -> bool:
        """Check if encryption is actually available (cryptography installed)."""
        if self._available is not None:
            return self._available
        try:
            from cryptography.fernet import Fernet  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _get_fernet(self) -> Any | None:
        """Lazy-init Fernet instance."""
        if self._fernet is not None:
            return self._fernet
        try:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(self._key)
        except ImportError:
            logger.warning("cryptography not installed — secret encryption unavailable")
            self._fernet = None
        return self._fernet

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext. Returns base64-encoded ciphertext.

        Falls back to ``[UNENCRYPTED]<plaintext>`` if cryptography
        is not installed, so data is never silently lost.
        """
        if not plaintext:
            return plaintext
        f = self._get_fernet()
        if f is None:
            return f"{_UNENCRYPTED_PREFIX}{plaintext}"
        try:
            token = f.encrypt(plaintext.encode("utf-8"))
            return str(token.decode("utf-8"))
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            return f"{_UNENCRYPTED_PREFIX}{plaintext}"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext. Handles fallback markers gracefully."""
        if not ciphertext:
            return ciphertext
        if ciphertext.startswith(_UNENCRYPTED_PREFIX):
            return ciphertext[len(_UNENCRYPTED_PREFIX) :]
        f = self._get_fernet()
        if f is None:
            logger.error("Cannot decrypt: cryptography not available")
            return _DECRYPTION_FAILED
        try:
            return str(f.decrypt(ciphertext.encode("utf-8")).decode("utf-8"))
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            return _DECRYPTION_FAILED

    def is_encrypted(self, text: str) -> bool:
        """Heuristic: check if text appears to be encrypted by this class."""
        if not text:
            return False
        if text.startswith(_UNENCRYPTED_PREFIX):
            return True
        # Fernet tokens start with "gAAAA" (base64 of version byte 0x80)
        return bool(text.startswith("gAAAA"))
