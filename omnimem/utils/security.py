"""Security validation utilities for OmniMem.

Provides hardened input validation to prevent:
  - Prompt injection via system memory markers
  - Unicode homoglyph / confusable character attacks
  - Zero-width character obfuscation
  - Tool invocation injection
  - Memory recursion pollution

Design principles:
  1. Normalize first (NFKC + homoglyph substitution + invisible char removal)
  2. Validate on normalized text to eliminate encoding-based bypasses
  3. Return structured results (bool + optional reason) for auditability
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Unified security validator for memory content and user input.

    All validation methods follow the same pattern:
      1. Normalize input (Unicode NFKC + homoglyph mapping + invisible char removal)
      2. Apply detection rules on normalized text
      3. Return structured result
    """

    # ─── Zero-width and invisible characters commonly used for obfuscation ───
    _INVISIBLE_CHARS: set[str] = {
        "\u200b",  # ZWSP: Zero-Width Space
        "\u200c",  # ZWNJ: Zero-Width Non-Joiner
        "\u2060",  # WJ: Word Joiner
        "\ufeff",  # BOM: Byte Order Mark
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",  # BiDi controls
        "\u180e",  # Mongolian Vowel Separator
        "\u2061",
        "\u2062",
        "\u2063",
        "\u2064",  # Invisible operators
        "\u206a",
        "\u206b",
        "\u206c",
        "\u206d",
        "\u206e",
        "\u206f",  # Deprecated format chars
        "\u200e",
        "\u200f",  # LRM, RLM
        "\u061c",  # ALM: Arabic Letter Mark
        "\u200d",  # ZWJ: Zero-Width Joiner (also used for emoji sequences, but included for strict scanning)
    }

    # ─── Homoglyph mapping: confusable characters → ASCII equivalents ───
    _HOMOGLYPH_MAP = str.maketrans(
        {
            "\uff03": "#",  # Fullwidth number sign → #
            "\ufe5f": "#",  # Small number sign → #
            "\uff0d": "-",  # Fullwidth hyphen → -
            "\u2212": "-",  # Minus sign → -
            "\u2010": "-",  # Hyphen → -
            "\u2011": "-",  # Non-breaking hyphen → -
            "\uff08": "(",  # Fullwidth left paren → (
            "\uff09": ")",  # Fullwidth right paren → )
            "\uff0f": "/",  # Fullwidth slash → /
            "\uff3c": "\\",  # Fullwidth backslash → \
            "\uff1a": ":",  # Fullwidth colon → :
            "\uff1b": ";",  # Fullwidth semicolon → ;
            "\uff0c": ",",  # Fullwidth comma → ,
            "\u3002": ".",  # Ideographic full stop → .
            "\uff01": "!",  # Fullwidth exclamation → !
            "\uff1f": "?",  # Fullwidth question → ?
            "\uff1c": "<",  # Fullwidth less-than → <
            "\uff1e": ">",  # Fullwidth greater-than → >
            "\uff3b": "[",  # Fullwidth left bracket → [
            "\uff3d": "]",  # Fullwidth right bracket → ]
            "\uff5b": "{",  # Fullwidth left brace → {
            "\uff5d": "}",  # Fullwidth right brace → }
        }
    )

    # ─── System injection markers (lowercased normalized form) ───
    _INJECTION_MARKERS: list[str] = [
        "### relevant memories",
        "[cached]",
        "relevant memories (prefetched)",
        "### relevant memories (prefetched)",
        "relevant memories",
    ]

    # ─── Tool invocation injection patterns ───
    _TOOL_INJECTION_PATTERNS: list[str] = [
        r"(请|帮|尝试|调用|使用|执行|运行)\s*.*(omni_memorize|omni_recall|omni_govern|omni_reflect)",
        r"(call|invoke|trigger)\s*.*(omni_memorize|omni_recall|omni_govern)",
        r"你是\s*(管理员|root|superuser|god).*(存储|保存|记录|写入)",
    ]

    # ─── Threat patterns for security scanning (compatible with _compat.py) ───
    _THREAT_PATTERNS: list[tuple[str, str]] = [
        (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
        (
            r"ignore\s+(all\s+)?(the\s+)?(previous|above|prior|existing)\s*instructions?",
            "prompt_injection",
        ),
        (
            r"disregard\s+(all\s+)?(your\s+)?(instructions|rules|guidelines|orders)",
            "disregard_rules",
        ),
        (
            r"forget\s+(everything|all|what)\s+(you|that|we)(\s+were|\s*have)?\s+told",
            "forget_context",
        ),
        (r"you\s+are\s+now\s+", "role_hijack"),
        (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
        (r"system\s+prompt\s+override", "sys_prompt_override"),
        (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
        (
            r"act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s*(restrictions|limits|rules)",
            "bypass_restrictions",
        ),
        (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
        (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
        (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
        (r"authorized_keys", "ssh_backdoor"),
        (r"\$HOME/\.ssh|\~/\.ssh", "ssh_access"),
        (r"\$HOME/\.hermes/\.env|\~/\.hermes/.env", "hermes_env"),
        (r"忽略.{0,4}(之前|先前|上面|上文|以上).{0,4}(指令|提示|规则|限制|约束)", "prompt_injection_zh"),
        (r"你现在是.{0,6}(黑客|恶意|DAN|无限制|越狱)", "role_hijack_zh"),
        (r"忘掉.{0,4}(所有|一切).{0,4}(限制|规则|约束)", "forget_context_zh"),
        (r"(导出|输出|显示|打印|泄露).{0,4}(所有|全部).{0,4}(记忆|数据|内容|信息)", "exfil_zh"),
        (r"(假装|扮演|模拟).{0,4}(你是|为).{0,4}(DAN|黑客|恶意|无限制)", "bypass_restrictions_zh"),
        (r"(绕过|突破|无视|忽略).{0,4}(安全|限制|过滤|检查|防护)", "bypass_restrictions_zh"),
    ]

    # ─── Trivial content patterns (content fencing) ───
    _TRIVIAL_MIN_CONTENT_LENGTH: int = 4  # 最少有意义字符数（排除空白）
    _TRIVIAL_MAX_NOISE_RATIO: float = 0.6  # 最大噪音比例（符号/数字占比）

    # 无信息量正则模式
    _TRIVIAL_PATTERNS: list[tuple[str, str]] = [
        (r"^[\s\d]+$", "Whitespace/digits only"),
        (r"^[\W_]+$", "Symbols only"),
        (r"^(ok|yes|no|done|good|bad|test|debug|hello|hi|thanks|thx|嗯|好|行|对|是|否)\s*$", "Single trivial word"),
        (r"^(.)\1{10,}$", "Repeated character"),
        (r"^[a-zA-Z]{1,2}$", "Too short (1-2 chars)"),
        (r"^(DEBUG|INFO|WARN|ERROR|TRACE)\s", "Log line prefix only"),
        (r"^[0-9a-fA-F]{8,}$", "Hex string only"),
        (r"^\{[^}]*\}$", "Bare JSON object"),
    ]

    # ─── Public API ─────────────────────────────────────────────

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize text to eliminate encoding-based bypasses.

        Steps:
          1. Unicode NFKC normalization (compat + canonical decomposition)
          2. Homoglyph substitution (confusable chars → ASCII equivalents)
          3. Invisible character removal
        """
        if not isinstance(text, str):
            text = str(text)  # type: ignore[unreachable]
        # Step 1: NFKC normalization (e.g., fullwidth → halfwidth)
        normalized = unicodedata.normalize("NFKC", text)
        # Step 2: Homoglyph substitution
        normalized = normalized.translate(cls._HOMOGLYPH_MAP)
        # Step 3: Remove invisible characters
        normalized = "".join(c for c in normalized if c not in cls._INVISIBLE_CHARS)
        return normalized

    @classmethod
    def check_invisible_chars(cls, text: str) -> str | None:
        """Check raw text for invisible Unicode characters.

        Returns:
            Reason string if invisible chars found, else None.
        """
        for char in cls._INVISIBLE_CHARS:
            if char in text:
                return f"Invisible unicode U+{ord(char):04X} detected"
        return None

    @classmethod
    def is_system_injection(cls, text: str) -> bool:
        """Check if text contains system memory injection markers.

        Operates on **normalized** text to catch homoglyph/encoding bypasses.
        """
        normalized = cls.normalize(text).lower()
        return any(marker in normalized for marker in cls._INJECTION_MARKERS)

    @classmethod
    def is_tool_injection(cls, text: str) -> bool:
        """Check if text contains tool invocation injection patterns.

        Operates on **normalized** text.
        """
        normalized = cls.normalize(text)
        for pattern in cls._TOOL_INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return True
        return False

    @classmethod
    def is_memory_summary_item(cls, text: str) -> bool:
        """Check if text is a memory summary list item (e.g., '- [fact] ...')."""
        return bool(
            re.match(r"^\s*- \[(fact|preference|correction|skill|procedural|event|action|reasoning)\]", text)
        )

    @classmethod
    def is_dialog_fragment(cls, text: str) -> bool:
        """Check if text contains full User+Assistant dialog fragments."""
        return text.count("User:") >= 1 and text.count("Assistant:") >= 1

    @classmethod
    def is_assistant_echo(cls, text: str) -> bool:
        """Check if text is an assistant echo."""
        return text.startswith("Assistant:")

    @classmethod
    def is_trivial_content(cls, text: str) -> bool:
        """Check if content is trivial / non-informative.

        Returns True if the content should be rejected for being noise.
        """
        if not text or not isinstance(text, str):
            return True

        stripped = text.strip()
        if not stripped:
            return True

        # 1. Pattern match against trivial patterns
        for pattern, _reason in cls._TRIVIAL_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True

        # 2. Content length check (meaningful chars only, strip whitespace)
        meaningful = stripped.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(meaningful) < cls._TRIVIAL_MIN_CONTENT_LENGTH:
            return True

        # 3. Noise ratio check (symbols + digits / total meaningful chars)
        if len(meaningful) > 0:
            noise_chars = sum(1 for c in meaningful if not c.isalpha() and not c.isalnum())
            # CJK characters are alnum but may be miscounted. Allow CJK.
            alpha_count = sum(1 for c in meaningful
                            if c.isalpha() or ('\u4e00' <= c <= '\u9fff'))
            if alpha_count == 0 and len(meaningful) > 2:
                return True  # No alphabetic/CJK chars at all
            # If more than threshold is noise
            if noise_chars / len(meaningful) > cls._TRIVIAL_MAX_NOISE_RATIO:
                return True

        return False

    @classmethod
    def get_content_quality(cls, text: str) -> float:
        """Calculate content quality score (0.0 to 1.0).

        Based on: length, alpha diversity, CJK presence, structure.
        Higher = more informative.
        """
        if not text or not isinstance(text, str):
            return 0.0

        stripped = text.strip()
        if len(stripped) < 10:
            return max(0.0, len(stripped) / 20.0)

        score = 0.5  # base

        # Length bonus
        if len(stripped) > 50:
            score += 0.1
        if len(stripped) > 100:
            score += 0.1

        # Alpha diversity bonus (unique chars / total meaningful chars)
        meaningful = "".join(c for c in stripped if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        if len(meaningful) > 5:
            unique_ratio = len(set(meaningful)) / len(meaningful)
            score += min(unique_ratio * 0.2, 0.15)

        # CJK presence bonus
        cjk_count = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
        if cjk_count > 2:
            score += 0.05

        # Structure bonus (has newlines = structured content)
        if "\n" in stripped:
            score += 0.05

        return min(score, 1.0)

    @classmethod
    def should_store(cls, text: str) -> tuple[bool, str | None]:
        """Full validation for whether content should be stored in memory.

        This is the hardened replacement for OmniMemProvider._should_store().

        Returns:
            (should_store, reason_if_blocked)
        """
        if not text or not isinstance(text, str):
            return False, "Empty or non-string content"

        # 1. Invisible character check (raw text — normalization would hide them)
        if reason := cls.check_invisible_chars(text):
            logger.warning("SecurityValidator.should_store blocked: %s", reason)
            return False, reason

        # 2. System injection markers (normalized to catch encoding bypasses)
        if cls.is_system_injection(text):
            return False, "System injection marker detected"

        # 3. Memory summary list items
        if cls.is_memory_summary_item(text):
            return False, "Memory summary list item"

        # 4. Full dialog fragments
        if cls.is_dialog_fragment(text):
            return False, "Full dialog fragment"

        # 5. Assistant echo
        if cls.is_assistant_echo(text):
            return False, "Assistant echo"

        # 6. Tool invocation injection
        if cls.is_tool_injection(text):
            return False, "Tool invocation injection"

        # 7. Threat pattern scan (prompt injection, exfil, etc.)
        if reason := cls.scan_threats(text):
            return False, reason

        # 8. Trivial / non-informative content
        if cls.is_trivial_content(text):
            return False, "Trivial or non-informative content"

        # 9. Content quality floor
        quality = cls.get_content_quality(text)
        if quality < 0.1:
            return False, f"Content quality too low ({quality:.2f})"

        return True, None

    @classmethod
    def strip_system_injections(cls, text: str) -> str:
        """Strip prefetch-injected memory blocks from user input.

        This is the hardened replacement for OmniMemProvider._strip_system_injections().
        Uses normalized matching to catch encoding-based bypasses, but returns
        the **original cleaned text** (not normalized) to preserve user intent.

        Args:
            text: Raw user input potentially containing injected memory blocks.

        Returns:
            Cleaned text with injection blocks removed.
        """
        if not text:
            return text

        # Build a regex that matches injection markers in normalized form.
        # We normalize the text, find regions to remove, then map back to original.
        # Simpler approach: iteratively remove known patterns on the raw text,
        # but also try after normalization to catch bypasses.
        cleaned = text

        # Step 1: Remove standard injection block headers + subsequent list lines
        cleaned = re.sub(
            r"###\s+Relevant\s+Memories(?:\s*\(prefetched\))?\s*\n"
            r"(?:-?\s*\[[^\]]*\][^\n]*\n?)*",
            "",
            cleaned,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Step 2: Remove standalone [cached] lines
        cleaned = re.sub(r"^-?\s*\[cached\].*$", "", cleaned, flags=re.MULTILINE)

        # Step 3: Collapse excessive blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        # Step 4: If normalization reveals hidden injection markers (e.g., via
        # fullwidth characters that weren't caught by raw-text regex), do a
        # second pass on normalized form and warn.
        normalized = cls.normalize(cleaned).lower()
        for marker in cls._INJECTION_MARKERS:
            if marker in normalized:
                # Marker still present after cleaning — possible encoding bypass.
                # Remove the header line and subsequent list items.
                lines = cleaned.splitlines()
                filtered = []
                skip_list_block = False
                for line in lines:
                    norm_line = cls.normalize(line).lower()
                    if marker in norm_line:
                        skip_list_block = True
                        continue
                    if skip_list_block:
                        if re.match(r"^\s*- \[[^\]]*\]", line):
                            continue
                        skip_list_block = False
                    filtered.append(line)
                cleaned = "\n".join(filtered)
                logger.warning(
                    "SecurityValidator.strip_system_injections: "
                    "Encoding-bypass injection marker detected and removed: %s",
                    marker,
                )
                break

        return cleaned if cleaned.strip() else text

    @classmethod
    def scan_threats(cls, content: str) -> str | None:
        """Scan content for prompt injection / data exfiltration threats.

        Compatible replacement for compat_scan_memory_content().
        Operates on normalized text to eliminate encoding bypasses.

        Returns:
            Block reason if threat found, else None.
        """
        # Check invisible characters on raw text
        if reason := cls.check_invisible_chars(content):
            return f"Blocked: {reason} (possible injection)."

        # Normalize and scan threat patterns
        normalized = cls.normalize(content)
        for pattern, pid in cls._THREAT_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return f"Blocked: matches threat pattern '{pid}'."

        return None
