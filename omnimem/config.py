"""OmniMem 配置 — 重导出到 config/ 包。

THIS FILE IS KEPT FOR BACKWARD COMPATIBILITY ONLY.
权威配置在 config/_config.py 的 _CONFIG_SCHEMA 中。
新增配置项请修改 config/_config.py，不要修改此文件。
"""
from omnimem.config._config import (  # noqa: F401
    _CONFIG_SCHEMA,
    DEFAULTS,
    OmniMemConfig,
)
