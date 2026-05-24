import pytest
pytest.importorskip("chromadb", reason="chromadb not installed")

import shutil
import uuid
from pathlib import Path

from omnimem.sdk import OmniMemSDK


def _make_tmpdir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix=f"omnimem_sdk_{uuid.uuid4().hex[:8]}_")


def test_sdk_memorize_and_recall():
    tmpdir = _make_tmpdir()
    try:
        sdk = OmniMemSDK(storage_dir=tmpdir)
        result = sdk.memorize("test sdk content", memory_type="fact")
        assert result["status"] in ("stored", "duplicate_skipped"), f"Unexpected: {result}"
        if result["status"] == "stored":
            assert "memory_id" in result
            mid = result["memory_id"]
        else:
            mid = result.get("existing_id", "")

        result2 = sdk.health_check()
        assert result2["status"] in ("healthy", "degraded")

        if mid:
            result3 = sdk.detail(memory_id=mid)
            assert result3.get("status") in ("found", "ok")

        result4 = sdk.detail_list()
        assert result4.get("status") in ("empty", "ok")

        sdk.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sdk_context_manager():
    tmpdir = _make_tmpdir()
    try:
        with OmniMemSDK(storage_dir=tmpdir) as sdk:
            result = sdk.memorize("context manager test", memory_type="fact")
            assert result["status"] in ("stored", "duplicate_skipped")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sdk_govern():
    tmpdir = _make_tmpdir()
    try:
        with OmniMemSDK(storage_dir=tmpdir) as sdk:
            result = sdk.govern("forgetting_status")
            assert result.get("status") == "ok"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sdk_compact():
    tmpdir = _make_tmpdir()
    try:
        with OmniMemSDK(storage_dir=tmpdir) as sdk:
            result = sdk.compact()
            assert result.get("status") == "ready"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_sdk_config_override():
    tmpdir = _make_tmpdir()
    try:
        with OmniMemSDK(storage_dir=tmpdir, config={"budget_tokens": 8000}) as sdk:
            assert sdk._provider._config.get("budget_tokens") == 8000
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
