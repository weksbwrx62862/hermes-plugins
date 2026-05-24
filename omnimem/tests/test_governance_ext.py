"""AuditLogger / KMSManager / RBACManager 单元测试。

覆盖: audit (log/query/time filter/close), kms (local/aws/azure/gcp/rotate/config),
       rbac (check_permission/assign/revoke/add_role/get_user_permissions)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from omnimem.governance.audit_log import AuditLogger
from omnimem.governance.kms import KMSManager
from omnimem.governance.rbac import RBACManager


# ──────────────────────────────────────────────
# AuditLogger
# ──────────────────────────────────────────────

class TestAuditLogger(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.gov_dir = Path(self.tmpdir) / "governance"
        self.gov_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLogger(self.gov_dir)

    def tearDown(self) -> None:
        self.audit.close()

    def test_table_created(self) -> None:
        db_path = self.gov_dir / "audit_log.db"
        self.assertTrue(db_path.exists())

    def test_log_and_query(self) -> None:
        self.audit.log("write", memory_id="m1", details={"action": "create"})
        self.audit.log("read", memory_id="m2", result="denied")
        results = self.audit.query(limit=10)
        self.assertEqual(len(results), 2)

    def test_query_by_operation(self) -> None:
        self.audit.log("write", memory_id="m1")
        self.audit.log("read", memory_id="m2")
        results = self.audit.query(operation="write")
        self.assertEqual(len(results), 1)

    def test_query_by_memory_id(self) -> None:
        self.audit.log("write", memory_id="target")
        self.audit.log("write", memory_id="other")
        results = self.audit.query(memory_id="target")
        self.assertEqual(len(results), 1)

    def test_query_time_filter(self) -> None:
        import time
        before = time.time()
        self.audit.log("write", memory_id="m1")
        after = time.time()

        # from_time filter
        results = self.audit.query(from_time=before, limit=10)
        self.assertGreaterEqual(len(results), 1)

        # to_time filter (far past)
        results = self.audit.query(to_time=before - 100, limit=10)
        self.assertEqual(len(results), 0)

    def test_query_returns_structured(self) -> None:
        self.audit.log("govern", memory_id="m1", details={"level": "team"})
        results = self.audit.query(limit=1)
        row = results[0]
        self.assertIn("id", row)
        self.assertIn("timestamp", row)
        self.assertIn("operation", row)
        self.assertIn("details", row)
        self.assertEqual(row["operation"], "govern")
        self.assertEqual(row["details"], {"level": "team"})

    def test_log_no_details(self) -> None:
        self.audit.log("compact")
        results = self.audit.query(limit=1)
        self.assertIsNone(results[0]["details"])


# ──────────────────────────────────────────────
# KMSManager
# ──────────────────────────────────────────────

class TestKMSManager(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.gov_dir = Path(self.tmpdir) / "governance"
        self.gov_dir.mkdir(parents=True, exist_ok=True)

    def test_local_key_generation(self) -> None:
        kms = KMSManager(self.gov_dir)
        key = kms.get_encryption_key("default")
        self.assertIsInstance(key, bytes)
        self.assertGreater(len(key), 10)

    def test_local_key_persistence(self) -> None:
        kms = KMSManager(self.gov_dir)
        key1 = kms.get_encryption_key("test-key")
        # Re-instantiate: should load same key
        kms2 = KMSManager(self.gov_dir)
        key2 = kms2.get_encryption_key("test-key")
        self.assertEqual(key1, key2)

    def test_rotate_key_local(self) -> None:
        kms = KMSManager(self.gov_dir)
        key1 = kms.get_encryption_key("rot-key")
        kms.rotate_key("rot-key")
        key2 = kms.get_encryption_key("rot-key")
        self.assertNotEqual(key1, key2)

    def test_configure_provider(self) -> None:
        kms = KMSManager(self.gov_dir)
        kms.configure_provider("aws", aws_key_id="arn:aws:kms:...")
        self.assertEqual(kms.provider, "aws")

    def test_configure_invalid_provider(self) -> None:
        kms = KMSManager(self.gov_dir)
        with self.assertRaises(ValueError):
            kms.configure_provider("invalid_provider")

    def test_default_provider_is_local(self) -> None:
        kms = KMSManager(self.gov_dir)
        self.assertEqual(kms.provider, "local")

    def test_aws_kms_fallback_to_local(self) -> None:
        import sys
        mock_boto3 = MagicMock()
        mock_boto3.client.side_effect = RuntimeError("AWS not available")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            kms = KMSManager(self.gov_dir)
            kms.configure_provider("aws", aws_key_id="arn:fake")
            key = kms.get_encryption_key("aws-fallback")
            self.assertIsInstance(key, bytes)
            self.assertGreater(len(key), 10)

    def test_azure_fallback_to_local(self) -> None:
        kms = KMSManager(self.gov_dir)
        kms.configure_provider("azure", azure_vault_url="https://fake.vault.azure.net")
        key = kms.get_encryption_key("azure-fallback")
        self.assertIsInstance(key, bytes)
        self.assertGreater(len(key), 10)

    def test_gcp_fallback_to_local(self) -> None:
        kms = KMSManager(self.gov_dir)
        kms.configure_provider("gcp", gcp_project_id="fake-project",
                               gcp_location="global", gcp_key_ring="fake-ring")
        key = kms.get_encryption_key("gcp-fallback")
        self.assertIsInstance(key, bytes)
        self.assertGreater(len(key), 10)


# ──────────────────────────────────────────────
# RBACManager
# ──────────────────────────────────────────────

class TestRBACManager(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.gov_dir = Path(self.tmpdir) / "governance"
        self.gov_dir.mkdir(parents=True, exist_ok=True)
        self.rbac = RBACManager(self.gov_dir)

    def test_default_roles_exist(self) -> None:
        # "default" user is "editor"
        self.assertTrue(self.rbac.check_permission("unknown_user", "read"))
        self.assertTrue(self.rbac.check_permission("unknown_user", "write"))

    def test_admin_permissions(self) -> None:
        self.rbac.assign_role("admin_user", "admin")
        self.assertTrue(self.rbac.check_permission("admin_user", "govern"))
        self.assertTrue(self.rbac.check_permission("admin_user", "audit"))

    def test_viewer_cannot_write(self) -> None:
        self.rbac.assign_role("viewer_user", "viewer")
        self.assertTrue(self.rbac.check_permission("viewer_user", "read"))
        self.assertFalse(self.rbac.check_permission("viewer_user", "write"))
        self.assertFalse(self.rbac.check_permission("viewer_user", "delete"))

    def test_auditor_role(self) -> None:
        self.rbac.assign_role("audit_guy", "auditor")
        self.assertTrue(self.rbac.check_permission("audit_guy", "audit"))
        self.assertFalse(self.rbac.check_permission("audit_guy", "write"))

    def test_assign_role_persistence(self) -> None:
        self.rbac.assign_role("persist_user", "admin")
        # Re-instantiate
        rbac2 = RBACManager(self.gov_dir)
        self.assertTrue(rbac2.check_permission("persist_user", "govern"))

    def test_revoke_role(self) -> None:
        self.rbac.assign_role("temp_user", "editor")
        self.rbac.revoke_role("temp_user", "editor")
        # After revoking, falls back to default
        perms = self.rbac.get_user_permissions("temp_user")
        self.assertNotIn("write", perms)

    def test_add_custom_role(self) -> None:
        self.rbac.add_role("custom", ["read", "export"])
        self.rbac.assign_role("custom_user", "custom")
        self.assertTrue(self.rbac.check_permission("custom_user", "read"))
        self.assertFalse(self.rbac.check_permission("custom_user", "write"))

    def test_get_user_permissions(self) -> None:
        self.rbac.assign_role("multi_user", "editor")
        self.rbac.assign_role("multi_user", "auditor")
        perms = self.rbac.get_user_permissions("multi_user")
        self.assertIn("audit", perms)
        self.assertIn("write", perms)

    def test_assign_role_dedup(self) -> None:
        self.rbac.assign_role("dedup_user", "editor")
        self.rbac.assign_role("dedup_user", "editor")
        perms = self.rbac.get_user_permissions("dedup_user")
        self.assertIn("write", perms)
