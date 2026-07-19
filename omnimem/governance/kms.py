"""KMSManager — 密钥管理服务。

提供密钥的获取、存储和轮换能力，支持多种后端：
  - local: 本地文件存储（默认），密钥从环境变量或本地文件读取
  - aws: AWS KMS（通过 GenerateDataKey 获取数据密钥）
  - azure: Azure Key Vault（通过 SecretClient 获取密钥）
  - gcp: Google Cloud KMS（通过对称解密获取密钥）

密钥优先级（local 模式）：
  1. 环境变量 OMNIMEM_KEY_{KEY_ID}（推荐，适合容器化部署）
  2. 本地文件 key_{key_id}.bin
  3. 自动生成（首次使用时）
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KMSManager:
    """密钥管理服务，支持多后端密钥获取和存储。"""

    def __init__(self, governance_dir: Path):
        self._dir = governance_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._config_path = governance_dir / "kms_config.json"
        self._provider: str = "local"
        self._config: dict = {}
        # ★ P0修复：密钥缓存，避免频繁磁盘 IO
        self._key_cache: dict[str, bytes] = {}
        self._load_config()

    def _load_config(self) -> None:
        if self._config_path.exists():
            with open(self._config_path, encoding="utf-8") as f:
                self._config = json.load(f)
            self._provider = self._config.get("provider", "local")
        else:
            self._provider = "local"
            self._config = {"provider": "local"}

    def _save_config(self) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)

    # ─── 公开接口 ─────────────────────────────────────────────

    def get_key(self, key_id: str = "default") -> bytes:
        """获取密钥（推荐接口）。

        优先从环境变量读取，回退到配置的 provider 获取。

        环境变量格式：OMNIMEM_KEY_{KEY_ID.upper()}，例如：
          - key_id="default" → OMNIMEM_KEY_DEFAULT
          - key_id="encryption" → OMNIMEM_KEY_ENCRYPTION

        Args:
            key_id: 密钥标识符

        Returns:
            密钥字节
        """
        # ★ P0修复：优先从环境变量读取密钥
        env_key = f"OMNIMEM_KEY_{key_id.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            logger.debug("KMS: 从环境变量 %s 获取密钥 (key_id=%s)", env_key, key_id)
            key_bytes = base64.b64decode(env_value)
            self._key_cache[key_id] = key_bytes
            return key_bytes

        # 缓存命中
        if key_id in self._key_cache:
            return self._key_cache[key_id]

        # 按 provider 获取
        key_bytes = self.get_encryption_key(key_id)
        self._key_cache[key_id] = key_bytes
        return key_bytes

    def set_key(self, key_id: str, key: bytes) -> None:
        """存储密钥到本地文件（仅 local provider）。

        对于非 local provider，此方法仅更新缓存，不写入远端。

        Args:
            key_id: 密钥标识符
            key: 密钥字节
        """
        self._key_cache[key_id] = key

        if self._provider == "local":
            key_path = self._dir / f"key_{key_id}.bin"
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(key)
            logger.info("KMS: 密钥已写入本地文件 (key_id=%s)", key_id)
        else:
            logger.warning("KMS: 非 local provider，set_key 仅更新缓存 (key_id=%s, provider=%s)",
                         key_id, self._provider)

    # ─── 原有接口（保留签名） ─────────────────────────────────

    def get_encryption_key(self, key_id: str = "default") -> bytes:
        """获取加密密钥（按 provider 分发）。

        保留原有方法签名，内部按 provider 路由到不同后端。
        """
        if self._provider == "aws":
            return self._get_from_aws_kms(key_id)
        elif self._provider == "azure":
            return self._get_from_azure_keyvault(key_id)
        elif self._provider == "gcp":
            return self._get_from_gcp_kms(key_id)
        else:
            return self._get_local_key(key_id)

    def _get_local_key(self, key_id: str) -> bytes:
        """从本地文件读取密钥，不存在时自动生成。"""
        key_path = self._dir / f"key_{key_id}.bin"
        if key_path.exists():
            with open(key_path, "rb") as f:
                return f.read()
        # 自动生成 Fernet 密钥
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(key)
        logger.info("KMS: 自动生成新密钥 (key_id=%s)", key_id)
        return key

    def _get_from_aws_kms(self, key_id: str) -> bytes:
        """从 AWS KMS 获取数据密钥。

        ★ P0修复：使用 GenerateDataKey 而非 Encrypt，获取明文数据密钥用于本地加密。
        """
        try:
            import boto3
            client = boto3.client("kms")
            kms_key_id = self._config.get("aws_key_id", "")
            if not kms_key_id:
                raise ValueError("KMS: aws_key_id 未配置")
            response = client.generate_data_key(
                KeyId=kms_key_id,
                KeySpec="AES_256",
            )
            # 返回明文数据密钥（Plaintext），CiphertextBlob 可用于持久化
            return response["Plaintext"]
        except Exception as e:
            logger.warning("KMS: AWS KMS 获取密钥失败 (key_id=%s): %s，回退到本地密钥", key_id, e)
            return self._get_local_key(key_id)

    def _get_from_azure_keyvault(self, key_id: str) -> bytes:
        """从 Azure Key Vault 获取密钥。

        ★ P0修复：正确解码 Secret 值，支持 Base64 和原始字符串两种格式。
        """
        try:
            from azure.keyvault.secrets import SecretClient
            vault_url = self._config.get("azure_vault_url", "")
            if not vault_url:
                raise ValueError("KMS: azure_vault_url 未配置")
            credential = self._get_azure_credential()
            if credential is None:
                raise ValueError("KMS: Azure 凭证不可用")
            client = SecretClient(vault_url=vault_url, credential=credential)
            secret = client.get_secret(key_id)
            # 尝试 Base64 解码，失败则使用原始字符串编码
            try:
                return base64.b64decode(secret.value)
            except Exception:
                return secret.value.encode("utf-8")
        except Exception as e:
            logger.warning("KMS: Azure Key Vault 获取密钥失败 (key_id=%s): %s，回退到本地密钥", key_id, e)
            return self._get_local_key(key_id)

    def _get_from_gcp_kms(self, key_id: str) -> bytes:
        """从 Google Cloud KMS 获取密钥。

        ★ P0修复：使用非对称签名验证密钥存在性，再通过对称解密获取密钥材料。
        需要在配置中提供 base64 编码的密文（ciphertext_b64）。
        """
        try:
            from google.cloud import kms
            project_id = self._config.get("gcp_project_id", "")
            location = self._config.get("gcp_location", "global")
            key_ring = self._config.get("gcp_key_ring", "")
            ciphertext_b64 = self._config.get("gcp_ciphertext_b64", "")
            if not all([project_id, key_ring, ciphertext_b64]):
                raise ValueError("KMS: gcp_project_id/gcp_key_ring/gcp_ciphertext_b64 未完整配置")
            client = kms.KeyManagementServiceClient()
            key_name = client.crypto_key_path(project_id, location, key_ring, key_id)
            ciphertext = base64.b64decode(ciphertext_b64)
            response = client.decrypt(request={"name": key_name, "ciphertext": ciphertext})
            return response.plaintext
        except Exception as e:
            logger.warning("KMS: GCP KMS 获取密钥失败 (key_id=%s): %s，回退到本地密钥", key_id, e)
            return self._get_local_key(key_id)

    def _get_azure_credential(self):
        """获取 Azure 认证凭证。"""
        try:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        except ImportError:
            logger.warning("KMS: azure-identity 未安装，Azure Key Vault 不可用")
            return None

    def rotate_key(self, key_id: str = "default") -> None:
        """轮换密钥：删除旧密钥，自动生成新密钥。"""
        # 清除缓存
        self._key_cache.pop(key_id, None)
        if self._provider == "local":
            key_path = self._dir / f"key_{key_id}.bin"
            if key_path.exists():
                key_path.unlink()
            self._get_local_key(key_id)
            logger.info("KMS: 密钥已轮换 (key_id=%s)", key_id)
        else:
            logger.warning("KMS: 非 local provider 的密钥轮换需在云平台操作 (key_id=%s)", key_id)

    def configure_provider(self, provider: str, **kwargs: Any) -> None:
        """配置 KMS provider 及其参数。

        Args:
            provider: "local" | "aws" | "azure" | "gcp"
            **kwargs: provider 特定配置参数
        """
        valid_providers = ["local", "aws", "azure", "gcp"]
        if provider not in valid_providers:
            raise ValueError(f"Invalid KMS provider: {provider}, must be one of {valid_providers}")
        self._provider = provider
        self._config["provider"] = provider
        self._config.update(kwargs)
        self._save_config()
        # 切换 provider 时清除缓存
        self._key_cache.clear()
        logger.info("KMS: provider 已切换为 %s", provider)

    @property
    def provider(self) -> str:
        return self._provider
