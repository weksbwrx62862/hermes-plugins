import json
import os
import hashlib
import base64
from pathlib import Path
from typing import Any

class KMSManager:
    def __init__(self, governance_dir: Path):
        self._dir = governance_dir
        self._config_path = governance_dir / "kms_config.json"
        self._provider: str = "local"
        self._config: dict = {}
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
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)
    
    def get_encryption_key(self, key_id: str = "default") -> bytes:
        if self._provider == "aws":
            return self._get_from_aws_kms(key_id)
        elif self._provider == "azure":
            return self._get_from_azure_keyvault(key_id)
        elif self._provider == "gcp":
            return self._get_from_gcp_kms(key_id)
        else:
            return self._get_local_key(key_id)
    
    def _get_local_key(self, key_id: str) -> bytes:
        key_path = self._dir / f"key_{key_id}.bin"
        if key_path.exists():
            with open(key_path, "rb") as f:
                return f.read()
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(key)
        return key
    
    def _get_from_aws_kms(self, key_id: str) -> bytes:
        try:
            import boto3
            client = boto3.client("kms")
            kms_key_id = self._config.get("aws_key_id", "")
            response = client.encrypt(KeyId=kms_key_id, Plaintext=b"omnimem-key-request-" + key_id.encode())
            return response["CiphertextBlob"]
        except Exception:
            return self._get_local_key(key_id)
    
    def _get_from_azure_keyvault(self, key_id: str) -> bytes:
        try:
            from azure.keyvault.secrets import SecretClient
            vault_url = self._config.get("azure_vault_url", "")
            credential = self._get_azure_credential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            secret = client.get_secret(key_id)
            return base64.b64decode(secret.value)
        except Exception:
            return self._get_local_key(key_id)
    
    def _get_from_gcp_kms(self, key_id: str) -> bytes:
        try:
            from google.cloud import kms
            project_id = self._config.get("gcp_project_id", "")
            location = self._config.get("gcp_location", "global")
            key_ring = self._config.get("gcp_key_ring", "")
            client = kms.KeyManagementServiceClient()
            key_name = client.crypto_key_path(project_id, location, key_ring, key_id)
            response = client.decrypt(request={"name": key_name, "ciphertext": b"omnimem"})
            return response.plaintext
        except Exception:
            return self._get_local_key(key_id)
    
    def _get_azure_credential(self):
        try:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        except ImportError:
            return None
    
    def rotate_key(self, key_id: str = "default") -> None:
        if self._provider == "local":
            key_path = self._dir / f"key_{key_id}.bin"
            if key_path.exists():
                key_path.unlink()
            self._get_local_key(key_id)
    
    def configure_provider(self, provider: str, **kwargs) -> None:
        valid_providers = ["local", "aws", "azure", "gcp"]
        if provider not in valid_providers:
            raise ValueError(f"Invalid KMS provider: {provider}, must be one of {valid_providers}")
        self._provider = provider
        self._config["provider"] = provider
        self._config.update(kwargs)
        self._save_config()
    
    @property
    def provider(self) -> str:
        return self._provider
