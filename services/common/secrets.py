"""
Secrets manager client for Python services.
Supports: Environment variables, AWS Secrets Manager, Azure Key Vault, HashiCorp Vault
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class SecretProvider(ABC):
    """Abstract base class for secret providers."""

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key."""
        pass


class EnvSecretProvider(SecretProvider):
    """Environment variable based secrets (for local development)."""

    def get_secret(self, key: str) -> Optional[str]:
        return os.environ.get(key)


class AwsSecretsProvider(SecretProvider):
    """AWS Secrets Manager provider."""

    def __init__(self, region: str = "us-east-1", prefix: str = "healthtech/"):
        self.region = region
        self.prefix = prefix
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("secretsmanager", region_name=self.region)
        return self._client

    def get_secret(self, key: str) -> Optional[str]:
        try:
            response = self.client.get_secret_value(SecretId=f"{self.prefix}{key}")
            secret_string = response.get("SecretString")
            
            # Try to parse as JSON (for structured secrets)
            try:
                secret_dict = json.loads(secret_string)
                return secret_dict.get("value", secret_string)
            except json.JSONDecodeError:
                return secret_string
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key} from AWS: {e}")
            return None


class AzureKeyVaultProvider(SecretProvider):
    """Azure Key Vault provider."""

    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            credential = DefaultAzureCredential()
            self._client = SecretClient(vault_url=self.vault_url, credential=credential)
        return self._client

    def get_secret(self, key: str) -> Optional[str]:
        try:
            # Azure Key Vault uses hyphens instead of underscores
            azure_key = key.replace("_", "-")
            secret = self.client.get_secret(azure_key)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key} from Azure: {e}")
            return None


class HashiCorpVaultProvider(SecretProvider):
    """HashiCorp Vault provider."""

    def __init__(
        self,
        vault_addr: str = "http://localhost:8200",
        vault_token: str = "",
        mount_path: str = "healthtech",
    ):
        self.vault_addr = vault_addr
        self.vault_token = vault_token
        self.mount_path = mount_path

    def get_secret(self, key: str) -> Optional[str]:
        try:
            import httpx
            
            response = httpx.get(
                f"{self.vault_addr}/v1/{self.mount_path}/data/{key}",
                headers={"X-Vault-Token": self.vault_token},
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            return data.get("data", {}).get("data", {}).get("value")
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key} from Vault: {e}")
            return None


class SecretsManager:
    """
    Main secrets manager class.
    
    Usage:
        secrets = SecretsManager()
        jwt_secret = secrets.get("JWT_SECRET")
    """

    _instance: Optional["SecretsManager"] = None
    _cache: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the appropriate secret provider."""
        provider_type = os.environ.get("SECRETS_PROVIDER", "env")
        
        if provider_type == "aws":
            region = os.environ.get("AWS_REGION", "us-east-1")
            prefix = os.environ.get("AWS_SECRETS_PREFIX", "healthtech/")
            self.provider = AwsSecretsProvider(region=region, prefix=prefix)
        elif provider_type == "azure":
            vault_url = os.environ.get("AZURE_KEYVAULT_URL", "")
            self.provider = AzureKeyVaultProvider(vault_url=vault_url)
        elif provider_type == "vault":
            self.provider = HashiCorpVaultProvider(
                vault_addr=os.environ.get("VAULT_ADDR", "http://localhost:8200"),
                vault_token=os.environ.get("VAULT_TOKEN", ""),
                mount_path=os.environ.get("VAULT_MOUNT_PATH", "healthtech"),
            )
        else:
            self.provider = EnvSecretProvider()
        
        logger.info(f"Secrets manager initialized with provider: {provider_type}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            key: The secret key
            default: Default value if secret not found
            
        Returns:
            The secret value or default
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        value = self.provider.get_secret(key)
        
        if value is not None:
            self._cache[key] = value
            return value
        
        return default

    def get_or_raise(self, key: str) -> str:
        """
        Get a secret value or raise an error.
        
        Args:
            key: The secret key
            
        Returns:
            The secret value
            
        Raises:
            ValueError: If secret not found
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required secret not found: {key}")
        return value

    def clear_cache(self):
        """Clear the secrets cache."""
        self._cache.clear()


# Convenience function
@lru_cache(maxsize=1)
def get_secrets_manager() -> SecretsManager:
    """Get the singleton secrets manager instance."""
    return SecretsManager()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret.
    
    Usage:
        from common.secrets import get_secret
        jwt_secret = get_secret("JWT_SECRET")
    """
    return get_secrets_manager().get(key, default)
