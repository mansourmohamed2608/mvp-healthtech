/**
 * Secrets Service for loading secrets from various providers
 * Supports: Environment variables, AWS Secrets Manager, Azure Key Vault, HashiCorp Vault
 */
import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

export interface SecretProvider {
  getSecret(key: string): Promise<string | undefined>;
}

@Injectable()
export class SecretsService implements OnModuleInit {
  private readonly logger = new Logger(SecretsService.name);
  private provider: SecretProvider;
  private cache = new Map<string, { value: string; expiry: number }>();
  private readonly cacheTTL = 5 * 60 * 1000; // 5 minutes

  constructor(private configService: ConfigService) {}

  async onModuleInit() {
    const providerType = this.configService.get<string>('SECRETS_PROVIDER', 'env');
    this.provider = this.createProvider(providerType);
    this.logger.log(`Secrets provider initialized: ${providerType}`);
  }

  private createProvider(type: string): SecretProvider {
    switch (type) {
      case 'aws':
        return new AwsSecretsProvider(this.configService);
      case 'azure':
        return new AzureKeyVaultProvider(this.configService);
      case 'vault':
        return new HashiCorpVaultProvider(this.configService);
      case 'env':
      default:
        return new EnvSecretProvider(this.configService);
    }
  }

  async getSecret(key: string): Promise<string | undefined> {
    // Check cache first
    const cached = this.cache.get(key);
    if (cached && cached.expiry > Date.now()) {
      return cached.value;
    }

    const value = await this.provider.getSecret(key);
    
    if (value) {
      this.cache.set(key, {
        value,
        expiry: Date.now() + this.cacheTTL,
      });
    }

    return value;
  }

  async getSecretOrThrow(key: string): Promise<string> {
    const value = await this.getSecret(key);
    if (!value) {
      throw new Error(`Required secret not found: ${key}`);
    }
    return value;
  }

  clearCache() {
    this.cache.clear();
  }
}

/**
 * Environment variable provider (default for local development)
 */
class EnvSecretProvider implements SecretProvider {
  constructor(private configService: ConfigService) {}

  async getSecret(key: string): Promise<string | undefined> {
    return this.configService.get<string>(key);
  }
}

/**
 * AWS Secrets Manager provider (stub - install @aws-sdk/client-secrets-manager to enable)
 */
class AwsSecretsProvider implements SecretProvider {
  constructor(private configService: ConfigService) {}

  async getSecret(_key: string): Promise<string | undefined> {
    throw new Error('AWS Secrets Manager provider requires @aws-sdk/client-secrets-manager to be installed');
  }
}

/**
 * Azure Key Vault provider (stub - install @azure/keyvault-secrets and @azure/identity to enable)
 */
class AzureKeyVaultProvider implements SecretProvider {
  constructor(private configService: ConfigService) {}

  async getSecret(_key: string): Promise<string | undefined> {
    throw new Error('Azure Key Vault provider requires @azure/keyvault-secrets and @azure/identity to be installed');
  }
}

/**
 * HashiCorp Vault provider
 */
class HashiCorpVaultProvider implements SecretProvider {
  private vaultAddr: string;
  private vaultToken: string;
  private mountPath: string;

  constructor(private configService: ConfigService) {
    this.vaultAddr = configService.get<string>('VAULT_ADDR', 'http://localhost:8200');
    this.vaultToken = configService.get<string>('VAULT_TOKEN', '');
    this.mountPath = configService.get<string>('VAULT_MOUNT_PATH', 'healthtech');
  }

  async getSecret(key: string): Promise<string | undefined> {
    try {
      const response = await fetch(
        `${this.vaultAddr}/v1/${this.mountPath}/data/${key}`,
        {
          headers: {
            'X-Vault-Token': this.vaultToken,
          },
        },
      );

      if (!response.ok) {
        return undefined;
      }

      const data = await response.json();
      return data.data?.data?.value;
    } catch (error) {
      console.error(`Failed to retrieve secret ${key} from Vault:`, error);
      return undefined;
    }
  }
}
