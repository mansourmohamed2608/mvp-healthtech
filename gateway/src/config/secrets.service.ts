/**
 * Secrets Service for loading secrets from various providers
 * Supports: Environment variables, AWS Secrets Manager, Azure Key Vault, HashiCorp Vault
 */

// Ambient declarations for optional cloud SDK dependencies (not installed by default)
declare module '@aws-sdk/client-secrets-manager';
declare module '@azure/keyvault-secrets';
declare module '@azure/identity';

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
 * AWS Secrets Manager provider
 */
class AwsSecretsProvider implements SecretProvider {
  private client: any;
  private prefix: string;

  constructor(private configService: ConfigService) {
    this.prefix = configService.get<string>('AWS_SECRETS_PREFIX', 'healthtech/');
    // Lazy load AWS SDK to avoid bundling if not needed
  }

  private async getClient() {
    if (!this.client) {
      const { SecretsManagerClient, GetSecretValueCommand } = await import(
        '@aws-sdk/client-secrets-manager'
      );
      this.client = new SecretsManagerClient({
        region: this.configService.get<string>('AWS_REGION', 'us-east-1'),
      });
      this.GetSecretValueCommand = GetSecretValueCommand;
    }
    return this.client;
  }

  private GetSecretValueCommand: any;

  async getSecret(key: string): Promise<string | undefined> {
    try {
      const client = await this.getClient();
      const response = await client.send(
        new this.GetSecretValueCommand({
          SecretId: `${this.prefix}${key}`,
        }),
      );
      return response.SecretString;
    } catch (error) {
      console.error(`Failed to retrieve secret ${key} from AWS:`, error);
      return undefined;
    }
  }
}

/**
 * Azure Key Vault provider
 */
class AzureKeyVaultProvider implements SecretProvider {
  private client: any;
  private vaultUrl: string;

  constructor(private configService: ConfigService) {
    this.vaultUrl = configService.get<string>('AZURE_KEYVAULT_URL', '');
  }

  private async getClient() {
    if (!this.client) {
      const { SecretClient } = await import('@azure/keyvault-secrets');
      // @ts-ignore
      const { DefaultAzureCredential } = await import('@azure/identity');
      this.client = new SecretClient(this.vaultUrl, new DefaultAzureCredential());
    }
    return this.client;
  }

  async getSecret(key: string): Promise<string | undefined> {
    try {
      const client = await this.getClient();
      const secret = await client.getSecret(key.replace(/_/g, '-'));
      return secret.value;
    } catch (error) {
      console.error(`Failed to retrieve secret ${key} from Azure:`, error);
      return undefined;
    }
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
