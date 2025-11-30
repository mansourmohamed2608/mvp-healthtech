import { Controller, Get } from '@nestjs/common';
import { KvCacheService } from './cache/kv-cache.service';

@Controller()
export class HealthController {
  constructor(private readonly kvCache: KvCacheService) {}

  @Get('/health')
  get() {
    return { ok: true, ts: Date.now() };
  }

  @Get('/ready')
  async ready() {
    const redisOk = await this.kvCache.ping();
    const requiredEnv = [
      'ASR_SERVICE_URL',
      'LLM_SERVICE_URL',
      'TTS_SERVICE_URL',
      'SOAP_SERVICE_URL',
      'FHIR_SERVICE_URL',
      'JWT_SECRET',
      'INTERNAL_SECRET',
    ];
    const configOk = requiredEnv.every((key) => !!process.env[key]);
    return { ready: redisOk && configOk, redis: redisOk, config: configOk };
  }
}
