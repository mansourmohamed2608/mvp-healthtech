// gateway/src/metrics/metrics.controller.ts
/**
 * Metrics Controller - Prometheus metrics export
 * Week 3 Day 19 (Oct 13, 2025)
 * Enhanced with custom metrics for ASR/LLM/TTS latency
 * NOTE: Metrics are global aggregates, not PHI, but TenantGuard added for defense-in-depth
 */
import { Controller, Get, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';
import {
  Registry,
  collectDefaultMetrics,
  Counter,
  Histogram,
  Gauge,
} from 'prom-client';

const registry = new Registry();
collectDefaultMetrics({ register: registry });

// Custom metrics
const asrLatency = new Histogram({
  name: 'asr_latency_seconds',
  help: 'ASR transcription latency in seconds',
  labelNames: ['endpoint', 'status'],
  buckets: [0.1, 0.3, 0.5, 1, 2, 5],
  registers: [registry],
});

const llmLatency = new Histogram({
  name: 'llm_latency_seconds',
  help: 'LLM inference latency in seconds',
  labelNames: ['endpoint', 'status'],
  buckets: [0.1, 0.3, 0.5, 1, 2, 5],
  registers: [registry],
});

const ttsLatency = new Histogram({
  name: 'tts_latency_seconds',
  help: 'TTS synthesis latency in seconds',
  labelNames: ['endpoint', 'status'],
  buckets: [0.1, 0.3, 0.5, 1, 2, 5],
  registers: [registry],
});

const activeConversations = new Gauge({
  name: 'active_conversations_total',
  help: 'Number of active conversations',
  registers: [registry],
});

const messagesProcessed = new Counter({
  name: 'messages_processed_total',
  help: 'Total number of messages processed',
  labelNames: ['role', 'status'],
  registers: [registry],
});

const twilioCallsTotal = new Counter({
  name: 'twilio_calls_total',
  help: 'Total number of Twilio calls',
  labelNames: ['status'],
  registers: [registry],
});

const soapLatency = new Histogram({
  name: 'soap_latency_seconds',
  help: 'SOAP service latency in seconds',
  labelNames: ['endpoint', 'status'],
  buckets: [0.1, 0.5, 1, 2, 5, 10],
  registers: [registry],
});

const fhirLatency = new Histogram({
  name: 'fhir_latency_seconds',
  help: 'FHIR write latency in seconds',
  labelNames: ['endpoint', 'status'],
  buckets: [0.1, 0.5, 1, 2, 5, 10],
  registers: [registry],
});

const pipelineLatency = new Histogram({
  name: 'va_pipeline_latency_seconds',
  help: 'End-to-end voice agent pipeline latency in seconds',
  labelNames: ['status'],
  buckets: [0.5, 1, 2, 3, 5, 8, 13, 21],
  registers: [registry],
});

const soapErrors = new Counter({
  name: 'soap_errors_total',
  help: 'Total SOAP errors',
  labelNames: ['endpoint'],
  registers: [registry],
});

const fhirErrors = new Counter({
  name: 'fhir_errors_total',
  help: 'Total FHIR errors',
  labelNames: ['endpoint'],
  registers: [registry],
});

@UseGuards(JwtAuthGuard, TenantGuard)
@Controller('metrics')
export class MetricsController {
  @Get()
  async metrics() {
    return registry.metrics();
  }

  // Export metrics for other services to use
  static getAsrLatency() {
    return asrLatency;
  }

  static getLlmLatency() {
    return llmLatency;
  }

  static getTtsLatency() {
    return ttsLatency;
  }

  static getActiveConversations() {
    return activeConversations;
  }

  static getMessagesProcessed() {
    return messagesProcessed;
  }

  static getTwilioCallsTotal() {
    return twilioCallsTotal;
  }

  static getSoapLatency() {
    return soapLatency;
  }

  static getFhirLatency() {
    return fhirLatency;
  }

  static getPipelineLatency() {
    return pipelineLatency;
  }

  static getSoapErrors() {
    return soapErrors;
  }

  static getFhirErrors() {
    return fhirErrors;
  }
}
