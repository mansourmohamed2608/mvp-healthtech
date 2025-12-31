// gateway/src/soap/soap.controller.ts
import {
  Controller,
  Post,
  Get,
  Body,
  Logger,
  UseGuards,
  Param,
  Patch,
  Query,
  BadRequestException,
  Req,
} from '@nestjs/common';
import axios from 'axios';
import { InternalHttpClient } from '../http/internal-http-client.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { Roles } from '../auth/roles.decorator';
import { Pool } from 'pg';
import { MetricsController } from '../metrics/metrics.controller';
import { wrapError, camelResponse } from '../utils/http-utils';
import type { Request } from 'express';
import { AuditService } from '../audit/audit.service';
import { safeLog } from '../utils/safe-logger';
import { AsrService } from '../asr/asr.service';

class CreateSoapDto {
  transcript!: string;
  sessionId!: string;
  patientContext?: any;
  patientId!: string;
  practitionerId!: string;
  encounterId?: string;
  templateId?: string;
  templateJson?: Record<string, any>;
  patientName?: string;
  providerName?: string;
  dateOfVisit?: string;
}

class UpdateSoapFieldDto {
  fieldPath!: string;
  audio?: string;
  transcript?: string;
  mode?: 'append' | 'replace';
  valueType?: 'string' | 'list';
  dialect?: string;
  language?: string;
}

class UpdateSoapSectionsDto {
  soapText?: string;
  subjective?: string;
  objective?: string;
  assessment?: string;
  plan?: string;
}

class PatientCreateDto {
  displayName!: string;
  externalId?: string;
}

class PatientDocumentDto {
  title?: string;
  content?: string;
  contentBase64?: string;
  fileName?: string;
  contentType?: string;
  source?: string;
  summarize?: boolean;
}

@UseGuards(JwtAuthGuard)
@Controller('soap')
export class SoapController {
  private readonly logger = new Logger(SoapController.name);
  private readonly soapServiceUrl =
    process.env.SOAP_SERVICE_URL || 'http://localhost:5003';
  private readonly headers = (() => {
    if (!process.env.INTERNAL_SECRET)
      throw new Error('INTERNAL_SECRET not set');
    return { 'x-internal-secret': process.env.INTERNAL_SECRET };
  })();
  private readonly pool: Pool | null;
  private readonly soapLatency = MetricsController.getSoapLatency();
  private readonly soapErrors = MetricsController.getSoapErrors();
  private readonly soapClient;
  private readonly fhirClient;

  constructor(
    private readonly auditService: AuditService,
    private readonly http: InternalHttpClient,
    private readonly asrService: AsrService,
  ) {
    const url = process.env.DATABASE_URL;
    this.pool = url ? new Pool({ connectionString: url }) : null;
    this.soapClient = this.http.getClient({
      baseUrl: this.soapServiceUrl,
      serviceName: 'soap',
    });
    this.fhirClient = this.http.getClient({
      baseUrl: process.env.FHIR_SERVICE_URL || 'http://localhost:5004',
      serviceName: 'fhir',
    });
  }

  @Roles('clinician')
  @Post('generate')
  async generateSoap(@Body() dto: CreateSoapDto, @Req() req: Request) {
    safeLog(this.logger, 'log', 'SOAP generate request', {
      sessionId: dto.sessionId || 'auto',
      patientId: dto.patientId || 'n/a',
      clinicianId: dto.practitionerId || 'n/a',
      correlationId: (req as any)?.correlationId,
    });
    const start = process.hrtime();
    try {
      if (!dto.transcript || !dto.patientId || !dto.practitionerId) {
        throw new BadRequestException(
          'transcript, patientId, practitionerId required',
        );
      }
      await this.validateEntities(dto.patientId, dto.practitionerId);
      const payload = {
        transcript: dto.transcript,
        sessionId: dto.sessionId || `soap-${Date.now()}`,
        patientContext: dto.patientContext,
        patientId: dto.patientId,
        clinicianId: dto.practitionerId,
        encounterId: dto.encounterId,
        templateId: dto.templateId,
        templateJson: dto.templateJson,
        patientName: dto.patientName,
        providerName: dto.providerName,
        dateOfVisit: dto.dateOfVisit,
      };
      const asyncEnabled =
        process.env.SOAP_ASYNC_ENABLED === '1' ||
        process.env.SOAP_ASYNC_ENABLED === 'true';
      if (asyncEnabled) {
        const queueUrl =
          process.env.SOAP_QUEUE_URL || process.env.REDIS_URL || '';
        if (!queueUrl)
          throw new BadRequestException('SOAP queue not configured');
        const jobId = await this.enqueueJob(
          payload,
          (req as any)?.correlationId,
        );
        this.soapLatency.observe(
          { endpoint: 'generate', status: 'ok' },
          this.durationSeconds(start),
        );
        return { jobId, status: 'pending' };
      }

      const response = await this.soapClient.post(`/generate`, payload, {
        headers: { 'x-correlation-id': (req as any)?.correlationId },
      });
      const soapNote = response.data;
      safeLog(this.logger, 'log', 'SOAP note created', {
        id: soapNote.id,
        sessionId: payload.sessionId,
      });
      const actor = (req as any).user?.sub || 'unknown';
      await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_CREATED',
        resourceType: 'soap_note',
        resourceId: soapNote.id,
        metadata: {
          sessionId: soapNote.session_id,
          patientId: soapNote.patient_id,
          clinicianId: soapNote.clinician_id,
        },
      });
      // FHIR is triggered on approval, not here
      this.soapLatency.observe(
        { endpoint: 'generate', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(soapNote);
    } catch (error) {
      this.soapErrors.inc({ endpoint: 'generate' });
      this.soapLatency.observe(
        { endpoint: 'generate', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Get('notes')
  @Roles('clinician')
  async getNotes(
    @Query() query?: { status?: string; clinicianId?: string },
    @Req() req?: Request,
  ) {
    this.logger.log('Get SOAP notes request');
    const start = process.hrtime();
    try {
      const response = await this.soapClient.get(`/notes`, {
        params: query || {},
        headers: { 'x-correlation-id': (req as any)?.correlationId },
      });
      this.soapLatency.observe(
        { endpoint: 'notes', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(response.data);
    } catch (error) {
      this.soapErrors.inc({ endpoint: 'notes' });
      this.soapLatency.observe(
        { endpoint: 'notes', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Get('job/:id')
  async getJobStatus(@Param('id') id: string) {
    if (this.pool) {
      const res = await this.pool.query(
        'SELECT * FROM soap_jobs WHERE job_id = $1',
        [id],
      );
     if ((res.rowCount ?? 0) > 0) {
        const row = res.rows[0];
        return {
          id,
          status: row.status,
          attempts: row.attempts,
          noteId: row.note_id,
          error: row.last_error,
          updatedAt: row.updated_at,
        };
      }
    }

    const redis = require('redis');
    const client = redis.createClient({
      url: process.env.SOAP_QUEUE_URL || process.env.REDIS_URL,
    });
    await client.connect();
    const data = await client.hGetAll(id);
    await client.quit();
    if (!data || Object.keys(data).length === 0) {
      throw new BadRequestException('Job not found');
    }
    const result = data.result ? JSON.parse(data.result) : undefined;
    return { id, status: data.status, attempts: data.attempts, result };
  }

  @Get('notes/:id')
  async getNote(@Param('id') id: string, @Req() req?: Request) {
    const start = process.hrtime();
    const response = await this.soapClient.get(`/notes/${id}`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    this.soapLatency.observe(
      { endpoint: 'note', status: 'ok' },
      this.durationSeconds(start),
    );
    return camelResponse(response.data);
  }

  @Get('templates')
  @Roles('clinician')
  async listTemplates(@Req() req?: Request) {
    const response = await this.soapClient.get(`/templates`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Get('templates/:id')
  @Roles('clinician')
  async getTemplate(@Param('id') id: string, @Req() req?: Request) {
    const response = await this.soapClient.get(`/templates/${id}`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Post('templates')
  @Roles('clinician')
  async createTemplate(@Body() dto: any, @Req() req?: Request) {
    const response = await this.soapClient.post(`/templates`, dto, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Get('patients')
  @Roles('clinician')
  async listPatients(@Req() req?: Request) {
    const response = await this.soapClient.get(`/patients`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Post('patients')
  @Roles('clinician')
  async createPatient(@Body() dto: PatientCreateDto, @Req() req?: Request) {
    const response = await this.soapClient.post(`/patients`, dto, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Get('patients/:id/documents')
  @Roles('clinician')
  async listPatientDocuments(@Param('id') id: string, @Req() req?: Request) {
    const response = await this.soapClient.get(`/patients/${id}/documents`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Post('patients/:id/documents')
  @Roles('clinician')
  async uploadPatientDocument(
    @Param('id') id: string,
    @Body() dto: PatientDocumentDto,
    @Req() req?: Request,
  ) {
    const response = await this.soapClient.post(`/patients/${id}/documents`, dto, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Post('patients/:id/documents/:docId/summary')
  @Roles('clinician')
  async summarizePatientDocument(
    @Param('id') id: string,
    @Param('docId') docId: string,
    @Req() req?: Request,
  ) {
    const response = await this.soapClient.post(
      `/patients/${id}/documents/${docId}/summary`,
      {},
      { headers: { 'x-correlation-id': (req as any)?.correlationId } },
    );
    return camelResponse(response.data);
  }

  @Get('patients/:id/context')
  @Roles('clinician')
  async getPatientContext(@Param('id') id: string, @Req() req?: Request) {
    const response = await this.soapClient.get(`/patients/${id}/context`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Get('patients/:id/rag')
  @Roles('clinician')
  async listPatientRag(@Param('id') id: string, @Req() req?: Request) {
    const response = await this.soapClient.get(`/patients/${id}/rag`, {
      headers: { 'x-correlation-id': (req as any)?.correlationId },
    });
    return camelResponse(response.data);
  }

  @Roles('clinician')
  @Patch('notes/:id/field')
  async updateNoteField(
    @Param('id') id: string,
    @Body() dto: UpdateSoapFieldDto,
    @Req() req?: Request,
  ) {
    const start = process.hrtime();
    try {
      if (!dto.fieldPath || (!dto.audio && !dto.transcript)) {
        throw new BadRequestException('fieldPath and audio/transcript required');
      }
      let transcript = dto.transcript?.trim() || '';
      if (dto.audio) {
        const asr = await this.asrService.transcribe(
          dto.audio,
          `soap-field-${id}-${Date.now()}`,
          {
            identifySpeakers: false,
            dialect: dto.dialect,
            language: dto.language || 'ar',
            enableDiarization: false,
            diarizeFirst: false,
          },
          (req as any)?.correlationId,
        );
        transcript = String(asr.text || '').trim();
      }
      if (!transcript) {
        throw new BadRequestException('ASR transcript empty');
      }
      const actor = (req as any)?.user?.sub || 'unknown';
      const response = await this.soapClient.patch(
        `/notes/${id}/field`,
        {
          fieldPath: dto.fieldPath,
          transcript,
          mode: dto.mode,
          valueType: dto.valueType,
          actorId: actor,
          source: dto.audio ? 'voice' : 'text',
        },
        { headers: { 'x-correlation-id': (req as any)?.correlationId } },
      );
      await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_FIELD_UPDATED',
        resourceType: 'soap_note',
        resourceId: id,
        metadata: {
          fieldPath: dto.fieldPath,
          mode: dto.mode || 'append',
        },
      });
      this.soapLatency.observe(
        { endpoint: 'field', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(response.data);
    } catch (error) {
      this.soapErrors.inc({ endpoint: 'field' });
      this.soapLatency.observe(
        { endpoint: 'field', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Roles('clinician')
  @Patch('notes/:id/sections')
  async updateNoteSections(
    @Param('id') id: string,
    @Body() dto: UpdateSoapSectionsDto,
    @Req() req?: Request,
  ) {
    const start = process.hrtime();
    try {
      if (!dto.soapText && !dto.subjective && !dto.objective && !dto.assessment && !dto.plan) {
        throw new BadRequestException('No section updates provided');
      }
      const actor = (req as any)?.user?.sub || 'unknown';
      const response = await this.soapClient.patch(
        `/notes/${id}/sections`,
        {
          soapText: dto.soapText,
          subjective: dto.subjective,
          objective: dto.objective,
          assessment: dto.assessment,
          plan: dto.plan,
          actorId: actor,
        },
        { headers: { 'x-correlation-id': (req as any)?.correlationId } },
      );
      await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_SECTIONS_UPDATED',
        resourceType: 'soap_note',
        resourceId: id,
        metadata: {},
      });
      this.soapLatency.observe(
        { endpoint: 'sections', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(response.data);
    } catch (error) {
      this.soapErrors.inc({ endpoint: 'sections' });
      this.soapLatency.observe(
        { endpoint: 'sections', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Roles('clinician')
  @Patch('notes/:id/approve')
  async approveNote(@Param('id') id: string, @Req() req?: Request) {
    this.logger.log(`Approve SOAP note ${id}`);
    const start = process.hrtime();
    const response = await this.soapClient.patch(
      `/notes/${id}/approve`,
      {},
      { headers: { 'x-correlation-id': (req as any)?.correlationId } },
    );
    const note = camelResponse(response.data);
    const noteSessionId = note.sessionId ?? note.session_id ?? '';
    const notePatientId = note.patientId ?? note.patient_id ?? '';
    const noteClinicianId = note.clinicianId ?? note.clinician_id ?? '';
    const noteEncounterId = note.encounterId ?? note.encounter_id ?? '';
    // Trigger FHIR write after approval
    try {
      const idempotencyKey = `${note.id || id}:${noteEncounterId || 'none'}`;
      const fhirAuthHeader = process.env.FHIR_BEARER_TOKEN
        ? { Authorization: `Bearer ${process.env.FHIR_BEARER_TOKEN}` }
        : {};
      const actor = (req as any)?.user?.sub || 'unknown';
      await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_APPROVED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: {
          sessionId: noteSessionId,
          patientId: notePatientId,
          clinicianId: noteClinicianId,
        },
      });
      await this.auditService.log({
        actorId: actor,
        action: 'FHIR_WRITE_ATTEMPTED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: {
          patientId: notePatientId,
          clinicianId: noteClinicianId,
          noteId: note.id,
        },
      });
      await this.fhirClient.post(
        `/write`,
        {
          soapNote: note,
          patientId: notePatientId,
          practitionerId: noteClinicianId,
          encounterId: noteEncounterId,
          sessionId: noteSessionId || `${note.id || id}`,
        },
        {
          headers: {
            'Idempotency-Key': idempotencyKey,
            'x-correlation-id': (req as any)?.correlationId,
            ...fhirAuthHeader,
          },
        },
      );
      await this.auditService.log({
        actorId: actor,
        action: 'FHIR_WRITE_SUCCEEDED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: {
          patientId: notePatientId,
          clinicianId: noteClinicianId,
          noteId: note.id,
          encounterId: noteEncounterId,
          httpStatus: 200,
        },
      });
      this.soapLatency.observe(
        { endpoint: 'approve', status: 'ok' },
        this.durationSeconds(start),
      );
    } catch (fhirErr) {
      const actor = (req as any)?.user?.sub || 'unknown';
      await this.auditService.log({
        actorId: actor,
        action: 'FHIR_WRITE_FAILED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: {
          patientId: note.patient_id,
          clinicianId: note.clinician_id,
          noteId: note.id,
        },
      });
      this.soapErrors.inc({ endpoint: 'approve' });
      this.soapLatency.observe(
        { endpoint: 'approve', status: 'error' },
        this.durationSeconds(start),
      );
      this.logger.warn(`FHIR write failed for note ${id}: ${fhirErr}`);
    }
    return note;
  }

  @Roles('clinician')
  @Patch('notes/:id/reject')
  async rejectNote(@Param('id') id: string, @Req() req?: Request) {
    this.logger.log(`Reject SOAP note ${id}`);
    const start = process.hrtime();
    const response = await axios.patch(
      `${this.soapServiceUrl}/notes/${id}/reject`,
      {},
      { headers: this.headers },
    );
    this.soapLatency.observe(
      { endpoint: 'reject', status: 'ok' },
      this.durationSeconds(start),
    );
    const note = camelResponse(response.data);
    const actor = (req as any)?.user?.sub || 'unknown';
    await this.auditService.log({
      actorId: actor,
      action: 'SOAP_NOTE_REJECTED',
      resourceType: 'soap_note',
      resourceId: note.id,
      metadata: {
        sessionId: note.session_id,
        patientId: note.patient_id,
        clinicianId: note.clinician_id,
      },
    });
    return note;
  }

  private async validateEntities(patientId?: string, clinicianId?: string) {
    if (!this.pool) return;
    if (patientId) {
      const res = await this.pool.query(
        'SELECT id FROM patients WHERE id = $1',
        [patientId],
      );
      if (res.rowCount === 0)
        throw new BadRequestException('Invalid patientId');
    }
    if (clinicianId) {
      const res = await this.pool.query(
        'SELECT id FROM clinicians WHERE id = $1',
        [clinicianId],
      );
      if (res.rowCount === 0)
        throw new BadRequestException('Invalid clinicianId');
    }
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }

  private async enqueueJob(
    payload: any,
    correlationId?: string,
  ): Promise<string> {
    const redis = require('redis');
    const client = redis.createClient({
      url: process.env.SOAP_QUEUE_URL || process.env.REDIS_URL,
    });
    await client.connect();
    const jobId = `soap:${Date.now()}:${Math.random().toString(36).slice(2, 7)}`;
    const job = {
      id: jobId,
      status: 'pending',
      payload,
      attempts: 0,
      correlationId: correlationId || null,
      createdAt: Date.now(),
    };
    if (this.pool) {
      await this.pool.query(
        `INSERT INTO soap_jobs (job_id, session_id, patient_id, clinician_id, status, attempts, correlation_id)
         VALUES ($1,$2,$3,$4,'pending',0,$5)`,
        [
          jobId,
          payload.sessionId,
          payload.patientId,
          payload.clinicianId,
          correlationId || null,
        ],
      );
    }
    await client.hSet(jobId, job);
    await client.lPush('soap_jobs', jobId);
    await client.quit();
    return jobId;
  }
}
