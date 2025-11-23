// gateway/src/soap/soap.controller.ts
import { Controller, Post, Get, Body, Logger, UseGuards, Param, Patch, Query, BadRequestException, Req } from '@nestjs/common';
import axios from 'axios';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { Roles } from '../auth/roles.decorator';
import { Pool } from 'pg';
import { MetricsController } from '../metrics/metrics.controller';
import { wrapError, camelResponse } from '../utils/http-utils';
import { Request } from 'express';
import { AuditService } from '../audit/audit.service';

class CreateSoapDto {
  transcript!: string;
  sessionId!: string;
  patientContext?: any;
  patientId!: string;
  practitionerId!: string;
  encounterId?: string;
}

@UseGuards(JwtAuthGuard)
@Controller('soap')
export class SoapController {
  private readonly logger = new Logger(SoapController.name);
  private readonly soapServiceUrl = process.env.SOAP_SERVICE_URL || 'http://localhost:5003';
  private readonly headers = (() => {
    if (!process.env.INTERNAL_SECRET) throw new Error('INTERNAL_SECRET not set');
    return { 'x-internal-secret': process.env.INTERNAL_SECRET };
  })();
  private readonly pool: Pool | null;
  private readonly soapLatency = MetricsController.getSoapLatency();
  private readonly soapErrors = MetricsController.getSoapErrors();

  constructor(private readonly auditService: AuditService) {
    const url = process.env.DATABASE_URL;
    this.pool = url ? new Pool({ connectionString: url }) : null;
  }

  @Roles('clinician')
  @Post('generate')
  async generateSoap(@Body() dto: CreateSoapDto, @Req() req: Request) {
    this.logger.log(`SOAP generate request (session=${dto.sessionId || 'auto'}, patient=${dto.patientId || 'n/a'})`);
    const start = process.hrtime();
    try {
      if (!dto.transcript || !dto.patientId || !dto.practitionerId) {
        throw new BadRequestException('transcript, patientId, practitionerId required');
      }
      await this.validateEntities(dto.patientId, dto.practitionerId);
      const payload = {
        transcript: dto.transcript,
        sessionId: dto.sessionId || `soap-${Date.now()}`,
        patientContext: dto.patientContext,
        patientId: dto.patientId,
        clinicianId: dto.practitionerId,
        encounterId: dto.encounterId,
      };
      const response = await axios.post(`${this.soapServiceUrl}/generate`, payload, { headers: this.headers });
      const soapNote = response.data;
      this.logger.log(`SOAP note created id=${soapNote.id} session=${payload.sessionId}`);
      const actor = (req as any).user?.sub || 'unknown';
      await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_CREATED',
        resourceType: 'soap_note',
        resourceId: soapNote.id,
        metadata: { sessionId: soapNote.session_id, patientId: soapNote.patient_id, clinicianId: soapNote.clinician_id },
      });
      // FHIR is triggered on approval, not here
      this.soapLatency.observe({ endpoint: 'generate', status: 'ok' }, this.durationSeconds(start));
      return camelResponse(soapNote);
    } catch (error) {
      this.soapErrors.inc({ endpoint: 'generate' });
      this.soapLatency.observe({ endpoint: 'generate', status: 'error' }, this.durationSeconds(start));
      wrapError(error, req);
    }
  }

  @Get('notes')
  async getNotes(@Query() query?: { status?: string; clinicianId?: string }, @Req() req?: Request) {
    this.logger.log('Get SOAP notes request');
    const start = process.hrtime();
    try {
      const response = await axios.get(`${this.soapServiceUrl}/notes`, { params: query || {}, headers: this.headers });
      this.soapLatency.observe({ endpoint: 'notes', status: 'ok' }, this.durationSeconds(start));
      return camelResponse(response.data);
    } catch (error) {
      this.soapErrors.inc({ endpoint: 'notes' });
      this.soapLatency.observe({ endpoint: 'notes', status: 'error' }, this.durationSeconds(start));
      wrapError(error, req);
    }
  }

  @Get('notes/:id')
  async getNote(@Param('id') id: string, @Req() req?: Request) {
    const start = process.hrtime();
    const response = await axios.get(`${this.soapServiceUrl}/notes/${id}`, { headers: this.headers });
    this.soapLatency.observe({ endpoint: 'note', status: 'ok' }, this.durationSeconds(start));
    return camelResponse(response.data);
  }

  @Roles('clinician')
  @Patch('notes/:id/approve')
  async approveNote(@Param('id') id: string, @Req() req?: Request) {
    this.logger.log(`Approve SOAP note ${id}`);
    const start = process.hrtime();
    const response = await axios.patch(`${this.soapServiceUrl}/notes/${id}/approve`, {}, { headers: this.headers });
    const note = camelResponse(response.data);
    // Trigger FHIR write after approval
    try {
      const idempotencyKey = `${note.id || id}:${note.encounter_id || 'none'}`;
      const fhirAuthHeader = process.env.FHIR_BEARER_TOKEN
        ? { Authorization: `Bearer ${process.env.FHIR_BEARER_TOKEN}` }
        : {};
      const actor = (req as any)?.user?.sub || 'unknown';
      await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_APPROVED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: { sessionId: note.session_id, patientId: note.patient_id, clinicianId: note.clinician_id },
      });
      await this.auditService.log({
        actorId: actor,
        action: 'FHIR_WRITE_ATTEMPTED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: { patientId: note.patient_id, clinicianId: note.clinician_id, noteId: note.id },
      });
      await axios.post(`${process.env.FHIR_SERVICE_URL || 'http://localhost:5004'}/write`, {
        soapNote: note,
        patientId: note.patient_id,
        practitionerId: note.clinician_id,
        encounterId: note.encounter_id,
        sessionId: note.session_id,
      }, {
        headers: {
          'Idempotency-Key': idempotencyKey,
          'x-internal-secret': process.env.INTERNAL_SECRET,
          ...fhirAuthHeader,
        },
      });
      await this.auditService.log({
        actorId: actor,
        action: 'FHIR_WRITE_SUCCEEDED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: { patientId: note.patient_id, clinicianId: note.clinician_id, noteId: note.id, encounterId: note.encounter_id, httpStatus: 200 },
      });
      this.soapLatency.observe({ endpoint: 'approve', status: 'ok' }, this.durationSeconds(start));
    } catch (fhirErr) {
      const actor = (req as any)?.user?.sub || 'unknown';
      await this.auditService.log({
        actorId: actor,
        action: 'FHIR_WRITE_FAILED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: { patientId: note.patient_id, clinicianId: note.clinician_id, noteId: note.id },
      });
      this.soapErrors.inc({ endpoint: 'approve' });
      this.soapLatency.observe({ endpoint: 'approve', status: 'error' }, this.durationSeconds(start));
      this.logger.warn(`FHIR write failed for note ${id}: ${fhirErr}`);
    }
    return note;
  }

  @Roles('clinician')
  @Patch('notes/:id/reject')
  async rejectNote(@Param('id') id: string, @Req() req?: Request) {
    this.logger.log(`Reject SOAP note ${id}`);
    const start = process.hrtime();
    const response = await axios.patch(`${this.soapServiceUrl}/notes/${id}/reject`, {}, { headers: this.headers });
    this.soapLatency.observe({ endpoint: 'reject', status: 'ok' }, this.durationSeconds(start));
    const note = camelResponse(response.data);
    const actor = (req as any)?.user?.sub || 'unknown';
    await this.auditService.log({
        actorId: actor,
        action: 'SOAP_NOTE_REJECTED',
        resourceType: 'soap_note',
        resourceId: note.id,
        metadata: { sessionId: note.session_id, patientId: note.patient_id, clinicianId: note.clinician_id },
    });
    return note;
  }

  private async validateEntities(patientId?: string, clinicianId?: string) {
    if (!this.pool) return;
    if (patientId) {
      const res = await this.pool.query('SELECT id FROM patients WHERE id = $1', [patientId]);
      if (res.rowCount === 0) throw new BadRequestException('Invalid patientId');
    }
    if (clinicianId) {
      const res = await this.pool.query('SELECT id FROM clinicians WHERE id = $1', [clinicianId]);
      if (res.rowCount === 0) throw new BadRequestException('Invalid clinicianId');
    }
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }
}
