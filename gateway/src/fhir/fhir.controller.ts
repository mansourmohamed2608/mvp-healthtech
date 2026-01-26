// gateway/src/fhir/fhir.controller.ts
import {
  Controller,
  Post,
  Get,
  Body,
  Param,
  Query,
  Logger,
  UseGuards,
} from '@nestjs/common';
import axios from 'axios';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';
import { Roles } from '../auth/roles.decorator';
import { MetricsController } from '../metrics/metrics.controller';
import { wrapError, camelResponse } from '../utils/http-utils';
import { Req } from '@nestjs/common';
import { Request } from 'express';

@UseGuards(JwtAuthGuard, TenantGuard)
@Roles('clinician')
@Controller('fhir')
export class FhirController {
  private readonly logger = new Logger(FhirController.name);
  private readonly fhirServiceUrl =
    process.env.FHIR_SERVICE_URL || 'http://localhost:5004';
  private readonly headers = (() => {
    if (!process.env.INTERNAL_SECRET)
      throw new Error('INTERNAL_SECRET not set');
    return {
      'x-internal-secret': process.env.INTERNAL_SECRET,
      ...(process.env.FHIR_BEARER_TOKEN
        ? { Authorization: `Bearer ${process.env.FHIR_BEARER_TOKEN}` }
        : {}),
    };
  })();
  private readonly fhirLatency = MetricsController.getFhirLatency();
  private readonly fhirErrors = MetricsController.getFhirErrors();

  @Post(':resourceType')
  @Roles('clinician')
  async createResource(
    @Param('resourceType') resourceType: string,
    @Body() data: any,
  ) {
    const start = process.hrtime();
    this.logger.log(`Create FHIR ${resourceType} request (body redacted)`);
    try {
      const response = await axios.post(
        `${this.fhirServiceUrl}/${resourceType}`,
        data,
        { headers: this.headers },
      );
      this.fhirLatency.observe(
        { endpoint: resourceType, status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(response.data);
    } catch (error) {
      this.fhirErrors.inc({ endpoint: resourceType });
      this.fhirLatency.observe(
        { endpoint: resourceType, status: 'error' },
        this.durationSeconds(start),
      );
      this.logger.error(`FHIR create ${resourceType} error:`, error);
      wrapError(error);
    }
  }

  @Get(':resourceType/:id')
  @Roles('clinician')
  async getResource(
    @Param('resourceType') resourceType: string,
    @Param('id') id: string,
  ) {
    this.logger.log(`Get FHIR ${resourceType}/${id} request`);
    try {
      const response = await axios.get(
        `${this.fhirServiceUrl}/${resourceType}/${id}`,
        { headers: this.headers },
      );
      return response.data;
    } catch (error) {
      this.logger.error(`FHIR get ${resourceType}/${id} error:`, error);
      throw error;
    }
  }

  @Get(':resourceType')
  async searchResource(
    @Param('resourceType') resourceType: string,
    @Query() params: Record<string, string>,
  ) {
    this.logger.log(`Search FHIR ${resourceType} request`);
    try {
      const response = await axios.get(
        `${this.fhirServiceUrl}/${resourceType}`,
        { params },
      );
      return response.data;
    } catch (error) {
      this.logger.error(`FHIR search ${resourceType} error:`, error);
      wrapError(error);
    }
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }
}
