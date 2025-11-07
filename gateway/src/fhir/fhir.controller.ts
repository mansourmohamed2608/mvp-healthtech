// gateway/src/fhir/fhir.controller.ts
import { Controller, Post, Get, Body, Param, Query, Logger } from '@nestjs/common';
import axios from 'axios';

@Controller('fhir')
export class FhirController {
  private readonly logger = new Logger(FhirController.name);
  private readonly fhirServiceUrl = process.env.FHIR_SERVICE_URL || 'http://localhost:5004';

  @Post(':resourceType')
  async createResource(
    @Param('resourceType') resourceType: string,
    @Body() data: any,
  ) {
    this.logger.log(`Create FHIR ${resourceType} request`);
    try {
      const response = await axios.post(
        `${this.fhirServiceUrl}/${resourceType}`,
        data,
      );
      return response.data;
    } catch (error) {
      this.logger.error(`FHIR create ${resourceType} error:`, error);
      throw error;
    }
  }

  @Get(':resourceType/:id')
  async getResource(
    @Param('resourceType') resourceType: string,
    @Param('id') id: string,
  ) {
    this.logger.log(`Get FHIR ${resourceType}/${id} request`);
    try {
      const response = await axios.get(
        `${this.fhirServiceUrl}/${resourceType}/${id}`,
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
      throw error;
    }
  }
}
