// gateway/src/soap/soap.controller.ts
import { Controller, Post, Get, Body, Logger } from '@nestjs/common';
import axios from 'axios';

class CreateSoapDto {
  transcript: string;
  sessionId?: string;
  patientContext?: any;
}

@Controller('soap')
export class SoapController {
  private readonly logger = new Logger(SoapController.name);
  private readonly soapServiceUrl = process.env.SOAP_SERVICE_URL || 'http://localhost:5003';

  @Post('generate')
  async generateSoap(@Body() dto: CreateSoapDto) {
    this.logger.log('SOAP generate request');
    try {
      const payload = {
        transcript: dto.transcript,
        sessionId: dto.sessionId || `soap-${Date.now()}`,
        patientContext: dto.patientContext,
      };
      const response = await axios.post(`${this.soapServiceUrl}/generate`, payload);
      return response.data;
    } catch (error) {
      this.logger.error('SOAP generate error:', error);
      throw error;
    }
  }

  @Get('notes')
  async getNotes() {
    this.logger.log('Get SOAP notes request');
    try {
      const response = await axios.get(`${this.soapServiceUrl}/notes`);
      return response.data;
    } catch (error) {
      this.logger.error('Get SOAP notes error:', error);
      throw error;
    }
  }
}
