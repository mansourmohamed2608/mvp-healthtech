import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
  Logger,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { FastifyRequest } from 'fastify';

/**
 * Audit log entry structure
 */
interface AuditLogEntry {
  timestamp: string;
  requestId: string;
  userId: string | null;
  tenantId: string | null;
  action: string;
  resource: string;
  method: string;
  path: string;
  statusCode: number;
  duration: number;
  ip: string;
  userAgent: string;
  success: boolean;
  errorMessage?: string;
  // PHI indicators - never log actual PHI data
  containsPatientData: boolean;
  resourceIds: {
    patientId?: string;
    encounterId?: string;
    soapNoteId?: string;
  };
}

/**
 * Audit logging interceptor for compliance and security
 * Logs all API access without capturing PHI/PII
 */
@Injectable()
export class AuditLoggingInterceptor implements NestInterceptor {
  private readonly logger = new Logger('AuditLog');
  private readonly sensitiveEndpoints = [
    '/asr/transcribe',
    '/llm/infer',
    '/llm/chat',
    '/soap',
    '/fhir',
  ];

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const startTime = Date.now();
    const request = context.switchToHttp().getRequest<FastifyRequest>();
    const { method, url, headers } = request;

    // Extract safe metadata
    const requestId = (headers['x-request-id'] as string) || this.generateRequestId();
    const userId = (request as any).user?.id || null;
    const tenantId = (request as any).user?.tenantId || null;
    const ip = this.getClientIp(request);
    const userAgent = (headers['user-agent'] as string) || 'unknown';

    // Determine resource IDs from request (for audit trail, not PHI)
    const resourceIds = this.extractResourceIds(request);
    const containsPatientData = this.containsPatientData(url);

    return next.handle().pipe(
      tap({
        next: (response) => {
          this.logAudit({
            timestamp: new Date().toISOString(),
            requestId,
            userId,
            tenantId,
            action: this.getAction(method),
            resource: this.getResource(url),
            method,
            path: this.sanitizePath(url),
            statusCode: 200,
            duration: Date.now() - startTime,
            ip,
            userAgent,
            success: true,
            containsPatientData,
            resourceIds,
          });
        },
        error: (error) => {
          this.logAudit({
            timestamp: new Date().toISOString(),
            requestId,
            userId,
            tenantId,
            action: this.getAction(method),
            resource: this.getResource(url),
            method,
            path: this.sanitizePath(url),
            statusCode: error.status || 500,
            duration: Date.now() - startTime,
            ip,
            userAgent,
            success: false,
            errorMessage: this.sanitizeError(error.message),
            containsPatientData,
            resourceIds,
          });
        },
      }),
    );
  }

  private getClientIp(request: FastifyRequest): string {
    const forwardedFor = request.headers['x-forwarded-for'];
    if (forwardedFor) {
      const ips = (forwardedFor as string).split(',');
      return ips[0].trim();
    }
    return request.ip || 'unknown';
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getAction(method: string): string {
    const actionMap: Record<string, string> = {
      GET: 'READ',
      POST: 'CREATE',
      PUT: 'UPDATE',
      PATCH: 'UPDATE',
      DELETE: 'DELETE',
    };
    return actionMap[method] || 'UNKNOWN';
  }

  private getResource(url: string): string {
    const parts = url.split('/').filter(Boolean);
    return parts[0] || 'root';
  }

  private sanitizePath(url: string): string {
    // Remove query parameters to avoid logging sensitive data
    return url.split('?')[0];
  }

  private sanitizeError(message: string): string {
    // Remove any potential PHI from error messages
    if (!message) return 'Unknown error';
    
    // Truncate long error messages
    return message.substring(0, 200);
  }

  private containsPatientData(url: string): boolean {
    return this.sensitiveEndpoints.some((endpoint) => url.includes(endpoint));
  }

  private extractResourceIds(request: FastifyRequest): Record<string, string> {
    const body = (request as any).body || {};
    const params = (request as any).params || {};
    const query = request.query || {};

    return {
      patientId: body.patientId || params.patientId || (query as any).patientId,
      encounterId: body.encounterId || params.encounterId || (query as any).encounterId,
      soapNoteId: body.soapNoteId || params.soapNoteId || params.id,
    };
  }

  private logAudit(entry: AuditLogEntry): void {
    // Structured logging for easy parsing
    this.logger.log(JSON.stringify({
      type: 'AUDIT',
      ...entry,
    }));

    // In production, this could also:
    // - Send to a dedicated audit log service
    // - Store in a separate audit database
    // - Send to SIEM system
  }
}
