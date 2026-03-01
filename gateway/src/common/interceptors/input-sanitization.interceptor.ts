import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
  BadRequestException,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { FastifyRequest } from 'fastify';

/**
 * Input sanitization interceptor
 * Protects against common injection attacks
 */
@Injectable()
export class InputSanitizationInterceptor implements NestInterceptor {
  // Patterns that may indicate injection attempts
  private readonly dangerousPatterns = [
    // SQL injection patterns
    /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b.*\b(FROM|INTO|TABLE|DATABASE)\b)/gi,
    /(--|#|\/\*|\*\/)/g, // SQL comments
    
    // NoSQL injection patterns
    /\$where|\$gt|\$lt|\$ne|\$or|\$and/gi,
    
    // Path traversal
    /\.\.\//g,
    /\.\.\\+/g,
    
    // XSS patterns
    /<script\b[^>]*>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi, // Event handlers like onclick=
    
    // Command injection
    /[;&|`$]/g, // Shell metacharacters (in specific contexts)
  ];

  // Fields that should never contain code-like content
  private readonly strictFields = [
    'patientId',
    'encounterId',
    'userId',
    'tenantId',
    'sessionId',
  ];

  // Fields that can contain rich content (less strict validation)
  private readonly richContentFields = [
    'transcript',
    'prompt',
    'content',
    'text',
    'notes',
  ];

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest<FastifyRequest>();
    
    // Sanitize body
    if (request.body && typeof request.body === 'object') {
      this.sanitizeObject(request.body as Record<string, any>);
    }

    // Sanitize query parameters
    if (request.query && typeof request.query === 'object') {
      this.validateQueryParams(request.query as Record<string, any>);
    }

    return next.handle();
  }

  private sanitizeObject(obj: Record<string, any>, path = ''): void {
    for (const [key, value] of Object.entries(obj)) {
      const currentPath = path ? `${path}.${key}` : key;

      if (value === null || value === undefined) {
        continue;
      }

      if (typeof value === 'string') {
        // Strict validation for ID fields
        if (this.strictFields.includes(key)) {
          this.validateStrictField(key, value);
        }
        // Check for dangerous patterns in non-rich-content fields
        else if (!this.richContentFields.includes(key)) {
          this.checkDangerousPatterns(key, value);
        }
        // Sanitize but allow rich content
        else {
          obj[key] = this.sanitizeRichContent(value);
        }
      } else if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i++) {
          if (typeof value[i] === 'string') {
            if (this.strictFields.includes(key)) {
              this.validateStrictField(`${key}[${i}]`, value[i]);
            }
          } else if (typeof value[i] === 'object') {
            this.sanitizeObject(value[i], `${currentPath}[${i}]`);
          }
        }
      } else if (typeof value === 'object') {
        this.sanitizeObject(value, currentPath);
      }
    }
  }

  private validateStrictField(field: string, value: string): void {
    // IDs should only contain alphanumeric characters, hyphens, and underscores
    const idPattern = /^[a-zA-Z0-9_-]+$/;
    if (!idPattern.test(value)) {
      throw new BadRequestException(`Invalid characters in ${field}`);
    }

    // Check for maximum length
    if (value.length > 100) {
      throw new BadRequestException(`${field} exceeds maximum length`);
    }
  }

  private checkDangerousPatterns(field: string, value: string): void {
    for (const pattern of this.dangerousPatterns) {
      if (pattern.test(value)) {
        // Reset regex lastIndex for global patterns
        pattern.lastIndex = 0;
        throw new BadRequestException(`Potentially dangerous content in ${field}`);
      }
    }
  }

  private sanitizeRichContent(value: string): string {
    // For rich content, we sanitize rather than reject
    let sanitized = value;

    // Remove script tags
    sanitized = sanitized.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '');
    
    // Remove event handlers
    sanitized = sanitized.replace(/\bon\w+\s*=\s*["'][^"']*["']/gi, '');
    
    // Remove javascript: URLs
    sanitized = sanitized.replace(/javascript:/gi, '');

    return sanitized;
  }

  private validateQueryParams(query: Record<string, any>): void {
    for (const [key, value] of Object.entries(query)) {
      if (typeof value === 'string') {
        // Query parameters should be simple values
        if (value.length > 500) {
          throw new BadRequestException(`Query parameter ${key} too long`);
        }
        
        // Check for injection attempts in query params
        this.checkDangerousPatterns(key, value);
      }
    }
  }
}
