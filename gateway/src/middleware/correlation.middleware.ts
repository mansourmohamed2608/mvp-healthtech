import { Injectable, NestMiddleware } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { Request, Response, NextFunction } from 'express';

@Injectable()
export class CorrelationMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    const incoming = (req.headers['x-correlation-id'] as string) || uuidv4();
    (req as any).correlationId = incoming;
    res.setHeader('x-correlation-id', incoming);
    next();
  }
}
