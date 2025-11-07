// gateway/src/clinical/clinical.module.ts
/**
 * Clinical Notes Module
 * Week 5 Day 30
 */
import { Module } from '@nestjs/common';
import { ClinicalController } from './clinical.controller';
import { ClinicalMetricsService } from './clinical-metrics.service';

@Module({
  controllers: [ClinicalController],
  providers: [ClinicalMetricsService],
  exports: [ClinicalMetricsService],
})
export class ClinicalModule {}
