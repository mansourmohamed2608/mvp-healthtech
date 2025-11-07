// gateway/src/clinical/clinical.controller.ts
/**
 * Clinical Notes Metrics Controller
 * Week 5 Day 30 (Oct 24, 2025)
 * Tracks note acceptance, edit distance, time to review
 */
import { Controller, Get, Post, Body, Param } from '@nestjs/common';
import { ClinicalMetricsService } from './clinical-metrics.service';

interface NoteReviewEvent {
  recordingId: string;
  accepted: boolean;
  editDistance: number;
  timeToReview: number; // seconds
  clinicianId?: string;
}

@Controller('clinical')
export class ClinicalController {
  constructor(private readonly metricsService: ClinicalMetricsService) {}

  @Get('metrics')
  async getMetrics() {
    return this.metricsService.getMetrics();
  }

  @Post('review')
  async recordReview(@Body() event: NoteReviewEvent) {
    await this.metricsService.recordReview(event);
    return { ok: true };
  }

  @Get('metrics/dashboard')
  async getDashboard() {
    const metrics = await this.metricsService.getMetrics();
    
    return {
      overview: {
        totalNotes: metrics.totalNotes,
        acceptanceRate: metrics.acceptanceRate,
        avgEditDistance: metrics.avgEditDistance,
        avgReviewTime: metrics.avgReviewTime,
      },
      trends: {
        daily: await this.metricsService.getDailyTrends(7),
        hourly: await this.metricsService.getHourlyTrends(24),
      },
      quality: {
        lowEditDistanceRate: metrics.lowEditDistanceRate,
        fastReviewRate: metrics.fastReviewRate,
      },
    };
  }
}
