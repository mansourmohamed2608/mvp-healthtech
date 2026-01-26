// gateway/src/clinical/clinical-metrics.service.ts
/**
 * Clinical Notes Metrics Service
 * Week 5 Day 30 - Tracks quality and acceptance metrics
 */
import { Injectable } from '@nestjs/common';

interface ReviewMetric {
  recordingId: string;
  accepted: boolean;
  editDistance: number;
  timeToReview: number;
  clinicianId?: string;
  timestamp: Date;
}

@Injectable()
export class ClinicalMetricsService {
  private reviews: ReviewMetric[] = [];

  // Thresholds for quality metrics
  private readonly LOW_EDIT_THRESHOLD = 50;
  private readonly FAST_REVIEW_THRESHOLD = 120; // 2 minutes

  async recordReview(event: {
    recordingId: string;
    accepted: boolean;
    editDistance: number;
    timeToReview: number;
    clinicianId?: string;
  }) {
    this.reviews.push({
      ...event,
      timestamp: new Date(),
    });

    // Keep only last 10,000 reviews in memory
    if (this.reviews.length > 10000) {
      this.reviews = this.reviews.slice(-10000);
    }
  }

  async getMetrics() {
    if (this.reviews.length === 0) {
      return {
        totalNotes: 0,
        acceptanceRate: 0,
        avgEditDistance: 0,
        avgReviewTime: 0,
        lowEditDistanceRate: 0,
        fastReviewRate: 0,
      };
    }

    const totalNotes = this.reviews.length;
    const accepted = this.reviews.filter((r) => r.accepted).length;
    const acceptanceRate = (accepted / totalNotes) * 100;

    const totalEditDistance = this.reviews.reduce(
      (sum, r) => sum + r.editDistance,
      0,
    );
    const avgEditDistance = totalEditDistance / totalNotes;

    const totalReviewTime = this.reviews.reduce(
      (sum, r) => sum + r.timeToReview,
      0,
    );
    const avgReviewTime = totalReviewTime / totalNotes;

    const lowEditCount = this.reviews.filter(
      (r) => r.editDistance <= this.LOW_EDIT_THRESHOLD,
    ).length;
    const lowEditDistanceRate = (lowEditCount / totalNotes) * 100;

    const fastReviewCount = this.reviews.filter(
      (r) => r.timeToReview <= this.FAST_REVIEW_THRESHOLD,
    ).length;
    const fastReviewRate = (fastReviewCount / totalNotes) * 100;

    return {
      totalNotes,
      acceptanceRate: Math.round(acceptanceRate * 10) / 10,
      avgEditDistance: Math.round(avgEditDistance * 10) / 10,
      avgReviewTime: Math.round(avgReviewTime * 10) / 10,
      lowEditDistanceRate: Math.round(lowEditDistanceRate * 10) / 10,
      fastReviewRate: Math.round(fastReviewRate * 10) / 10,
    };
  }

  async getDailyTrends(days: number) {
    const now = new Date();
    const trends: Array<{
      date: string;
      count: number;
      acceptanceRate: number;
      avgEditDistance: number;
    }> = [];

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      date.setHours(0, 0, 0, 0);

      const nextDate = new Date(date);
      nextDate.setDate(nextDate.getDate() + 1);

      const dayReviews = this.reviews.filter(
        (r) => r.timestamp >= date && r.timestamp < nextDate,
      );

      if (dayReviews.length > 0) {
        const accepted = dayReviews.filter((r) => r.accepted).length;
        const avgEdit =
          dayReviews.reduce((sum, r) => sum + r.editDistance, 0) /
          dayReviews.length;

        trends.push({
          date: date.toISOString().split('T')[0],
          count: dayReviews.length,
          acceptanceRate:
            Math.round((accepted / dayReviews.length) * 1000) / 10,
          avgEditDistance: Math.round(avgEdit * 10) / 10,
        });
      } else {
        trends.push({
          date: date.toISOString().split('T')[0],
          count: 0,
          acceptanceRate: 0,
          avgEditDistance: 0,
        });
      }
    }

    return trends;
  }

  async getHourlyTrends(hours: number) {
    const now = new Date();
    const trends: Array<{
      hour: string;
      count: number;
      accepted: number;
    }> = [];

    for (let i = hours - 1; i >= 0; i--) {
      const hour = new Date(now);
      hour.setHours(hour.getHours() - i, 0, 0, 0);

      const nextHour = new Date(hour);
      nextHour.setHours(nextHour.getHours() + 1);

      const hourReviews = this.reviews.filter(
        (r) => r.timestamp >= hour && r.timestamp < nextHour,
      );

      trends.push({
        hour: hour.toISOString(),
        count: hourReviews.length,
        accepted: hourReviews.filter((r) => r.accepted).length,
      });
    }

    return trends;
  }
}
