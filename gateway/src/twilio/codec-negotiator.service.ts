// gateway/src/twilio/codec-negotiator.service.ts
import { Injectable, Logger } from '@nestjs/common';

export enum Codec {
  OPUS = 'opus',
  PCMU = 'pcmu',
  PCMA = 'pcma',
}

interface NetworkConditions {
  bandwidth: number; // kbps
  latency: number; // ms
  packetLoss: number; // percentage 0-100
}

interface CodecPreference {
  codec: Codec;
  priority: number;
  bitrate?: number;
  sampleRate: number;
}

@Injectable()
export class CodecNegotiatorService {
  private readonly logger = new Logger(CodecNegotiatorService.name);

  // Codec specifications
  private readonly codecSpecs = {
    [Codec.OPUS]: {
      minBandwidth: 6, // kbps
      maxBandwidth: 510, // kbps
      adaptiveBitrate: true,
      sampleRates: [8000, 12000, 16000, 24000, 48000],
      latency: 'low', // 20-40ms
      complexity: 'medium',
      quality: 'excellent',
    },
    [Codec.PCMU]: {
      bandwidth: 64, // kbps (fixed)
      adaptiveBitrate: false,
      sampleRates: [8000],
      latency: 'very-low', // <10ms
      complexity: 'low',
      quality: 'good',
    },
    [Codec.PCMA]: {
      bandwidth: 64, // kbps (fixed)
      adaptiveBitrate: false,
      sampleRates: [8000],
      latency: 'very-low', // <10ms
      complexity: 'low',
      quality: 'good',
    },
  };

  /**
   * Select best codec based on network conditions
   */
  selectCodec(conditions?: Partial<NetworkConditions>): CodecPreference[] {
    const network: NetworkConditions = {
      bandwidth: conditions?.bandwidth ?? 100, // Default 100 kbps
      latency: conditions?.latency ?? 50, // Default 50ms
      packetLoss: conditions?.packetLoss ?? 1, // Default 1%
    };

    this.logger.log(
      `Selecting codec for conditions: ${JSON.stringify(network)}`,
    );

    const preferences: CodecPreference[] = [];

    // Decision logic based on conditions
    if (this.shouldUseOpus(network)) {
      preferences.push({
        codec: Codec.OPUS,
        priority: 1,
        bitrate: this.getOptimalOpusBitrate(network),
        sampleRate: this.getOptimalSampleRate(network),
      });
      // PCMU as fallback
      preferences.push({
        codec: Codec.PCMU,
        priority: 2,
        sampleRate: 8000,
      });
    } else {
      // Use PCMU for poor networks or low-latency requirements
      preferences.push({
        codec: Codec.PCMU,
        priority: 1,
        sampleRate: 8000,
      });
      // Opus as fallback if network improves
      preferences.push({
        codec: Codec.OPUS,
        priority: 2,
        bitrate: 16, // Low bitrate for fallback
        sampleRate: 16000,
      });
    }

    this.logger.log(
      `Codec preferences: ${preferences.map((p) => `${p.codec}(${p.priority})`).join(', ')}`,
    );
    return preferences;
  }

  /**
   * Determine if Opus should be used
   */
  private shouldUseOpus(network: NetworkConditions): boolean {
    // Use Opus if:
    // 1. Good bandwidth (> 20 kbps available)
    // 2. Acceptable latency (< 100ms)
    // 3. Low packet loss (< 5%)

    const hasGoodBandwidth = network.bandwidth > 20;
    const hasAcceptableLatency = network.latency < 100;
    const hasLowPacketLoss = network.packetLoss < 5;

    return hasGoodBandwidth && hasAcceptableLatency && hasLowPacketLoss;
  }

  /**
   * Get optimal Opus bitrate based on bandwidth
   */
  private getOptimalOpusBitrate(network: NetworkConditions): number {
    const { bandwidth, packetLoss } = network;

    // Reserve 30% of bandwidth for overhead
    const availableBandwidth = bandwidth * 0.7;

    // Adjust for packet loss (use lower bitrate if high loss)
    const lossAdjustment = Math.max(0.5, 1 - packetLoss / 20);
    const targetBitrate = availableBandwidth * lossAdjustment;

    // Clamp to valid range for speech
    // Speech optimal range: 16-32 kbps
    // Wideband speech: 24-40 kbps
    if (targetBitrate < 16) return 16;
    if (targetBitrate > 40) return 40;
    return Math.round(targetBitrate);
  }

  /**
   * Get optimal sample rate based on conditions
   */
  private getOptimalSampleRate(network: NetworkConditions): number {
    const { bandwidth, latency } = network;

    // Higher sample rates need more bandwidth and add latency
    if (bandwidth > 50 && latency < 80) {
      return 24000; // Wideband for good conditions
    } else if (bandwidth > 30) {
      return 16000; // Standard wideband
    } else {
      return 8000; // Narrowband for poor conditions
    }
  }

  /**
   * Convert codec enum to Twilio Call.Codec
   */
  toTwilioCodec(codec: Codec): string {
    switch (codec) {
      case Codec.OPUS:
        return 'opus';
      case Codec.PCMU:
        return 'pcmu';
      case Codec.PCMA:
        return 'pcma';
      default:
        return 'opus';
    }
  }

  /**
   * Get codec list for Twilio Device initialization
   */
  getTwilioCodecPreferences(conditions?: Partial<NetworkConditions>): string[] {
    const preferences = this.selectCodec(conditions);
    return preferences
      .sort((a, b) => a.priority - b.priority)
      .map((p) => this.toTwilioCodec(p.codec));
  }

  /**
   * Estimate bandwidth usage for a codec
   */
  estimateBandwidth(codec: Codec, bitrate?: number): number {
    switch (codec) {
      case Codec.OPUS:
        // Opus: bitrate + ~20% overhead for RTP/UDP/IP
        return (bitrate || 24) * 1.2;
      case Codec.PCMU:
      case Codec.PCMA:
        // PCMU/PCMA: 64 kbps + ~20% overhead
        return 64 * 1.2;
      default:
        return 80; // Default estimate
    }
  }

  /**
   * Get codec info for monitoring
   */
  getCodecInfo(codec: Codec) {
    return this.codecSpecs[codec] || null;
  }

  /**
   * Monitor and log codec performance
   */
  logCodecMetrics(
    codec: Codec,
    metrics: {
      actualBitrate?: number;
      jitter?: number;
      packetLoss?: number;
      roundTripTime?: number;
    },
  ): void {
    this.logger.log(
      `Codec: ${codec} | ` +
        `Bitrate: ${metrics.actualBitrate || 'N/A'} kbps | ` +
        `Jitter: ${metrics.jitter || 'N/A'} ms | ` +
        `Loss: ${metrics.packetLoss || 'N/A'}% | ` +
        `RTT: ${metrics.roundTripTime || 'N/A'} ms`,
    );
  }

  /**
   * Recommend codec change based on runtime metrics
   */
  shouldChangeCodec(
    currentCodec: Codec,
    metrics: {
      packetLoss: number;
      bandwidth: number;
      latency: number;
    },
  ): { shouldChange: boolean; recommendedCodec?: Codec; reason?: string } {
    // If using Opus and experiencing high packet loss
    if (currentCodec === Codec.OPUS && metrics.packetLoss > 5) {
      return {
        shouldChange: true,
        recommendedCodec: Codec.PCMU,
        reason: 'High packet loss detected, switching to PCMU for reliability',
      };
    }

    // If using PCMU and conditions improved
    if (
      currentCodec === Codec.PCMU &&
      metrics.packetLoss < 2 &&
      metrics.bandwidth > 30 &&
      metrics.latency < 80
    ) {
      return {
        shouldChange: true,
        recommendedCodec: Codec.OPUS,
        reason:
          'Network conditions improved, switching to Opus for better quality',
      };
    }

    return { shouldChange: false };
  }
}
