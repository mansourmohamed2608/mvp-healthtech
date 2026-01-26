// gateway/src/audio/audio-processor.service.ts
import { Injectable, Logger } from '@nestjs/common';

interface AudioProcessingOptions {
  targetVolume?: number; // dB level (-20 to 0)
  removeSilence?: boolean;
  silenceThreshold?: number; // 0-1
  applyNoiseReduction?: boolean;
  normalizeVolume?: boolean;
}

@Injectable()
export class AudioProcessorService {
  private readonly logger = new Logger(AudioProcessorService.name);

  // Default processing options for medical audio
  private readonly defaultOptions: AudioProcessingOptions = {
    targetVolume: -16, // Target -16 dB for speech
    removeSilence: true,
    silenceThreshold: 0.01, // 1% amplitude threshold
    applyNoiseReduction: true,
    normalizeVolume: true,
  };

  /**
   * Process audio buffer with normalization, silence removal, and filtering
   */
  async processAudio(
    audioBuffer: Buffer,
    options: Partial<AudioProcessingOptions> = {},
  ): Promise<Buffer> {
    const opts = { ...this.defaultOptions, ...options };

    try {
      let processedBuffer = audioBuffer;

      // Step 1: Normalize volume
      if (opts.normalizeVolume) {
        processedBuffer = this.normalizeVolume(
          processedBuffer,
          opts.targetVolume!,
        );
      }

      // Step 2: Remove silence
      if (opts.removeSilence) {
        processedBuffer = this.removeSilence(
          processedBuffer,
          opts.silenceThreshold!,
        );
      }

      // Step 3: Apply noise reduction (basic high-pass filter)
      if (opts.applyNoiseReduction) {
        processedBuffer = this.applyNoiseReduction(processedBuffer);
      }

      this.logger.debug(
        `Audio processed: ${audioBuffer.length} -> ${processedBuffer.length} bytes`,
      );
      return processedBuffer;
    } catch (error) {
      this.logger.error('Error processing audio', error);
      // Return original buffer on error
      return audioBuffer;
    }
  }

  /**
   * Normalize audio volume to target dB level
   * Assumes 16-bit PCM audio
   */
  private normalizeVolume(buffer: Buffer, targetDb: number): Buffer {
    const samples = new Int16Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.length / 2,
    );

    // Calculate RMS (Root Mean Square) of the audio
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      const normalized = samples[i] / 32768; // Normalize to -1 to 1
      sumSquares += normalized * normalized;
    }
    const rms = Math.sqrt(sumSquares / samples.length);

    if (rms === 0) {
      return buffer; // Silence, no normalization needed
    }

    // Calculate current dB level
    const currentDb = 20 * Math.log10(rms);

    // Calculate gain needed to reach target
    const gainDb = targetDb - currentDb;
    const gainLinear = Math.pow(10, gainDb / 20);

    // Apply gain (with limiting to prevent clipping)
    const maxGain = 4.0; // Limit gain to prevent excessive amplification
    const actualGain = Math.min(gainLinear, maxGain);

    const outputSamples = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const amplified = samples[i] * actualGain;
      // Soft clipping to prevent harsh distortion
      outputSamples[i] = Math.max(
        -32768,
        Math.min(32767, Math.round(amplified)),
      );
    }

    return Buffer.from(outputSamples.buffer);
  }

  /**
   * Remove silent sections from audio
   * Keeps only segments above the silence threshold
   */
  private removeSilence(buffer: Buffer, threshold: number): Buffer {
    const samples = new Int16Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.length / 2,
    );
    const chunkSize = 160; // 10ms at 16kHz
    const keptChunks: Int16Array[] = [];

    for (let i = 0; i < samples.length; i += chunkSize) {
      const chunk = samples.slice(i, Math.min(i + chunkSize, samples.length));

      // Calculate chunk energy
      let sumSquares = 0;
      for (let j = 0; j < chunk.length; j++) {
        const normalized = chunk[j] / 32768;
        sumSquares += normalized * normalized;
      }
      const rms = Math.sqrt(sumSquares / chunk.length);

      // Keep chunk if above threshold
      if (rms > threshold) {
        keptChunks.push(chunk);
      }
    }

    // Combine kept chunks
    if (keptChunks.length === 0) {
      return buffer; // All silence, return original
    }

    const totalLength = keptChunks.reduce(
      (sum, chunk) => sum + chunk.length,
      0,
    );
    const result = new Int16Array(totalLength);
    let offset = 0;
    for (const chunk of keptChunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return Buffer.from(result.buffer);
  }

  /**
   * Apply basic noise reduction (high-pass filter)
   * Removes low-frequency noise below 80 Hz
   */
  private applyNoiseReduction(buffer: Buffer): Buffer {
    const samples = new Int16Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.length / 2,
    );

    // Simple first-order high-pass filter
    // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
    // alpha = RC / (RC + dt), where fc = 1/(2*pi*RC)

    const sampleRate = 16000; // Assume 16kHz
    const cutoffFreq = 80; // Hz
    const rc = 1 / (2 * Math.PI * cutoffFreq);
    const dt = 1 / sampleRate;
    const alpha = rc / (rc + dt);

    const filtered = new Int16Array(samples.length);
    let prevInput = samples[0];
    let prevOutput = 0;

    for (let i = 0; i < samples.length; i++) {
      const input = samples[i] / 32768; // Normalize
      const output = alpha * (prevOutput + input - prevInput);

      filtered[i] = Math.max(
        -32768,
        Math.min(32767, Math.round(output * 32768)),
      );

      prevInput = input;
      prevOutput = output;
    }

    return Buffer.from(filtered.buffer);
  }

  /**
   * Convert mulaw to PCM 16-bit
   * Twilio sends mulaw, we need PCM for processing
   */
  mulawToPcm(mulawBuffer: Buffer): Buffer {
    const pcmSamples = new Int16Array(mulawBuffer.length);

    for (let i = 0; i < mulawBuffer.length; i++) {
      pcmSamples[i] = this.mulawDecode(mulawBuffer[i]);
    }

    return Buffer.from(pcmSamples.buffer);
  }

  /**
   * Convert PCM 16-bit to mulaw
   * Convert back to mulaw for Twilio
   */
  pcmToMulaw(pcmBuffer: Buffer): Buffer {
    const samples = new Int16Array(
      pcmBuffer.buffer,
      pcmBuffer.byteOffset,
      pcmBuffer.length / 2,
    );
    const mulawBuffer = Buffer.alloc(samples.length);

    for (let i = 0; i < samples.length; i++) {
      mulawBuffer[i] = this.mulawEncode(samples[i]);
    }

    return mulawBuffer;
  }

  /**
   * Mulaw decode (8-bit mulaw to 16-bit PCM)
   */
  private mulawDecode(mulaw: number): number {
    mulaw = ~mulaw;
    const sign = mulaw & 0x80;
    const exponent = (mulaw >> 4) & 0x07;
    const mantissa = mulaw & 0x0f;

    let sample = ((mantissa << 3) + 0x84) << exponent;
    if (sign !== 0) sample = -sample;

    return sample;
  }

  /**
   * Mulaw encode (16-bit PCM to 8-bit mulaw)
   */
  private mulawEncode(pcm: number): number {
    const sign = pcm < 0 ? 0x80 : 0x00;
    let magnitude = Math.abs(pcm);

    magnitude += 0x84;
    if (magnitude > 0x7fff) magnitude = 0x7fff;

    let exponent = 7;
    for (let exp = 0; exp < 8; exp++) {
      if (magnitude <= 0xff << exp) {
        exponent = exp;
        break;
      }
    }

    const mantissa = (magnitude >> (exponent + 3)) & 0x0f;
    return ~(sign | (exponent << 4) | mantissa);
  }

  /**
   * Analyze audio quality metrics
   */
  async analyzeAudio(buffer: Buffer): Promise<{
    duration: number; // seconds
    averageVolume: number; // dB
    peakVolume: number; // dB
    silenceRatio: number; // 0-1
    clippingRatio: number; // 0-1
  }> {
    const samples = new Int16Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.length / 2,
    );
    const sampleRate = 16000; // Assume 16kHz

    let sumSquares = 0;
    let peakAmplitude = 0;
    let silentSamples = 0;
    let clippedSamples = 0;
    const silenceThreshold = 0.01;

    for (let i = 0; i < samples.length; i++) {
      const normalized = Math.abs(samples[i]) / 32768;
      sumSquares += normalized * normalized;
      peakAmplitude = Math.max(peakAmplitude, normalized);

      if (normalized < silenceThreshold) silentSamples++;
      if (Math.abs(samples[i]) >= 32767) clippedSamples++;
    }

    const rms = Math.sqrt(sumSquares / samples.length);
    const avgDb = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
    const peakDb =
      peakAmplitude > 0 ? 20 * Math.log10(peakAmplitude) : -Infinity;

    return {
      duration: samples.length / sampleRate,
      averageVolume: avgDb,
      peakVolume: peakDb,
      silenceRatio: silentSamples / samples.length,
      clippingRatio: clippedSamples / samples.length,
    };
  }
}
