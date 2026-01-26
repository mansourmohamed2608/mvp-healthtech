// TypeScript declarations for Twilio SDK
declare module 'twilio' {
  export function validateRequest(
    authToken: string,
    twilioSignature: string,
    url: string,
    params: Record<string, any>,
  ): boolean;

  export namespace twiml {
    export class VoiceResponse {
      constructor();
      say(
        message: string,
        options?: { voice?: string; language?: string },
      ): this;
      play(url: string, options?: { loop?: number }): this;
      pause(options?: { length?: number }): this;
      dial(number: string, options?: any): this;
      hangup(): this;
      record(options?: any): this;
      gather(options?: any): this;
      redirect(url: string): this;
      start(): Start;
      toString(): string;
    }

    export class Start {
      stream(options: { url: string; track?: string }): Stream;
    }

    export class Stream {
      parameter(options: { name: string; value: string }): this;
    }
  }
}

export interface TwilioWebhookBody {
  CallSid: string;
  AccountSid: string;
  From: string;
  To: string;
  CallStatus?: string;
  Direction?: string;
  [key: string]: any;
}

export interface TwilioMediaFrame {
  event: 'start' | 'media' | 'stop' | 'mark';
  sequenceNumber?: string;
  streamSid?: string;
  media?: {
    track: string;
    chunk: string;
    timestamp: string;
    payload: string;
  };
  start?: {
    streamSid: string;
    accountSid: string;
    callSid: string;
    tracks: string[];
    mediaFormat: {
      encoding: string;
      sampleRate: number;
      channels: number;
    };
  };
  stop?: {
    accountSid: string;
    callSid: string;
  };
}
