// API Client for HealthTech Backend Services
// Supports both Gateway and Direct Service modes

// Force all calls through the gateway to keep a single contract
const envVars = (import.meta as any)?.env || {};
const USE_DIRECT_SERVICES = false;
const API_BASE_URL = envVars.VITE_API_URL || 'http://localhost:3000';

// Direct service URLs (when not using gateway)
const SERVICE_URLS = {
  asr: envVars.VITE_ASR_URL || 'http://localhost:5000',
  llm: envVars.VITE_LLM_URL || 'http://localhost:5001',
  tts: envVars.VITE_TTS_URL || 'http://localhost:5002',
  soap: envVars.VITE_SOAP_URL || 'http://localhost:5003',
  fhir: envVars.VITE_FHIR_URL || 'http://localhost:5004',
};

class ApiClient {
  private baseUrl: string;
  private useDirect: boolean;

  constructor(baseUrl: string, useDirect = false) {
    this.baseUrl = baseUrl;
    this.useDirect = useDirect;
  }

  private getServiceUrl(service: keyof typeof SERVICE_URLS): string {
    if (this.useDirect) {
      return SERVICE_URLS[service];
    }
    return this.baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const handleError = async (response: Response) => {
      const errorData = await response.json().catch(() => ({}));
      const err: any = new Error(errorData.message || `HTTP error! status: ${response.status}`);
      err.code = errorData.code;
      err.correlationId = errorData.correlationId;
      throw err;
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        await handleError(response);
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  }

  private async serviceRequest<T>(
    service: keyof typeof SERVICE_URLS,
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const baseUrl = this.getServiceUrl(service);
    const url = `${baseUrl}${endpoint}`;
    const handleError = async (response: Response) => {
      const errorData = await response.json().catch(() => ({}));
      const err: any = new Error(errorData.message || `HTTP error! status: ${response.status}`);
      err.code = errorData.code;
      err.correlationId = errorData.correlationId;
      throw err;
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        await handleError(response);
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  }

  // ASR (Automatic Speech Recognition) Service
  async transcribeAudio(audioData: any, callSid?: string, dialect?: string) {
    const endpoint = '/asr/transcribe';
    return this.request<{
      text: string;
      dialect?: string;
      auto_detected?: boolean;
      segments?: Array<{
        text: string;
        start: number;
        end: number;
        speaker?: string;
      }>;
      speakers?: string[];
      roles?: Array<{
        speaker_id: string;
        role: string;
        confidence: number;
        reasoning: string;
      }>;
      primary_doctor?: string;
      primary_patient?: string;
      duration?: number;
      processing_time?: number;
      rtf?: number;
    }>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ audio: audioData, callSid, dialect }),
    });
  }

  async streamAudio(audioData: string, callSid: string, dialect?: string) {
    const endpoint = '/asr/stream';
    return this.request<{ partial?: string; final?: string }>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ audio: audioData, callSid, dialect }),
    });
  }

  // LLM (Language Model) Service
  async inferMessage(message: string, sessionId: string, intent?: string) {
    const endpoint = '/llm/infer';
    return this.request<{ intent: string; reply: string }>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ message, sessionId, intent: intent || 'general' }),
    });
  }

  async generateSOAP(transcript: string) {
    const endpoint = this.useDirect ? '/soap' : '/llm/soap';
    const service = 'llm';

    if (this.useDirect) {
      return this.serviceRequest<{ soap: any }>(
        service,
        endpoint,
        {
          method: 'POST',
          body: JSON.stringify({ transcript }),
        }
      );
    }

    return this.request<{ soap: any }>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ transcript }),
    });
  }

  // TTS (Text-to-Speech) Service
  async synthesizeSpeech(text: string, voice?: string) {
    const endpoint = this.useDirect ? '/synthesize' : '/tts/synthesize';
    const service = 'tts';

    if (this.useDirect) {
      return this.serviceRequest<{ audio: string; duration: number; sampleRate: number }>(
        service,
        endpoint,
        {
          method: 'POST',
          body: JSON.stringify({ text, voice }),
        }
      );
    }

    return this.request<{ audio: string; duration: number; sampleRate: number }>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ text, voice }),
    });
  }

  // SOAP Notes Service
  async createSOAPNote(data: {
    transcript: string;
    sessionId?: string;
    patientContext?: any;
    patientId?: string;
    practitionerId?: string;
    encounterId?: string;
  }) {
    const endpoint = '/soap/generate';
    const service = 'soap';

    if (this.useDirect) {
      return this.serviceRequest<{
        subjective: string;
        objective: string;
        assessment: string;
        plan: string;
        icd_codes?: string[];
        cpt_codes?: string[];
      }>(service, '/generate', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    }

    return this.request<{
      subjective: string;
      objective: string;
      assessment: string;
      plan: string;
      icd_codes?: string[];
      cpt_codes?: string[];
    }>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getSOAPNotes(status?: string, clinicianId?: string) {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (clinicianId) params.append('clinicianId', clinicianId);
    const endpoint = `/soap/notes${params.toString() ? `?${params}` : ''}`;
    return this.request<{ notes: any[] }>(endpoint, {
      method: 'GET',
    });
  }

  async getSOAPNote(id: string) {
    return this.request<any>(`/soap/notes/${id}`, { method: 'GET' });
  }

  async approveSOAPNote(id: string) {
    return this.request<any>(`/soap/notes/${id}/approve`, { method: 'PATCH' });
  }

  async rejectSOAPNote(id: string) {
    return this.request<any>(`/soap/notes/${id}/reject`, { method: 'PATCH' });
  }

  // FHIR Integration Service
  async createFHIRResource(resourceType: string, data: any) {
    const endpoint = `/fhir/${resourceType}`;
    const service = 'fhir';

    if (this.useDirect) {
      return this.serviceRequest<any>(service, `/${resourceType}`, {
        method: 'POST',
        body: JSON.stringify(data),
      });
    }

    return this.request<any>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getFHIRResource(resourceType: string, id: string) {
    const endpoint = `/fhir/${resourceType}/${id}`;
    const service = 'fhir';

    if (this.useDirect) {
      return this.serviceRequest<any>(service, `/${resourceType}/${id}`, {
        method: 'GET',
      });
    }

    return this.request<any>(endpoint, {
      method: 'GET',
    });
  }

  async searchFHIR(resourceType: string, params: Record<string, string>) {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = `/fhir/${resourceType}?${queryString}`;
    const service = 'fhir';

    if (this.useDirect) {
      return this.serviceRequest<any>(service, `/${resourceType}?${queryString}`, {
        method: 'GET',
      });
    }

    return this.request<any>(endpoint, {
      method: 'GET',
    });
  }

  // Clinical Notes
  async getClinicalNotes() {
    return this.request<Array<any>>('/clinical/notes', {
      method: 'GET',
    });
  }

  async createClinicalNote(note: any) {
    return this.request<any>('/clinical/notes', {
      method: 'POST',
      body: JSON.stringify(note),
    });
  }

  // Metrics & Analytics
  async getMetrics() {
    return this.request<any>('/metrics', {
      method: 'GET',
    });
  }

  // Health Check
  async healthCheck() {
    return this.request<{ status: string; services: any }>('/health', {
      method: 'GET',
    });
  }

  // Direct service health checks (for testing)
  async checkASRHealth() {
    if (this.useDirect) {
      return this.serviceRequest<any>('asr', '/health', { method: 'GET' });
    }
    return this.request<any>('/asr/health', { method: 'GET' });
  }

  async checkLLMHealth() {
    if (this.useDirect) {
      return this.serviceRequest<any>('llm', '/health', { method: 'GET' });
    }
    return this.request<any>('/llm/health', { method: 'GET' });
  }

  async checkTTSHealth() {
    if (this.useDirect) {
      return this.serviceRequest<any>('tts', '/health', { method: 'GET' });
    }
    return this.request<any>('/tts/health', { method: 'GET' });
  }

  async checkSOAPHealth() {
    if (this.useDirect) {
      return this.serviceRequest<any>('soap', '/health', { method: 'GET' });
    }
    return this.request<any>('/soap/health', { method: 'GET' });
  }

  async checkFHIRHealth() {
    if (this.useDirect) {
      return this.serviceRequest<any>('fhir', '/health', { method: 'GET' });
    }
    return this.request<any>('/fhir/health', { method: 'GET' });
  }

  // Twilio Voice Service
  async getTwilioToken(identity?: string) {
    return this.request<{ token: string; identity: string }>('/twilio/token', {
      method: 'POST',
      headers: identity ? { 'X-Twilio-Identity': identity } : {},
    });
  }

  // Convenience methods for Clinical Notes page
  async transcribeAudioFile(formData: FormData) {
    const service = 'asr';

    if (this.useDirect) {
      const response = await fetch(`${SERVICE_URLS[service]}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.statusText}`);
      }
      return await response.json();
    }

    const response = await fetch(`${this.baseUrl}/asr/transcribe`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`Transcription failed: ${response.statusText}`);
    }
    return await response.json();
  }

  async generateSoapNote(transcript: string) {
    const endpoint = '/soap/generate';
    const service = 'soap';

    if (this.useDirect) {
      return this.serviceRequest<{
        soapNote?: string;
        soap?: string;
        subjective: string;
        objective: string;
        assessment: string;
        plan: string;
      }>(service, '/generate', {
        method: 'POST',
        body: JSON.stringify({ transcript }),
      });
    }

    return this.request<{
      soapNote?: string;
      soap?: string;
      subjective: string;
      objective: string;
      assessment: string;
      plan: string;
    }>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ transcript }),
    });
  }

  async convertToFHIR(data: {
    soapNote: any;
    patientId: string;
    practitionerId: string;
    sessionId: string;
  }) {
    const endpoint = '/fhir/convert';
    const service = 'fhir';

    if (this.useDirect) {
      return this.serviceRequest<{ documentReferenceId: string }>(service, '/convert', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    }

    return this.request<{ documentReferenceId: string }>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

export const api = new ApiClient(API_BASE_URL, USE_DIRECT_SERVICES);
export default api;
