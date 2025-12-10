export function validateEnv() {
  const required = [
    'JWT_SECRET',
    'INTERNAL_SECRET',
    'ASR_SERVICE_URL',
    'LLM_SERVICE_URL',
    'TTS_SERVICE_URL',
    'SOAP_SERVICE_URL',
    'FHIR_SERVICE_URL',
    'TWILIO_AUTH_TOKEN',
    'TWILIO_ACCOUNT_SID',
    'TWILIO_API_SECRET',
    'FHIR_BASE_URL',
  ];
  const missing = required.filter((key) => !process.env[key]);
  if (missing.length) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }

  if ((process.env.SOAP_ASYNC_ENABLED || '').toLowerCase() === 'true' && !process.env.SOAP_QUEUE_URL) {
    throw new Error('SOAP_ASYNC_ENABLED is true but SOAP_QUEUE_URL is not set');
  }
}
