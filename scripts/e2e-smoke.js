#!/usr/bin/env node
// Simple E2E smoke. Requires stack running via docker-compose.
const fetch = global.fetch || require('node-fetch');
const jwt = require('jsonwebtoken');
if (!process.env.JWT_SECRET) {
  throw new Error('JWT_SECRET must be set for e2e smoke');
}
const secret = process.env.JWT_SECRET;
const token = jwt.sign({ sub: 'tester', roles: ['clinician'] }, secret);
const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` };

(async () => {
  try {
    // Voice: ASR -> LLM -> TTS
    const asr = await fetch('http://localhost:3000/asr/transcribe', {
      method: 'POST', headers,
      body: JSON.stringify({ audio: 'UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=' })
    });
    console.log('ASR status', asr.status);
    // SOAP generate
    const soap = await fetch('http://localhost:3000/soap/generate', {
      method: 'POST', headers,
      body: JSON.stringify({ transcript: 'demo note', sessionId: 'sess-e2e', patientId: 'p1', practitionerId: 'c1' })
    });
    const note = await soap.json();
    console.log('SOAP status', soap.status, note.id);
    // Approve -> FHIR
    if (note.id) {
      const approve = await fetch(`http://localhost:3000/soap/notes/${note.id}/approve`, { method: 'PATCH', headers });
      console.log('Approve status', approve.status);
    }
  } catch (e) {
    console.error('E2E smoke failed', e);
    process.exit(1);
  }
})();
