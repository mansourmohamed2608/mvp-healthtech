# ✅ Service Connection Checklist

## Before Starting Frontend

### Backend Services (Python FastAPI)
- [ ] **ASR** running on http://localhost:5000
  - Check: `curl http://localhost:5000/health`
  - Terminal message: "Starting ASR service on http://0.0.0.0:5000..."

- [ ] **LLM** running on http://localhost:5001
  - Check: `curl http://localhost:5001/health`
  - Terminal message: "Model loaded successfully!"

- [ ] **TTS** running on http://localhost:5002
  - Check: `curl http://localhost:5002/health`
  - Terminal message: "TTS Service ready"

- [ ] **SOAP** running on http://localhost:5003
  - Check: `curl http://localhost:5003/health`
  - Terminal message: "Application startup complete"

- [ ] **FHIR** running on http://localhost:5004
  - Check: `curl http://localhost:5004/health`
  - Terminal message: "Application startup complete"

### Gateway (Optional - NestJS)
- [ ] **Gateway** running on http://localhost:3001
  - Check: `curl http://localhost:3001/health`
  - Terminal message: "Gateway listening on port 3001"
  - Note: Only needed if using gateway mode

### Configuration
- [ ] `frontend-vite/.env` configured
  - [ ] `VITE_USE_DIRECT_SERVICES` set (true or false)
  - [ ] Service URLs configured
  - [ ] Gateway URL configured (if using gateway)

- [ ] `gateway/.env` configured (if using gateway)
  - [ ] All service URLs point to correct ports
  - [ ] PORT set to 3001

## Starting Frontend

### Installation
- [ ] Navigate to `frontend-vite` directory
- [ ] Run `pnpm install` (first time only)
- [ ] Dependencies installed successfully

### Launch
- [ ] Run `pnpm run dev` or `.\start-frontend.ps1`
- [ ] Frontend starts on http://localhost:5173
- [ ] No compilation errors
- [ ] Browser opens automatically

## Testing Connections

### Via Browser
- [ ] Navigate to http://localhost:5173/test
- [ ] Click "Check All Services"
- [ ] All services show green/online status
- [ ] Run individual functional tests
- [ ] All tests pass successfully

### Via Code
```typescript
// In browser console at http://localhost:5173
import api from './src/utils/api';

// Test each service
await api.checkASRHealth();
await api.checkLLMHealth();
await api.checkTTSHealth();
await api.checkSOAPHealth();
await api.checkFHIRHealth();
```

## Common Issues & Solutions

### ❌ Service shows "offline"
- **Solution**: Check if service terminal shows success message
- **Solution**: Verify service is running on correct port
- **Solution**: Test direct URL in browser

### ❌ CORS error
- **Solution**: Service might not be running
- **Solution**: Check service has CORS enabled (should be by default)
- **Solution**: Verify URL in .env matches actual port

### ❌ "Connection refused"
- **Solution**: Service not started yet
- **Solution**: Port blocked by firewall
- **Solution**: Check port not used by another application

### ❌ Gateway can't reach services
- **Solution**: Verify `gateway/.env` has correct URLs
- **Solution**: Services must be running before gateway starts
- **Solution**: Try direct mode instead

### ❌ Frontend won't compile
- **Solution**: Run `pnpm install` again
- **Solution**: Delete `node_modules` and reinstall
- **Solution**: Check Node.js version (should be 18+)

## Quick Commands

### Check All Ports (PowerShell)
```powershell
netstat -an | Select-String "5000|5001|5002|5003|5004|3001|5173"
```

### Test All Services
```powershell
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
curl http://localhost:5004/health
curl http://localhost:3001/health
```

### Kill Port (if stuck)
```powershell
# Find process on port
Get-NetTCPConnection -LocalPort 5000 | Select-Object -Property OwningProcess

# Kill process
Stop-Process -Id <ProcessId> -Force
```

## Success Indicators

### You're ready when:
✅ All service health checks return 200 OK
✅ Test page shows all services online
✅ Functional tests pass
✅ No errors in browser console
✅ No errors in service terminals

### You can now:
✅ Use the main application features
✅ Test voice transcription
✅ Generate SOAP notes
✅ Integrate with FHIR
✅ Use all API methods in your components

---

**Current Progress:**
- [x] ASR Service: Running on 5000 ✅
- [ ] LLM Service: Not started yet
- [ ] TTS Service: Not started yet
- [ ] SOAP Service: Not started yet
- [ ] FHIR Service: Not started yet
- [ ] Gateway: Not started yet
- [ ] Frontend: Not started yet

**Next Step:** Start LLM service in a new terminal!
