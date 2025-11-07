# 🚀 QUICK START - Testing Frontend with Backend

## ⚡ Start Everything (2 Commands)

```powershell
# Terminal 1: Start ALL backend services
cd D:\Downloads\HealthTech\mvp-healthtech\infra
docker-compose up -d

# Terminal 2: Start frontend (if not already running)
cd D:\Downloads\HealthTech\mvp-healthtech\frontend-vite
npx vite
```

## 🎯 Main Testing URL

Open in browser: **http://localhost:3000/demo** (or 5173)

This one page lets you test ALL 5 backend services!

## 🧪 Test Each Service (In Order)

### 1. Test ASR (Voice to Text)
- Click "Voice to Text" tab
- Select dialect (Egyptian recommended)
- Enter text: "Patient has chest pain for 3 days"
- Click "Test ASR Service"
- ✅ Should see transcription in Response panel

### 2. Test LLM (AI Assistant)
- Click "AI Assistant" tab
- Enter question: "What are the symptoms of diabetes?"
- Click "Test LLM Service"
- ✅ Should see AI response with confidence score

### 3. Test SOAP Notes
- Click "SOAP Notes" tab
- Fields are pre-filled with medical example
- Click "Generate SOAP Note"
- ✅ Should see structured SOAP note in Response

### 4. Test FHIR Integration
- Click "FHIR Integration" tab
- Select "Patient" resource type
- Edit JSON data if needed
- Click "Create FHIR Resource"
- ✅ Should see created FHIR resource

### 5. Test TTS (Text to Speech)
- Click "Text to Speech" tab
- Enter text: "Hello, this is a test"
- Click "Synthesize Speech"
- ✅ Should see audio player and hear voice

## ❌ If You See Errors

### "Failed to fetch" or "Network Error"
**Problem:** Backend not running

**Solution:**
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\infra
docker-compose up -d
docker ps  # Check all services are running
```

### "CORS Error"
**Problem:** Gateway doesn't allow frontend origin

**Solution:** Check `gateway/src/main.ts` has:
```typescript
app.enableCors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true,
});
```

### Service-specific errors
Check individual service logs:
```powershell
docker-compose logs gateway
docker-compose logs asr
docker-compose logs llm
docker-compose logs tts
docker-compose logs soap
docker-compose logs fhir
```

## 🎨 Other Pages to Check

| URL | What It Shows |
|-----|---------------|
| `http://localhost:3000/` | Home page with hero |
| `http://localhost:3000/features` | All features overview |
| `http://localhost:3000/dashboard` | Metrics & analytics |
| `http://localhost:3000/demo` | **⭐ Test all services** |

## 🔍 How to Debug

### 1. Open Browser DevTools (F12)
- **Console Tab**: See API calls and errors
- **Network Tab**: See all HTTP requests
- Filter by "XHR" to see only API calls

### 2. Check Request Details
- Click any request in Network tab
- See full URL, headers, payload, response

### 3. Test Backend Directly
```powershell
# Test gateway health
curl http://localhost:3000/health

# Test specific service
curl http://localhost:5000/health  # ASR
curl http://localhost:5001/health  # LLM
```

## ✅ Success Indicators

You know it's working when you see:
- ✅ Green checkmark in Response panel
- ✅ JSON response with data
- ✅ No red error messages
- ✅ Console shows `200 OK` status
- ✅ Services show "Online" status on Dashboard

## 🎯 5-Minute Test Workflow

1. **Start services**: `docker-compose up -d` (30 sec)
2. **Open Demo page**: `http://localhost:3000/demo`
3. **Test ASR**: Select dialect → Test → See result (30 sec)
4. **Test LLM**: Ask question → Test → See AI answer (30 sec)
5. **Test SOAP**: Pre-filled → Generate → See note (30 sec)
6. **Test FHIR**: Select Patient → Create → See resource (30 sec)
7. **Test TTS**: Enter text → Synthesize → Hear audio (30 sec)
8. **Check Dashboard**: See all metrics and status (1 min)

**Total: 5 minutes to test everything!** ⚡

## 📚 Full Documentation

- `TESTING_GUIDE.md` - Detailed testing instructions
- `INTEGRATION_SUMMARY.md` - Complete feature list
- `README.md` - Project overview
- `QUICKSTART.md` - Setup guide

## 🆘 Still Not Working?

1. Check all Docker containers: `docker ps`
2. Restart services: `docker-compose restart`
3. Check logs: `docker-compose logs --tail=50`
4. Verify ports: `netstat -an | findstr "3000 5000 5001 5002 5003 5004"`
5. Restart frontend: Stop Vite (Ctrl+C) → `npx vite`

## 💡 Pro Tips

- Use Demo page for quick testing
- Watch Network tab to see actual API calls
- Check Console for detailed error messages
- Dashboard shows if services are online
- Try both light and dark themes!
- Switch to Arabic to test RTL layout

---

**You're all set!** Open `http://localhost:3000/demo` and start testing! 🎉
