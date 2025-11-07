# 🎤 Voice Agent Setup Guide

## Problem: "Application Error" on Twilio Call

**What's happening:**
- ✅ Frontend works (call connects)
- ✅ Gateway works (token generated)
- ✅ Twilio works (call starts)
- ❌ **Twilio can't send audio to gateway** (gateway is on localhost, Twilio is on internet)

## Solution: Expose Gateway with ngrok

### Step 1: Install ngrok

```powershell
# Download from https://ngrok.com/download
# Or use winget:
winget install ngrok
```

### Step 2: Start ngrok

```powershell
# Expose gateway port 3001
ngrok http 3001
```

You'll see output like:
```
Session Status                online
Account                       Your Name (Plan: Free)
Version                       3.5.0
Region                        United States (us)
Latency                       45ms
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abcd-1234-5678.ngrok-free.app -> http://localhost:3001
```

### Step 3: Update .env with ngrok URL

Copy the **https** URL from ngrok (e.g., `https://abcd-1234-5678.ngrok-free.app`)

```properties
# In gateway/.env
GATEWAY_PUBLIC_URL=wss://abcd-1234-5678.ngrok-free.app

# ⚠️ Change https:// to wss:// (WebSocket Secure)
```

### Step 4: Update Twilio TwiML App

1. Go to https://console.twilio.com/us1/develop/voice/manage/twiml-apps
2. Find your TwiML App (SID: `APd8a04cfcf25c57ff10019f304256b583`)
3. Set **Voice Request URL** to:
   ```
   https://abcd-1234-5678.ngrok-free.app/twilio/voice/start
   ```
4. Set **HTTP Method** to: `POST`
5. Click **Save**

### Step 5: Restart Gateway

```powershell
cd gateway
pnpm run start:dev
```

### Step 6: Test Voice Call

1. Open frontend: http://localhost:5173/voice-agent
2. Click "Start Call"
3. Speak in Arabic: "عندي صداع"
4. AI should respond!

---

## Why This is Needed

```
┌─────────────────────────────────────────────────────────┐
│                     HOW IT WORKS                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Frontend]                                             │
│  localhost:5173 ──┐                                     │
│                   │                                     │
│                   ▼                                     │
│  [Gateway]        ┌──────────────────┐                 │
│  localhost:3001 ◄─┤  Twilio Voice    │                 │
│      ▲            │  (needs PUBLIC   │                 │
│      │            │   URL to send    │                 │
│      │            │   audio stream)  │                 │
│      │            └──────────────────┘                 │
│      │                   ▲                             │
│      │                   │                             │
│  [ngrok]                 │                             │
│  public URL ─────────────┘                             │
│  wss://abc.ngrok-free.app                              │
│                                                         │
│  [Python Services]                                     │
│  ASR: localhost:5000                                   │
│  LLM: localhost:5001                                   │
│  TTS: localhost:5002                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Without ngrok:**
- Twilio tries to connect to `wss://your-domain.ngrok.io`
- This is NOT a real URL
- Twilio can't send audio
- Error: "Application error has occurred"

**With ngrok:**
- Twilio connects to `wss://abcd-1234.ngrok-free.app`
- ngrok forwards to `localhost:3001`
- Gateway receives audio
- ✅ Everything works!

---

## Alternative: Use Twilio's Built-in Testing

If you don't want to use ngrok, you can test locally with Twilio's test credentials:

1. Use **Twilio Test Credentials** (no real calls):
   - Test Account SID: `AC...test...`
   - Test Auth Token: `test_token`

2. Or use **Twilio Simulator**: https://www.twilio.com/console/voice/calls/simulator

---

## Troubleshooting

### "ngrok not found"
```powershell
# Add to PATH or use full path:
C:\path\to\ngrok.exe http 3001
```

### "Gateway still shows old URL"
```powershell
# Restart gateway after updating .env:
cd gateway
pnpm run start:dev
```

### "Call connects but no AI response"
Check logs:
```powershell
# Gateway logs (Terminal: node)
# Should show:
# ✅ "WebSocket connected: CAxxxx"
# ✅ "Stream started: MZxxxx for call: CAxxxx"
# ✅ "Transcribed (CAxxxx): عندي صداع"
# ✅ "LLM response (CAxxxx): يمكن علاج الصداع..."
```

### "Redis errors"
Redis is optional, gateway works without it:
```
❌ Redis Client Error: ECONNREFUSED
✅ Gateway still works (uses in-memory storage)
```

---

## Quick Commands

```powershell
# Terminal 1: Start ngrok
ngrok http 3001

# Terminal 2: Start Gateway (after updating .env)
cd gateway
pnpm run start:dev

# Terminal 3: Start ASR
cd services/asr
python app.py

# Terminal 4: Start LLM
cd services/llm
python app.py

# Terminal 5: Start TTS
cd services/tts
python app.py

# Terminal 6: Start Frontend
cd frontend-vite
pnpm run dev
```

---

## Expected Behavior

### 1. Call Starts
```
User clicks "Start Call"
→ Frontend requests token from gateway
→ Gateway generates Twilio token
→ Frontend connects to Twilio
→ Twilio plays: "مرحبا بك في النظام الصحي"
```

### 2. User Speaks
```
User says: "عندي صداع"
→ Twilio sends audio via WebSocket to gateway
→ Gateway forwards to ASR service
→ ASR transcribes: "عندي صداع"
```

### 3. AI Responds
```
Gateway sends transcript to LLM
→ LLM responds: "يمكن علاج الصداع بـ..."
→ Gateway sends text to TTS
→ TTS synthesizes Arabic voice
→ Gateway sends audio to Twilio
→ User hears AI response
```

---

## Next Steps

1. ✅ Install ngrok: https://ngrok.com/download
2. ✅ Run `ngrok http 3001`
3. ✅ Copy the https URL (e.g., `https://abcd-1234.ngrok-free.app`)
4. ✅ Update `gateway/.env`: `GATEWAY_PUBLIC_URL=wss://abcd-1234.ngrok-free.app`
5. ✅ Update Twilio TwiML App Voice URL: `https://abcd-1234.ngrok-free.app/twilio/voice/start`
6. ✅ Restart gateway: `cd gateway && pnpm run start:dev`
7. ✅ Test call again!

**Happy voice calling! 🎉**
