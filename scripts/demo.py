#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════╗
║       مركز علاجك — المساعد الصوتي ليان              ║
║       Alaajak Medical Center — Investor Demo         ║
╚══════════════════════════════════════════════════════╝

Pre-scripted investor demo: a patient calls to book a dermatology appointment.
The VA (ليان) responds with real AI inference + Saudi XTTS voice.

Usage (from repo root):
    python scripts/demo.py

Requirements:
    pip install requests
    SSH key at %USERPROFILE%\\.ssh\\google_compute_engine
    Python 3.8+, Windows (winsound) or Linux/macOS (aplay/afplay)
"""

import sys
import os
import io
import time
import wave
import json
import struct
import socket
import subprocess
import threading
import base64

import requests  # pip install requests

# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these if the VM address or SSH key changes
# ─────────────────────────────────────────────────────────────────────────────
VM_HOST  = "manso@34.26.235.26"
SSH_KEY  = os.path.join(os.path.expanduser("~"), ".ssh", "google_compute_engine")
LLM_PORT = 5007
TTS_PORT = 5002
LLM_URL  = f"http://localhost:{LLM_PORT}"
TTS_URL  = f"http://localhost:{TTS_PORT}"
DIALECT  = "saudi"
SESSION  = "demo-investor-2026"
TIMEOUT  = 60   # seconds per LLM request (model inference can be slow)

# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours (ANSI — works in Windows Terminal & PowerShell 7)
# ─────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-scripted patient turns (inputs only — VA replies are live AI)
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_TURNS = [
    "أبغى أحجز موعد جلدية",          # Turn 1: specialty
    "الثلاثاء",                        # Turn 2: pick day from what VA offers
    "الساعة الثالثة مساءً",           # Turn 3: pick time
    "اسمي منصور محمد منصور",          # Turn 4: name
    "01095013536",                     # Turn 5: phone
    "26/08/2001",                      # Turn 6: DOB  →  triggers booking
]

# ─────────────────────────────────────────────────────────────────────────────
# μ-law → 16-bit PCM lookup table (pure-Python, no audioop/ffmpeg needed)
# ─────────────────────────────────────────────────────────────────────────────
def _build_ulaw_table() -> list:
    table = []
    for i in range(256):
        u = (~i) & 0xFF
        s = (u & 0x80)
        e = (u >> 4) & 0x07
        m = u & 0x0F
        val = ((m << 3) | 0x84) << e
        table.append(-val if s else val)
    return table

_ULAW_TABLE = _build_ulaw_table()


def mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    """Decode 8 kHz μ-law bytes → a valid 16-bit PCM WAV blob."""
    pcm = struct.pack(f"<{len(mulaw_bytes)}h",
                      *[_ULAW_TABLE[b] for b in mulaw_bytes])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    return buf.getvalue()


def play_audio(wav_bytes: bytes) -> None:
    """Play WAV bytes — Windows (winsound) or Linux/macOS fallback."""
    if not wav_bytes:
        return
    try:
        if sys.platform == "win32":
            import winsound
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            try:
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        elif sys.platform == "darwin":
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            subprocess.call(["afplay", tmp_path])
            os.remove(tmp_path)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            subprocess.call(["aplay", "-q", tmp_path])
            os.remove(tmp_path)
    except Exception as e:
        print(f"{DIM}  [audio playback failed: {e}]{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# SSH tunnel management
# ─────────────────────────────────────────────────────────────────────────────
def start_ssh_tunnel() -> subprocess.Popen:
    """Start background SSH tunnel for LLM-VA:5007 and TTS:5002."""
    cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-N",                                   # no remote command
        "-L", f"{LLM_PORT}:localhost:{LLM_PORT}",
        "-L", f"{TTS_PORT}:localhost:{TTS_PORT}",
        VM_HOST,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _port_open(port: int) -> bool:
    """Return True if localhost:port is accepting connections."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_tunnel(timeout: int = 30) -> bool:
    """Poll until both tunnelled ports are open."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(LLM_PORT) and _port_open(TTS_PORT):
            return True
        time.sleep(0.5)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# API calls
# ─────────────────────────────────────────────────────────────────────────────
def call_llm(message: str, history: list, slots: dict) -> dict:
    """Call LLM-VA /chat. Returns {"reply": str, "slots": dict}."""
    payload = {
        "message":  message,
        "history":  history,
        "sessionId": SESSION,
        "mode":     "va",
        "slots":    slots,
        "dialect":  DIALECT,
        "tenantId": "default",
    }
    resp = requests.post(
        f"{LLM_URL}/chat",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def call_tts(text: str) -> bytes:
    """Call TTS /synthesize. Returns WAV bytes (decoded from mulaw base64)."""
    payload = {
        "text":  text,
        "voice": "saudi-tts",
        "sessionId": SESSION,
        "format": "mulaw",
    }
    resp = requests.post(
        f"{TTS_URL}/synthesize",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    mulaw_b64 = data.get("audio", "")
    if not mulaw_b64:
        return b""
    mulaw_bytes = base64.b64decode(mulaw_b64)
    return mulaw_to_wav(mulaw_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def print_banner() -> None:
    print(f"\n{MAGENTA}{BOLD}")
    print("╔══════════════════════════════════════════════════════╗")
    print("║       مركز علاجك — المساعد الصوتي ليان              ║")
    print("║       Alaajak Medical Center — Live AI Demo          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(RESET)


def divider() -> None:
    print(f"\n{DIM}{'─' * 58}{RESET}")


def typewriter(text: str, color: str, delay: float = 0.025) -> None:
    """Print text one character at a time for a dramatic live effect."""
    print(color, end="", flush=True)
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print(RESET)


def spinner(label: str, done_event: threading.Event) -> None:
    """Show a spinner while the LLM is thinking."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not done_event.is_set():
        print(f"\r{YELLOW}{frames[i % len(frames)]}  {label}...{RESET}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r{' ' * 40}\r", end="", flush=True)   # clear spinner line


# ─────────────────────────────────────────────────────────────────────────────
# Booking summary banner (shown after last turn)
# ─────────────────────────────────────────────────────────────────────────────
def print_booking_summary(slots: dict) -> None:
    from datetime import date as _date, timedelta
    # Compute next Tuesday for display
    today = _date.today()
    days_ahead = (1 - today.weekday()) % 7  # Tuesday = weekday 1
    if days_ahead == 0:
        days_ahead = 7
    appt_date = (today + timedelta(days=days_ahead)).strftime("%d/%m/%Y")

    print(f"\n{GREEN}{BOLD}")
    print("┌─────────────────────────────────────────────────┐")
    print("│         ✅  تأكيد الحجز — Booking Confirmed     │")
    print("├─────────────────────────────────────────────────┤")
    print(f"│  المريض  : {slots.get('name',  'منصور محمد منصور'):<36}│")
    print(f"│  الهاتف  : {slots.get('phone', '01095013536'):<36}│")
    print(f"│  الميلاد : {slots.get('dob',   '26/08/2001'):<36}│")
    print(f"│  التخصص  : {slots.get('specialty','جلدية'):<36}│")
    print(f"│  الطبيب  : {'دكتور علي الغامدي':<36}│")
    print(f"│  التاريخ : {slots.get('date', appt_date):<36}│")
    print(f"│  الوقت   : {slots.get('time', '15:00'):<36}│")
    print("└─────────────────────────────────────────────────┘")
    print(RESET)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print_banner()

    # ── Start SSH tunnel ──────────────────────────────────────────────────
    print(f"{YELLOW}⚙  Connecting to VM via SSH tunnel...{RESET}")
    tunnel = start_ssh_tunnel()

    if not wait_for_tunnel(timeout=30):
        tunnel.terminate()
        print(f"{RED}✗  Could not reach LLM-VA or TTS on VM. Check SSH key / VM status.{RESET}")
        sys.exit(1)

    print(f"{YELLOW}✓  SSH tunnel established  (LLM-VA:{LLM_PORT}  TTS:{TTS_PORT})\n{RESET}")
    time.sleep(0.5)

    # ── Initial greeting ──────────────────────────────────────────────────
    divider()
    print(f"{MAGENTA}{BOLD}📞  المكالمة بدأت — Call Connected{RESET}")
    divider()

    history: list = []
    slots:   dict = {}

    for turn_idx, patient_text in enumerate(PATIENT_TURNS):

        # Show patient input
        divider()
        print(f"\n{CYAN}{BOLD}👤  المريض:{RESET}  {CYAN}{patient_text}{RESET}")
        time.sleep(0.6)

        # Call LLM with spinner
        done   = threading.Event()
        spin_t = threading.Thread(target=spinner, args=("ليان تفكر", done), daemon=True)
        spin_t.start()

        try:
            result   = call_llm(patient_text, history, slots)
            va_reply = result.get("reply", "")
            slots    = result.get("slots", slots) or slots
        except Exception as e:
            done.set()
            spin_t.join()
            print(f"\n{RED}✗  LLM call failed: {e}{RESET}")
            tunnel.terminate()
            sys.exit(1)
        finally:
            done.set()
            spin_t.join()

        # Show VA reply with typewriter effect
        print(f"\n{GREEN}{BOLD}🤖  ليان:{RESET}  ", end="")
        typewriter(va_reply, GREEN, delay=0.02)

        # Synthesise and play audio in parallel while staying "live"
        try:
            audio_done = threading.Event()

            def _play():
                try:
                    wav = call_tts(va_reply)
                    play_audio(wav)
                except Exception as ae:
                    print(f"\n{DIM}  [audio: {ae}]{RESET}")
                finally:
                    audio_done.set()

            audio_thread = threading.Thread(target=_play, daemon=True)
            audio_thread.start()
            audio_done.wait(timeout=90)    # audio plays in foreground
        except Exception:
            pass

        # Append to history for context
        history.append({"role": "user",      "content": patient_text})
        history.append({"role": "assistant", "content": va_reply})

        # Brief pause between turns (feels natural)
        time.sleep(0.8)

    # ── Booking summary ───────────────────────────────────────────────────
    divider()
    print_booking_summary(slots)
    divider()
    print(f"{MAGENTA}{BOLD}📞  انتهت المكالمة — Call Ended{RESET}\n")

    tunnel.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Demo interrupted.{RESET}")
        sys.exit(0)
