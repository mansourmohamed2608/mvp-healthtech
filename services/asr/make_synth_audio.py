# services/asr/make_synth_audio.py
import os, uuid, time, math, asyncio, pandas as pd
import edge_tts

VOICE       = "ar-EG-SalmaNeural"
CSV_IN      = "data/medical_text.csv"
OUT_DIR     = "data/tts_ar_med"
CONCURRENCY = 4          # try 2–6; lower if you see throttling
LIMIT       = None       # e.g., 2000 while testing
SAVE_EVERY  = 100        # checkpoint manifest every N items

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load texts ----
df = pd.read_csv(CSV_IN).dropna()
if LIMIT:
    df = df.head(LIMIT)
texts = df["text"].astype(str).str.strip().tolist()
texts = [t for t in texts if t]
if not texts:
    print(f"No rows found in {CSV_IN}")
    raise SystemExit(0)

# ---- one synthesis ----
async def synth_one(text: str, path: str):
    comm = edge_tts.Communicate(text, VOICE)
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

# ---- worker coroutine ----
async def worker(queue: asyncio.Queue, out_rows: list):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, text = item
        path = os.path.join(OUT_DIR, f"{uuid.uuid4().hex}.mp3")
        try:
            await synth_one(text, path)
            out_rows.append({"audio_filepath": path, "text": text})
        except Exception:
            # skip failed line; you can log the exception if you want
            pass
        finally:
            queue.task_done()

# ---- main loop with safe ETA ----
async def main():
    total = len(texts)
    q = asyncio.Queue()
    for i, t in enumerate(texts):
        await q.put((i, t))
    for _ in range(CONCURRENCY):
        await q.put(None)

    results = []
    tasks = [asyncio.create_task(worker(q, results)) for _ in range(CONCURRENCY)]
    start = time.perf_counter()
    last_saved = 0

    while any(not t.done() for t in tasks):
        await asyncio.sleep(1.0)
        done = len(results)
        elapsed = time.perf_counter() - start
        rate = (done / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, total - done)

        # SAFE ETA: show only after we have a non-zero rate
        if rate > 0:
            eta_sec = remaining / rate
            m, s = divmod(int(eta_sec), 60)
            eta_str = f"{m}m {s}s"
        else:
            eta_str = "--:--"

        print(
            f"\r{done}/{total} done | elapsed {int(elapsed)}s | rate {rate:.2f}/s | ETA {eta_str}",
            end="", flush=True
        )

        if done - last_saved >= SAVE_EVERY:
            pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
            last_saved = done

    await asyncio.gather(*tasks)
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
