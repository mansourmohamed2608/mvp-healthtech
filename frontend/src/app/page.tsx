"use client";
import { useState } from "react";

export default function Home() {
  const [sid, setSid] = useState<string>("");
  async function createSession() {
    const res = await fetch("http://localhost:3000/session", { method: "POST" });
    const data = await res.json();
    setSid(data.sessionId);
  }
  return (
    <main style={{ padding: 24 }}>
      <h1>Voice Agent MVP</h1>
      <button onClick={createSession}>Create Session</button>
      {sid && <p>Session: {sid}</p>}
    </main>
  );
}
