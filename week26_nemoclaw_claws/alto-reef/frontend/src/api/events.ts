import { useEffect, useRef, useState } from "react";
import { API_BASE, api } from "./client";
import type { ReefEvent } from "./types";

/** SSE stream with polling fallback (deploy proxies may not stream). Merges with the seeded history. */
export function useReefEvents(limit = 200) {
  const [events, setEvents] = useState<ReefEvent[]>([]);
  const [live, setLive] = useState<"sse"|"poll"|"down">("down");
  const seen = useRef<Set<number>>(new Set());
  const push = (incoming: ReefEvent[]) => setEvents(prev => {
    const fresh = incoming.filter(e => !seen.current.has(e.id)); fresh.forEach(e => seen.current.add(e.id));
    if (!fresh.length) return prev;
    return [...prev, ...fresh].sort((a,b)=>a.at-b.at).slice(-limit);
  });
  useEffect(() => {
    let es: EventSource | null = null; let poll: number | undefined; let dead = false;
    api<ReefEvent[]>(`/api/events?limit=${limit}`).then(push).catch(()=>{});
    const startPoll = () => { setLive("poll"); poll = window.setInterval(() => api<ReefEvent[]>(`/api/events?limit=50`).then(push).catch(()=>setLive("down")), 5000); };
    try {
      es = new EventSource(`${API_BASE}/api/stream`);
      es.addEventListener("hello", () => setLive("sse"));
      es.addEventListener("reef", (m) => push([JSON.parse((m as MessageEvent).data)]));
      es.onerror = () => { if (dead) return; es?.close(); es = null; if (!poll) startPoll(); };
    } catch { startPoll(); }
    return () => { dead = true; es?.close(); if (poll) clearInterval(poll); };
  }, [limit]);
  return { events, live };
}
