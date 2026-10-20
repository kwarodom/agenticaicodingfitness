#!/usr/bin/env python3
"""Lab 07-2 · The memory agent: a local Gemma 4 that searches photos, voice and notes before it answers.

An agent loop on this machine only. The LLM (Ollama, gemma4:12b if present) gets one tool, `search_memory`,
backed by Lab 07-1's index and EmbeddingGemma 2. The tool fixes the modality gap from Labs 04-3 and 05-2 the
simple way: it searches EACH kind separately (notes, photos, voice, video, incident records) and returns the
best hits of every kind, each with its source, so the model can cite a voicemail even when notes score higher.
Two night-shift questions (one English, one Thai), at most 6 LLM calls, inside the egress guard.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_2_memory_agent.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import egress  # noqa: E402
import embedkit as ek  # noqa: E402
from vectorstore import VectorStore  # noqa: E402

RUNS = ek.runs(__file__)
DB = RUNS / "edge_memory.sqlite"
MAX_CALLS = 6
QUESTIONS = ["Night shift handover: what happened in plant room B2, and what is our procedure for it?",
             "แขกห้อง 1203 แจ้งเรื่องอะไรเข้ามาบ้าง และควรทำอะไรต่อ"]
TOOLS = [{"type": "function", "function": {
    "name": "search_memory",
    "description": ("Search the hotel's private, offline memory: ops notes, photos, CCTV, guest voicemails and "
                    "incident records. Returns the best matches of each kind with a source id to cite."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "what to look for, in plain words (any language)"},
        "kinds": {"type": "array", "items": {"type": "string", "enum": ["note", "image", "audio", "video", "record"]},
                  "description": "optional: only these kinds"}},
        "required": ["query"]}}}]
SYSTEM = ("You are Alto Edge Memory, the night-shift assistant of a hotel engineering team. Always call "
          "search_memory before answering. Answer only from what it returns, cite sources in square brackets "
          "exactly as given (e.g. [media/cctv_plantroom.mp4] or [hotel_kb.json#2]), say what you could not find, "
          "and answer in the language of the question in 3-6 short sentences.")


def search_memory(model, store, query: str, kinds=None, per_kind: int = 2) -> list[dict]:
    q = model.encode(query, prompt_name="SearchQuery")
    wanted = kinds or ["note", "image", "audio", "video", "record"]
    hits = []
    for k in wanted:                                      # per-kind search: the modality-gap fix
        for h in store.search(q, k=per_kind, kinds={k}):
            hits.append({"kind": h["kind"], "title": h["title"], "source": h["source"],
                         "text": (h["text"] or "")[:240], "score": round(h["score"], 3)})
    return hits


ek.banner("Lab 07-2 · the memory agent", "local LLM + search_memory tool over Alto Edge Memory · egress guarded")
if not DB.exists():
    ek.warn("run Lab 07-1 first — it builds .runs/edge_memory.sqlite")
    sys.exit(1)
if not ek.ollama_up():
    ek.warn("Ollama is not running on localhost:11434 — start it to run the agent")
    sys.exit(0)
chat_model = ek.pick_chat_model()
ek.note(f"agent model: {chat_model} (local) · memory: {DB.name}")

run_log = {"model": chat_model, "questions": []}
calls = 0
with egress.guard(allow_ports={11434}) as log:
    model = ek.load(("text",))                            # queries are text: the text model is enough here
    store = VectorStore(DB, dim=512)
    for n, question in enumerate(QUESTIONS, 1):
        ek.step(n, question)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        entry = {"question": question, "tool_calls": [], "answer": ""}
        while calls < MAX_CALLS:
            msg = ek.ollama_chat(chat_model, messages, tools=TOOLS, max_tokens=2048, think=False)
            calls += 1
            tcs = msg.get("tool_calls") or []
            if not tcs:
                entry["answer"] = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function", {})
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                hits = search_memory(model, store, args.get("query", question), args.get("kinds"))
                entry["tool_calls"].append({"args": args, "hits": [h["source"] for h in hits]})
                print(f"  → search_memory({json.dumps(args, ensure_ascii=False)})")
                for h in hits:
                    print(f"      {h['score']:.3f}  {h['kind']:6s} {h['title'][:58]}  [{h['source']}]")
                messages.append({"role": "tool", "content": json.dumps(hits, ensure_ascii=False)})
        print("  · ANSWER " + entry["answer"].replace("\n", "\n           "))
        run_log["questions"].append(entry)
s = egress.summary(log)
run_log.update(llm_calls=calls, egress=s)
(RUNS / "agent_run.json").write_text(json.dumps(run_log, indent=2, ensure_ascii=False))
ek.step(len(QUESTIONS) + 1, "evidence")
egress.report(log)
ek.ok(f"{calls} LLM calls · left this machine: {s['outbound_allowed']} connections · log → .runs/agent_run.json")
ek.result("an agent with a private, multimodal memory — the photo, the CCTV clip, the Thai voicemail and the "
          "procedure — and not one byte of it left the building.")
