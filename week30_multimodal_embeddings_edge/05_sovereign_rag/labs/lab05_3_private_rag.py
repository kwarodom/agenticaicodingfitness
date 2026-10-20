#!/usr/bin/env python3
"""Lab 05-3 · Private RAG: local embeddings find the notes, a local Gemma 4 writes the answer.

Retrieval-augmented generation with nothing leaving the machine: EmbeddingGemma 2 retrieves the top notes
from the SYNTHETIC hotel knowledge base, and a local model in Ollama (gemma4:12b if you have it) answers ONLY
from those notes, citing them by number. The whole run is inside the egress guard with one exception —
Ollama on localhost:11434. Three questions, three LLM calls.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_3_private_rag.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import egress  # noqa: E402
import embedkit as ek  # noqa: E402

QUESTIONS = ["There's water on the floor under an air handler in the basement. What do I do first?",
             "แขกห้อง 1203 บอกว่าแอร์มีกลิ่นไหม้ ต้องทำอะไรบ้าง",
             "A supplier emailed asking us to pay their invoice into a new bank account. Can I just update it?"]
SYSTEM = ("You are the night-shift assistant of a hotel engineering team. Answer ONLY from the numbered notes. "
          "Cite the notes you used like [1]. If the notes do not cover it, say so. Answer in the language of the "
          "question, in 2-4 short sentences.")

ek.banner("Lab 05-3 · private RAG", "EmbeddingGemma 2 retrieves · a local Ollama model answers · egress guarded")
if not ek.ollama_up():
    ek.warn("Ollama is not running on localhost:11434 — start it (ollama serve) to run the answer step")
    sys.exit(0)
chat_model = ek.pick_chat_model()
ek.note(f"answer model: {chat_model} (local, via Ollama)")

with egress.guard(allow_ports={11434}) as log:
    model = ek.load(("text",))
    kb = ek.load_kb()
    D = model.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])
    for n, question in enumerate(QUESTIONS, 1):
        ek.step(n, question)
        hits = ek.top_k(model.encode(question, prompt_name="QuestionAnswering"), D, k=3)
        context = "\n".join(f"[{i + 1}] {kb['docs'][j]['title']}: {kb['docs'][j]['text']}"
                            for i, (j, _s) in enumerate(hits))
        for i, (j, s) in enumerate(hits):
            print(f"  [{i + 1}] {s:.3f}  {kb['docs'][j]['title']}")
        msg = ek.ollama_chat(chat_model, [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"}],
                             max_tokens=1536, think=False)
        answer = (msg.get("content") or "").strip()
        print("  · ANSWER " + answer.replace("\n", "\n           "))
        ek.note(f"{msg.get('_eval_count')} tokens in {msg.get('_secs', 0):.1f}s on this machine"
                + (" (includes loading the model into memory)" if n == 1 else ""))
s = egress.summary(log)
ek.step(len(QUESTIONS) + 1, "what crossed the network")
egress.report(log)
(ek.ok if s["outbound_allowed"] == 0 else ek.warn)(f"left this machine: {s['outbound_allowed']} connections")
ek.result("the whole RAG loop — embedding, retrieval, generation — ran on hardware you control. That is the "
          "difference between 'hosted in our region' and sovereign.")
