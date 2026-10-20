#!/usr/bin/env python3
"""Lab 04-3 · Thai and English in one space: cross-lingual and cross-modal at once.

EmbeddingGemma 2 was trained on 100+ languages. A Thai question should find an English note, an English
question should find a Thai note, and either should find the Thai voicemail — without translation, without
speech-to-text. This lab measures all four directions on the SYNTHETIC hotel knowledge base and voicemails.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_3_cross_lingual.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402

ek.banner("Lab 04-3 · Thai ⇄ English", "cross-lingual text search + Thai voicemail retrieval · SYNTHETIC data")
model = ek.load(("text", "audio"))
kb = ek.load_kb()
docs = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]]
D = model.encode(docs)
titles = [d["title"] for d in kb["docs"]]

ek.step(1, "Thai questions → English notes")
th_to_en = [("ปั๊มสระว่ายน้ำตัดบ่อย ทำยังไงดี", "Pool pump trip"),
            ("มีคนติดอยู่ในลิฟต์", "Lift entrapment"),
            ("ใบแจ้งหนี้ขอเปลี่ยนเลขบัญชีธนาคาร", "Supplier bank detail changes"),
            ("ห้องพักร้อนมาก แขกบ่น", "Guest room too hot")]
ok = 0
for q, want in th_to_en:
    i, s = ek.top_k(model.encode(q, prompt_name="SearchQuery"), D, 1)[0]
    ok += titles[i] == want
    print(f"  {'✓' if titles[i] == want else '✕'} {q:34s} → {titles[i]} ({s:.3f})")

ek.step(2, "English questions → Thai notes")
en_to_th = [("the air conditioner smells like something is burning", "แอร์มีกลิ่นไหม้"),
            ("water is dripping from the ceiling", "น้ำรั่วจากเพดาน")]
for q, want in en_to_th:
    i, s = ek.top_k(model.encode(q, prompt_name="SearchQuery"), D, 1)[0]
    ok += titles[i] == want
    print(f"  {'✓' if titles[i] == want else '✕'} {q:52s} → {titles[i]} ({s:.3f})")
(ek.ok if ok == len(th_to_en) + len(en_to_th) else ek.warn)(f"{ok}/{len(th_to_en) + len(en_to_th)} cross-lingual "
                                                             "questions found the right note first")

ek.step(3, "the Thai voicemail, asked in Thai and in English")
AUDIO = ["voicemail_flood_en.wav", "voicemail_burning_th.wav", "request_towels_en.wav"]
A = model.encode([{"audio": ek.media(f)} for f in AUDIO])
for q in ("แขกโทรมาบอกว่าแอร์มีกลิ่นไหม้", "a guest called about a burning smell from the AC"):
    scores = ek.renorm(A) @ ek.renorm(model.encode(q, prompt_name="SearchQuery"))
    best = int(np.argmax(scores))
    mark = "✓" if AUDIO[best] == "voicemail_burning_th.wav" else "✕"
    print(f"  {mark} {q[:48]:48s} → {AUDIO[best]}  " + " · ".join(f"{a.split('_')[1]} {x:.3f}" for a, x in zip(AUDIO, scores)))
ek.note("Lab 04-1 asked 'a Thai speaker reports a burning smell…' and found it — naming the language helps when the "
        "item is speech in that language")

ek.step(4, "does the Thai voicemail sit near its Thai note or its English twin?")
th_note = D[titles.index("แอร์มีกลิ่นไหม้")]
en_note = model.encode("title: Burning smell from the AC | text: If a guest reports a burning smell from the air "
                       "conditioner, switch it off, move the guest and send an electrician.")
ek.table([["Thai voicemail ↔ Thai note", f"{ek.cosine(A[1], th_note):.3f}"],
          ["Thai voicemail ↔ English note (same meaning)", f"{ek.cosine(A[1], en_note):.3f}"],
          ["Thai voicemail ↔ towel request voicemail", f"{ek.cosine(A[1], A[2]):.3f}"]], ["pair", "cosine"])
ek.result("text ⇄ text works across Thai and English. Audio ⇄ text works, but with a MODALITY GAP: voicemails sit "
          "closer to other voicemails than to notes that mean the same thing. Rank within one modality, or calibrate "
          "scores per modality before merging lists — Module 07's agent does exactly that.")
