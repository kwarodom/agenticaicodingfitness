#!/usr/bin/env python3
"""Lab 05-2 · A local vector store: one SQLite file holds notes, photos, voicemails and clips.

Builds Alto Edge Memory's index with common/vectorstore.py: 18 SYNTHETIC notes + 6 images + 3 voicemails +
1 CCTV clip, stored at 256 dimensions in float16 (Lab 03 showed 256-d keeps the quality at a third of the
size). Then it closes the file, reopens it as a fresh process would, and searches it — with and without a
`kinds` filter, which is how you work around the modality gap from Lab 04-3.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_2_local_vector_store.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import egress  # noqa: E402
import embedkit as ek  # noqa: E402
from vectorstore import VectorStore  # noqa: E402

DB = ek.runs(__file__) / "alto_memory.sqlite"
MEDIA = [("image", "leak_under_ahu.jpg", "Water pooling under AHU-3"),
         ("image", "plantroom_normal.jpg", "Plant room B2, normal"),
         ("image", "plantroom_smoke.jpg", "Plant room B2, haze"),
         ("image", "thermostat_room1203.png", "Thermostat, room 1203"),
         ("image", "invoice_bank_change.png", "Supplier invoice, bank change"),
         ("image", "bms_trend_ahu3.png", "BMS trend, AHU-3"),
         ("audio", "voicemail_flood_en.wav", "Voicemail: bathroom flooding, 1507"),
         ("audio", "voicemail_burning_th.wav", "Voicemail (Thai): AC burning smell, 1203"),
         ("audio", "request_towels_en.wav", "Voicemail: towels for 812"),
         ("video", "cctv_plantroom.mp4", "CCTV: plant room B2, 10 s")]

ek.banner("Lab 05-2 · local vector store", "SQLite + numpy · 256-d float16 · notes, photos, voice, video · offline")
DB.unlink(missing_ok=True)

with egress.guard() as log:
    ek.step(1, "embed and store everything")
    model = ek.load(("text", "image", "audio"))
    store = VectorStore(DB, dim=256)
    kb = ek.load_kb()
    with ek.timer() as t:
        notes = model.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])
        for d, v in zip(kb["docs"], notes):
            store.add(kind="note", title=d["title"], text=d["text"], source="hotel_kb.json", vector=v)
        for kind, f, title in MEDIA:
            v = model.encode([{kind: ek.media(f)}])[0]
            store.add(kind=kind, title=title, source=f"media/{f}", vector=v, meta={"file": f})
    ek.ok(f"{len(store)} items · {store.kinds()} · {t.s:.1f}s · {store.bytes_on_disk() / 1024:.0f} KiB on disk")
    store.close()

    ek.step(2, "reopen the file, as tomorrow's process would, and search")
    store = VectorStore(DB, dim=256)
    checks = [("water leaking under an air handler", None, "AHU condensate leak"),
              ("water leaking under an air handler", {"image"}, "Water pooling under AHU-3"),
              ("guest called about a burning smell from the AC", None, "แอร์มีกลิ่นไหม้"),
              ("guest called about a burning smell from the AC", {"audio"}, "Voicemail (Thai): AC burning smell, 1203"),
              ("แขกโทรมาบอกว่าแอร์มีกลิ่นไหม้", {"audio"}, "Voicemail (Thai): AC burning smell, 1203"),
              ("haze on the plant room camera", {"video", "image"}, "CCTV: plant room B2, 10 s")]
    good = 0
    for q, kinds, want in checks:
        hits = store.search(model.encode(q, prompt_name="SearchQuery"), k=2, kinds=kinds)
        f = "all kinds" if not kinds else "+".join(sorted(kinds))
        mark = "✓" if hits and hits[0]["title"] == want else "✕"
        good += mark == "✓"
        print(f"  {mark} {q[:44]:44s} [{f:11s}] → " + " · ".join(f"{h['kind']}:{h['title'][:28]} {h['score']:.3f}" for h in hits))
    (ek.ok if good == len(checks) else ek.warn)(f"{good}/{len(checks)} found the expected item first")
    ek.note("the English 'burning smell' question lands on an English AC note before the Thai one, and misses the "
            "Thai voicemail even inside the audio filter — a filter removes the notes, it does not close the "
            "language + modality gap. Asked in Thai, the voicemail is found first.")
s = egress.summary(log)
ek.ok(f"left this machine: {s['outbound_allowed']} connections (attempts {s['attempts']}, blocked {s['blocked']})")
ek.note(f"index file: week30_multimodal_embeddings_edge/05_sovereign_rag/.runs/{DB.name} — copy it to another edge box and search there")
ek.result("a whole hotel's multimodal memory fits in a file you can back up, audit and delete. Filter by kind to "
          "stop notes crowding out media, and remember the gap: speech is found best in its own language.")
