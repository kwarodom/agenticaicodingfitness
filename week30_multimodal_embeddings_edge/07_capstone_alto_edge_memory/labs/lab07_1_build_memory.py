#!/usr/bin/env python3
"""Lab 07-1 · Build Alto Edge Memory: one offline index of notes, photos, voicemails, CCTV and incidents.

The capstone's index. Everything from Modules 02–05 in one place: 18 SYNTHETIC ops notes (titled documents),
6 photos, 3 voicemails (one Thai), 1 CCTV clip and 2 interleaved incident records, embedded with every
encoder and stored at 512 dimensions (multimodal quality drops faster than text below 512 — model card) in a
single SQLite file. The build runs inside the egress guard and writes a manifest the grader (Lab 07-3) checks.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_1_build_memory.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import egress  # noqa: E402
import embedkit as ek  # noqa: E402
from vectorstore import VectorStore  # noqa: E402

RUNS = ek.runs(__file__)
DB = RUNS / "edge_memory.sqlite"
DIM = 512
MEDIA = [("image", "leak_under_ahu.jpg", "Photo: water pooling under AHU-3"),
         ("image", "plantroom_normal.jpg", "Photo: plant room B2, normal"),
         ("image", "plantroom_smoke.jpg", "Photo: plant room B2, haze"),
         ("image", "thermostat_room1203.png", "Photo: thermostat in room 1203 reads 29.1 °C"),
         ("image", "invoice_bank_change.png", "Scan: supplier invoice asking for a bank change"),
         ("image", "bms_trend_ahu3.png", "Screenshot: BMS trend for AHU-3"),
         ("audio", "voicemail_flood_en.wav", "Voicemail: room 1507 bathroom flooding"),
         ("audio", "voicemail_burning_th.wav", "Voicemail (Thai): room 1203, AC not cold, burning smell"),
         ("audio", "request_towels_en.wav", "Voicemail: towels for room 812, no rush"),
         ("video", "cctv_plantroom.mp4", "CCTV: plant room B2, haze appears after ~4 s")]
RECORDS = [
    ("Incident 0412 · plant room B2 haze", {
        "text": "Incident 0412, plant room B2, 02:10. Night shift saw haze: <|image|> CCTV, same camera: <|video|>",
        "image": ["plantroom_smoke.jpg"], "video": "cctv_plantroom.mp4"}),
    ("Incident 0413 · AHU-3 condensate leak", {
        "text": "Incident 0413, basement AHU-3, 06:45. Engineer's photo: <|image|> Drain pan blocked, unit isolated.",
        "image": ["leak_under_ahu.jpg"]}),
]

ek.banner("Lab 07-1 · build Alto Edge Memory", f"every encoder · {DIM}-d float16 · one SQLite file · egress guarded")
DB.unlink(missing_ok=True)
t_start = time.time()
with egress.guard() as log:
    model = ek.load(("text", "image", "audio"))
    store = VectorStore(DB, dim=DIM)

    ek.step(1, "ops notes")
    kb = ek.load_kb()
    vecs = model.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])
    for i, (d, v) in enumerate(zip(kb["docs"], vecs)):
        store.add(kind="note", title=d["title"], text=d["text"], source=f"hotel_kb.json#{i}", vector=v)
    ek.ok(f"{len(kb['docs'])} notes")

    ek.step(2, "photos, voicemails, CCTV")
    for kind, f, title in MEDIA:
        v = model.encode([{kind: ek.media(f)}])[0]
        store.add(kind=kind, title=title, source=f"media/{f}", vector=v, meta={"file": f})
        print(f"  + {kind:5s} {title}")

    ek.step(3, "interleaved incident records")
    for title, rec in RECORDS:
        item = dict(rec)
        item["image"] = [ek.media(x) for x in rec["image"]]
        if "video" in rec:
            item["video"] = ek.media(rec["video"])
        v = model.encode([item])[0]
        files = rec["image"] + ([rec["video"]] if "video" in rec else [])
        store.add(kind="record", title=title, text=rec["text"], source="incident log", vector=v, meta={"files": files})
        print(f"  + record {title}")
s = egress.summary(log)
manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t_start, 1),
            "dim": DIM, "items": len(store), "kinds": store.kinds(), "bytes": store.bytes_on_disk(),
            "egress": s, "model": ek.MODEL_ID, "machine": ek.machine()}
(RUNS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

ek.step(4, "summary")
ek.table([[k, v] for k, v in manifest["kinds"].items()], ["kind", "items"])
ek.ok(f"{manifest['items']} items · {manifest['bytes'] / 1024:.0f} KiB · built in {manifest['seconds']}s · "
      f"left this machine: {s['outbound_allowed']} connections")
ek.result("week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/.runs/edge_memory.sqlite is the whole memory — Lab 07-2 gives "
          "it to an agent.")
