#!/usr/bin/env python3
"""Lab 07-3 · Grade the capstone on evidence: index, retrieval, sovereignty, agent citations.

The grader reads what Labs 07-1 and 07-2 left in .runs/ and re-runs a fixed retrieval check against the
index. No opinions: every point comes from a file or a measurement. 100 points:
  20 index built (all five kinds present, manifest written)     · 25 retrieval (8 fixed questions, per-kind search)
  20 sovereignty (0 connections left the machine, build + agent) · 35 agent (both questions answered, every
     citation is a real source in the index, the plant-room answer cites the procedure AND a camera/photo item,
     the Thai answer cites the Thai voicemail)

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_3_grade.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402
from vectorstore import VectorStore  # noqa: E402

RUNS = ek.runs(__file__)
CHECKS = [("procedure for smoke in a plant room", "note", "Smoke in the plant room"),
          ("haze seen on the plant room camera", "video", "CCTV: plant room B2, haze appears after ~4 s"),
          ("photo of water under an air handler", "image", "Photo: water pooling under AHU-3"),
          ("thermostat display showing a hot room", "image", "Photo: thermostat in room 1203 reads 29.1 °C"),
          ("แขกแจ้งว่าแอร์มีกลิ่นไหม้", "audio", "Voicemail (Thai): room 1203, AC not cold, burning smell"),
          ("guest bathroom is flooding", "audio", "Voicemail: room 1507 bathroom flooding"),
          ("incident report about haze in B2", "record", "Incident 0412 · plant room B2 haze"),
          ("supplier wants payment to a new bank account", "note", "Supplier bank detail changes")]

ek.banner("Lab 07-3 · grade the capstone", "evidence only: files in .runs/ + a fixed retrieval check")
score = 0

ek.step(1, "index (20)")
man_p, db_p = RUNS / "manifest.json", RUNS / "edge_memory.sqlite"
if man_p.exists() and db_p.exists():
    man = json.loads(man_p.read_text())
    kinds = set(man["kinds"])
    pts = 10 + (10 if kinds >= {"note", "image", "audio", "video", "record"} else 0)
    ek.ok(f"{man['items']} items · kinds {sorted(kinds)} · {man['dim']}-d · +{pts}")
    score += pts
else:
    ek.warn("no index — run Lab 07-1")
    man = None

ek.step(2, "retrieval (25) — per-kind search, expected item first")
if man:
    model = ek.load(("text",))
    store = VectorStore(db_p, dim=man["dim"])
    good = 0
    for q, kind, want in CHECKS:
        top = store.search(model.encode(q, prompt_name="SearchQuery"), k=1, kinds={kind})
        hit = top[0]["title"] if top else ""
        good += hit == want
        print(f"  {'✓' if hit == want else '✕'} [{kind:6s}] {q[:42]:42s} → {hit[:46]}")
    pts = round(25 * good / len(CHECKS))
    ek.ok(f"{good}/{len(CHECKS)} · +{pts}")
    score += pts

ek.step(3, "sovereignty (20)")
run_p = RUNS / "agent_run.json"
run = json.loads(run_p.read_text()) if run_p.exists() else None
out_build = man["egress"]["outbound_allowed"] if man else None
out_agent = run["egress"]["outbound_allowed"] if run else None
pts = (10 if out_build == 0 else 0) + (10 if out_agent == 0 else 0)
ek.ok(f"connections that left the machine — build: {out_build} · agent: {out_agent} · +{pts}")
score += pts

ek.step(4, "agent (35)")
if run and man:
    sources = {r[0] for r in store.db.execute("SELECT source FROM items").fetchall()}
    answers = [q["answer"] for q in run["questions"]]
    pts = 10 if len(answers) == 2 and all(len(a) > 40 for a in answers) else 0
    cited = [set(re.findall(r"\[([^\]]+)\]", a)) for a in answers]
    all_cites = set().union(*cited) if cited else set()
    real = {c for c in all_cites if c in sources}
    fake = all_cites - real
    pts += 5 if all_cites and not fake else 0
    print(f"  citations: {len(all_cites)} · real sources: {len(real)} · not in the index: {sorted(fake) or 'none'}")
    b2 = cited[0] if cited else set()
    b2_ok = any(c.startswith("hotel_kb.json#2") for c in b2) and bool(b2 & {"media/cctv_plantroom.mp4",
                                                                          "media/plantroom_smoke.jpg", "incident log"})
    pts += 10 if b2_ok else 0
    print(f"  {'✓' if b2_ok else '✕'} plant-room answer cites the smoke procedure AND a camera/photo/incident item")
    th_ok = len(cited) > 1 and "media/voicemail_burning_th.wav" in cited[1]
    pts += 10 if th_ok else 0
    print(f"  {'✓' if th_ok else '✕'} Thai answer cites the Thai voicemail")
    ek.ok(f"+{pts}")
    score += pts
else:
    ek.warn("no agent run — run Lab 07-2")

ek.step(5, "result")
grade = "distinction" if score >= 90 else "pass" if score >= 70 else "not yet"
print(f"  {ek.bar(score, 100, 40)} {score}/100")
ek.result(f"{score}/100 — {grade}. Every point above came from a file in .runs/ or a measurement just now.")
