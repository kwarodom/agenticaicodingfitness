#!/usr/bin/env python3
"""Lab 05-1 · Egress proof: load, embed and search with the network fenced off — and show the log.

"It runs locally" is a claim; a log of every connection attempt is evidence. This lab wraps the whole embed
pipeline in common/egress.py's guard (only loopback allowed, every attempt recorded), then deliberately tries
two things a careless pipeline does — fetch the model card from Hugging Face, post a vector to a cloud API —
and shows both being blocked. The guard is a teaching tripwire, not a firewall (see the module text).

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_1_egress_proof.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import egress  # noqa: E402
import embedkit as ek  # noqa: E402

ek.banner("Lab 05-1 · egress proof", "the whole embed pipeline inside a loopback-only guard · every attempt logged")

ek.step(1, "settings that keep Hugging Face libraries offline")
ek.table([[k, os.environ.get(k, "(unset)")] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")],
         ["variable", "value"])
ek.note("embedkit sets HF_HUB_OFFLINE=1 once the model is in the local cache")

ek.step(2, "load + embed + search inside the guard")
with egress.guard() as log:
    model = ek.load(("text", "image", "audio"))
    kb = ek.load_kb()
    D = model.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])
    img = model.encode([{"image": ek.media("leak_under_ahu.jpg")}])
    q = model.encode("water under the air handler", prompt_name="SearchQuery")
    hit = ek.top_k(q, D, 1)[0]
s = egress.summary(log)
ek.ok(f"pipeline finished · top hit: {kb['docs'][hit[0]]['title']} ({hit[1]:.3f})")
(ek.ok if s["blocked"] == 0 and s["outbound_allowed"] == 0 else ek.warn)(
    f"connection attempts during the pipeline: {s['attempts']} · blocked {s['blocked']} · left this machine: "
    f"{s['outbound_allowed']}")

ek.step(3, "now try to leak — the guard must block both")
with egress.guard() as log2:
    for what, call in (
        ("fetch the model card from huggingface.co",
         lambda: urllib.request.urlopen("https://huggingface.co/google/embeddinggemma-2", timeout=5)),
        ("POST a query vector to a cloud embedding API",
         lambda: urllib.request.urlopen(urllib.request.Request(
             "https://api.example-embeddings.com/v1/store", data=json.dumps({"v": q[:8].tolist()}).encode(),
             method="POST"), timeout=5)),
    ):
        try:
            call()
            ek.warn(f"{what}: it went through?!")
        except Exception as e:  # noqa: BLE001
            reason = getattr(e, "reason", e)
            ek.ok(f"{what}: blocked — {str(reason)[:70]}")
egress.report(log2)

ek.step(4, "what WOULD have left the building with a cloud embedding API")
ek.table([["every guest voicemail, photo and note", "sent to embed"],
          ["every staff question", "sent to embed"],
          ["the vectors themselves", "stored by the provider — and text can be largely rebuilt from them "
                                     "(Morris et al., EMNLP 2023)"]], ["data", "fate"])
ek.result("local embedding is the step that keeps sovereignty cheap: the bulky, personal data (voices, faces, "
          "rooms) never crosses the wire, and neither do the vectors.")
