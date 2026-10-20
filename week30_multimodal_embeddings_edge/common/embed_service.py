#!/usr/bin/env python3
"""embed_service — a tiny always-on EmbeddingGemma 2 service for a Mac mini (Week 30, Module 08).

The Mac does the embedding; small devices on the same network (an ESP32, a Pi, a phone app, a BMS gateway) send
text and get vectors or search hits back — the video's "Arduino asks the Mac Studio" pattern, for embeddings.
MLX 4-bit text model, the SYNTHETIC hotel knowledge base pre-embedded at start-up.

    week30_multimodal_embeddings_edge/.venv-mlx/bin/python week30_multimodal_embeddings_edge/common/embed_service.py                     # 127.0.0.1:8765 (this Mac only)
    EMBED_TOKEN=$(openssl rand -hex 24) week30_multimodal_embeddings_edge/.venv-mlx/bin/python week30_multimodal_embeddings_edge/common/embed_service.py --host 0.0.0.0

Safety: it binds to loopback by default. It refuses to listen on any other address unless EMBED_TOKEN is set
(24+ characters),
and then every request must send `Authorization: Bearer <token>`. Anyone who can reach it can read the hotel
knowledge base through /search — put it on a trusted network only.

    POST /embed   {"texts": [...], "task": "search"|"document"|null, "dim": 768|512|256|128}  → {"vectors": [...]}
    POST /search  {"q": "...", "k": 3}                                                         → {"hits": [...]}
    GET  /health                                                                                → {"ok": true, ...}
"""

import argparse
import hmac
import ipaddress
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedkit as ek  # noqa: E402
import mackit  # noqa: E402

PREFIX = {"search": "task: search result | query: ", "document": "title: none | text: ", None: ""}


def build_app(variant: str = "4bit"):
    import numpy as np
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel

    embedder = mackit.MLXEmbedder(variant)
    kb = ek.load_kb()
    doc_vecs = embedder.encode([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]])
    token = os.environ.get("EMBED_TOKEN", "")
    started = time.time()
    app = FastAPI(title="Alto Edge Memory — embedding service")

    def check(req: Request) -> None:
        if not token:
            return
        sent = req.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(sent, token):
            raise HTTPException(401, "missing or wrong token")

    class EmbedReq(BaseModel):
        texts: list[str]
        task: str | None = "search"
        dim: int = 768

    class SearchReq(BaseModel):
        q: str
        k: int = 3

    @app.get("/health")
    def health(req: Request):
        check(req)
        return {"ok": True, "model": embedder.repo, "docs": len(kb["docs"]), "uptime_s": round(time.time() - started)}

    @app.post("/embed")
    def embed(body: EmbedReq, req: Request):
        check(req)
        if body.dim not in ek.DIMS or body.task not in PREFIX or not 0 < len(body.texts) <= 64:
            raise HTTPException(422, "dim ∈ 768/512/256/128 · task ∈ search/document/null · 1–64 texts")
        v = embedder.encode([PREFIX[body.task] + t[:4000] for t in body.texts])[:, :body.dim]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return {"dim": body.dim, "vectors": np.round(v, 5).tolist()}

    @app.post("/search")
    def search(body: SearchReq, req: Request):
        check(req)
        q = embedder.encode([PREFIX["search"] + body.q[:2000]])[0]
        return {"hits": [{"title": kb["docs"][i]["title"], "score": round(s, 3)}
                         for i, s in ek.top_k(q, doc_vecs, k=max(1, min(body.k, 10)))]}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--variant", default="4bit", choices=sorted(mackit.MLX_REPOS))
    a = ap.parse_args()
    loopback = a.host == "localhost" or ipaddress.ip_address(a.host).is_loopback
    if not loopback and len(os.environ.get("EMBED_TOKEN", "")) < 24:
        sys.exit("✕ refusing to listen on a network address without a long EMBED_TOKEN (24+ characters) — "
                 "e.g. EMBED_TOKEN=$(openssl rand -hex 24)")
    import uvicorn
    uvicorn.run(build_app(a.variant), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
