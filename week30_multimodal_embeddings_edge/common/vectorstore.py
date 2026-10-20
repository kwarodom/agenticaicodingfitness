"""vectorstore — a tiny, local, multimodal vector store on SQLite + numpy (Week 30).

One file on disk, no server, no network: the shape of a sovereign edge index. Each row keeps what a person
needs to trust a hit — what kind of item it is (note, image, audio, video, record), where it came from, a
human-readable title and text — next to its vector. Vectors are stored at a chosen Matryoshka dimension in
float16 (half the bytes of float32, plenty for cosine ranking), re-normalised after the cut.

    store = VectorStore(path, dim=256)
    store.add(kind="note", title="AHU condensate leak", text="…", source="kb#1", vector=v768)
    store.search(q768, k=3, kinds={"note", "image"})
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

KINDS = ("note", "image", "audio", "video", "record")


def _cut(v, dim: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)[:dim]
    return v / max(float(np.linalg.norm(v)), 1e-12)


class VectorStore:
    def __init__(self, path: str | Path, dim: int = 768):
        if dim not in (768, 512, 256, 128):
            raise ValueError("EmbeddingGemma 2 Matryoshka dims are 768, 512, 256, 128")
        self.path, self.dim = Path(path), dim
        self.db = sqlite3.connect(self.path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, title TEXT, text TEXT, source TEXT,
            meta TEXT, dim INTEGER NOT NULL, vec BLOB NOT NULL, added REAL NOT NULL)""")
        self.db.commit()
        self._cache: tuple | None = None

    def add(self, *, kind: str, vector, title: str = "", text: str = "", source: str = "", meta: dict | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        v = _cut(vector, self.dim).astype(np.float16)
        cur = self.db.execute("INSERT INTO items(kind,title,text,source,meta,dim,vec,added) VALUES(?,?,?,?,?,?,?,?)",
                              (kind, title, text, source, json.dumps(meta or {}, ensure_ascii=False), self.dim,
                               v.tobytes(), time.time()))
        self.db.commit()
        self._cache = None
        return int(cur.lastrowid)

    def __len__(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def bytes_on_disk(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def _matrix(self):
        if self._cache is None:
            rows = self.db.execute("SELECT id, kind, title, text, source, meta, vec FROM items WHERE dim = ? ORDER BY id",
                                   (self.dim,)).fetchall()
            mat = (np.vstack([np.frombuffer(r[6], dtype=np.float16) for r in rows]).astype(np.float32)
                   if rows else np.zeros((0, self.dim), dtype=np.float32))
            self._cache = (rows, mat)
        return self._cache

    def search(self, query_vector, k: int = 3, kinds: set[str] | None = None) -> list[dict]:
        rows, mat = self._matrix()
        if not rows:
            return []
        q = _cut(query_vector, self.dim)
        scores = mat @ q
        order = np.argsort(-scores)
        out = []
        for i in order:
            r = rows[int(i)]
            if kinds and r[1] not in kinds:
                continue
            out.append({"id": r[0], "kind": r[1], "title": r[2], "text": r[3], "source": r[4],
                        "meta": json.loads(r[5] or "{}"), "score": float(scores[int(i)])})
            if len(out) >= k:
                break
        return out

    def kinds(self) -> dict:
        return dict(self.db.execute("SELECT kind, COUNT(*) FROM items GROUP BY kind").fetchall())

    def close(self) -> None:
        self.db.close()
