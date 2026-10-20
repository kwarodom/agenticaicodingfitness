#!/usr/bin/env python3
"""Lesson 05.1 — RAG: "chat with your docs" from scratch.

Run:  python week15/code/05_rag/01_chat_with_docs.py

The four moving parts of RAG:
  1. CHUNK    — split docs into bite-sized pieces
  2. EMBED    — turn each chunk's MEANING into a vector
  3. RETRIEVE — embed the question, find the nearest chunks
  4. GENERATE — paste those chunks in the prompt; answer from them ONLY

To stay runnable with zero extra dependencies, we use a simple TF-IDF +
cosine "embedding". Real systems use a neural embedding model + a vector DB
(Chroma/FAISS/pgvector) — but the FLOW is identical, and that's the lesson.

Watch it correctly say "I don't know" when the answer isn't in the docs.
"""
import math
import re
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
DOCS_DIR = Path(__file__).parent / "sample_docs"


# ---- 1. CHUNK -------------------------------------------------------------
def load_and_chunk(folder, max_words=80):
    """Read every .md/.txt file and split into ~max_words chunks (by paragraph)."""
    chunks = []
    for path in sorted(folder.glob("*")):
        if path.suffix not in (".md", ".txt"):
            continue
        for para in re.split(r"\n\s*\n", path.read_text()):
            para = para.strip()
            if not para:
                continue
            words = para.split()
            # split overly long paragraphs into max_words slices
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
    return chunks


# ---- 2. "EMBED" (TF-IDF) --------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(chunks):
    """Return (vectors, idf). Each vector is a {term: tf-idf} dict."""
    docs_tokens = [tokenize(c) for c in chunks]
    df = Counter()
    for toks in docs_tokens:
        for term in set(toks):
            df[term] += 1
    n = len(chunks)
    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}
    vectors = []
    for toks in docs_tokens:
        tf = Counter(toks)
        total = sum(tf.values()) or 1
        vectors.append({t: (c / total) * idf.get(t, 0.0) for t, c in tf.items()})
    return vectors, idf


def embed_query(question, idf):
    tf = Counter(tokenize(question))
    total = sum(tf.values()) or 1
    return {t: (c / total) * idf.get(t, 0.0) for t, c in tf.items()}


def cosine(a, b):
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---- 3. RETRIEVE ----------------------------------------------------------
def retrieve(question, chunks, vectors, idf, k=3):
    qv = embed_query(question, idf)
    scored = sorted(
        ((cosine(qv, v), c) for v, c in zip(vectors, chunks)),
        key=lambda x: x[0], reverse=True,
    )
    return [(score, chunk) for score, chunk in scored[:k] if score > 0]


# ---- 4. GENERATE ----------------------------------------------------------
def answer(question, retrieved):
    context = "\n\n".join(chunk for _, chunk in retrieved) or "(no relevant context found)"
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}"""
    return client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text


if __name__ == "__main__":
    chunks = load_and_chunk(DOCS_DIR)
    vectors, idf = build_index(chunks)
    print(f"Indexed {len(chunks)} chunks from {DOCS_DIR.name}/\n")

    questions = [
        "What temperature should the lobby be kept at?",          # in the docs
        "How do I reset my password?",                            # NOT in the docs
        "What is the energy savings target for the hotel?",       # in the docs
    ]
    for q in questions:
        print(f"Q: {q}")
        hits = retrieve(q, chunks, vectors, idf, k=3)
        print(f"   (top match score: {hits[0][0]:.2f})" if hits else "   (no matches)")
        print(f"A: {answer(q, hits)}\n")

    print("Try your own:  edit the questions list, or drop a new .md into sample_docs/")
