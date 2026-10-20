#!/usr/bin/env python3
"""Lab 19-3 · Playbook finder: "which playbook should I use for X?", answered with citations.

The retrieval half of RAG in ~100 lines of standard-library Python: split every
DGX Spark playbook README into "## " sections, score them with TF-IDF + cosine
similarity, and return the best playbooks with a citation (file § heading) for each.
No embeddings, no vector database, no new packages, so every number is inspectable.

Then two things a real RAG system needs:
  • an eval: eight questions with known right answers → hit@1 and hit@3
  • a grounded answer: the top 3 sections go to an LLM that must cite them as [1]-[3],
    and the lab checks every citation points at a source it was given

The multi-agent chatbot's search_documents tool does the same job with Qwen3-Embedding
vectors in Milvus (k=8, chunks of 1000 characters, overlap 200).

Run: .venv/bin/python week25/19_multi_agent_rag_kg/labs/lab03_playbook_finder.py [--ask "..."] [--no-llm]
"""
import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, banner, chat_any, note, pick_laptop_model, result, show_chat, step, table, warn  # noqa: E402

PB = ROOT / "dgx-spark-playbooks" / "nvidia"
SPARK_MODEL = "qwen3.6:35b-a3b-mtp-q4_K_M"      # the model Module 18 pulled on the Spark
STOP = set("""a an and are as at be by can for from has have how if in into is it its of on or that the this to
use using with you your will when which what we our not all any more can do does each also than then there these
they via see step run""".split())
EVAL = [  # question → playbook folder names that count as a correct answer
    ("fine-tune a vision language model on my own images", {"vlm-finetuning"}),
    ("high-throughput LLM serving with tensor parallelism across two Sparks", {"vllm", "sglang", "trt-llm"}),
    ("turn my documents into a knowledge graph I can query", {"txt2kg"}),
    ("run Claude Code against a local model", {"cli-coding-agent", "local-coding-agent"}),
    ("cable two Sparks together for distributed workloads", {"connect-two-sparks"}),
    ("sandbox and govern an AI agent's network and file access", {"openshell", "nemoclaw"}),
    ("a supervisor agent that delegates to coding and RAG agents", {"multi-agent-chatbot"}),
    ("code completion in VS Code from a model on my Spark", {"vibe-coding"}),
]


def tokens(text: str) -> list[str]:
    return [t.strip(".-") for t in re.findall(r"[a-z0-9][a-z0-9+.\-]*", text.lower())
            if t.strip(".-") and t.strip(".-") not in STOP and len(t.strip(".-")) > 1]


def load_sections() -> list[dict]:
    secs = []
    for readme in sorted(PB.glob("playbook-*/README.md")):
        name = readme.parent.name.removeprefix("playbook-")
        text = readme.read_text(encoding="utf-8", errors="replace")
        title = (re.search(r"^# (.+)$", text, re.M) or [None, name])[1].strip()
        for part in re.split(r"^## ", text, flags=re.M)[1:]:
            head, _, body = part.partition("\n")
            if head.strip() in ("Table of Contents",) or len(body.strip()) < 40:
                continue
            prose = re.sub(r"```.*?```", " ", body, flags=re.S)
            secs.append({"playbook": name, "title": title, "heading": head.strip(), "body": body.strip(),
                         "snippet": re.sub(r"\s+", " ", re.sub(r"[#*`>|\[\]]", "", prose)).strip()[:150],
                         "tf": Counter(tokens(f"{title} {name.replace('-', ' ')} {head} {body}"))})
    return secs


def build_index(secs: list[dict]) -> dict:
    df = Counter(t for s in secs for t in s["tf"])
    n = len(secs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
    for s in secs:
        vec = {t: (1 + math.log(c)) * idf[t] for t, c in s["tf"].items()}
        s["norm"] = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        s["vec"] = vec
    return idf


def search(q: str, secs: list[dict], idf: dict, k: int = 3) -> list[tuple[float, dict]]:
    """Top-k PLAYBOOKS (best section each) by cosine similarity."""
    qv = {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in Counter(tokens(q)).items()}
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    best = {}
    for s in secs:
        score = sum(w * s["vec"].get(t, 0) for t, w in qv.items()) / (qn * s["norm"])
        if score > best.get(s["playbook"], (0, None))[0]:
            best[s["playbook"]] = (score, s)
    return sorted(best.values(), key=lambda x: -x[0])[:k]


def cite(s: dict) -> str:
    return f"nvidia/playbook-{s['playbook']}/README.md § {s['heading']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default="", help="your own question")
    ap.add_argument("--no-llm", action="store_true", help="retrieval only, no grounded answer")
    args = ap.parse_args()

    banner("Lab 19-3 · playbook finder — TF-IDF retrieval with citations, an eval, and a grounded answer",
           "standard library only · every DGX Spark playbook README · hit@1 / hit@3")
    if not PB.is_dir():
        warn(f"playbooks not found at {PB}.")
        print("$ git clone https://github.com/NVIDIA/dgx-spark-playbooks   # run at the repo root (gitignored)")
        result("Clone the playbooks, then rerun this lab.")
        return

    step(1, "index: split every playbook README on '## ' headings")
    secs = load_sections()
    idf = build_index(secs)
    print(f"◆ {len({s['playbook'] for s in secs})} playbooks · {len(secs)} sections · {len(idf):,} distinct terms")

    step(2, "eval: eight questions with known answers")
    rows, h1, h3 = [], 0, 0
    for q, gold in EVAL:
        hits = search(q, secs, idf)
        names = [s["playbook"] for _, s in hits]
        r1, r3 = names[:1] and names[0] in gold, bool(gold & set(names))
        h1, h3 = h1 + r1, h3 + r3
        rows.append(["✓" if r1 else ("≈" if r3 else "✕"), q[:46], ", ".join(names), f"{hits[0][0]:.2f}"])
    table(rows, ["", "question", "top 3 playbooks", "score"])
    print(f"◆ hit@1 = {h1}/{len(EVAL)} · hit@3 = {h3}/{len(EVAL)}   (✓ right at #1 · ≈ right in top 3 · ✕ missed)")

    q = args.ask or "Which playbook should I use to build a knowledge graph from my PDFs and ask it questions?"
    step(3, f"retrieve for: {q!r}")
    hits = search(q, secs, idf)
    for i, (score, s) in enumerate(hits, 1):
        print(f"[{i}] {score:.3f}  {s['title']}  —  {cite(s)}")
        print(f"      {s['snippet']}…")

    if args.no_llm:
        result("Retrieval only (--no-llm). Every answer above carries its source.")
        return
    step(4, "grounded answer: the LLM may use ONLY these three sources, and must cite them")
    context = "\n\n".join(f"[{i}] {s['title']} — {s['heading']}\n{s['body'][:900]}" for i, (_, s) in enumerate(hits, 1))
    r = chat_any("ollama", SPARK_MODEL, [
        {"role": "system", "content": "Answer the question using ONLY the numbered sources. Name the playbook to "
                                      "use and cite every claim like [1]. If the sources do not answer it, say so. "
                                      "At most 4 sentences."},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {q}"}],
        max_tokens=200, laptop_model=pick_laptop_model(["gemma4:12b", "nemotron-3-nano"]))
    show_chat(r)
    cited = sorted({int(c) for c in re.findall(r"\[(\d+)\]", r.get("text") or "")})
    if r["source"] in ("spark", "laptop", "recorded"):
        bad = [c for c in cited if not 1 <= c <= len(hits)]
        if not cited:
            warn("the answer cites nothing — a grounded answer must point at its sources")
        elif bad:
            warn(f"the answer cites [{bad[0]}], which it was never given — a hallucinated citation")
        else:
            print(f"✓ every citation {cited} points at a source it was given: "
                  + " · ".join(cite(hits[c - 1][1]) for c in cited))
    if r["source"] == "laptop":
        note("LAPTOP STAND-IN for the answer; the retrieval is plain arithmetic and identical on every machine.")
    elif r["source"] == "reference":
        note("DRY: no model endpoint, so no grounded answer. Steps 1–3 above are real and need no model.")
    result("Retrieval decides what the model can know; citations let you check it. Measure retrieval "
           "(hit@k) before you tune the prompt.")


if __name__ == "__main__":
    main()
