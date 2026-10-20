#!/usr/bin/env python3
"""Lab 21-2 · Which playbook for my goal? A TF-IDF recommender over the playbook READMEs.

Describe what you want to do in plain words ("turn my webcam into something a model can describe"). The lab
scores every playbook's overview (title, tagline, "Basic idea", "What you'll accomplish", prerequisites)
with TF-IDF and cosine similarity — standard library only, no model, no network — and prints the top three
with the words that matched, whether it runs on a DGX Spark, and which Week 25 module teaches it.

Run:  .venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py
      .venv/bin/python week25/21_playbook_atlas/labs/lab21_2_goal_recommender.py "fine-tune a robot policy" --spark
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparkkit import banner, note, result, step, table  # noqa: E402
from lab21_1_playbook_atlas import build_link, load_playbooks  # noqa: E402

# Words every README shares (hardware boilerplate, markdown, verbs of setup) carry no signal about the goal.
STOPWORDS = set("""
a an and are as at be by can for from has have how if in into is it its of on or that the this to up use i me my
using used with you your yours will via than then them they we our not no do does all any more most each
other one two also only such so these those what when where which who why about after before over under
hardware platform platforms supported dgx spark station memory unified os linux gb tb step steps playbook
playbooks run running runs set setup install installed installing instructions overview prerequisites
required optional basic idea accomplish know before starting familiarity experience understanding
recommended default local settings multi node capable matrix confirm whether applies see table above
time risk rollback estimated min minutes hour hours https http www com github nvidia md html
""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words of 2+ characters, split on anything that is not a letter or digit, minus stopwords."""
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def doc_text(pb: dict) -> str:
    """What the recommender reads per playbook. Title and tagline count extra because they state the purpose."""
    return " ".join([pb["title"]] * 3 + [pb["tagline"]] * 2 + [pb["overview"]])


def build_index(pbs: list[dict]) -> tuple[list[dict], dict[str, float]]:
    """TF-IDF vectors, one per playbook. idf = log((1+N)/(1+df)) + 1 (smoothed, like scikit-learn)."""
    toks = [tokenize(doc_text(pb)) for pb in pbs]
    n = len(toks)
    df = Counter(w for t in toks for w in set(t))
    idf = {w: math.log((1 + n) / (1 + d)) + 1 for w, d in df.items()}
    vecs = []
    for t in toks:
        tf = Counter(t)
        v = {w: (1 + math.log(c)) * idf[w] for w, c in tf.items()}          # sub-linear tf
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs, idf


def recommend(query: str, pbs: list[dict], index, k: int = 3, spark_only: bool = False) -> list[tuple]:
    """→ [(score, playbook, [matched words by weight]), …] best first."""
    vecs, idf = index
    q = Counter(w for w in tokenize(query) if w in idf)
    qv = {w: (1 + math.log(c)) * idf[w] for w, c in q.items()}
    qn = math.sqrt(sum(x * x for x in qv.values())) or 1.0
    scored = []
    for pb, v in zip(pbs, vecs):
        if spark_only and not pb["spark"]:
            continue
        contrib = {w: qv[w] / qn * v[w] for w in qv if w in v}
        if contrib:
            words = [w for w, _ in sorted(contrib.items(), key=lambda kv: -kv[1])]
            scored.append((sum(contrib.values()), pb, words))
    return sorted(scored, key=lambda s: -s[0])[:k]


DEMO_GOALS = [
    "generate images and short videos from text prompts",
    "describe what my webcam sees in real time",
    "search and summarize hours of security camera footage",
    "train a humanoid robot with reinforcement learning in simulation",
    "speed up pandas and scikit-learn without changing my code",
    "optimize a stock portfolio for tail risk",
    "analyze single-cell RNA sequencing data on the GPU",
    "write my own GPU kernels in Python and benchmark them",
    "let my team reach my machine and share GPU access remotely",
    "serve an LLM with an OpenAI-compatible API and tool calling",
]


def show(query: str, pbs, index, spark_only: bool) -> None:
    hits = recommend(query, pbs, index, 3, spark_only)
    print(f"→ goal: \"{query}\"")
    if not hits:
        print("  (no playbook shares a meaningful word with this goal — rephrase with concrete nouns)")
        return
    rows = [[f"{s:.3f}", pb["slug"], "✓" if pb["spark"] else "✕", "Module " + pb["module"],
             ", ".join(words[:4])] for s, pb, words in hits]
    table(rows, ["score", "playbook", "Spark?", "taught in", "matched words"])
    best = hits[0][1]
    print(f"· cite: {best['title']} — {build_link(best)}")
    print(f"        source: dgx-spark-playbooks/nvidia/playbook-{best['slug']}/README.md")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    spark_only = "--spark" in sys.argv
    banner("Lab 21-2 · which playbook for my goal?",
           "TF-IDF over the READMEs · standard library · offline", status=False)

    step(1, "index every playbook overview")
    pbs = load_playbooks()
    index = build_index(pbs)
    vocab = len(index[1])
    note(f"{len(pbs)} playbooks · vocabulary {vocab} words after stopwords · "
         f"{'Spark-supported only' if spark_only else 'all platforms (add --spark to filter)'}")

    goals = [" ".join(args)] if args else DEMO_GOALS
    step(2, f"recommend for {len(goals)} goal{'s' if len(goals) > 1 else ''}")
    for g in goals:
        print()
        show(g, pbs, index, spark_only)

    step(3, "where keyword search breaks — two goals it gets wrong (all platforms)")
    for g, want in (("split one GPU between several users", "mig"),
                    ("chat with a model I trained from scratch", "nanochat")):
        print()
        show(g, pbs, index, False)                    # all platforms: the right answers are Station-only
        got = [pb["slug"] for _, pb, _ in recommend(g, pbs, index, 3)]
        print(f"◆ a person would pick '{want}': it is "
              + (f"#{got.index(want) + 1} here" if want in got else "not in the top three"))
    note("MIG's README says 'partition' and 'isolated instances', never 'split' or 'users'; 'scratch' is a "
         "rare word that hermes-agent happens to use. A keyword recommender only knows the author's words. "
         "Exercise 21 turns this into a test set.")
    result("Scores are relative (cosine of TF-IDF vectors), not probabilities. Read the top three, then the README.")


if __name__ == "__main__":
    main()
