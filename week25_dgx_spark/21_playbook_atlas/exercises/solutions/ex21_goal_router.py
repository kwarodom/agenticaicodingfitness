#!/usr/bin/env python3
"""Exercise 21 · reference solution — the goal → playbook router and a test set it passes.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/21_playbook_atlas/exercises/ex21_goal_router.py

The checker is free and offline: it tests your tokenizer and IDF on known inputs, indexes the 64 local
playbook READMEs with them, runs three built-in goals, then runs YOUR test set. Every expected playbook
must appear in the top three. Stuck? Compare with exercises/solutions/.
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs"))
from sparkkit import banner, check  # noqa: E402
from lab21_2_goal_recommender import STOPWORDS, doc_text  # noqa: E402
from lab21_1_playbook_atlas import load_playbooks  # noqa: E402


# ── TODO 1 ── lower-case the text, split on every run of characters that is not a-z or 0-9,
#   keep words of 2+ characters that are not in STOPWORDS.
#   tokenize("Fine-Tune FLUX.1 on the GPU!") → ["fine", "tune", "flux", "gpu"]
def tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


# ── TODO 2 ── smoothed inverse document frequency for every word in `docs` (a list of token lists):
#   idf(w) = log((1 + N) / (1 + df(w))) + 1, where N = number of docs, df(w) = docs that contain w.
#   A word in every document gets 1.0; a rare word gets more.
def idf(docs: list[list[str]]) -> dict[str, float]:
    n = len(docs)
    df = Counter(w for d in docs for w in set(d))
    return {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}


# ── TODO 3 ── your test set: at least 8 (goal, expected playbook slug) pairs.
#   Rules the checker enforces: ≥ 5 different playbooks, ≥ 3 of them from this atlas (Module 21), and
#   "describe the goal, don't name the tool": a goal may not contain any 4+ letter word of its slug
#   (so no "portfolio" for portfolio-optimization, no "kernels" for cutile-kernels).
TEST_SET: list[tuple[str, str]] = [
    ("stream my laptop camera to a vision model and read what it says", "live-vlm-webui"),
    ("turn a text prompt into a picture with a node graph", "comfyui"),
    ("ask questions about hours of recorded video footage", "vss"),
    ("simulate a humanoid robot and train a locomotion policy", "isaac"),
    ("RNA sequencing clustering with scanpy on GPU", "single-cell"),
    ("minimize conditional value at risk for thousands of assets", "portfolio-optimization"),
    ("pandas and scikit-learn on the GPU with zero code changes", "cuda-x-data-science"),
    ("numpy style arrays spread across two machines", "cupynumeric"),
    ("benchmark flash attention written in a python tile DSL", "cutile-kernels"),
    ("a small desk robot that takes pictures, talks and restyles them", "spark-reachy-photo-booth"),
    ("serve a model with an OpenAI-compatible endpoint and high throughput", "vllm"),
]


# ─────────────────────────── checker — no need to edit below ────────────────
GOLD = [("describe what my webcam sees in real time", "live-vlm-webui"),
        ("optimize a stock portfolio for tail risk", "portfolio-optimization"),
        ("analyze single-cell RNA sequencing data on the GPU", "single-cell")]


def recommend(goal: str, pbs: list[dict], vecs: list[dict], weights: dict, k: int = 3) -> list[str]:
    """Cosine similarity between the goal and each playbook's TF-IDF vector (sub-linear tf)."""
    q = {w: (1 + math.log(c)) * weights[w] for w, c in Counter(tokenize(goal)).items() if w in weights}
    qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
    scores = [(sum(q[w] / qn * v.get(w, 0.0) for w in q), pb["slug"]) for pb, v in zip(pbs, vecs)]
    return [s for sc, s in sorted(scores, reverse=True)[:k] if sc > 0]


def build(pbs: list[dict]) -> tuple[list[dict], dict]:
    docs = [tokenize(doc_text(pb)) for pb in pbs]
    weights = idf(docs)
    vecs = []
    for t in docs:
        v = {w: (1 + math.log(c)) * weights[w] for w, c in Counter(t).items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / n for w, x in v.items()})
    return vecs, weights


def rule_breaks(goal: str, slug: str) -> list[str]:
    words = set(re.split(r"[^a-z0-9]+", goal.lower()))
    return [p for p in slug.split("-") if len(p) >= 4 and p in words]


def main() -> None:
    banner("Exercise 21 · goal → playbook router", "offline checker · free · no Spark needed", status=False)
    ok = True
    t = tokenize("Fine-Tune FLUX.1 on the GPU!")
    ok &= check(t == ["fine", "tune", "flux", "gpu"],
                "tokenize: 'Fine-Tune FLUX.1 on the GPU!' → fine, tune, flux, gpu",
                f"TODO 1: tokenize('Fine-Tune FLUX.1 on the GPU!') should be ['fine', 'tune', 'flux', 'gpu'] (got {t!r})")
    w = idf([["cuda", "kernel"], ["cuda"]])
    good = isinstance(w, dict) and abs(w.get("cuda", 0) - 1.0) < 1e-9 and abs(w.get("kernel", 0) - 1.4055) < 1e-3
    ok &= check(good, "idf: word in every doc = 1.0 · word in 1 of 2 docs ≈ 1.405",
                f"TODO 2: idf([['cuda','kernel'],['cuda']]) should be {{'cuda': 1.0, 'kernel': 1.405…}} (got {w!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    pbs = load_playbooks()
    slugs = {pb["slug"]: pb for pb in pbs}
    vecs, weights = build(pbs)
    gold_hits = [e in recommend(g, pbs, vecs, weights) for g, e in GOLD]
    ok &= check(all(gold_hits), f"built-in goals: {sum(gold_hits)}/{len(GOLD)} land in the top three",
                "your tokenize/idf index the READMEs, but the built-in goals miss — compare with the solution")

    bad_slug = [e for _, e in TEST_SET if e not in slugs]
    distinct = {e for _, e in TEST_SET if e in slugs}
    atlas = {e for e in distinct if slugs[e]["module"] == "21"}
    named = [(g, e, rule_breaks(g, e)) for g, e in TEST_SET if rule_breaks(g, e)]
    shape = len(TEST_SET) >= 8 and not bad_slug and len(distinct) >= 5 and len(atlas) >= 3 and not named
    ok &= check(shape, f"test set: {len(TEST_SET)} goals · {len(distinct)} playbooks · {len(atlas)} from the atlas · "
                       "no goal names its tool",
                "TODO 3: need ≥ 8 goals, ≥ 5 playbooks, ≥ 3 atlas playbooks"
                + (f" · unknown slugs: {', '.join(bad_slug)}" if bad_slug else "")
                + "".join(f" · '{g}' names '{'/'.join(b)}'" for g, _, b in named)
                + f" (now {len(TEST_SET)} goals, {len(distinct)} playbooks, {len(atlas)} atlas)")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your test set against your router (top three)")
    top1 = misses = 0
    for g, e in TEST_SET:
        got = recommend(g, pbs, vecs, weights)
        mark = "✓ #1" if got[:1] == [e] else (f"✓ #{got.index(e) + 1}" if e in got else "✕ miss")
        top1 += got[:1] == [e]
        misses += e not in got
        print(f"│ {mark:6s} {e:24s} ← \"{g}\"   (got {', '.join(got)})")
    ok = check(misses == 0, f"all {len(TEST_SET)} expected playbooks in the top three · top-1 {top1}/{len(TEST_SET)}",
               f"{misses} goal(s) miss: reword them with the words a README would use, or accept that "
               "keyword search cannot find them and pick another goal")
    if not ok:
        sys.exit(1)
    print("\n═ A test set is how you notice a recommender drifting. Add a goal the router gets wrong on purpose "
          "(lab 21-2, step 3) and see which word would fix it.")


if __name__ == "__main__":
    main()
