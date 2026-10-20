#!/usr/bin/env python3
"""PART 2 · Curate — turn raw logs into training data  [INTERMEDIATE]

Raw production logs are 90% noise. NeMo Curator is the pipeline that turns them into
a clean, safe, labeled dataset: dedup → quality filter → PII scrub → label. This is
where "self-evolving" earns its keep — garbage traffic in would mean a worse model out.

Run:  python demos/step02_curate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
import curatorlab  # noqa: E402

STAGES = [
    ("raw logs",       1_000_000, "everything the agent saw in production"),
    ("dedup",            420_000, "drop near-identical prompts (semantic + exact)"),
    ("quality filter",   180_000, "keep coherent, on-task, successful traces"),
    ("PII scrub",        180_000, "redact secrets/PII — nothing sensitive is trained on"),
    ("label (LLM-judge)", 62_000, "teacher model scores/labels the keepers"),
]

CODE = """\
# NeMo Curator (sketch) — runs on the DGX, data never leaves the box
from nemo_curator import Pipeline, Dedup, QualityFilter, PIIRedactor, LLMLabeler
pipe = Pipeline([
    Dedup(method="semantic", threshold=0.92),
    QualityFilter(min_score=0.6),
    PIIRedactor(entities=["EMAIL","PHONE","SECRET","NAME"]),
    LLMLabeler(judge="nemotron-3-super", rubric="was the agent correct + helpful?"),
])
dataset = pipe.run("s3://prod-logs/agent/*.jsonl")   # → curated, labeled, safe
"""


def _real() -> None:
    """REAL: run an actual NeMo Curator CPU pipeline via the py3.12 side-venv."""
    print("Running a REAL NeMo Curator (CPU) pipeline on sample agent logs")
    print("(read → unicode/newline clean → word-count + repeating-n-gram filters → write)…\n")
    r = curatorlab.curate_via_sidevenv("/tmp/curatorlab_demo")
    if not r:
        _sim(note=False)
        return
    print(f"  INPUT rows: {r['in']}   KEPT rows: {r['kept']}   "
          f"(dropped {r['in'] - r['kept']} low-signal/spam)\n")
    print("  Kept (substantive) traces:")
    for row in r["rows"]:
        print(f"    · {row[:66]}")
    print("\n  This is the same funnel as the SIM, run for real on this DGX — $0, data never")
    print("  left the box. GPU-gated here (need RAPIDS cuda12): semantic/fuzzy dedup + the")
    print("  DeBERTa quality/domain/safety classifiers. Add those on a cuda12 GPU.")


def _sim(note: bool = True) -> None:
    print("The curation funnel (1M raw traces → a clean training set):\n")
    print(f"  {'stage':<20}{'rows':>12}   what it does")
    print("  " + "─" * 78)
    for name, rows, what in STAGES:
        bar = "█" * max(1, round(rows / 1_000_000 * 24))
        print(f"  {name:<20}{rows:>12,}   {what}")
        print(f"  {'':<20}{bar}")
    print()
    print(CODE)
    if note:
        print("(Tip: create the py3.12 side-venv + `uv pip install nemo-curator[text-cpu]` to run")
        print(" this funnel for REAL on the DGX — see week20/11_data_flywheel/curatorlab.py.)\n")


def main() -> None:
    view.banner("PART 2", "Curate — raw logs into training data", "INTERMEDIATE")
    view.mode_line()

    if curatorlab.sidevenv_ready():
        _real()
    else:
        _sim()
    print("\nTakeaway: Curator is the quality gate of self-evolution — messy logs become")
    print("clean, labeled, safe examples. Next: fine-tune a small model on them.")


if __name__ == "__main__":
    main()
