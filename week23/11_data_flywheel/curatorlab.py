#!/usr/bin/env python3
"""REAL NeMo Curator (CPU) curation pipeline for App 11 — the flywheel's Curator step.

The flywheel demos SHOW a `nemo_curator` sketch (Pipeline([Dedup, QualityFilter, …])).
That is pedagogical; the real 1.2 API is stage-based. This module runs a **real,
CPU-only** curation pipeline on this DGX Spark so you can try the curation path for
real, on agent logs — no cloud, $0.

IMPORTANT — run this with the py3.12 side-venv, NOT the main tutorial venv::

    uv venv --python 3.12 .venv-curator
    sudo apt-get install -y python3.12-dev              # fasttext needs Python.h
    VIRTUAL_ENV=.venv-curator uv pip install "nemo-curator[text-cpu]"
    .venv-curator/bin/python week20/11_data_flywheel/curatorlab.py

Why a side-venv: `nemo-curator` requires Python <3.13 (this repo's .venv is 3.13).

What is REAL here vs. GPU-gated on this box (GB10 / CUDA 13 / Blackwell sm_121):
  • CPU-real (works now): JSONL read/write, heuristic quality filters (word-count,
    non-alnum, repeated n-grams), unicode/newline cleaning modifiers.
  • GPU-only (NOT available): exact/fuzzy/**semantic dedup** and the DeBERTa
    quality/domain/safety classifiers — they need RAPIDS (`cudf`/`cupy`/`pynvml`),
    whose wheels are cuda12-only. On a cuda12 datacenter GPU you'd add those stages;
    here we stay CPU-heuristic. (This is the caveat App 11's README should state.)
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# The side-venv this module must run under (py3.12 + nemo-curator). App 11's server
# runs in the main py3.13 venv, so it shells OUT to here via curate_via_sidevenv().
REPO = Path(__file__).resolve().parents[2]
SIDE_VENV_PY = REPO / ".venv-curator" / "bin" / "python"

SAMPLE = [
    "The HVAC agent set chiller 2 to 7C and verified the setpoint held for ten minutes across three sensors.",
    "ok",
    "Room 1203 alarm: smoke detected. Agent dispatched security, notified the guest, and logged a full timeline.",
    "hi",
    "Curation removes short low-signal rows so the customizer trains on substantive agent turns that mutated state.",
    "asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf",  # repetitive spam
]


def _write_sample(indir: Path) -> None:
    indir.mkdir(parents=True, exist_ok=True)
    with (indir / "agent_logs.jsonl").open("w") as f:
        for t in SAMPLE:
            f.write(json.dumps({"text": t}) + "\n")


def build_pipeline(indir: str, outdir: str):
    """A real CPU curation pipeline: read → clean → heuristic quality filters → write."""
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.stages.text.filters.score_filter import ScoreFilter
    from nemo_curator.stages.text.filters.heuristic.string import WordCountFilter
    from nemo_curator.stages.text.filters.heuristic.repetition import RepeatingTopNGramsFilter
    from nemo_curator.stages.text.modifiers.modifier import Modify
    from nemo_curator.stages.text.modifiers.string.newline_normalizer import NewlineNormalizer
    from nemo_curator.stages.text.modifiers.unicode.unicode_reformatter import UnicodeReformatter

    return Pipeline(name="agent-log-curation", description="CPU heuristic curation of agent logs", stages=[
        JsonlReader(file_paths=indir),
        Modify([UnicodeReformatter(), NewlineNormalizer()]),                       # clean
        ScoreFilter(WordCountFilter(min_words=5, max_words=10000), text_field="text"),   # drop low-signal
        ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.18),
                    text_field="text"),                                            # drop repetitive spam
        JsonlWriter(path=outdir),
    ])


def curate(indir: str, outdir: str) -> dict:
    """Run the pipeline and return before/after counts + the kept rows."""
    n_in = sum(1 for f in glob.glob(f"{indir}/*.jsonl") for _ in open(f))
    build_pipeline(indir, outdir).run()                     # default XennaExecutor (local Ray, CPU)
    kept = [json.loads(l) for f in glob.glob(f"{outdir}/*.jsonl") for l in open(f)]
    return {"in": n_in, "kept": len(kept), "rows": kept}


def sidevenv_ready() -> bool:
    """Fast check: does the py3.12 side-venv exist with nemo_curator installed?"""
    if not SIDE_VENV_PY.exists():
        return False
    return bool(glob.glob(str(REPO / ".venv-curator" / "lib" / "python3.12" / "site-packages" / "nemo_curator")))


def curate_via_sidevenv(base: str | Path = "/tmp/curatorlab") -> dict | None:
    """Run THIS file under the side-venv and parse its result — for callers on py3.13.

    Returns {in, kept, rows} or None if the side-venv isn't ready / the run failed.
    """
    if not sidevenv_ready():
        return None
    env = {**os.environ, "RAY_DISABLE_USAGE_STATS": "1"}
    proc = subprocess.run([str(SIDE_VENV_PY), __file__, str(base)],
                          capture_output=True, text=True, env=env, timeout=300)
    m = re.search(r"INPUT rows:\s*(\d+)\s+KEPT rows:\s*(\d+)", proc.stdout)
    if not m:
        return None
    rows = re.findall(r"^\s*·\s+(.*)$", proc.stdout, flags=re.MULTILINE)
    return {"in": int(m.group(1)), "kept": int(m.group(2)), "rows": rows}


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/curatorlab")
    indir, outdir = base / "in", base / "out"
    if not glob.glob(f"{indir}/*.jsonl"):
        _write_sample(indir)
    print(f"NeMo Curator (CPU) · in={indir} out={outdir}\n")
    r = curate(str(indir), str(outdir))
    print(f"\nINPUT rows: {r['in']}   KEPT rows: {r['kept']}   (dropped {r['in'] - r['kept']} low-signal/spam)")
    for row in r["rows"]:
        print("  ·", row["text"][:72])
    print("\nGPU-gated on this cu13/Blackwell box (need RAPIDS cuda12): semantic/fuzzy dedup,")
    print("DeBERTa quality/domain/safety classifiers. Add those stages on a cuda12 GPU.")


if __name__ == "__main__":
    main()
