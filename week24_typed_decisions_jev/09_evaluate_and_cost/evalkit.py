#!/usr/bin/env python3
"""evalkit — the small evaluation helpers shared by Lab 09's scripts.

Not a lab itself (the runner only lists labs/ and exercises/). Plain Python:
    load_rows()       read a labeled JSONL file
    run_intent_eval() ask Jev the intent questions for every row (one call per row)
    gate()            the accept / send-to-review decision
    metrics()         accuracy, coverage, accepted accuracy, wrong-accepted
    confusion()       gold × predicted counts
    percentile()      nearest-rank p50 / p95
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent                   # …/week24/09_evaluate_and_cost
WEEK24 = MODULE.parent
sys.path.insert(0, str(WEEK24 / "common"))
sys.path.insert(0, str(WEEK24 / "jev_lab"))
import jevkit      # noqa: E402
import jev_lab     # noqa: E402  (the reference lab: same questions + prepare())

DATA = MODULE / "data" / "intent_eval.jsonl"
RUNS = MODULE / ".runs"
QUESTIONS = jev_lab.QUESTIONS["intent"]                    # 7 questions: 1 choice, 5 nouls, 1 score
LABELS = list(QUESTIONS["intent"]["criteria"])


def load_rows(path: Path = DATA) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if r.get("expected", {}).get("intent") not in LABELS:
            raise ValueError(f"row {r.get('id')}: expected.intent must be one of {LABELS}")
    return rows


def run_intent_eval(rows: list[dict], *, quiet: bool = True) -> list[dict]:
    """One Jev call per row. Returns one record per row — failures included,
    because a failed call still counts against coverage."""
    out = []
    for r in rows:
        rec = {"id": r["id"], "language": r.get("language", "en"), "gold": r["expected"]["intent"]}
        try:
            state = jev_lab.prepare("intent", r)             # allow-lists fields: labels never sent
            resp = jevkit.ask(state, QUESTIONS, quiet=quiet)
            a = jevkit.validate(resp, QUESTIONS)["intent"]
            rec.update(ok=True, pred=a["choice"], probabilities=a["probabilities"],
                       confidence=a["confidence"], latency_ms=resp.get("_latency_ms"),
                       input_tokens=resp["usage"]["input_tokens"], source=resp["_source"],
                       model=resp["model"])
        except Exception as e:  # noqa: BLE001 — record, never crash the eval
            rec.update(ok=False, error=str(e)[:160])
        out.append(rec)
    return out


def gate(rec: dict, threshold: float | None = None) -> bool:
    """Accept for automatic routing? threshold=None → the reference lab's
    illustrative gate (confidence ≥ .75, top ≥ .80, margin ≥ .20).
    A number → a simple top-probability gate. 'unknown' is never auto-routed."""
    if not rec.get("ok") or rec["pred"] == "unknown":
        return False
    if threshold is None:
        return jev_lab.accepted({"confidence": rec["confidence"], "probabilities": rec["probabilities"]})
    return max(rec["probabilities"].values()) >= threshold


def metrics(recs: list[dict], threshold: float | None = None) -> dict:
    good = [r for r in recs if r.get("ok")]
    acc = [r for r in good if gate(r, threshold)]
    right = lambda rs: sum(r["pred"] == r["gold"] for r in rs)
    return {
        "rows": len(recs),
        "failures": len(recs) - len(good),
        "raw_accuracy": right(good) / len(good) if good else None,
        "coverage": len(acc) / len(recs) if recs else 0.0,
        "accepted": len(acc),
        "accepted_accuracy": right(acc) / len(acc) if acc else None,
        "wrong_accepted": len(acc) - right(acc),
    }


def confusion(recs: list[dict]) -> dict:
    m: dict = {}
    for r in recs:
        if r.get("ok"):
            m.setdefault(r["gold"], {}).setdefault(r["pred"], 0)
            m[r["gold"]][r["pred"]] += 1
    return m


def percentile(values: list[float], q: float):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[max(0, math.ceil(q * len(vals)) - 1)]


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def save_run(recs: list[dict], name: str = "intent_answers.json") -> Path:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / name
    path.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_run(name: str = "intent_answers.json") -> list[dict] | None:
    path = RUNS / name
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
