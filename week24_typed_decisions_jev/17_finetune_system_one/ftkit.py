#!/usr/bin/env python3
"""ftkit — fine-tune, evaluate and ship local System One models (Module 17).

    import ftkit as F               # first: re-runs the lab inside week24_typed_decisions_jev/.venv-local (via Module 16's localkit)
    train, val = F.split("train"), F.split("val")
    preds, src = F.predict("laya_ft", F.split("test"))      # LOCAL if the checkpoint is here, else RECORDED
    F.report(F.split("test"), preds)                         # accuracy, macro-F1, per language, safety recall

Everything a lab needs and nothing it should re-invent:
  • the ONE question set (Q) every engine is trained and evaluated on, identical to Lab 16-5
  • the data splits and their two training formats (Laya JSONL, GLiNER2 JSONL), built from Q
  • engines: zero-shot bases, your fine-tuned checkpoints, Jev, and two cheap baselines
  • metrics with honest error bars: accuracy, macro-F1, recall, ECE, Brier, bootstrap CIs, paired wins/losses
  • a prediction cache (.runs/, gitignored) and recordings (recorded/, committed) so the ▶ Run buttons stay fast
    and a machine without the models still learns from the 2026-10-17 run

Checkpoints live in 17_finetune_system_one/.runs/ (gitignored: ~1.3 GB Laya, ~1.1 GB full GLiNER, ~5 MB LoRA).
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK24 = HERE.parent
sys.path.insert(0, str(WEEK24 / "16_local_system_one_models"))
import localkit as L  # noqa: E402  (re-execs into .venv-local; also puts week24_typed_decisions_jev/common on sys.path)
import jevkit  # noqa: E402
from jevkit import choice, noul, score  # noqa: E402

DATA = HERE / "data"
RUNS = HERE / ".runs"
RECORDED = HERE / "recorded"
CKPT = {"laya_ft": RUNS / "laya-ft", "gliner_lora": RUNS / "gliner-lora", "gliner_full": RUNS / "gliner-full"}
LAYA_BASE = "multilingual"                     # one checkpoint for English AND Thai (mmBERT-base, 322M)
GLINER_BASE = L.DECIDE_MULTI                   # fastino/GLiNER2.5-multi-Decide (mDeBERTa-v3-base, 287M)

# ── the one question set — identical to Lab 16-5, so Module 16's numbers stay comparable ─────
QUEUES = ["hvac", "housekeeping", "plumbing", "safety", "front_desk"]
Q = {
    "queue": choice("Which team should handle `message`?", {
        "hvac": "air conditioning, heating, temperature", "housekeeping": "towels, cleaning, amenities",
        "plumbing": "water leaks, drains, toilets, showers",
        "safety": "smoke, burning smell, fire, sparks, or water near electricity",
        "front_desk": "check-in, check-out, billing, room changes, stay extensions"}),
    "safety": noul("Does `message` describe smoke, burning, fire, sparks or water near electricity?"),
    "urgency": score("How urgent is `message`?", [
        "routine — whenever convenient, or nothing needs doing", "today — a problem that should be fixed today",
        "immediate — danger or damage happening now"]),
}
FIELDS = ("queue", "safety", "urgency")


# ── data ──────────────────────────────────────────────────────────────────────
def split(name: str) -> list[dict]:
    """train / val (generated, Lab 17-1) · test (hand-written) · m16 (Module 16's 18 tickets) · probe."""
    if name == "test":
        return json.loads((DATA / "test_handwritten.json").read_text(encoding="utf-8"))["tickets"]
    if name == "m16":
        return json.loads((WEEK24 / "16_local_system_one_models" / "tickets.json").read_text(encoding="utf-8"))["tickets"]
    if name == "probe":
        return json.loads((DATA / "forgetting_probe.json").read_text(encoding="utf-8"))
    path = DATA / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"✗ {path.relative_to(WEEK24)} is missing — run Lab 17-1 first (labs/lab01_build_dataset.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def to_laya_rows(tickets: list[dict]) -> list[dict]:
    """Laya's training format: the SAME state shape and question dict you serve with."""
    return [{"state": {"message": t["text"]}, "questions": Q,
             "expected": {"queue": t["queue"], "safety": bool(t["safety"]), "urgency": int(t["urgency"])}}
            for t in tickets]


def to_gliner_records(tickets: list[dict]) -> list[dict]:
    """GLiNER2's training format, built through Module 16's adapter so training sees exactly the
    labels, descriptions and prompts that classify_text() sends at inference."""
    tasks = L.to_gliner_schema(Q)
    out = []
    for t in tickets:
        gold = {"queue": t["queue"], "safety": "yes" if t["safety"] else "no",
                "urgency": tasks["urgency"]["labels"][int(t["urgency"])]}
        cls = []
        for k, task in tasks.items():
            labels = task["labels"]
            c = {"task": k, "labels": list(labels), "true_label": gold[k], "prompt": task["prompt"]}
            if isinstance(labels, dict):
                c["label_descriptions"] = dict(labels)
            cls.append(c)
        out.append({"input": t["text"], "output": {"classifications": cls}})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ── leakage: is anything in the test set a near-copy of training data? ────────
def _norm(text: str) -> str:
    import re
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


@functools.lru_cache(maxsize=None)
def shingles(text: str, n: int = 5) -> frozenset:
    s = _norm(text)
    return frozenset(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


def nearest(text: str, pool: list[dict], n: int = 5) -> tuple[float, dict | None]:
    """Highest character-5-gram Jaccard similarity to any ticket in `pool` (room numbers ignored)."""
    a, best, who = shingles(text, n), 0.0, None
    for t in pool:
        b = shingles(t["text"], n)
        j = len(a & b) / max(1, len(a | b))
        if j > best:
            best, who = j, t
    return best, who


# ── engines ───────────────────────────────────────────────────────────────────
ENGINES = {
    "majority": "always the most common training label (the floor)",
    "keywords": "hand-written keyword rules (what you'd write without ML)",
    "laya_router": "Laya zero-shot · Router (EN checkpoint for English, multilingual for Thai) — Module 16",
    "laya_base": "Laya zero-shot · multilingual checkpoint only — what you fine-tune",
    "laya_ft": "Laya fine-tuned (Lab 17-4) · multilingual + your 456 tickets",
    "gliner_base": "GLiNER2.5-multi-Decide zero-shot",
    "gliner_lora": "GLiNER2.5-multi-Decide + LoRA adapter (Lab 17-5, 5 MB)",
    "gliner_full": "GLiNER2.5-multi-Decide fully fine-tuned (Lab 17-5, 1.1 GB)",
    "jev": "Jev 1.13 hosted, zero-shot — the bar from Module 16",
}
_MODELS: dict = {}


def gl_dir(engine: str) -> Path:
    """The GLiNER checkpoint to serve: `best` (lowest validation loss) if the run saved one, else `final`."""
    best = CKPT[engine] / "best"
    return best if best.is_dir() else CKPT[engine] / "final"


def available(engine: str) -> bool:
    if engine in ("majority", "keywords", "jev"):
        return True
    if engine in CKPT:
        marker = CKPT[engine] / "rl_agent_config.json" if engine == "laya_ft" else gl_dir(engine)
        if not marker.exists():
            return False
    return bool(L.where())


def _load(engine: str):
    if engine in _MODELS:
        return _MODELS[engine]
    t0 = time.perf_counter()
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        if engine == "laya_router":
            m = L.laya()
        elif engine == "laya_base":
            import laya
            m = laya.load(LAYA_BASE)
        elif engine == "laya_ft":
            import laya
            m = laya.load(str(CKPT["laya_ft"]))
        elif engine == "gliner_base":
            m = L.decide(multi=True)
        elif engine == "gliner_lora":
            from gliner2 import AutoExtractor
            from peft import PeftModel                   # the boundary architecture loads adapters through PEFT
            m = PeftModel.from_pretrained(AutoExtractor.from_pretrained(GLINER_BASE), str(gl_dir("gliner_lora")))
        elif engine == "gliner_full":
            from gliner2 import AutoExtractor
            m = AutoExtractor.from_pretrained(str(gl_dir("gliner_full")))
        else:
            raise KeyError(engine)
    if engine not in ("laya_router", "gliner_base"):     # localkit prints its own load line
        print(f"   ◇ loaded {engine} in {time.perf_counter() - t0:.1f} s (local, offline)")
    _MODELS[engine] = m
    return m


def _norm_laya(a: dict) -> dict:
    q = a["queue"]
    return {"queue": q["choice"], "queue_p": q.get("answer_confidence", max(q["probabilities"].values())),
            "safety_p": a["safety"]["noul"], "safety": a["safety"]["noul"] >= 0.5,
            "urgency": int(round(a["urgency"]["score"])),
            "urgency_p": a["urgency"].get("answer_confidence")}


def _norm_gliner(a: dict) -> dict:
    return {"queue": a["queue"]["choice"], "queue_p": a["queue"]["confidence"], "safety_p": None,
            "safety": a["safety"]["label"] == "yes", "urgency": a["urgency"]["level"],
            "urgency_p": a["urgency"]["confidence"]}


KEYWORDS = {
    "safety": ["smoke", "burn", "spark", "fire", "shock", "ควัน", "ไหม้", "ไฟดูด", "ประกายไฟ", "ไฟแลบ"],
    "plumbing": ["toilet", "drain", "shower", "tap", "pipe", "sink", "water", "ชักโครก", "ท่อ", "ก๊อก", "ฝักบัว", "น้ำ"],
    "hvac": ["ac ", "aircon", "air con", "thermostat", "heat", "hot", "warm", "cold", "แอร์", "ร้อน"],
    "housekeeping": ["towel", "pillow", "clean", "sheet", "blanket", "ผ้า", "หมอน", "ทำความสะอาด"],
}


def _keywords(text: str) -> dict:
    s = text.lower() + " "
    q = next((k for k, words in KEYWORDS.items() if any(w in s for w in words)), "front_desk")
    urgent = any(w in s for w in ("now", "hurry", "urgent", "immediately", "ด่วน", "เดี๋ยวนี้"))
    return {"queue": q, "queue_p": None, "safety": q == "safety", "safety_p": None,
            "urgency": 2 if (urgent or q == "safety") else 1, "urgency_p": None}


def _one(engine: str, text: str) -> dict:
    state = {"message": text}
    if engine == "keywords":
        return _keywords(text)
    if engine == "jev":
        r = jevkit.ask(state, Q, quiet=True)
        a = jevkit.validate(r, Q)
        return {"queue": a["queue"]["choice"], "queue_p": max(a["queue"]["probabilities"].values()),
                "safety_p": a["safety"]["noul"], "safety": a["safety"]["noul"] >= 0.5,
                "urgency": int(round(a["urgency"]["score"])), "urgency_p": None,
                "_src": r.get("_source", jevkit.mode())}
    m = _load(engine)
    if engine.startswith("laya"):
        return _norm_laya(m.predict(state, Q)["answers"])
    return _norm_gliner(L.from_gliner(Q, m.classify_text(text, L.to_gliner_schema(Q), include_confidence=True)))


def fingerprint(engine: str) -> str:
    """Changes when a checkpoint is re-trained, so stale cached predictions are never reused."""
    if engine not in CKPT:
        return "base"
    f = CKPT[engine] / "model.safetensors" if engine == "laya_ft" else gl_dir(engine)
    if engine != "laya_ft":
        f = next(iter(sorted(f.glob("*.safetensors"))), f)
    return f"{int(f.stat().st_mtime)}-{f.stat().st_size}" if f.exists() else "missing"


def predict(engine: str, tickets: list[dict], *, majority: str = "hvac") -> tuple[dict, str]:
    """{ticket_id: prediction} and its source: LOCAL (computed now), CACHED (.runs, same checkpoint),
    RECORDED (the 2026-10-17 run, committed) or DRY/LIVE for Jev."""
    if engine == "majority":
        return {t["id"]: {"queue": majority, "queue_p": None, "safety": False, "safety_p": None,
                          "urgency": 1, "urgency_p": None} for t in tickets}, "RULE"
    if engine == "keywords":
        return {t["id"]: _keywords(t["text"]) for t in tickets}, "RULE"
    rec_path = RECORDED / f"preds_{engine}.json"
    recorded = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else {}
    cache_path = RUNS / "preds" / f"{engine}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    fp = fingerprint(engine)
    if cache.get("_fp") != fp:
        cache = {"_fp": fp}
    out, live, src = {}, available(engine), set()
    for t in tickets:
        k = sha(t["text"])
        if k in cache and engine != "jev":
            out[t["id"]], s = cache[k], "CACHED"
        elif live and engine not in ("jev",):
            t0 = time.perf_counter()
            p = _one(engine, t["text"])
            p["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            cache[k] = out[t["id"]] = p
            s = "LOCAL"
        elif engine == "jev":
            p = _one(engine, t["text"])
            s = (p.pop("_src", "dry") or "dry").upper()
            if s == "PLACEHOLDER":                         # no recording, no key: flat 0.5s are not answers
                continue
            out[t["id"]] = p
        elif k in recorded:
            out[t["id"]], s = recorded[k], "RECORDED"
        else:
            continue
        src.add(s)
        if os.environ.get("JEV_RECORD") == "1" and s in ("LOCAL", "CACHED"):
            recorded[k] = out[t["id"]]
    if any(s in ("LOCAL",) for s in src):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if os.environ.get("JEV_RECORD") == "1" and engine != "jev" and src & {"LOCAL", "CACHED"}:
        RECORDED.mkdir(exist_ok=True)
        recorded["_recorded_at"] = time.strftime("%Y-%m-%d")
        rec_path.write_text(json.dumps(recorded, ensure_ascii=False, indent=0), encoding="utf-8")
    label = "+".join(sorted(src)) if src else "MISSING"
    return out, label


class Recorder(L.Recorder):
    """Module 16's Recorder, writing to 17_finetune_system_one/recorded/<name>.json (JEV_RECORD=1 only)."""

    def __init__(self, name: str):
        self.path = RECORDED / f"{name}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.dirty = False

    def save(self) -> None:
        if self.dirty and os.environ.get("JEV_RECORD") == "1":
            RECORDED.mkdir(exist_ok=True)
            self.data["_recorded_at"] = time.strftime("%Y-%m-%d")
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


# ── metrics ───────────────────────────────────────────────────────────────────
def correct(t: dict, p: dict | None, field: str) -> bool:
    return p is not None and p.get(field) == (bool(t[field]) if field == "safety" else t[field])


def accuracy(tickets, preds, field="queue", lang=None) -> tuple[int, int]:
    rows = [t for t in tickets if (lang is None or t["lang"] == lang) and t["id"] in preds]
    return sum(correct(t, preds[t["id"]], field) for t in rows), len(rows)


def macro_f1(tickets, preds, field="queue", labels=QUEUES) -> float:
    f1s = []
    for c in labels:
        tp = sum(1 for t in tickets if t["id"] in preds and preds[t["id"]][field] == c and t[field] == c)
        fp = sum(1 for t in tickets if t["id"] in preds and preds[t["id"]][field] == c and t[field] != c)
        fn = sum(1 for t in tickets if t["id"] in preds and preds[t["id"]][field] != c and t[field] == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return statistics.mean(f1s)


def safety_recall(tickets, preds) -> tuple[int, int]:
    """Of the tickets that ARE dangerous, how many did the engine flag? The number that must not drop."""
    rows = [t for t in tickets if t["safety"] and t["id"] in preds]
    return sum(bool(preds[t["id"]]["safety"]) for t in rows), len(rows)


def bootstrap_ci(tickets, preds, field="queue", n=2000, seed=17) -> tuple[float, float]:
    """95 % interval for accuracy by resampling tickets. With 48 tickets it is WIDE — that is the lesson."""
    rows = [correct(t, preds.get(t["id"]), field) for t in tickets if t["id"] in preds]
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    accs = sorted(sum(rng.choice(rows) for _ in rows) / len(rows) for _ in range(n))
    return accs[int(0.025 * n)], accs[int(0.975 * n) - 1]


def paired(tickets, a: dict, b: dict, field="queue") -> dict:
    """Same tickets, two engines: who fixed what, who broke what (+ an exact McNemar p-value)."""
    fixed = [t for t in tickets if not correct(t, a.get(t["id"]), field) and correct(t, b.get(t["id"]), field)]
    broke = [t for t in tickets if correct(t, a.get(t["id"]), field) and not correct(t, b.get(t["id"]), field)]
    n, k = len(fixed) + len(broke), min(len(fixed), len(broke))
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"fixed": fixed, "broke": broke, "p": p}


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """Expected calibration error: |confidence − accuracy|, averaged over confidence bins."""
    pairs = [(c, y) for c, y in pairs if c is not None]
    if not pairs:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        cell = [(c, y) for c, y in pairs if lo < c <= hi or (b == 0 and c == 0)]
        if cell:
            tot += len(cell) / len(pairs) * abs(statistics.mean(c for c, _ in cell) - statistics.mean(y for _, y in cell))
    return tot


def queue_ece(tickets, preds) -> float | None:
    return ece([(preds[t["id"]].get("queue_p"), correct(t, preds[t["id"]], "queue")) for t in tickets if t["id"] in preds])


def brier_safety(tickets, preds) -> float | None:
    ps = [(preds[t["id"]]["safety_p"], t["safety"]) for t in tickets if t["id"] in preds and preds[t["id"]].get("safety_p") is not None]
    return statistics.mean((p - (1.0 if y else 0.0)) ** 2 for p, y in ps) if ps else None


def threshold_for(tickets, preds, target_error=0.05, field="queue") -> float | None:
    """Lowest confidence cut whose accepted answers have error ≤ target on THESE tickets (fit on val!)."""
    rows = sorted(((preds[t["id"]].get(f"{field}_p"), correct(t, preds[t["id"]], field)) for t in tickets
                   if t["id"] in preds and preds[t["id"]].get(f"{field}_p") is not None), reverse=True)
    best = None
    for i in range(1, len(rows) + 1):
        acc = rows[:i]
        if sum(not y for _, y in acc) / len(acc) <= target_error:
            best = acc[-1][0]
    return best


def coverage_at(tickets, preds, cut: float | None, field="queue") -> tuple[float, float]:
    """(share of tickets answered automatically, error among them) at a confidence cut."""
    rows = [(preds[t["id"]].get(f"{field}_p"), correct(t, preds[t["id"]], field)) for t in tickets if t["id"] in preds]
    if cut is None or not rows:
        return 0.0, 0.0
    acc = [y for c, y in rows if c is not None and c >= cut]
    return len(acc) / len(rows), (sum(not y for y in acc) / len(acc) if acc else 0.0)


def median_ms(preds: dict) -> float | None:
    ms = [p["ms"] for p in preds.values() if p.get("ms") is not None]
    return statistics.median(ms) if ms else None


def summary_row(name: str, tickets: list[dict], preds: dict, src: str) -> list:
    ok, n = accuracy(tickets, preds)
    lo, hi = bootstrap_ci(tickets, preds)
    en, ne = accuracy(tickets, preds, lang="en")
    th, nt = accuracy(tickets, preds, lang="th")
    sr, ns = safety_recall(tickets, preds)
    fa = sum(1 for t in tickets if t["id"] in preds and preds[t["id"]]["safety"] and not t["safety"])
    return [name, f"{ok}/{n}" if n else "—", f"{lo:.0%}–{hi:.0%}" if n else "—", f"{macro_f1(tickets, preds):.2f}" if n else "—",
            f"{en}/{ne}", f"{th}/{nt}", f"{sr}/{ns}", fa, src]


SUMMARY_HEADERS = ["engine", "queue", "95% CI", "macro-F1", "EN", "TH", "safety recall", "false alarms", "source"]


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    import platform
    import subprocess
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def mem_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            import subprocess
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (OSError, ValueError):
        return None


# ── the ship/hold gate (Lab 17-7; Exercise 17 asks you to write this yourself) ─
def ship_decision(metrics: dict, criteria: list[dict]) -> tuple[str, list[tuple[str, str, str]]]:
    """('SHIP' | 'HOLD', [(criterion id, 'PASS' | 'FAIL' | 'MISSING', detail), …]).
    A criterion with no measurement is MISSING, and MISSING holds the release just like FAIL:
    'we did not measure it' is not 'it passed'."""
    rows = []
    for c in criteria:
        v = metrics.get(c["metric"])
        if v is None:
            rows.append((c["id"], "MISSING", f"{c['metric']} not measured"))
            continue
        ok = (c.get("min") is None or v >= c["min"]) and (c.get("max") is None or v <= c["max"])
        bound = f"≥ {c['min']}" if c.get("min") is not None else f"≤ {c['max']}"
        rows.append((c["id"], "PASS" if ok else "FAIL", f"{c['metric']} = {v:g} (needs {bound})"))
    return ("SHIP" if rows and all(s == "PASS" for _, s, _ in rows) else "HOLD"), rows
