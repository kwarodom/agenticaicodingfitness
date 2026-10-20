#!/usr/bin/env python3
"""localkit — run open-weights System One models on THIS machine (Module 16).

    import localkit as L            # first: re-runs the lab inside week24_typed_decisions_jev/.venv-local if needed
    laya = L.laya()                 # Router (English + multilingual checkpoints), or None
    decide = L.decide(multi=False)  # GLiNER2.5-Decide (English) / GLiNER2.5-multi-Decide, or None
    ner = L.gliner()                # classic GLiNER (multilingual span extractor), or None

Why a second venv? The course .venv stays light (stdlib Jev client, no torch). The local models
need torch + transformers, and classic `gliner` pins transformers < 5.17, so they live in
week24_typed_decisions_jev/.venv-local. Create it once (≈ 1 min, ≈ 1.5 GB of packages):

    uv venv --python 3.13 week24_typed_decisions_jev/.venv-local
    uv pip install --python week24_typed_decisions_jev/.venv-local/bin/python laya "gliner2[local]" gliner tiktoken sentencepiece pillow

then download the weights once from the ⌨ terminal (≈ 3 GB, a few minutes — longer than the
runner's 150 s lab limit, so don't do it from a ▶ Run button):

    week24_typed_decisions_jev/.venv-local/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab00_download_models.py

Labs load weights with HF_HUB_OFFLINE=1, so a lab never starts a surprise download. If a model
is missing, the lab says so and replays answers recorded on 2026-10-17 (labelled RECORDED).
Nothing here sends your text anywhere: local inference is the point.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK24 = HERE.parent
LOCAL_PY = WEEK24 / ".venv-local" / "bin" / "python"
RECORDED = HERE / "recorded"
sys.path.insert(0, str(WEEK24 / "common"))

LAYA_REPO = "convaiinnovations/laya"
DECIDE_EN = "fastino/GLiNER2.5-Decide"
DECIDE_MULTI = "fastino/GLiNER2.5-multi-Decide"
GLINER_NER = "urchade/gliner_multi-v2.1"
ALL_REPOS = [LAYA_REPO, DECIDE_EN, DECIDE_MULTI, GLINER_NER]


# ── 1 · make sure we run where the packages are ──────────────────────────────
def _has(pkg: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


def use_local_venv() -> None:
    """If laya/gliner2 aren't importable here but week24_typed_decisions_jev/.venv-local exists, re-run this script there."""
    if _has("laya") and _has("gliner2"):
        return
    if LOCAL_PY.is_file() and not os.environ.get("LOCALKIT_REEXEC"):
        os.environ["LOCALKIT_REEXEC"] = "1"
        sys.stdout.flush()
        os.execv(str(LOCAL_PY), [str(LOCAL_PY), *sys.argv])


use_local_venv()
os.environ.setdefault("USE_TF", "0")                  # Laya's card: transformers + TF can deadlock
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if not os.environ.get("LOCALKIT_ALLOW_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # never download inside a lab
warnings.filterwarnings("ignore")


def where() -> str:
    if _has("laya") and _has("gliner2"):
        return "week24_typed_decisions_jev/.venv-local" if ".venv-local" in sys.executable else sys.executable
    return ""


def device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        return "no torch"
    return "cpu"


# ── 2 · loaders — None + a printed reason when unavailable ───────────────────
_CACHE: dict = {}


def _quiet_load(name: str, fn):
    if name in _CACHE:
        return _CACHE[name]
    t0 = time.perf_counter()
    try:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            obj = fn()
        print(f"   ◇ loaded {name} in {time.perf_counter() - t0:.1f} s (local, offline)")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:110]
        print(f"   ◇ {name} not available here → replaying recordings  ({type(e).__name__}: {msg})")
        obj = None
    _CACHE[name] = obj
    return obj


def laya():
    if not _has("laya"):
        return _quiet_load("Laya", lambda: (_ for _ in ()).throw(ImportError("pip install laya")))
    from laya import Router
    return _quiet_load("Laya Router", lambda: Router())


def decide(multi: bool = False):
    repo = DECIDE_MULTI if multi else DECIDE_EN
    if not _has("gliner2"):
        return _quiet_load(repo, lambda: (_ for _ in ()).throw(ImportError('pip install "gliner2[local]"')))
    from gliner2 import AutoExtractor
    return _quiet_load(repo.split("/")[1], lambda: AutoExtractor.from_pretrained(repo))


def gliner():
    if not _has("gliner"):
        return _quiet_load(GLINER_NER, lambda: (_ for _ in ()).throw(ImportError("pip install gliner")))
    from gliner import GLiNER
    return _quiet_load("gliner_multi-v2.1", lambda: GLiNER.from_pretrained(GLINER_NER))


# ── 3 · adapters: one Jev-style question set → each engine's input ────────────
def to_gliner_schema(questions: dict) -> dict:
    """Jev questions → GLiNER2.5-Decide `classify_text` tasks.

    choice → {"labels": {option: description}, "prompt": instructions}
    noul   → {"labels": ["yes", "no"],          "prompt": instructions}
    score  → {"labels": [level, …],             "prompt": instructions}   (ordinal = plain classes)
    Backtick field names (`message`) are dropped: GLiNER reads one text, not a state object.
    """
    out = {}
    for k, q in questions.items():
        prompt = str(q.get("instructions", "")).replace("`", "")
        t = q.get("type")
        if t == "choice":
            crit = q.get("criteria") or {}
            labels = {o: (d if isinstance(d, str) and d else o) for o, d in crit.items()}
            out[k] = {"labels": labels, "prompt": prompt}
        elif t == "noul":
            out[k] = {"labels": ["yes", "no"], "prompt": prompt}
        elif t == "score":
            out[k] = {"labels": [str(c) for c in q.get("criteria") or []], "prompt": prompt}
        else:
            raise ValueError(f"{k}: GLiNER2.5-Decide has no equivalent of a {t!r} question")
    return out


def from_gliner(questions: dict, raw: dict) -> dict:
    """GLiNER output → Jev-like answers. Only the LABEL is comparable; its confidence is one
    score for the winning label, not a probability distribution (Fastino says so too)."""
    answers = {}
    for k, q in questions.items():
        r = raw.get(k) or {}
        label, conf = (r.get("label"), r.get("confidence")) if isinstance(r, dict) else (r, None)
        if q["type"] == "noul":
            answers[k] = {"type": "noul", "label": label, "yes": label == "yes", "confidence": conf}
        elif q["type"] == "score":
            levels = [str(c) for c in q["criteria"]]
            answers[k] = {"type": "score", "level": levels.index(label) if label in levels else None,
                          "label": label, "confidence": conf}
        else:
            answers[k] = {"type": "choice", "choice": label, "confidence": conf}
    return answers


def run_laya(router, state, questions: dict, **kw) -> tuple[dict, float]:
    t0 = time.perf_counter()
    res = router.predict(state, questions, **kw)
    return res, (time.perf_counter() - t0) * 1000


def run_decide(model, text: str, questions: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    raw = model.classify_text(text, to_gliner_schema(questions), include_confidence=True)
    return from_gliner(questions, raw), (time.perf_counter() - t0) * 1000


# ── 4 · recordings: local results are saved so a machine without the models can still learn ──
class Recorder:
    """recorded/<name>.json = {key: value}. Write when JEV_RECORD=1 (maintainers), always read."""

    def __init__(self, name: str):
        self.path = RECORDED / f"{name}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.dirty = False

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        if os.environ.get("JEV_RECORD") == "1":
            self.data[key] = value
            self.dirty = True

    def save(self) -> None:
        if self.dirty:
            RECORDED.mkdir(exist_ok=True)
            self.data["_recorded_at"] = time.strftime("%Y-%m-%d")
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


def tag(source: str) -> str:
    return {"local": "LOCAL", "recorded": "RECORDED", "live": "LIVE"}.get(source, source.upper())


def banner_local() -> None:
    w = where()
    if w:
        print(f"▣ LOCAL models · python {sys.version.split()[0]} in {w} · Laya on {device()} (picks the best "
              f"device itself) · GLiNER on cpu (its default)")
    else:
        print("◈ no week24_typed_decisions_jev/.venv-local — local answers replay from recordings (see localkit.py to set it up)")


def as_resp(answers: dict, model: str, source: str, ms: float | None) -> dict:
    """Wrap local answers so jevkit.show() can draw them (no tokens, no bill)."""
    return {"model": model, "answers": answers, "usage": {}, "_source": source,
            "_latency_ms": round(ms, 1) if ms is not None else None}
