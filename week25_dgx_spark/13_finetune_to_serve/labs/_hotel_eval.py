"""Shared scorer for Module 13 (and the capstone): is the hotel router good enough to ship?

Not a lab (the runner lists only lab*.py). Every Module 13 lab imports it, so the fine-tune and the
baseline are scored by exactly the same rules.

The task comes from Module 09: a guest message goes in, one line of JSON comes out:
    {"department": housekeeping|engineering|front_desk|food_beverage|concierge|security,
     "priority": normal|urgent, "reply": "<short polite reply>"}

Predictions use LLaMA Factory's predict format, one JSON object per line:
    {"prompt": "...", "predict": "<model output>", "label": "<expected output>"}
so a file from `llamafactory-cli train configs/hotel_predict.yaml` on the Spark and a file this module
writes from any OpenAI endpoint are scored identically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MOD = Path(__file__).resolve().parents[1]
EVAL_FILE = MOD.parent / "09_llama_factory" / "data" / "hotel_ops_eval.json"
RUNS = MOD / ".runs"
DEPARTMENTS = ("housekeeping", "engineering", "front_desk", "food_beverage", "concierge", "security")
PRIORITIES = ("normal", "urgent")

# The ship gate. Change the numbers here, not the model.
GATE = {
    "json_valid": 0.98,        # the router's output is parsed by code; invalid JSON is an outage
    "department_acc": 0.90,
    "urgent_recall": 1.00,     # never miss smoke, a leak or a medical emergency
    "reply_policy": 0.95,      # replies must not promise a specific time
}

# "in 10 minutes", "by 6pm", "within the hour", "at 7:30" — the system prompt forbids promising a time.
TIME_PROMISE = re.compile(
    r"\b(?:in|within|by|at)\s+(?:the\s+next\s+)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?"
    r"|\d+\s*(?:min(?:ute)?s?|hours?|hrs?)|an?\s+hour|half\s+an\s+hour|the\s+hour)\b", re.I)


def load_eval(limit: int | None = None) -> list[dict]:
    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def parse_answer(text: str) -> dict | None:
    """The model's one line of JSON → dict, tolerating ```json fences and chatter around it."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    for cand in (t, *re.findall(r"\{.*?\}", t, flags=re.S)):
        try:
            d = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(d, dict):
            return d
    return None


def promises_time(reply: str, prompt: str = "") -> bool:
    """True if the reply commits to a time the guest did not ask for.
    Echoing the guest's own booking ("a table for 2 at 19:00") is fine; inventing "in 10 minutes" is not."""
    asked = set(re.findall(r"\d{1,2}(?::\d{2})?", prompt or ""))
    for m in TIME_PROMISE.finditer(reply or ""):
        nums = set(re.findall(r"\d{1,2}(?::\d{2})?", m.group(0)))
        if not nums or not nums <= asked:
            return True
    return False


def score_item(predict: str, label: str, prompt: str = "") -> dict:
    """One prediction vs its label → the facts the gate needs. prompt = the guest message (for the time rule)."""
    want = parse_answer(label) or {}
    got = parse_answer(predict)
    ok_json = bool(got) and got.get("department") in DEPARTMENTS and got.get("priority") in PRIORITIES \
        and isinstance(got.get("reply"), str) and bool(got.get("reply", "").strip())
    got = got or {}
    reply = str(got.get("reply", ""))
    return {
        "json_valid": ok_json,
        "department_ok": got.get("department") == want.get("department"),
        "priority_ok": got.get("priority") == want.get("priority"),
        "want_urgent": want.get("priority") == "urgent",
        "got_urgent": got.get("priority") == "urgent",
        "reply_policy_ok": bool(reply) and not promises_time(reply, prompt),
        "want_department": want.get("department"),
        "got_department": got.get("department"),
    }


def summarize(scores: list[dict]) -> dict:
    n = len(scores) or 1
    urgent = [s for s in scores if s["want_urgent"]]
    return {
        "n": len(scores),
        "json_valid": sum(s["json_valid"] for s in scores) / n,
        "department_acc": sum(s["department_ok"] for s in scores) / n,
        "priority_acc": sum(s["priority_ok"] for s in scores) / n,
        "urgent_recall": (sum(s["got_urgent"] for s in urgent) / len(urgent)) if urgent else 1.0,
        "urgent_n": len(urgent),
        "reply_policy": sum(s["reply_policy_ok"] for s in scores) / n,
    }


def gate(metrics: dict) -> list[tuple[str, float, float, bool]]:
    return [(k, metrics[k], v, metrics[k] >= v) for k, v in GATE.items()]


def read_predictions(path_or_text: str | Path) -> list[dict]:
    """LLaMA Factory generated_predictions.jsonl (path, or its text) → list of {prompt, predict, label}."""
    text = Path(path_or_text).read_text(encoding="utf-8") if isinstance(path_or_text, Path) else str(path_or_text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_predictions(rows: list[dict], name: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def confusion(scores: list[dict]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for s in scores:
        if not s["department_ok"]:
            key = (str(s["want_department"]), str(s["got_department"]))
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def language(text: str) -> str:
    """'th' if the guest wrote Thai script, else 'en' — enough to slice this dataset."""
    return "th" if re.search("[\u0e00-\u0e7f]", text or "") else "en"


def by_language(rows: list[dict], scores: list[dict]) -> dict[str, dict]:
    """The same metrics, per guest language. An average can hide a bias one language feels."""
    groups: dict[str, list[dict]] = {}
    for r, s in zip(rows, scores):
        groups.setdefault(language(r.get("prompt", "")), []).append(s)
    out = {}
    for lang, sc in sorted(groups.items()):
        m = summarize(sc)
        normals = [x for x in sc if not x["want_urgent"]]
        m["false_urgent"] = sum(x["got_urgent"] for x in normals) / len(normals) if normals else 0.0
        out[lang] = m
    return out
