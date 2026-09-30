#!/usr/bin/env python3
"""tracekit — Module 05's small, standard-library reader for NAT traces (and a few shared paths).

NAT 1.9's `file` tracing exporter is a RAW exporter: it writes one JSON line per IntermediateStep event
(WORKFLOW_START, FUNCTION_START, LLM_START, LLM_NEW_TOKEN, LLM_END, TOOL_START, TOOL_END, … _END), not
finished OpenTelemetry spans. A span is a START and an END that share the same payload UUID. This module
pairs them, so the labs can count LLM vs TOOL spans, add up tokens and show durations.

    from tracekit import load_events, spans, summary
    ev = load_events(".runs/traces/lab05_1_trace.jsonl")
    sp = spans(ev)            # [{kind, name, start, end, dur_s, tokens, uuid, open}]
    print(summary(sp))

The same event shape streams from `nat serve` on /v1/workflow/full (as `intermediate_data:` SSE lines,
with the event inside a JSON string field `payload`). `events_from_sse()` turns that stream into the same list.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).resolve().parent                 # …/week26/05_tracing
RUNS = MODULE / ".runs"
CONFIGS = MODULE / "configs"
TRACES = RUNS / "traces"


def load_events(path: str | Path) -> list[dict]:
    """The `payload` dict of every line the NAT file exporter wrote (bad lines are skipped, not fatal)."""
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pl = rec.get("payload") if isinstance(rec, dict) else None
        if isinstance(pl, dict) and "event_type" in pl:
            out.append(pl)
    return out


def events_from_sse(text: str) -> list[dict]:
    """Events from a /v1/workflow/full stream: lines `intermediate_data: {"type": …, "payload": "<json>"}`."""
    out = []
    for line in text.splitlines():
        if not line.startswith("intermediate_data:"):
            continue
        try:
            item = json.loads(line.split(":", 1)[1])
            pl = json.loads(item.get("payload") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(pl, dict) and "event_type" in pl:
            out.append(pl)
    return out


def _kind(event_type: str) -> str:
    return event_type.rsplit("_", 1)[0]                   # LLM_END → LLM, FUNCTION_START → FUNCTION


def spans(events: list[dict]) -> list[dict]:
    """Pair *_START / *_END events by UUID. A START without an END is returned with open=True."""
    starts, ends = {}, {}
    for e in events:
        et = e.get("event_type", "")
        if et.endswith("_START"):
            starts[e.get("UUID")] = e
        elif et.endswith("_END"):
            ends[e.get("UUID")] = e
    out = []
    for uid, s in starts.items():
        e = ends.get(uid)
        t0 = float(s.get("event_timestamp") or 0)
        t1 = float(e.get("event_timestamp") or 0) if e else None
        tu = ((e or {}).get("usage_info") or {}).get("token_usage") or {}
        out.append({"kind": _kind(s.get("event_type", "")), "name": s.get("name") or "?", "start": t0, "end": t1,
                    "dur_s": (t1 - t0) if t1 else None, "uuid": uid, "open": e is None,
                    "tokens": {k: int(tu.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    "output": ((e or {}).get("data") or {}).get("output")})
    for uid, e in ends.items():                            # an END whose START was filtered out (filter_steps)
        if uid not in starts:
            t1 = float(e.get("event_timestamp") or 0)
            t0 = float(e.get("span_event_timestamp") or t1)
            tu = (e.get("usage_info") or {}).get("token_usage") or {}
            out.append({"kind": _kind(e.get("event_type", "")), "name": e.get("name") or "?", "start": t0, "end": t1,
                        "dur_s": t1 - t0, "uuid": uid, "open": False,
                        "tokens": {k: int(tu.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                        "output": (e.get("data") or {}).get("output")})
    return sorted(out, key=lambda s: s["start"])


def summary(sp: list[dict], events: list[dict] | None = None) -> dict:
    llm = [s for s in sp if s["kind"] == "LLM" and not s["open"]]
    tool = [s for s in sp if s["kind"] == "TOOL" and not s["open"]]
    wf = [s for s in sp if s["kind"] == "WORKFLOW" and not s["open"]]
    return {
        "events": len(events) if events is not None else None,
        "event_types": dict(Counter(e.get("event_type") for e in (events or []))),
        "spans": len(sp),
        "open_spans": sum(1 for s in sp if s["open"]),
        "llm_spans": len(llm),
        "tool_spans": len(tool),
        "tools": [s["name"] for s in tool],
        "prompt_tokens": sum(s["tokens"]["prompt_tokens"] for s in llm),
        "completion_tokens": sum(s["tokens"]["completion_tokens"] for s in llm),
        "total_tokens": sum(s["tokens"]["total_tokens"] for s in llm),
        "llm_s": round(sum(s["dur_s"] or 0 for s in llm), 3),
        "tool_s": round(sum(s["dur_s"] or 0 for s in tool), 3),
        "workflow_s": round(wf[0]["dur_s"], 3) if wf else None,
    }


def with_output_path(cfg_text: str, new_path: str) -> str:
    """Point every `output_path:` in a config at `new_path` (so two labs never write the same trace file)."""
    return re.sub(r"(?m)^(\s*output_path:\s*).*$", lambda m: m.group(1) + new_path, cfg_text)
