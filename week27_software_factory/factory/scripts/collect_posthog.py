#!/usr/bin/env python3
"""Emit PostHog signal candidates (from telemetry/posthog.json or a live export in the same shape)."""
import json
from collect_common import ROOT, args, load, parse_since, today
a = args(); T = today(a); since = parse_since(a.since, T)
data = load(a.file or ROOT / "telemetry" / "posthog.json")
out = []
for s in data["signals"]:
    hint = "drop-fixed" if s.get("stopped") else "file"
    out.append({"source": "posthog", "key": s["key"], "kind": s["kind"], "title": s["title"], "route": s["route"],
                "count": s.get("count"), "trend": s.get("trend"), "sample_session": s.get("sample_session"), "hint": hint, "note": s.get("note")})
print(json.dumps({"since": since.isoformat(), "candidates": out}, indent=1))
