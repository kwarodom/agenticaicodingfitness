#!/usr/bin/env python3
"""Emit Sentry candidates (from telemetry/sentry.json or a live export in the same shape) with a pre-verdict hint.
The hint is advisory; the fitness-collect skill still sends each candidate to a fresh verifier."""
import datetime as dt, json
from collect_common import ROOT, args, load, parse_since, today
a = args(); T = today(a); since = parse_since(a.since, T)
data = load(a.file or ROOT / "telemetry" / "sentry.json"); board = load(a.board)
open_titles = " ".join(i["content"]["title"].lower() for i in board["items"] if i["status"] != "Done")
seen_fp = {}
out = []
for i in data["issues"]:
    last = dt.date.fromisoformat(i["last_seen"]); hint = "file"
    if i.get("status") == "resolved" and i.get("merged_at") and last <= dt.date.fromisoformat(i["merged_at"]): hint = "drop-fixed"
    elif last < since: hint = "drop-stale"
    elif i["count"] <= 3 and "deploy" in i.get("note", ""): hint = "drop-noise"
    elif i["fingerprint"] in seen_fp: hint = f"duplicate:{seen_fp[i['fingerprint']]}"
    elif any(w in open_titles for w in i["title"].lower().split()[:2]) and "kwh_rollup" in i["fingerprint"]: hint = "duplicate:#51"
    seen_fp.setdefault(i["fingerprint"], i["id"])
    out.append({"source": "sentry", "id": i["id"], "fingerprint": i["fingerprint"], "title": i["title"], "count": i["count"],
                "first_seen": i["first_seen"], "last_seen": i["last_seen"], "route": i["route"], "frames": i["frames"][:3], "hint": hint})
print(json.dumps({"since": since.isoformat(), "candidates": out}, indent=1))
