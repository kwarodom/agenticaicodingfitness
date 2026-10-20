#!/usr/bin/env python3
"""Lab 03 — export a real OTel trace to Phoenix, then curate for the flywheel.

demos/step04_export_loop.py narrates the fan-out; here you DO one leg of it:
take the OTel-shaped trace lab01 wrote, POST it to a live Phoenix over
OTLP-HTTP (:6006/v1/traces, canonical protobuf encoding when the
opentelemetry-proto package is present), and open the span tree in the UI. Then the
'learn' half: filter the trace for slow/expensive spans and emit them as
JSONL — exactly the feedstock App 11's Data Flywheel curates into training data.

Run:  cd <repo root> && .venv/bin/python week23/08_nemo_relay/labs/lab03_otel_export.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PHOENIX = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006").rstrip("/")
SLOW_MS, PRICEY_TOK = 1000, 100          # curation thresholds


def canned_trace() -> dict:
    """A minimal OTel doc (clearly canned) so the lab teaches even without lab01."""
    now = time.time()
    def span(name, kind, dur_s, parent=None, extra=None):
        a = [{"key": "openinference.span.kind", "value": {"stringValue": kind}}]
        for k, v in (extra or {}).items():
            a.append({"key": k, "value": {"intValue": str(v)} if isinstance(v, int)
                      else {"stringValue": str(v)}})
        return {"traceId": "c" * 32, "spanId": os.urandom(8).hex(),
                **({"parentSpanId": parent} if parent else {}), "name": name, "kind": 1,
                "startTimeUnixNano": str(int((now - dur_s) * 1e9)),
                "endTimeUnixNano": str(int(now * 1e9)), "attributes": a}
    root = span("hermes-turn (canned)", "AGENT", 2.6)
    kids = [span("llm · plan (canned)", "LLM", 1.4, root["spanId"],
                 {"gen_ai.usage.output_tokens": 180, "gen_ai.request.model": "canned"}),
            span("tool · terminal (canned)", "TOOL", 0.2, root["spanId"])]
    return {"resourceSpans": [{"resource": {"attributes": [
        {"key": "service.name", "value": {"stringValue": "lab-relay"}}]},
        "scopeSpans": [{"scope": {"name": "lab03_canned"}, "spans": [root] + kids}]}]}


def load_trace() -> tuple[dict, bool]:
    path = config.PKG / ".sandbox" / "trace_lab01.json"
    if path.exists():
        print(f"▣ using the REAL trace lab01 measured → {path}")
        return json.loads(path.read_text()), True
    print("▣ no .sandbox/trace_lab01.json — run lab01 first for a real trace.")
    print("  ◈ using a clearly-CANNED trace so the export mechanics still teach.")
    return canned_trace(), False


def phoenix_up() -> bool:
    try:
        with urllib.request.urlopen(PHOENIX, timeout=2) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def encode(doc: dict) -> tuple[bytes, str]:
    """OTLP-HTTP's canonical encoding is protobuf. Installing arize-phoenix
    already brought in opentelemetry-proto — use it; else fall back to JSON
    (which some Phoenix versions reject with 415)."""
    try:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest)
    except ImportError:
        print("  ◈ opentelemetry-proto not installed — POSTing JSON (may be rejected).")
        return json.dumps(doc).encode(), "application/json"

    def _set(v, jv):    # OTLP-JSON attr value → protobuf AnyValue
        if "intValue" in jv:
            v.int_value = int(jv["intValue"])
        else:
            v.string_value = str(jv.get("stringValue", ""))
    req = ExportTraceServiceRequest()
    for rs in doc.get("resourceSpans", []):
        prs = req.resource_spans.add()
        for a in rs.get("resource", {}).get("attributes", []):
            kv = prs.resource.attributes.add()
            kv.key = a["key"]; _set(kv.value, a["value"])
        for ss in rs.get("scopeSpans", []):
            pss = prs.scope_spans.add()
            pss.scope.name = ss.get("scope", {}).get("name", "")
            for s in ss.get("spans", []):
                p = pss.spans.add()
                p.trace_id = bytes.fromhex(s["traceId"])      # OTLP-JSON ids are hex
                p.span_id = bytes.fromhex(s["spanId"])
                if s.get("parentSpanId"):
                    p.parent_span_id = bytes.fromhex(s["parentSpanId"])
                p.name, p.kind = s["name"], s.get("kind", 1)
                p.start_time_unix_nano = int(s["startTimeUnixNano"])
                p.end_time_unix_nano = int(s["endTimeUnixNano"])
                for a in s.get("attributes", []):
                    kv = p.attributes.add()
                    kv.key = a["key"]; _set(kv.value, a["value"])
    return req.SerializeToString(), "application/x-protobuf"


def export(doc: dict) -> None:
    body, ctype = encode(doc)
    req = urllib.request.Request(PHOENIX + "/v1/traces", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=8) as r:
        enc = "protobuf" if "protobuf" in ctype else "JSON"
        print(f"  ✓ HTTP {r.status} from Phoenix — trace accepted over OTLP-HTTP ({enc}).")


def curate(doc: dict, is_real: bool) -> None:
    print("\n◈ 'Learn' — curate slow/expensive spans (the Data Flywheel's feedstock):")
    tag = "real" if is_real else "canned"
    n = 0
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                ms = (int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"])) / 1e6
                toks = next((int(a["value"].get("intValue", 0)) for a in s.get("attributes", [])
                             if a["key"] == "gen_ai.usage.output_tokens"), 0)
                reasons = [r for r, hit in (("slow", ms > SLOW_MS), ("pricey", toks > PRICEY_TOK)) if hit]
                if reasons:
                    n += 1
                    print("  " + json.dumps({"span": s["name"], "latency_ms": round(ms),
                                             "out_tokens": toks, "why": reasons,
                                             "source": tag}, ensure_ascii=False))
    print(f"  → {n} candidate span(s). Scrub PII before any export — spans carry prompts.")
    print("  App 11 fine-tunes on curated turns like these; the router then re-routes.")


def main() -> None:
    print("━" * 64)
    print("  ▣ Lab 03 — OTel export to Phoenix + flywheel curation")
    print("━" * 64 + "\n")
    doc, is_real = load_trace()

    print(f"\n▣ probing Phoenix at {PHOENIX} …")
    if phoenix_up():
        try:
            export(doc)
            print(f"  open {PHOENIX} → project 'default' → Traces, and click the span tree.")
        except urllib.error.HTTPError as e:
            print(f"  ✗ Phoenix answered HTTP {e.code} — OTLP-HTTP's canonical encoding is")
            print("    protobuf; if JSON is rejected, use the phoenix.otel SDK path from")
            print("    TUTORIAL.md step 3 (register() + OpenAIInstrumentor) instead.")
    else:
        print("  ◈ [no Phoenix — showing expected output] Nothing was exported. Start it:")
        print("      uv pip install arize-phoenix")
        print("      .venv/bin/python -m phoenix.server.main serve   # UI + OTLP on :6006")
        print("    (on a Spark: run it there, then  ssh -L 6006:localhost:6006 <spark>)")
        print("    then re-run this lab. Expected:")
        print("      ✓ HTTP 200 from Phoenix — trace accepted over OTLP-HTTP (protobuf).")
        print("    Keep Phoenix on the box/VPN only — traces contain prompts (PII).")

    curate(doc, is_real)
    print("\n✓ One instrumentation, many backends: the same OTLP doc could go to Datadog")
    print("  or LangSmith by swapping the endpoint. That is the whole 'Relay' idea.")


if __name__ == "__main__":
    main()
