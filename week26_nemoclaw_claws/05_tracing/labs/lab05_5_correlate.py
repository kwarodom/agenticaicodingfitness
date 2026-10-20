#!/usr/bin/env python3
"""Lab 05-5 · Correlate: one request, three planes.

Research tutorial Lab 4.5 (L4.5). On THIS laptop, for real:
  1. start `nat serve` for Alto Ops Claw (with the file exporter) on a free port near 8001, in the background;
  2. POST /v1/workflow/full?filter_steps=LLM_END,TOOL_END — count the LLM and TOOL end events in the stream;
     (if `nat serve` cannot start here, the lab says exactly why and runs the same route function in-process —
     week26/05_tracing/fullstream.py — labelled as such);
  3. read the file exporter's trace for the SAME run and compare its LLM span count with the stream's;
  4. predict the sandboxed case: one `inspect_for_inference` per LLM span — and show how to check it on the
     Spark (read-only, bounded; DRY → EXAMPLE);
  5. the mapping table of the three planes (agent traces · policy logs · harness logs).
One request = 2 LLM calls with nemotron-3-nano on Ollama (LAPTOP STAND-IN). The server is always stopped.

Run: .venv/bin/python week26/05_tracing/labs/lab05_5_correlate.py
"""
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policykit as pk  # noqa: E402
from clawkit import (NAT, NAT_PY, ROOT, background, banner, free_port, laptop, note, ok, result,  # noqa: E402
                     sandbox, sh, step, table, warn)
from tracekit import CONFIGS, MODULE, RUNS, TRACES, events_from_sse, load_events, spans, summary, with_output_path  # noqa: E402

QUESTION = "Plant status last 6 hours?"
FILTER = "LLM_END,TOOL_END"
SB = sandbox("alto-ops")
TRACES.mkdir(parents=True, exist_ok=True)

banner("Lab 05-5 · correlate one request", "nat serve + /v1/workflow/full + file exporter for real · Spark check read-only")

trace = TRACES / "lab05_5_trace.jsonl"
cfg = RUNS / "lab05_5_workflow.yml"
cfg.write_text(with_output_path((CONFIGS / "workflow.traced.yml").read_text(encoding="utf-8"),
                                str(trace.relative_to(ROOT))), encoding="utf-8")
if trace.exists():
    trace.unlink()

step(1, "start `nat serve` (background, always stopped at the end)")
port = free_port(8001)
print(f"◆ port {port}" + ("" if port == 8001 else " (8001 is busy — another lab or server is using it)"))
log = RUNS / "lab05_5_nat_serve.log"
body = ""
how = ""
t0 = time.time()
try:
    with background([NAT, "serve", "--config_file", cfg, "--port", str(port)], ready_url=f"http://localhost:{port}/docs",
                    log=log, cwd=ROOT, timeout=180,
                    show=f"nat serve --config_file week26/05_tracing/.runs/{cfg.name} --port {port}"):
        step(2, f"POST /v1/workflow/full?filter_steps={FILTER}")
        url = f"http://localhost:{port}/v1/workflow/full?filter_steps={FILTER}"
        print(f"$ curl -s -X POST '{url}' -H 'Content-Type: application/json' "
              f"-d '{{\"input_message\": \"{QUESTION}\"}}'   [this laptop]")
        t0 = time.time()
        req = Request(url, data=json.dumps({"input_message": QUESTION}).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=400) as resp:                         # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            body = getattr(e, "read", lambda: b"")().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            warn(f"the request failed: {e} {body[:300]}")
        how = "nat serve · HTTP"
        time.sleep(2.0)                                                     # exporter background writes
except RuntimeError as e:
    tail = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else str(e)
    if "greenlet" in tail:
        warn("`nat serve` exited before it was ready: NAT 1.9.0's FastAPI front end imports its async job store "
             "(SQLAlchemy asyncio) at start-up, and SQLAlchemy asyncio needs the `greenlet` package, which is not "
             "in week26/.venv-nat on this laptop.")
        print("→ fix (the lab does not install anything): "
              "uv pip install --python week26/.venv-nat/bin/python greenlet   — then run this lab again")
    else:
        warn(f"`nat serve` did not come up: {str(e)[:400]}")
    step(2, f"fallback — the same route function in-process: generate_streaming_response_full(filter_steps={FILTER})")
    t0 = time.time()
    r = laptop([NAT_PY, MODULE / "fullstream.py", cfg, QUESTION, FILTER], cwd=ROOT, quiet=True, timeout=400,
               show=f'week26/.venv-nat/bin/python week26/05_tracing/fullstream.py week26/05_tracing/.runs/{cfg.name} '
                    f'"{QUESTION}" {FILTER}')
    body = r.out
    how = "in-process twin of /v1/workflow/full (nat serve could not start)"
wall = time.time() - t0

stream_ev = events_from_sse(body)
answer = "".join(json.loads(ln[5:]).get("value", "") for ln in body.splitlines()
                 if ln.startswith("data:") and ln[5:].strip().startswith("{") and '"value"' in ln)
types = [e["event_type"] for e in stream_ev]
n_llm_stream, n_tool_stream = types.count("LLM_END"), types.count("TOOL_END")
if not stream_ev:
    print(f"✕ no intermediate steps in the response ({how}) — tail of the output:\n{body[-1200:]}")
    sys.exit(1)
ok(f"{how}: {len(stream_ev)} intermediate step(s) after the filter in {wall:.1f}s → LLM_END ×{n_llm_stream}, "
   f"TOOL_END ×{n_tool_stream}")
table([[e["event_type"], e.get("name", "?"),
        f"{(e.get('usage_info') or {}).get('token_usage', {}).get('total_tokens', 0)}" if e["event_type"] == "LLM_END"
        else "", f"{float(e['event_timestamp']) - float(e.get('span_event_timestamp') or e['event_timestamp']):.3f} s"]
       for e in stream_ev], ["event (stream)", "name", "tokens", "duration"])
if answer:
    print("· ANSWER " + answer.replace("\n", " ")[:360])

step(3, "the same run in the file exporter's trace")
fev = load_events(trace)
fsp = spans(fev)
fs = summary(fsp, fev)
table([["LLM spans", n_llm_stream, fs["llm_spans"]], ["TOOL spans", n_tool_stream, fs["tool_spans"]],
       ["total tokens (LLM)", sum((e.get("usage_info") or {}).get("token_usage", {}).get("total_tokens", 0)
                                  for e in stream_ev if e["event_type"] == "LLM_END"), fs["total_tokens"]],
       ["open spans", "—", fs["open_spans"]], ["workflow span", "—", f"{fs['workflow_s']} s" if fs["workflow_s"] else "—"]],
      ["measure (LAPTOP STAND-IN)", "/v1/workflow/full stream", "file exporter"])
same = n_llm_stream == fs["llm_spans"] and n_tool_stream == fs["tool_spans"]
if same:
    ok(f"both views of the run agree: {fs['llm_spans']} LLM span(s), {fs['tool_spans']} TOOL span(s)")
else:
    warn("the stream and the file disagree — look for open spans (a lost tail) in the file")
if fs["open_spans"] == 0:
    ok("no open spans: the process stayed alive after the run, so the exporter finished writing (compare lab 05-1)")

step(4, "the policy plane, predicted: one inspect_for_inference per LLM span")
d, why = pk.decide({"network_policies": {}}, {"op": "http", "host": "inference.local", "port": 443,
                                             "binary": "/usr/bin/python3.12", "method": "POST",
                                             "path": "/v1/chat/completions"})
print(f"◆ policykit: POST inference.local/v1/chat/completions → {d} ({why[:80]}…)")
predicted = fs["llm_spans"]
ok(f"prediction for the SANDBOXED run of this request: {predicted} inspect_for_inference event(s) in `openshell "
   f"term` / `openshell logs {SB}` — one per LLM span. Verify on the Spark:")
sh(f"curl -s -X POST 'http://localhost:8001/v1/workflow/full?filter_steps={FILTER}' "
   f"-H 'Content-Type: application/json' -d '{{\"input_message\": \"{QUESTION}\"}}' "
   "| grep -c '\"type\":\"LLM_END\"'", timeout=400, example="2")
sh(f"openshell logs {SB} -n 200 --since 2m --source sandbox | grep -c inspect_for_inference", timeout=60,
   example="2")
d2, why2 = pk.decide({"network_policies": {}}, {"op": "connect", "host": "api.open-meteo.com", "port": 443,
                                               "binary": "/usr/bin/python3.12"})
note(f"research step 4 (a denied call): api.open-meteo.com → {d2} ({why2}). Alto Ops Claw has no web tool, so on "
     "the laptop the agent cannot even try; a harness that does try shows a FAILED tool span in the agent plane at "
     "the same moment OpenShell logs `deny` in the policy plane.")

step(5, "the three planes of ONE request — how to join them")
table([
    ["agent traces", "NAT exporters (file · Phoenix · OTel · Langfuse)", "workflow_run_id / trace id, input text, "
     "timestamps", f"{fs['llm_spans']} LLM span(s), {fs['tool_spans']} TOOL span(s), {fs['total_tokens']} tokens"],
    ["policy logs", "openshell logs / term, OCSF findings", "sandbox name + timestamp window",
     f"{predicted} inspect_for_inference (predicted), deny = a failed tool span"],
    ["harness logs", f"nemoclaw <sandbox> logs, /tmp/gateway.log (OpenClaw)", "timestamp, channel / session id",
     "channel events, pairing, crashes"],
], ["plane", "source", "join key", "for this request"])
(RUNS / "lab05_5_summary.json").write_text(json.dumps({
    "how": how, "stream_llm_end": n_llm_stream, "stream_tool_end": n_tool_stream, "file": fs,
    "llm_spans": [{"name": s["name"], "dur_s": round(s["dur_s"] or 0, 3), **s["tokens"]}
                  for s in fsp if s["kind"] == "LLM" and not s["open"]],
    "tool_spans": [{"name": s["name"], "dur_s": round(s["dur_s"] or 0, 3)} for s in fsp if s["kind"] == "TOOL"],
    "wall_s": round(wall, 1)}, indent=1), encoding="utf-8")
result(f"One request: {n_llm_stream} LLM_END in the stream = {fs['llm_spans']} LLM span(s) in the file → predict "
       f"{predicted} inspect_for_inference in the sandbox.")
