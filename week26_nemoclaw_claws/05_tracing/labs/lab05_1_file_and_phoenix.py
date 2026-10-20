#!/usr/bin/env python3
"""Lab 05-1 · Agent traces: NAT's file exporter for real, Phoenix only when it is really there.

Research tutorial Lab 4.1 (L4.1). On THIS laptop, for real:
  1. `nat info components -t tracing` — which exporters NAT 1.9.0 has registered here;
  2. the 1.9 YAML keys of the `file` tracing exporter, read from the installed NAT (not from memory), and
     `nat validate` on the research tutorial's block vs the course's block;
  3. a probe of localhost:6006 — Phoenix is added as a SECOND exporter only if something answers there;
  4. one `nat run` of Alto Ops Claw (2 LLM calls with nemotron-3-nano on Ollama — a LAPTOP STAND-IN);
  5. the trace file parsed: span names, LLM vs TOOL spans, token counts, durations, open spans.
The Spark half (Phoenix in Docker next to vLLM) is shown with sh()/change(): DRY → EXAMPLE, never a fake trace.

Run: .venv/bin/python week26/05_tracing/labs/lab05_1_file_and_phoenix.py
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import NAT, NAT_PY, ROOT, banner, change, laptop, note, ok, result, sh, step, table, warn  # noqa: E402
from tracekit import CONFIGS, RUNS, TRACES, load_events, spans, summary, with_output_path  # noqa: E402

QUESTION = "Plant status last 6 hours?"
TRACES.mkdir(parents=True, exist_ok=True)

banner("Lab 05-1 · file exporter + Phoenix", "NAT 1.9.0 on this laptop for real · Phoenix only if it answers")


def phoenix_up(url: str = "http://localhost:6006/") -> bool:
    try:
        with urlopen(url, timeout=2):                                  # noqa: S310
            return True
    except HTTPError:
        return True                                                    # any HTTP answer = something listens
    except Exception:  # noqa: BLE001
        return False


step(1, "which tracing exporters does THIS NAT install have?")
r = laptop([NAT, "info", "components", "-t", "tracing"], quiet=True, env={"COLUMNS": "250"}, timeout=120,
           show="nat info components -t tracing")
rows = []
for ln in r.out.splitlines():
    cells = [c.strip() for c in ln.strip().strip("│").split("│")]
    if len(cells) >= 4 and cells[2] == "tracing" and cells[3]:
        rows.append([cells[3], cells[0], cells[1]])
if rows:
    table(rows, ["component_name (_type)", "package", "version"])
    ok(f"{len(rows)} tracing exporters registered: {', '.join(r_[0] for r_ in rows)}")
else:
    warn(f"could not read the component table (exit {r.code}) — run `nat info components -t tracing` yourself")

step(2, "the 1.9 keys of the `file` TRACING exporter — asked from the installed NAT, not from memory")
probe = ("from nat.observability.register import FileTelemetryExporterConfig as C\n"
         "for k, v in C.model_fields.items():\n"
         "    print(f'{k}|' + ('REQUIRED' if v.is_required() else repr(v.default)))")
r = laptop([NAT_PY, "-c", probe], quiet=True, timeout=120,
           show="week26/.venv-nat/bin/python -c 'from nat.observability.register import FileTelemetryExporterConfig …'")
fields = [ln.split("|", 1) for ln in r.out.splitlines() if "|" in ln and not ln.startswith(" ")]
if fields:
    table([[k, v] for k, v in fields], ["field", "default"])
required = [k for k, v in fields if v == "REQUIRED"]
note(f"required in 1.9.0: {', '.join(required) or '?'} — the research tutorial's `file_backup: {{_type: file}}` "
     "has neither (its comment says only `# path etc.`)")

TUTORIAL_BLOCK = """general:
  telemetry:
    tracing:
      file_backup:
        _type: file
        # path etc.
""" + "functions:" + (CONFIGS / "workflow.traced.yml").read_text(encoding="utf-8").split("\nfunctions:", 1)[1]
tut = RUNS / "lab05_1_tutorial_block.yml"
tut.write_text(TUTORIAL_BLOCK, encoding="utf-8")
r = laptop([NAT, "validate", "--config_file", tut], quiet=True, timeout=120,
           show="nat validate --config_file week26/05_tracing/.runs/lab05_1_tutorial_block.yml")
err = next((ln.strip() for ln in r.out.splitlines() if ln.startswith("Invalid configuration")), "")
print(("✕ " if r.code else "✓ ") + f"research tutorial's block → exit {r.code}" + (f" · {err}" if err else ""))
r = laptop([NAT, "validate", "--config_file", CONFIGS / "workflow.traced.yml"], quiet=True, timeout=120,
           show="nat validate --config_file week26/05_tracing/configs/workflow.traced.yml")
print(("✓ " if r.ok else "✕ ") + f"course block (output_path + project + mode) → exit {r.code}")

step(3, "Phoenix: is anything answering on localhost:6006?")
use_phoenix = phoenix_up()
if use_phoenix:
    ok("something answers on :6006 → this run exports to Phoenix AND to the file (two exporters at once)")
    src = CONFIGS / "workflow.phoenix.yml"
else:
    warn("nothing on localhost:6006 → Phoenix is SKIPPED on this run (no Phoenix trace is shown or implied); "
         "the file exporter is the always-works path")
    src = CONFIGS / "workflow.traced.yml"
print("→ to add Phoenix on this laptop, start Docker Desktop and run it yourself in the ⌨ terminal:")
print("$ docker run -d --name phoenix-nat -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest   [not run by the lab]")
print("  then run this lab again: it switches to configs/workflow.phoenix.yml automatically")
note("on the Spark, Phoenix runs next to vLLM (same command). Starting a container is a change → change():")
change("docker run -d --name phoenix-nat -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest",
       preview="docker ps -a --filter name=phoenix-nat --format '{{.Names}} {{.Status}}'",
       example="phoenix-nat Up 3 minutes")
sh("curl -s -o /dev/null -w '%{http_code}\\n' http://localhost:6006/", example="200", timeout=20)

step(4, f"one `nat run` with the file exporter{' + Phoenix' if use_phoenix else ''} (2 LLM calls expected)")
trace = TRACES / "lab05_1_trace.jsonl"
cfg = RUNS / "lab05_1_workflow.yml"
cfg.write_text(with_output_path(src.read_text(encoding="utf-8"), str(trace.relative_to(ROOT))), encoding="utf-8")
if trace.exists():
    trace.unlink()
t0 = time.time()
r = laptop([NAT, "run", "--config_file", cfg, "--input", QUESTION], cwd=ROOT, quiet=True, timeout=400,
           show=f'nat run --config_file week26/05_tracing/.runs/{cfg.name} --input "{QUESTION}"')
wall = time.time() - t0
answer = r.out.split("Workflow Result:", 1)[-1].strip().strip("-").strip() if "Workflow Result:" in r.out else ""
answer = re.sub(r"\x1b\[[0-9;]*m", "", answer).split("\n-----", 1)[0].strip()
if r.ok and answer:
    ok(f"nat run exit 0 in {wall:.1f}s (LAPTOP STAND-IN: nemotron-3-nano on Ollama, not vLLM on a GB10)")
    print("· ANSWER " + answer.replace("\n", " ")[:420])
else:
    print(f"✕ nat run exit {r.code} — is Ollama running with nemotron-3-nano:latest?")
    print(r.out[-1200:])
    sys.exit(1)
time.sleep(0.5)

step(5, f"parse the trace file: {trace.relative_to(ROOT)}")
ev = load_events(trace)
sp = spans(ev)
s = summary(sp, ev)
if not ev:
    print("✕ the trace file is empty — check the tracing block and the log in .runs/alto_ops.log")
    sys.exit(1)
et = s["event_types"]
table([[k, v] for k, v in sorted(et.items(), key=lambda kv: -kv[1])], ["event_type", "count"])
note(f"{et.get('LLM_NEW_TOKEN', 0)} of {len(ev)} lines are LLM_NEW_TOKEN — one per streamed token. The file "
     "exporter is RAW: IntermediateStep events, not finished OTel spans. A span = START + END with one UUID.")
shown = [x for x in sp if x["kind"] in ("WORKFLOW", "LLM", "TOOL") or x["open"]]
table([[x["kind"], x["name"], "OPEN (no END)" if x["open"] else f"{x['dur_s']:.3f} s",
        (f"{x['tokens']['prompt_tokens']} + {x['tokens']['completion_tokens']} = {x['tokens']['total_tokens']}"
         if x["kind"] == "LLM" and not x["open"] else "")]
       for x in shown], ["kind", "name", "duration", "tokens (prompt + completion)"])
table([["LLM calls started (LLM_START)", et.get("LLM_START", 0)],
       ["LLM spans closed (START + END)", s["llm_spans"]],
       ["TOOL spans", f"{s['tool_spans']} ({', '.join(s['tools']) or '—'})"],
       ["tokens (prompt / completion / total)", f"{s['prompt_tokens']} / {s['completion_tokens']} / {s['total_tokens']}"],
       ["time in LLM spans", f"{s['llm_s']:.2f} s"], ["time in TOOL spans", f"{s['tool_s']:.3f} s"],
       ["workflow span", f"{s['workflow_s']:.2f} s" if s["workflow_s"] else "no WORKFLOW_END in the file"],
       ["open spans (START, no END)", s["open_spans"]]], ["measure (LAPTOP STAND-IN)", "value"])
if s["open_spans"]:
    warn(f"{s['open_spans']} span(s) have no END line. NAT 1.9's exporter stop() does not wait for background "
         "export tasks, so a short-lived `nat run` can exit before the last events are written. The model did "
         "answer; the TAIL of the trace was lost. Lab 05-5 uses `nat serve` (a long-lived process) and compares.")
else:
    ok("every START has its END — the whole run reached the file this time")
(RUNS / "lab05_1_summary.json").write_text(json.dumps({**s, "phoenix": use_phoenix, "wall_s": round(wall, 1)},
                                                        indent=1), encoding="utf-8")
if use_phoenix:
    note("open http://localhost:6006 → project alto-ops-claw: the same run as a trace tree. Compare its LLM span "
         f"count with the file's ({s['llm_spans']}) — Part 4 exercise 1.")
llm_started = et.get("LLM_START", 0)
result(f"One run → {llm_started} LLM call(s) started, {s['llm_spans']} closed in the file · {s['tool_spans']} TOOL "
       f"span(s) · {s['total_tokens']} tokens on closed LLM spans, "
       f"written by the file exporter{' and sent to Phoenix' if use_phoenix else ' (Phoenix skipped: not running)'}.")
