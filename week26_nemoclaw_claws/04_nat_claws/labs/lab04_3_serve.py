#!/usr/bin/env python3
"""Lab 04-3 · Serve it: `nat serve` — REST, OpenAI-compatible, and the step-by-step /full stream.

Research tutorial L3.5. Starts Alto Ops Claw (workflow.laptop.yml) as a FastAPI server on this laptop
(free_port(8001), stopped at the end), lists the routes NAT 1.9 really creates and diffs them against the
tutorial's list, then makes three calls: POST /v1/workflow, POST /v1/chat/completions (the OpenAI client's
request shape, sent with urllib) and POST /v1/workflow/full?filter_steps=LLM_END,TOOL_END — whose
intermediate steps it counts (Module 05 builds on this). The steps are saved to .runs/lab04_3_full_steps.json.

LLM budget: 3 agent requests ≈ 5–6 laptop LLM calls.
Run: .venv/bin/python week26/04_nat_claws/labs/lab04_3_serve.py
"""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import NAT, background, banner, free_port, laptop_models, note, ok, result, step, table, warn  # noqa: E402
from natkit import (GREENLET_FIX, LAPTOP_YML, RUNS, get_json, greenlet_env, parse_sse, post,  # noqa: E402
                    rel)

banner("Lab 04-3 · serve it", "nat serve on this laptop · REST + OpenAI-compatible + /full steps · stopped at the end",
       status=False)
QUESTION = "Is the chiller plant efficient over the last 6 hours?"
# The routes the research tutorial lists (it cites the NAT 1.8 API-server page).
TUTORIAL_ROUTES = ["/v1/workflow", "/v1/workflow/stream", "/v1/workflow/full", "/v1/workflow/async", "/v1/chat",
                   "/v1/chat/stream", "/v1/chat/completions", "/feedback", "/monitor/users", "/generate", "/chat"]

step(0, "a NAT 1.9.0 quirk first: `nat serve` needs the `greenlet` package")
env, how = greenlet_env()
if env is None:
    warn("week26/.venv-nat has no greenlet, so `nat serve` stops with: \"The SQLAlchemy asyncio module requires that "
         "the Python 'greenlet' library is installed\".")
    print(f"→ fix it once, in the ⌨ terminal:  {GREENLET_FIX}")
    result("Skipped: install greenlet and rerun. (`nat run` and `nat mcp serve` do not need it.)")
    sys.exit(0)
if env:
    warn("week26/.venv-nat has no greenlet (NAT 1.9.0 + SQLAlchemy 2.1 import it on `nat serve`). This run borrows "
         f"the one in week25/.venv-nat through PYTHONPATH=.runs/pyshim. The real fix: {GREENLET_FIX}")
else:
    ok(how)
if not any(m.startswith("nemotron-3-nano") for m in laptop_models()):
    warn("no nemotron-3-nano on this laptop's Ollama — `ollama pull nemotron-3-nano` and rerun.")
    sys.exit(0)

port = free_port(8001)
base = f"http://127.0.0.1:{port}"
note(f"port {port}" + ("" if port == 8001 else " (8001 was busy — another lab is running)"))

step(1, f"nat serve --config_file {rel(LAPTOP_YML)} --port {port}")
steps_saved = {}
try:
    with background([NAT, "serve", "--config_file", LAPTOP_YML, "--host", "127.0.0.1", "--port", str(port)],
                    ready_url=f"{base}/health", log=RUNS / "lab04_3_nat_serve.log", env={"PYTHONWARNINGS": "ignore", **env},
                    show=f"week26/.venv-nat/bin/nat serve --config_file {rel(LAPTOP_YML)} --host 127.0.0.1 --port {port}"):
        print(f"→ GET /health → {json.dumps(get_json(base + '/health'))}")

        step(2, "which routes did NAT 1.9 create? (GET /openapi.json)")
        paths = get_json(base + "/openapi.json")["paths"]
        table([[p, ", ".join(m.upper() for m in paths[p])] for p in paths], ["route", "methods"])
        missing = [p for p in TUTORIAL_ROUTES if p not in paths]
        extra = [p for p in paths if p not in TUTORIAL_ROUTES and not p.startswith("/auth")]
        print(f"◆ in the tutorial's list but NOT served here: {', '.join(missing) or 'none'}")
        print(f"◆ served here but not in the tutorial's list: {', '.join(extra) or 'none'}")
        note("/v1/workflow/async needs the optional dask extra; /monitor/users needs general.enable_per_user_monitoring: "
             "true; /feedback is not registered by this config on 1.9.0 (check /docs on your install).")

        step(3, "POST /v1/workflow — the plain REST call")
        body = {"input_message": QUESTION}
        print(f"→ POST {base}/v1/workflow  {json.dumps(body)}")
        raw, secs = post(base + "/v1/workflow", body)
        d = json.loads(raw)
        print(f"◆ response keys: {sorted(d)}")
        print("· ANSWER  " + str(d.get("value", d))[:600])
        print(f"◆ LAPTOP STAND-IN · {secs:.1f}s for the whole agent loop (LLM calls + tool call)")
        steps_saved["workflow_s"] = round(secs, 1)

        step(4, "POST /v1/chat/completions — what the OpenAI client sends (base_url=…/v1, dummy key)")
        body = {"model": "alto-ops", "messages": [{"role": "user", "content": "Plant status?"}], "stream": False}
        print(f"→ POST {base}/v1/chat/completions  {json.dumps(body)}")
        raw, secs = post(base + "/v1/chat/completions", body)
        d = json.loads(raw)
        msg = d["choices"][0]["message"]
        print(f"◆ object={d.get('object')} · model={d.get('model')!r} · finish_reason={d['choices'][0].get('finish_reason')}"
              f" · usage={json.dumps(d.get('usage'))}")
        print("· ANSWER  " + (msg.get("content") or "")[:600].replace("\n", "\n          "))
        print(f"◆ LAPTOP STAND-IN · {secs:.1f}s")
        note("The `model` you send is not echoed (1.9.0 answers model='unknown-model') and prompt_tokens is 0 — count "
             "tokens from the /full steps instead. 'Plant status?' is vague: the agent may ask back rather than call a tool.")
        steps_saved["chat_s"] = round(secs, 1)

        step(5, "POST /v1/workflow/full?filter_steps=LLM_END,TOOL_END — the answer AND the steps")
        body = {"input_message": QUESTION}
        print(f"→ POST {base}/v1/workflow/full?filter_steps=LLM_END,TOOL_END  {json.dumps(body)}")
        raw, secs = post(base + "/v1/workflow/full?filter_steps=LLM_END,TOOL_END", body)
        (RUNS / "lab04_3_full_raw.txt").write_bytes(raw)
        events = parse_sse(raw)
        chunks = [e for k, e in events if k == "data"]
        inter = [e for k, e in events if k == "intermediate_data"]
        print(f"◆ it is a stream (server-sent events): {len(chunks)} `data:` answer chunks + {len(inter)} "
              "`intermediate_data:` steps")
        rows, llm_n, tool_n, tok = [], 0, 0, 0
        saved = []
        for e in inter:
            p = json.loads(e.get("payload") or "{}")
            dur = (p.get("event_timestamp") or 0) - (p.get("span_event_timestamp") or 0)
            usage = ((p.get("usage_info") or {}).get("token_usage") or {})
            if e["type"] == "LLM_END":
                llm_n += 1
                tok += int(usage.get("total_tokens") or 0)
            elif e["type"] == "TOOL_END":
                tool_n += 1
            out = str((p.get("data") or {}).get("output", ""))[:70].replace("\n", " ")
            rows.append([e["type"], e["name"], f"{dur:.2f}", usage.get("prompt_tokens", "—"),
                         usage.get("completion_tokens", "—"), out])
            saved.append({"type": e["type"], "name": e["name"], "seconds": round(dur, 3),
                          "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
        table(rows, ["step", "name", "s", "prompt tok", "completion tok", "output (start)"])
        answer = "".join(str(c.get("value", "")) for c in chunks).strip()
        print("· ANSWER  " + answer[:400])
        print(f"◆ LAPTOP STAND-IN · {secs:.1f}s · LLM_END × {llm_n} · TOOL_END × {tool_n} · {tok} LLM tokens")
        ok(f"one question = {llm_n} LLM call(s) + {tool_n} tool call(s): plan → chiller_kpi → summarise")
        steps_saved.update({"full_s": round(secs, 1), "llm_end": llm_n, "tool_end": tool_n, "tokens": tok,
                            "steps": saved})
        (RUNS / "lab04_3_full_steps.json").write_text(json.dumps(steps_saved, indent=1), encoding="utf-8")
        note(f"saved → {rel(RUNS / 'lab04_3_full_steps.json')} (Module 05 compares these with a trace)")
except (RuntimeError, HTTPError, URLError, TimeoutError) as e:
    warn(f"nat serve lab stopped early: {type(e).__name__}: {str(e)[:400]}")
    sys.exit(1)

result("One YAML, one server: /v1/workflow for apps, /v1/chat/completions for OpenAI clients, /full for debugging.")
