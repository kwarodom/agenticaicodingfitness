#!/usr/bin/env python3
"""Lab 04-1 · Hello, NAT: install check, a scaffold, and the minimal ReAct workflow on laptop Ollama.

Research tutorial L3.1–L3.3. On THIS laptop, for real: `nat --version`, `nat workflow create --no-install` into
.runs/, `nat validate`, then `nat run` of the tutorial's minimal react_agent + current_datetime workflow against
laptop Ollama (nemotron-3-nano) — first exactly as written, then with the one line a thinking model needs.
On the Spark (read-only or gated by change()): the uv install of NAT and the Nemotron 3 Nano vLLM container.

LLM budget: at most 3 laptop LLM calls.
Run: .venv/bin/python week26/04_nat_claws/labs/lab04_1_hello_workflow.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import banner, change, laptop_models, note, ok, result, sh, step, table, warn  # noqa: E402
from natkit import CONFIGS, RUNS, error_kind, error_line, nat, rel, workflow_result  # noqa: E402

banner("Lab 04-1 · hello, NAT", "laptop NAT 1.9 for real · the Spark install + vLLM via change() (or DRY)")
MODEL = "nemotron-3-nano:latest"

step(1, "L3.1 — the NAT CLI on this laptop")
r, _ = nat(["--version"])
version = next((ln for ln in r.out.splitlines() if ln.startswith("nat, version")), "")
if not version:
    warn("no NAT CLI — see '0 · Before you start' in TUTORIAL.md")
    sys.exit(1)
ok(f"{version} — the research tutorial cites the NAT 1.8 docs; flags below were checked on 1.9.0")

step(2, "L3.1 — the same install on the Spark (arm64, uv, Python 3.12)")
change("mkdir -p ~/works/alto-ops-claw && cd ~/works/alto-ops-claw\n"
       "uv venv --python 3.12 && source .venv/bin/activate\n"
       "uv pip install 'nvidia-nat[langchain,mcp,profiler,phoenix,opentelemetry]'\n"
       "nat --version",
       example="Using CPython 3.12.x\nCreating virtual environment at: .venv\nResolved … packages in …\n"
               "Installed … packages in …\nnat, version <the release uv resolved>")
note("nvidia-nat is a pure-Python wheel, so arm64 needs no compiler (the tutorial, citing Classmethod's DGX Spark "
     "write-up). This laptop's venv also has the eval + ragas extras.")

step(3, "L3.2 — a local vLLM for NAT on the Spark (Nemotron 3 Nano needs --trust-remote-code)")
change("docker run -d --name vllm-nat --gpus all --shm-size=16g -p 8000:8000 \\\n"
       "  -v \"$HOME/.cache/huggingface:/root/.cache/huggingface\" \\\n"
       "  nvcr.io/nvidia/vllm:26.01-py3 \\\n"
       "  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\\n"
       "  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85",
       example="<container id>")
sh("curl -s http://localhost:8000/v1/models | python3 -m json.tool",
   example='{\n    "object": "list",\n    "data": [\n        {\n'
           '            "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",\n            "object": "model",\n'
           '            "max_model_len": 8192\n        }\n    ]\n}')
note("Per the tutorial (citing Classmethod): without --trust-remote-code the nemotron_h model crashes on a pydantic "
     "ValidationError; loading took ~4 min + ~2 min torch.compile. This laptop uses Ollama as a LAPTOP STAND-IN.")

step(4, "L3.3 — scaffold a workflow package (no install) into .runs/")
wf_dir = RUNS / "workflows"
shutil.rmtree(wf_dir / "alto_ops", ignore_errors=True)
wf_dir.mkdir(parents=True, exist_ok=True)       # NAT 1.9 refuses a --workflow-dir that does not exist yet
print(f"$ mkdir -p {rel(wf_dir)}   [this laptop]")
nat(["workflow", "create", "--no-install", "--workflow-dir", wf_dir, "alto_ops", "--description", "Alto Ops Claw"],
    show=f"nat workflow create --no-install --workflow-dir {rel(wf_dir)} alto_ops --description \"Alto Ops Claw\"")
pkg = wf_dir / "alto_ops"
files = sorted(str(p.relative_to(pkg)) for p in pkg.rglob("*") if p.is_file())
table([[f] for f in files], [f"files under {rel(pkg)}/"])
reg = (pkg / "src" / "alto_ops" / "alto_ops.py").read_text(encoding="utf-8")
imports = [ln for ln in reg.splitlines() if ln.startswith("from nat")]
for ln in imports:
    print(f"│ {ln}")
cfg_txt = (pkg / "src" / "alto_ops" / "configs" / "config.yml").read_text(encoding="utf-8")
llm_type = next((ln.strip() for ln in cfg_txt.splitlines() if "_type:" in ln and "nim" in ln), "")
note(f"1.9 scaffold imports everything from `nat.plugin_api` and its config.yml uses `{llm_type}` (a hosted NIM) — "
     "we replace that llms block with a local OpenAI-compatible server, as the tutorial does.")

step(5, "L3.3 — the minimal workflow, pointed at this laptop's Ollama: validate it (no LLM call)")
print(f"│ {rel(CONFIGS / 'hello.laptop.yml')}")
for ln in (CONFIGS / "hello.laptop.yml").read_text(encoding="utf-8").splitlines():
    if not ln.startswith("#"):
        print(f"│   {ln}")
r, _ = nat(["validate", "--config_file", CONFIGS / "hello.laptop.yml"])
if r.code != 0:
    warn("validation failed — read the error above")
    sys.exit(1)

if not any(m.startswith("nemotron-3-nano") for m in laptop_models()):
    warn(f"Ollama on this laptop does not list {MODEL} — `ollama pull nemotron-3-nano` and rerun. Skipping the runs.")
    result("Scaffold + validate done; the two `nat run` steps need the laptop model.")
    sys.exit(0)

QUESTION = "What time is it in Bangkok right now?"
step(6, "L3.3 — nat run, exactly the tutorial's shape (react_agent)")
r1, t1 = nat(["run", "--config_file", CONFIGS / "hello.laptop.yml", "--input", QUESTION], log="lab04_1_run1.log")
ans1 = workflow_result(r1.out)
if ans1:
    print(f"· ANSWER  {ans1}")
    ok(f"LAPTOP STAND-IN · {MODEL} · react_agent · {t1:.1f}s wall (includes NAT start-up)")
else:
    print(f"✕ {error_line(r1.out) or 'no Workflow Result'}")
    note(f"LAPTOP STAND-IN · {t1:.1f}s · a THINKING model on Ollama returned an empty content field, so the ReAct "
         "parser found no 'Thought:/Action:' text. Nothing is wrong with the YAML.")

step(7, "L3.3 — the same workflow + `reasoning_effort: none` (thinking off)")
r2, t2 = nat(["run", "--config_file", CONFIGS / "hello.laptop.nothink.yml", "--input", QUESTION],
             log="lab04_1_run2.log")
ans2 = workflow_result(r2.out)
if not ans2:
    print(f"✕ {error_line(r2.out) or 'no Workflow Result'}")
    warn("the agent did not answer — see .runs/lab04_1_run2.log")
    sys.exit(1)
print(f"· ANSWER  {ans2}")
ok(f"LAPTOP STAND-IN · {MODEL} · react_agent · {t2:.1f}s wall (includes NAT start-up)")
if "+0000" in ans2 or "UTC" in ans2:
    warn("look at the offset: current_datetime returns UTC (+0000). Bangkok is UTC+7. The agent passed the tool's "
         "time straight through as 'Bangkok time' — a tool result is only as good as the question it answers.")

table([["tutorial shape", "react_agent", "✓ answered" if ans1 else "✕ " + error_kind(r1.out),
        f"{t1:.1f}"],
       ["+ reasoning_effort: none", "react_agent", "✓ answered", f"{t2:.1f}"]],
      ["config", "agent", "outcome", "seconds (LAPTOP STAND-IN)"])
result("NAT = YAML (functions · llms · workflow) + `_type`. Validate before you run; thinking models need a knob.")
