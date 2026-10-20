#!/usr/bin/env python3
"""Exercise 14 · Wire the hotel agent's NAT workflow yourself.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/14_nat_agents/exercises/ex14_wire_the_agent.py

Each TODO returns one section of a NAT workflow file as a Python dict. The checker is offline and
free: it lints the assembled config, writes it to week25/14_nat_agents/.runs/, and — if NAT is
installed in week25/.venv-nat — asks the real `nat validate` whether NAT accepts it (no model is
called). Stuck? Compare with exercises/solutions/ and with configs/hotel_agent.yml.
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import WEEK, banner, check  # noqa: E402

MOD = Path(__file__).resolve().parents[1]                 # week25/14_nat_agents


# ── TODO 1 ── the `llms.hotel_llm` block for an OpenAI-compatible server.
#   _type "openai", base_url, model_name, api_key "not-needed", temperature 0.0, max_tokens 512.
#   Thinking off: target "ollama" → reasoning_effort "none";
#                 target "vllm"   → extra_body {"chat_template_kwargs": {"enable_thinking": False}}.
def llm_block(target: str, base_url: str, model: str) -> dict:
    return None


# ── TODO 2 ── the `functions` section: the two hotel_ops_nat tools plus NAT's built-in clock.
#   keys room_temperature, create_maintenance_ticket, current_datetime; each value has a `_type`
#   equal to its registered name; create_maintenance_ticket also gets ticket_log=<ticket_log>.
def functions_block(ticket_log: str) -> dict:
    return None


# ── TODO 3 ── the `workflow` section: a tool_calling_agent that uses llm `llm_name` and EVERY
#   function in `functions`, stops after at most 6 tool rounds (max_iterations), and has a system_prompt.
def workflow_block(llm_name: str, functions: dict) -> dict:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
AGENTS = ("tool_calling_agent", "react_agent", "rewoo_agent")


def assemble(target: str) -> dict:
    base = {"ollama": "http://localhost:11434/v1", "vllm": "http://spark-a:8000/v1"}[target]
    model = {"ollama": "nemotron-3-nano:latest", "vllm": "nvidia/Qwen3.6-35B-A3B-NVFP4"}[target]
    fns = functions_block(".runs/tickets.jsonl")
    return {"llms": {"hotel_llm": llm_block(target, base, model)}, "functions": fns,
            "workflow": workflow_block("hotel_llm", fns or {})}


def lint(cfg: dict) -> list[str]:
    """The mistakes NAT would reject (or silently mis-run) — checked without NAT."""
    probs = []
    llms, fns, wf = cfg.get("llms") or {}, cfg.get("functions") or {}, cfg.get("workflow") or {}
    for name, b in llms.items():
        if not isinstance(b, dict) or b.get("_type") != "openai":
            probs.append(f"llms.{name}._type must be 'openai'")
            continue
        if not str(b.get("base_url", "")).rstrip("/").endswith("/v1"):
            probs.append(f"llms.{name}.base_url should end in /v1 (got {b.get('base_url')!r})")
        if not b.get("model_name"):
            probs.append(f"llms.{name}.model_name is missing")
    for name, b in fns.items():
        if not isinstance(b, dict) or b.get("_type") != name:
            probs.append(f"functions.{name}._type should be {name!r}")
    if wf.get("_type") not in AGENTS:
        probs.append(f"workflow._type should be one of {AGENTS}")
    if wf.get("llm_name") not in llms:
        probs.append(f"workflow.llm_name {wf.get('llm_name')!r} is not a key of llms")
    missing = [t for t in wf.get("tool_names") or [] if t not in fns]
    if missing:
        probs.append(f"workflow.tool_names {missing} are not defined under functions")
    return probs


def main() -> None:
    banner("Exercise 14 · wire the hotel agent", "offline checker · free · no model is called", status=False)
    ok = True
    o, v = llm_block("ollama", "http://x:11434/v1", "m"), llm_block("vllm", "http://x:8000/v1", "m")
    ok &= check(isinstance(o, dict) and o.get("_type") == "openai" and o.get("reasoning_effort") == "none"
                and isinstance(v, dict) and v.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
                and "reasoning_effort" not in v and o.get("api_key") and o.get("temperature") == 0.0,
                "llm_block: openai type · Ollama thinks off via reasoning_effort · vLLM via chat_template_kwargs",
                f"TODO 1: llm_block('ollama', …) gave {o!r}")
    f = functions_block(".runs/t.jsonl")
    ok &= check(isinstance(f, dict) and set(f) == {"room_temperature", "create_maintenance_ticket", "current_datetime"}
                and all(isinstance(b, dict) and b.get("_type") == k for k, b in f.items())
                and f["create_maintenance_ticket"].get("ticket_log") == ".runs/t.jsonl",
                "functions_block: 3 tools, each `_type` = its registered name, ticket_log passed through",
                f"TODO 2: functions_block(...) gave {f!r}")
    w = workflow_block("hotel_llm", f if isinstance(f, dict) else {})
    ok &= check(isinstance(w, dict) and w.get("_type") == "tool_calling_agent" and w.get("llm_name") == "hotel_llm"
                and isinstance(f, dict) and sorted(w.get("tool_names") or []) == sorted(f)
                and 1 <= int(w.get("max_iterations") or 0) <= 6 and bool(w.get("system_prompt")),
                "workflow_block: tool_calling_agent · every function as a tool · max_iterations ≤ 6 · system prompt",
                f"TODO 3: workflow_block(...) gave {w!r}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your config, assembled and linted for both targets")
    out_dir = MOD / ".runs"
    out_dir.mkdir(exist_ok=True)
    for target in ("ollama", "vllm"):
        cfg = assemble(target)
        probs = lint(cfg)
        path = out_dir / f"ex14_{target}.yml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        ok &= check(not probs, f"lint {target:6s}: no problems → {path.relative_to(WEEK.parent)}",
                    f"lint {target}: " + "; ".join(probs))

    nat = WEEK / ".venv-nat" / "bin" / "nat"
    if nat.exists():
        print("\n▣ the real test: does NeMo Agent Toolkit accept it? (`nat validate`, offline)")
        env = {**os.environ, "PYTHONWARNINGS": "ignore", "NO_COLOR": "1"}
        for target in ("ollama", "vllm"):
            p = subprocess.run([str(nat), "validate", "--config_file", f".runs/ex14_{target}.yml"], cwd=MOD,
                               capture_output=True, text=True, timeout=300, env=env)
            good = p.returncode == 0 and "is valid" in p.stdout + p.stderr
            ok &= check(good, f"nat validate ex14_{target}.yml: valid",
                        f"nat validate ex14_{target}.yml: " + (p.stdout + p.stderr).strip().splitlines()[-1][:200])
    else:
        print("\n⚠ NAT is not installed in week25/.venv-nat — skipped `nat validate` (lab 14-1 shows the install).")
    if not ok:
        sys.exit(1)
    print("\n═ Run it: cd week25/14_nat_agents && ../.venv-nat/bin/nat run --config_file .runs/ex14_ollama.yml "
          "--input \"Check room 808\"")


if __name__ == "__main__":
    main()
