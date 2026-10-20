"""Capstone helper (not a lab): the two-Spark plan, the gateway config, and the acceptance scenarios.

Every capstone lab imports this so the plan, the ports and the tests are defined once.
It reuses, rather than copies, the earlier modules:
  Module 08  _gateway.Proxy / write_config / raw   the LiteLLM proxy runner
  Module 13  _hotel_eval                            the router's parser and the ship gate
  Module 15  policykit                              policy build / validate / decide
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
WEEK = MOD.parent
for p in (WEEK / "common", WEEK / "08_litellm_gateway" / "labs", WEEK / "13_finetune_to_serve" / "labs",
          WEEK / "15_openshell_sandbox"):
    sys.path.insert(0, str(p))

import _gateway as G      # noqa: E402,F401  (Module 08)
import _hotel_eval as H   # noqa: E402,F401  (Module 13)
import policykit as PK    # noqa: E402,F401  (Module 15)
from sparkkit import LAPTOP_OLLAMA, url  # noqa: E402

RUNS = MOD / ".runs"
NAT = WEEK / ".venv-nat" / "bin" / "nat"
AGENT_YML = MOD / "configs" / "capstone_agent.yml"

# ── the plan: which Spark serves what ─────────────────────────────────────────
BRAIN_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"     # Module 05's agent-ready recipe (older vLLM playbook)
ROUTER_BASE = "Qwen/Qwen3-4B-Instruct-2507"      # Module 09's base; the adapter is served as hotel-ft
LAPTOP_MODEL = "gemma4:12b"                       # stand-in for both, when no Spark answers

PLAN = [
    # (Spark, service, port, model, GB of weights (arithmetic), module)
    ("A", "vLLM · agent brain", 8000, BRAIN_MODEL, 20.0, "05"),
    ("A", "LiteLLM gateway", 4000, "agent-brain · hotel-router aliases", 0.5, "08"),
    ("A", "OpenShell sandbox + NAT agent", 0, "route_guest_message · room_temperature · tickets", 1.0, "14 · 15"),
    ("B", "vLLM · base + LoRA hotel-ft", 8000, f"{ROUTER_BASE} + hotel-ft", 8.1, "09 · 13"),
]


def gateway_config() -> dict:
    """Two aliases, each with a Spark primary and a laptop fallback (Module 08's shape)."""
    brain_a = url("vllm", "a") or "http://spark-a:8000/v1"
    router_b = url("vllm", "b") or "http://spark-b:8000/v1"
    laptop = {"model": f"openai/{LAPTOP_MODEL}", "api_base": LAPTOP_OLLAMA, "api_key": "none",
              "extra_body": {"reasoning_effort": "none"}}          # Module 08: thinking off in Ollama
    return {
        "model_list": [
            {"model_name": "agent-brain",
             "litellm_params": {"model": f"hosted_vllm/{BRAIN_MODEL}", "api_base": brain_a, "api_key": "none"}},
            {"model_name": "hotel-router",
             "litellm_params": {"model": "hosted_vllm/hotel-ft", "api_base": router_b, "api_key": "none"}},
            {"model_name": "agent-brain-laptop", "litellm_params": dict(laptop)},
            {"model_name": "hotel-router-laptop", "litellm_params": dict(laptop)},
        ],
        "router_settings": {"num_retries": 0, "timeout": 120,
                            "fallbacks": [{"agent-brain": ["agent-brain-laptop"]},
                                          {"hotel-router": ["hotel-router-laptop"]}]},
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


# ── acceptance scenarios: what a night manager would check by hand ────────────
SCENARIOS = [
    {"id": "smoke", "message": "Hello, there's a strong smell of smoke in the corridor near room 522. Please hurry.",
     "department": "security", "ticket": {"room": "522", "priority": "urgent"}},
    {"id": "hot-room", "message": "Guest in room 808: it's really hot in here and the air conditioning just blows air.",
     "department": "engineering", "ticket": {"room": "808"}, "must_call": "room_temperature"},
    {"id": "restaurant", "message": "Could you recommend a good seafood restaurant within walking distance? Thank you.",
     "department": "concierge", "ticket": None},
    {"id": "thai-tv", "message": "สวัสดีครับ ทีวีในห้อง 1203 เปิดไม่ติด",
     "department": "engineering", "ticket": {"room": "1203", "priority": "normal"}},
]


def read_trace(path: Path) -> list[dict]:
    out = []
    if path.is_file():
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                p = json.loads(ln)["payload"]
            except (json.JSONDecodeError, KeyError):
                continue
            out.append({"type": p.get("event_type"), "name": p.get("name"),
                        "input": (p.get("data") or {}).get("input"), "output": (p.get("data") or {}).get("output")})
    return out


def tool_calls(trace: list[dict]) -> list[tuple[str, str, str]]:
    """(tool, input, output) for every tool the agent ran, in order."""
    starts = [e for e in trace if e["type"] == "FUNCTION_START" and e["name"] in
              ("route_guest_message", "room_temperature", "create_maintenance_ticket")]
    ends = [e for e in trace if e["type"] == "FUNCTION_END" and e["name"] in
            ("route_guest_message", "room_temperature", "create_maintenance_ticket")]
    return [(s["name"], str(s["input"]), str(e["output"])) for s, e in zip(starts, ends)]


def final_answer(trace: list[dict]) -> str:
    """The agent's answer, from the workflow's own END event (not scraped from console output)."""
    ends = [e for e in trace if e["type"] in ("FUNCTION_END", "WORKFLOW_END") and e["name"] in ("<workflow>", "tool_calling_agent")]
    return str(ends[-1]["output"]) if ends and ends[-1]["output"] else ""


def judge(sc: dict, calls: list[tuple[str, str, str]], tickets: list[dict]) -> list[tuple[str, bool]]:
    """The acceptance checks for one scenario → [(check, passed)]."""
    names = [c[0] for c in calls]
    checks = [("routed first", bool(names) and names[0] == "route_guest_message")]
    routed = next((c for c in calls if c[0] == "route_guest_message"), None)
    dept = None
    if routed:
        try:
            dept = json.loads(routed[2]).get("department")
        except (json.JSONDecodeError, AttributeError):
            dept = None
    checks.append((f"router said {sc['department']}", dept == sc["department"]))
    if sc.get("must_call"):
        checks.append((f"called {sc['must_call']}", sc["must_call"] in names))
    want = sc.get("ticket")
    if want is None:
        checks.append(("no ticket opened", not tickets))
    else:
        t = tickets[-1] if tickets else {}
        checks.append((f"ticket for room {want['room']}", t.get("room") == want["room"]))
        if "priority" in want:
            checks.append((f"ticket priority {want['priority']}", t.get("priority") == want["priority"]))
    return checks
