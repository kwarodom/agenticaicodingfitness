#!/usr/bin/env python3
"""Lab 19-1 · Stack map: services, ports, GPUs and memory of the three playbook stacks.

Before you `docker compose up` two big stacks on one Spark, read what they start.
This lab parses the playbooks' own Compose files (from the dgx-spark-playbooks clone
at the repo root) and prints, per stack: every service, its image, the host ports it
publishes and whether it reserves the GPU. Then it finds port collisions — between
the stacks, and with the servers earlier modules left running (Ollama :11434,
vLLM :8000, …) — and adds up the memory the playbooks say each stack needs.

On a Spark it also lists running containers and listening ports (read-only).

Run: .venv/bin/python week25/19_multi_agent_rag_kg/labs/lab01_stack_map.py
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import PORTS, ROOT, SPEC, banner, note, result, sh, step, table, warn, weights_gb  # noqa: E402

PB = ROOT / "dgx-spark-playbooks" / "nvidia"
STACKS = {   # stack → compose files, in the order the playbook passes them
    "multi-agent chatbot": ["playbook-multi-agent-chatbot/assets/docker-compose.yml",
                            "playbook-multi-agent-chatbot/assets/docker-compose-models.yml"],
    "txt2kg (./start.sh)": ["playbook-txt2kg/assets/deploy/compose/docker-compose.yml"],
    "txt2kg --neo4j": ["playbook-txt2kg/assets/deploy/compose/docker-compose.neo4j.yml"],
    "txt2kg --vllm": ["playbook-txt2kg/assets/deploy/compose/docker-compose.vllm.yml"],
}
COURSE_OWNER = {"ollama": "Modules 03/18 (host Ollama)", "vllm": "Module 05 (vLLM)", "nim": "Module 06 (NIM)",
                "sglang": "Module 06", "trtllm": "Modules 06-07", "llamacpp": "Module 04", "lmstudio": "Module 04",
                "litellm": "Module 08", "openwebui": "Module 03", "dashboard": "DGX Dashboard"}


def host_ports(svc: dict) -> list[int]:
    out = []
    for p in svc.get("ports") or []:
        parts = str(p).split(":")
        if len(parts) >= 2 and parts[-2].isdigit():
            out.append(int(parts[-2]))
    return out


def load(stack: str) -> list[dict]:
    rows = []
    for rel in STACKS[stack]:
        d = yaml.safe_load((PB / rel).read_text(encoding="utf-8")) or {}
        for name, s in (d.get("services") or {}).items():
            gpu = bool(((s.get("deploy") or {}).get("resources") or {}).get("reservations"))
            rows.append({"stack": stack, "service": name, "image": s.get("image") or "(built locally)",
                         "ports": host_ports(s), "gpu": gpu, "profiles": s.get("profiles") or []})
    return rows


def main() -> None:
    banner("Lab 19-1 · stack map — what the multi-agent chatbot and txt2kg actually start",
           "parsed from the playbooks' docker-compose files · ports · GPUs · memory")
    if not PB.is_dir():
        warn(f"playbooks not found at {PB}.")
        print("$ git clone https://github.com/NVIDIA/dgx-spark-playbooks   # run at the repo root (gitignored)")
        result("Clone the playbooks, then rerun this lab.")
        return

    step(1, "every service, per stack (from the Compose files)")
    services = [r for s in STACKS for r in load(s)]
    table([[r["stack"], r["service"], r["image"][:44], ",".join(map(str, r["ports"])) or "—",
            "GPU" if r["gpu"] else "", "profile: " + ",".join(r["profiles"]) if r["profiles"] else ""]
           for r in services], ["stack", "service", "image", "host ports", "gpu", "note"])
    note("The multi-agent model servers publish no host port: the backend reaches them by container name on the "
         "chatbot-net network (http://gpt-oss-120b:8000/v1, http://deepseek-coder:8000/v1, …).")

    step(2, "port collisions — between stacks, and with servers earlier modules started")
    course = {}
    for kind, port in PORTS.items():
        course.setdefault(port, []).append(COURSE_OWNER.get(kind, kind))
    by_port = {}
    for r in services:
        short = "chatbot" if r["stack"].startswith("multi") else "txt2kg"
        tag = f"{short}:{r['service']}" + (" (opt)" if r["profiles"] else "")
        for p in r["ports"]:
            by_port.setdefault(p, set()).add(tag)
    rows = []
    for p in sorted(by_port):
        users = sorted(by_port[p])
        two_stacks = len({u.split(":")[0] for u in users}) > 1
        if two_stacks or p in course:
            rows.append([p, ", ".join(users), "yes" if two_stacks else "—", " / ".join(course.get(p, [])) or "—"])
    table(rows, ["port", "published by", "both stacks?", "course server on that port"])
    note("(opt) = only with --vector-search. Docker refuses to start a container whose host port is taken "
         "('port is already allocated'). Stop the earlier server first, or run one stack at a time.")

    step(3, "memory: what the playbooks say each stack needs")
    llama8 = weights_gb(8, "q4_k_m")
    table([["multi-agent chatbot", "~120 GB (playbook: 'uses ~120 GB of memory by default')", "REFERENCE"],
           ["  downloads", "gpt-oss-120B ~63 GB · Deepseek-Coder 6.7B ~7 GB · Qwen3-Embedding-4B ~4 GB", "REFERENCE"],
           ["txt2kg (Ollama stacks)", f"llama3.1:8b default → ~{llama8:.1f} GB of Q4_K_M weights + KV cache", "arithmetic"],
           ["txt2kg --vllm", "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 → ~49 GB of FP8 weights", "arithmetic"],
           ["Module 18 coding model", "qwen3.6:35b-a3b-mtp-q4_K_M ~23 GB", "REFERENCE"],
           ["one Spark", f"{SPEC['memory_gb']} GB unified (≈119 GiB in free -g)", "spec"]],
          ["what", "memory", "source"])
    note("The chatbot alone fills a Spark. Stop it (docker compose … down) before starting txt2kg, and unload "
         "Module 18's model (ollama stop <model>), or switch the supervisor to gpt-oss-20B (playbook Step 8).")

    step(4, "on the Spark: what is running and listening right now (read-only)")
    sh("docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
       example="NAMES          STATUS                   PORTS\n"
               "frontend       Up 3 minutes             0.0.0.0:3000->3000/tcp\n"
               "backend        Up 3 minutes             0.0.0.0:8000->8000/tcp\n"
               "gpt-oss-120b   Up 3 minutes\n"
               "qwen2.5-vl     Up 3 minutes (unhealthy)")
    sh("ss -ltn | grep -E ':(3000|3001|5432|7474|8000|8001|8529|11434|19530)\\b' || echo 'none of these ports is in use'",
       example="none of these ports is in use")
    result("Map first, then start: one heavy stack per Spark at a time, and free :8000 and :11434 before you "
           "start a stack that publishes them.")


if __name__ == "__main__":
    main()
