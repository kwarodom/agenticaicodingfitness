#!/usr/bin/env python3
"""Exercise 05 · Write a correct vLLM launch command: memory, tool calling, and a LoRA adapter.

Fill in the four TODOs, save, then run:
    .venv/bin/python week25/05_vllm/exercises/ex05_launch_command.py

The checker is free and offline. It builds a full `docker run … vllm serve …` command from your
functions, parses it like a shell would, and checks it against the vLLM playbook's rules and Module 01's
memory math. Stuck? Compare with exercises/solutions/.
"""
import math  # noqa: F401 — you will need math.floor in TODO 2
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

SPARK_GB = 128
RUNTIME_GB = 4          # activations + CUDA graphs inside vLLM's slice (same assumption as lab 05-1)


# ── TODO 1 ── the container flags from the playbook's base configuration, as a list of tokens.
#   GPU access, shared memory for the engine (--ipc host), the OpenAI port (8000 on the Spark → 8000 in
#   the container), and the Hugging Face cache mounted at /root/.cache/huggingface so downloads persist.
def docker_flags() -> list[str]:
    return []


# ── TODO 2 ── how many FULL-LENGTH sequences fit? (this becomes --max-num-seqs)
#   slice = util × 128 GB. KV room = slice − weights_gb − RUNTIME_GB. Return floor(KV room ÷ kv_gb_per_seq),
#   or 0 when nothing fits.
def max_full_length_seqs(weights_gb: float, kv_gb_per_seq: float, util: float) -> int:
    return None


# ── TODO 3 ── tool-calling flags for a model family, as a list of tokens.
#   Use the parsers the playbooks pair with each family:
#     "qwen3.6" → qwen3_xml (agent-ready Qwen3.6 recipe) · "nemotron" → qwen3_coder · "gemma4" → gemma4
#   Auto tool choice must be switched on too, or vLLM rejects requests that send `tools`.
def tool_flags(family: str) -> list[str]:
    return []


# ── TODO 4 ── serve a LoRA adapter next to the base model, as a list of tokens.
#   vLLM flags (from vLLM's LoRA docs — the playbook does not cover LoRA): switch LoRA on, register the
#   adapter as NAME=PATH (PATH is where the adapter sits INSIDE the container), and allow adapters up to
#   `rank`. Clients then call it with "model": NAME.
def lora_flags(name: str, path: str, rank: int) -> list[str]:
    return []


# ─────────────────────────── checker — no need to edit below ────────────────
IMAGE = "vllm/vllm-openai:latest"
AGENT = "nvidia/Qwen3.6-35B-A3B-NVFP4"           # the playbook's agent-ready model for DGX Spark
BASE = "Qwen/Qwen3-8B"                           # stand-in base for a LoRA: use your adapter's base model


def q(t: str) -> str:
    return f'"{t}"' if "$" in t else shlex.quote(t)          # keep $HOME expandable by the Spark's shell


def build(model: str, util: float, ctx: int, seqs: int, extra_mounts: list[str], tail: list[str]) -> list[str]:
    serve = ["vllm", "serve", model, "--max-model-len", str(ctx), "--gpu-memory-utilization", str(util),
             "--max-num-seqs", str(seqs)]
    return (["docker", "run", "-d", "--name", "vllm-server"] + (docker_flags() or []) + extra_mounts
            + ["--entrypoint", "", IMAGE] + serve + tail)


def pretty(toks: list[str]) -> str:
    """One flag (and its value) per line, like the playbook prints its commands."""
    i, img = 3, toks.index(IMAGE)
    lines = ["$ docker run -d"]
    while i < img:
        if i + 1 < img and not toks[i + 1].startswith("-"):
            lines.append(f"  {toks[i]} {q(toks[i + 1])}")
            i += 2
        else:
            lines.append(f"  {toks[i]}")
            i += 1
    lines.append(f"  {IMAGE}")
    lines.append("  " + " ".join(toks[img + 1:img + 4]))
    i = img + 4
    while i < len(toks):
        if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
            lines.append(f"    {toks[i]} {q(toks[i + 1])}")
            i += 2
        else:
            lines.append(f"    {toks[i]}")
            i += 1
    return " \\\n".join(lines)


def has_pair(toks: list[str], flag: str, value: str | None = None) -> bool:
    """flag present (as `--f v`, `--f=v` or bare `--f`), optionally with this exact value."""
    for i, t in enumerate(toks):
        if t == flag:
            return value is None or (i + 1 < len(toks) and toks[i + 1] == value)
        if t.startswith(flag + "="):
            return value is None or t.split("=", 1)[1] == value
    return False


def main() -> None:
    banner("Exercise 05 · a correct vLLM launch command", "offline checker · free · no Spark needed", status=False)
    ok = True

    d = docker_flags() or []
    ok &= check(has_pair(d, "--gpus", "all") and (has_pair(d, "--ipc", "host") or "--ipc=host" in d)
                and has_pair(d, "-p", "8000:8000")
                and any(t.endswith(":/root/.cache/huggingface") for t in d),
                "docker_flags: --gpus all · --ipc host · -p 8000:8000 · HF cache → /root/.cache/huggingface",
                f"TODO 1: docker_flags() is missing a flag (got {d!r})")

    got = [max_full_length_seqs(*a) for a in ((18.4, 8.59, 0.8), (39.7, 10.74, 0.8), (141.2, 10.74, 0.8),
                                             (20.0, 2.0, 0.4))]
    ok &= check(got == [9, 5, 0, 13],
                "max_full_length_seqs: Qwen3-32B@0.8 → 9 · Llama-70B-NVFP4@0.8 → 5 · 70B-bf16 → 0 · 20 GB@0.4 → 13",
                f"TODO 2: expected [9, 5, 0, 13] (got {got!r})")

    t = tool_flags("qwen3.6") or []
    parsers = [tool_flags(f) or [] for f in ("nemotron", "gemma4")]
    ok &= check("--enable-auto-tool-choice" in t and has_pair(t, "--tool-call-parser", "qwen3_xml")
                and has_pair(parsers[0], "--tool-call-parser", "qwen3_coder")
                and has_pair(parsers[1], "--tool-call-parser", "gemma4"),
                "tool_flags: auto tool choice on · qwen3.6→qwen3_xml · nemotron→qwen3_coder · gemma4→gemma4",
                f"TODO 3: tool_flags('qwen3.6') gave {t!r}")

    lf = lora_flags("hotel-ft", "/adapters/hotel-ft", 16) or []
    rank_ok = False
    for i, tok in enumerate(lf):
        v = tok.split("=", 1)[1] if tok.startswith("--max-lora-rank=") else (
            lf[i + 1] if tok == "--max-lora-rank" and i + 1 < len(lf) else None)
        if v is not None:
            rank_ok = v.isdigit() and int(v) >= 16
    ok &= check("--enable-lora" in lf and has_pair(lf, "--lora-modules", "hotel-ft=/adapters/hotel-ft") and rank_ok,
                "lora_flags: --enable-lora · --lora-modules hotel-ft=/adapters/hotel-ft · --max-lora-rank ≥ 16",
                f"TODO 4: lora_flags(...) gave {lf!r}")

    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    # A · the agent-ready server (a trimmed version of the playbook recipe: memory + tool flags only)
    a = build(AGENT, 0.4, 262_144, 4, [], ["--kv-cache-dtype", "fp8"] + (tool_flags("qwen3.6") or []))
    # B · a LoRA fine-tune on its base model (Qwen3-8B bf16: 16.4 GB weights, 4.83 GB KV per 32K sequence)
    seqs = max(1, max_full_length_seqs(16.4, 4.83, 0.5))
    b = build(BASE, 0.5, 32_768, seqs, ["-v", "$HOME/w25/adapters:/adapters"],
              lora_flags("hotel-ft", "/adapters/hotel-ft", 16) or [])
    parsed = True
    for toks in (shlex.split(" ".join(q(t) for t in a)), shlex.split(" ".join(q(t) for t in b))):
        parsed &= toks[toks.index(IMAGE) + 1:toks.index(IMAGE) + 3] == ["vllm", "serve"] and "--entrypoint" in toks
    ok &= check(parsed, "both assembled commands parse: docker flags, IMAGE, then `vllm serve <model>` flags",
                "an assembled command does not parse as docker run … IMAGE vllm serve …")

    print("\n▣ A · agent-ready server (assembled from your functions; the full recipe adds speed flags)")
    print(pretty(a))
    print(f"\n▣ B · base model + your LoRA adapter (--max-num-seqs {seqs} = your TODO 2 at util 0.5, 32K context)")
    print(pretty(b))
    print("\n◆ --max-num-seqs 4 in A is the playbook recipe's own value for 262K-token contexts.")
    print("═ In B, GET /v1/models lists both ids: the base and hotel-ft. Send \"model\": \"hotel-ft\" to use the adapter.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
