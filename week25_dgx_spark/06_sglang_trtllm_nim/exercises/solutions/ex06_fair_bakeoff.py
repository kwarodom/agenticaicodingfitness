#!/usr/bin/env python3
"""Exercise 06 · reference solution — plan a fair engine bake-off: ports, clashes, fairness rules, and honest summaries.

Fill in the four TODOs, save, then run:
    .venv/bin/python week25/06_sglang_trtllm_nim/exercises/ex06_fair_bakeoff.py

The checker is free and offline. It tests your port map against the playbooks, finds engines that cannot
run at the same time, rejects unfair comparisons, and summarises repeated runs the honest way.
Stuck? Compare with exercises/solutions/.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from sparkkit import banner, check  # noqa: E402


# ── TODO 1 ── the OpenAI port each engine's DGX Spark playbook uses (the host side of -p / --port).
#   vLLM (base configuration), SGLang, TensorRT-LLM (single-node Instructions), NIM for LLMs, and the
#   TensorRT-LLM recipe for Nemotron Super (it uses its own port — look in the Nemotron playbook).
PORTS = {
    "vllm": 8000,
    "sglang": 30000,
    "trtllm": 8355,
    "nim": 8000,
    "trtllm-nemotron-super": 8123,
}


# ── TODO 2 ── which engines cannot run side by side on one Spark without changing a -p mapping?
#   Return every pair that shares a port, each pair sorted alphabetically, the list sorted.
#   Example: {"a": 1, "b": 1, "c": 2} → [("a", "b")]
def port_conflicts(ports: dict) -> list[tuple[str, str]]:
    names = sorted(ports)
    return [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if ports[x] == ports[y]]


# ── TODO 3 ── is comparing run A with run B fair? Return a list of problems (empty list = fair).
#   Each run is a dict with: model, precision, prompts (a list), max_tokens, temperature, concurrency,
#   warmup (bool), other_engines_running (int).
#   Rules — add one short string per broken rule, e.g. "precision differs":
#     same model · same precision · same prompts · same max_tokens · same concurrency
#     temperature 0 in both (greedy decoding: same work every time)
#     a warm-up request in both (the first call pays for loading and CUDA-graph capture)
#     no other engine running during either run (they share the 128 GB and the bandwidth)
def fairness_problems(a: dict, b: dict) -> list[str]:
    problems = [f"{k} differs" for k in ("model", "precision", "prompts", "max_tokens", "concurrency") if a[k] != b[k]]
    if a["temperature"] != 0 or b["temperature"] != 0:
        problems.append("temperature is not 0 in both runs")
    if not (a["warmup"] and b["warmup"]):
        problems.append("a run had no warm-up request")
    if a["other_engines_running"] or b["other_engines_running"]:
        problems.append("another engine was running")
    return problems


# ── TODO 4 ── summarise repeated runs of one engine: the MEDIAN of each metric, never the best run.
#   runs is a list of dicts with keys "ttft_ms" and "tok_s". Return {"ttft_ms": median, "tok_s": median}.
def summarise(runs: list[dict]) -> dict:
    return {"ttft_ms": statistics.median(r["ttft_ms"] for r in runs),
            "tok_s": statistics.median(r["tok_s"] for r in runs)}


# ─────────────────────────── checker — no need to edit below ────────────────
BASE_RUN = {"model": "Llama-3.1-8B-Instruct", "precision": "nvfp4", "prompts": ["p1", "p2", "p3"],
            "max_tokens": 128, "temperature": 0.0, "concurrency": 1, "warmup": True, "other_engines_running": 0}


def main() -> None:
    banner("Exercise 06 · a fair engine bake-off", "offline checker · free · no Spark needed", status=False)
    ok = True

    want = {"vllm": 8000, "sglang": 30000, "trtllm": 8355, "nim": 8000, "trtllm-nemotron-super": 8123}
    wrong = {k: PORTS.get(k) for k in want if PORTS.get(k) != want[k]}
    ok &= check(not wrong, "PORTS: vLLM 8000 · SGLang 30000 · TensorRT-LLM 8355 · NIM 8000 · Nemotron Super TRT-LLM 8123",
                f"TODO 1: these ports do not match the playbooks: {wrong}")

    pc = port_conflicts({"a": 1, "b": 1, "c": 2, "d": 1})
    real = port_conflicts(want) if pc is not None else None
    ok &= check(pc == [("a", "b"), ("a", "d"), ("b", "d")] and real == [("nim", "vllm")],
                "port_conflicts: finds every shared port · on a Spark: NIM and vLLM both want :8000",
                f"TODO 2: port_conflicts({{'a':1,'b':1,'c':2,'d':1}}) should be [('a','b'), ('a','d'), ('b','d')] (got {pc!r})")

    cases = [({}, 0), ({"precision": "fp8"}, 1), ({"temperature": 0.7}, 1), ({"warmup": False}, 1),
             ({"other_engines_running": 1}, 1), ({"max_tokens": 256, "concurrency": 8}, 2),
             ({"prompts": ["p1"], "model": "Qwen3-8B"}, 2)]
    got = []
    for change, _ in cases:
        probs = fairness_problems(BASE_RUN, {**BASE_RUN, **change})
        got.append(len(probs) if isinstance(probs, list) else None)
    ok &= check(got == [n for _, n in cases],
                "fairness_problems: identical → fair · precision, temperature, warm-up, a second engine, "
                "max_tokens + concurrency, model + prompts all caught",
                f"TODO 3: expected problem counts {[n for _, n in cases]} (got {got})")

    s = summarise([{"ttft_ms": 120, "tok_s": 40.0}, {"ttft_ms": 900, "tok_s": 12.0}, {"ttft_ms": 140, "tok_s": 38.0}])
    ok &= check(s == {"ttft_ms": 140, "tok_s": 38.0},
                "summarise: median of 3 runs (TTFT 140 ms, 38.0 tok/s) — one slow outlier does not decide the result",
                f"TODO 4: summarise(...) should be {{'ttft_ms': 140, 'tok_s': 38.0}} (got {s!r})")

    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your plan, applied to four engines serving Llama 3.1 8B Instruct")
    order = sorted(["vllm", "sglang", "trtllm", "nim"], key=lambda e: PORTS[e])
    for e in order:
        print(f"│ {e:7s} :{PORTS[e]:<5d}  start → warm-up → 3 prompts → stop")
    for x, y in port_conflicts({e: PORTS[e] for e in order}):
        print(f"◆ {x} and {y} share :{PORTS[x]} — run them one after the other, or map one to -p 8001:{PORTS[x]}")
    print("═ One engine at a time, same model and precision, temperature 0, warm-up, medians. Then lab 06-1.")


if __name__ == "__main__":
    main()
