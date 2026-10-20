#!/usr/bin/env python3
"""Lab 07-4 · Speculative decoding calculator: acceptance rate → tokens per step → speed-up.

Pure arithmetic plus a seeded simulation, runs anywhere. It shows why verifying k drafted
tokens costs about the same as generating one on a Spark (decode is memory-bound), derives
the expected tokens per target step, E = (1 − α^(k+1)) / (1 − α), checks the formula
against a Monte Carlo run, and finds the best draft length for the playbook's two setups.

Run: .venv/bin/python week25/07_nvfp4_speculative/labs/lab04_speculative_calculator.py
     add --alpha 0.7 --cost 0.1 to try your own acceptance rate and draft cost
     add --measured 1.8 to back out the acceptance rate from a speed-up you measured with lab 03
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import SPEC, banner, bar, note, result, step, table, weights_gb  # noqa: E402


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens produced per target forward pass with k drafted tokens, each accepted with probability alpha.
    The target always adds one token of its own (the correction, or a bonus token when all k are accepted)."""
    return float(k + 1) if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    """Tokens per unit time vs plain decoding. One step = 1 target pass + k draft passes costing c each."""
    return expected_tokens(alpha, k) / (1 + k * c)


def simulate(alpha: float, k: int, steps: int, rng: random.Random) -> float:
    total = 0
    for _ in range(steps):
        accepted = 0
        while accepted < k and rng.random() < alpha:     # drafts are checked left to right; first miss stops
            accepted += 1
        total += accepted + 1                             # + the target's own token
    return total / steps


def arg(flag: str, default: float) -> float:
    try:
        return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
    except (IndexError, ValueError):
        return default


banner("Lab 07-4 · speculative decoding calculator", "arithmetic + a seeded simulation · no Spark needed",
       status=False)

step(1, "why checking 6 tokens costs about the same as generating 1 (Llama 3.3 70B, NVFP4, one Spark)")
params = 70.6e9
read_gb = weights_gb(70.6, "nvfp4")
t_mem = read_gb / SPEC["mem_bw_gbs"] * 1000                          # ms to stream every weight once
flops_tok = 2 * params                                                # ≈ 2 FLOPs per parameter per token
rows = []
for n in (1, 6, 32):
    t_cmp = n * flops_tok / (SPEC["fp4_pflops"] * 1e15) * 1000        # ms at the 1 PFLOP (sparse FP4) peak
    rows.append([f"{n} token{'s' if n > 1 else ''}", f"{read_gb:.1f} GB", f"{t_mem:6.1f} ms",
                 f"{n * flops_tok / 1e12:6.2f} TFLOP", f"{t_cmp:6.2f} ms", "memory" if t_mem > t_cmp else "compute"])
table(rows, ["tokens in one pass", "weights read", "time to read", "math", "time at peak", "bound by"])
note("One forward pass reads all the weights once, whether it scores 1 token or 6. Even if the real compute rate "
     "is 10× below the peak, the math for 6 tokens stays far below the ~145 ms memory read. The extra "
     "positions are nearly free. That is the gap speculative decoding fills.")

step(2, "tokens per target step: E = (1 − α^(k+1)) / (1 − α)")
ks = [1, 2, 3, 4, 5, 8]
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
rows = [[f"α = {a}"] + [f"{expected_tokens(a, k):.2f}" for k in ks] for a in alphas]
table(rows, ["acceptance"] + [f"k={k}" for k in ks])
note("α = probability that the target agrees with one drafted token. k = max_draft_len. Even α = 0.5 gives "
     "~2 tokens per pass with enough drafts; the ceiling is 1 ÷ (1 − α) tokens per pass, however large k gets.")

step(3, "check the formula: simulate 20,000 steps per cell (seed 25)")
rng = random.Random(25)
rows = []
for a, k in [(0.6, 3), (0.7, 4), (0.8, 5), (0.9, 5)]:
    sim, exact = simulate(a, k, 20_000, rng), expected_tokens(a, k)
    rows.append([f"α={a} k={k}", f"{exact:.3f}", f"{sim:.3f}", f"{abs(sim - exact) / exact * 100:.2f} %"])
table(rows, ["case", "formula", "simulated", "difference"])

step(4, "the draft is not free: speed-up = E ÷ (1 + k·c), c = cost of one draft pass ÷ one target pass")
alpha, c_user = arg("--alpha", 0.7), arg("--cost", -1.0)
setups = [
    ("Draft-Target: 8B FP4 drafts for 70B FP4", weights_gb(8.0, "nvfp4") / weights_gb(70.6, "nvfp4"), 4),
    ("EAGLE-3 head on gpt-oss-120b (assumed c)", 0.05, 5),
]
if c_user >= 0:
    setups.append((f"your setup (--cost {c_user})", c_user, 4))
for name, c, playbook_k in setups:
    print(f"\n│ {name} · c = {c:.3f} · α = {alpha} · playbook max_draft_len = {playbook_k}")
    best_k = max(range(1, 11), key=lambda k: speedup(alpha, k, c))
    for k in (1, 2, 3, 4, 5, 6, 8):
        s = speedup(alpha, k, c)
        mark = " · ".join(t for t, hit in (("best", k == best_k), ("playbook", k == playbook_k)) if hit)
        mark = f"  ← {mark}" if mark else ""
        print(f"│   k={k}  E={expected_tokens(alpha, k):4.2f}  speed-up {s:4.2f}×  {bar(s, 3.5)}{mark}")
note("Draft-Target's c comes from arithmetic: the 8B draft reads 8 ÷ 70.6 of the target's bytes per pass. The "
     "EAGLE-3 c = 0.05 is an ASSUMPTION for illustration (a head of about one decoder layer). Change it with --cost.")

step(5, "what α do you need for a 2× speed-up at k = 4?")
for name, c, _ in setups[:2]:
    need = next((a / 100 for a in range(1, 100) if speedup(a / 100, 4, c) >= 2), None)
    print(f"│ {name:44s} α ≥ {need}" if need else f"│ {name:44s} not reachable at k = 4")

measured = arg("--measured", -1.0)
if measured > 0:
    step(6, f"back out α from YOUR measured speed-up ({measured}× = eagle3 tok/s ÷ baseline tok/s, from lab 03)")
    for name, c, k in setups[:2]:
        lo, hi = 0.0, 0.999
        for _ in range(60):                                           # bisection: speedup() grows with α
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if speedup(mid, k, c) < measured else (lo, mid)
        fits = speedup(0.0, k, c) <= measured <= speedup(0.999, k, c)
        print(f"│ {name:44s} k={k}  α ≈ {lo:.2f}" if fits else f"│ {name:44s} k={k}  outside this model's range")
    note("This inverts the simple model (independent acceptances, c fixed), so it is an estimate. The playbook "
         "asks you to monitor acceptance rates; the TensorRT-LLM speculative decoding docs cover how.")

result("Speculative decoding keeps the target's output: the target checks every drafted token and keeps "
       "only what it would have produced (exactly, for greedy decoding; the same distribution, when sampling). "
       "What changes is how many tokens each expensive pass yields. Measure α on your own prompts: code and structured text accept well, creative text less.")
