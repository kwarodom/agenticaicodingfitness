#!/usr/bin/env python3
"""Lab 01 · Build a verifiable-reward environment and score REAL rollouts.

An environment = a task sampler + a deterministic reward_fn. The demos score
canned answers; HERE the rollouts come from a live model on your endpoint and
YOUR verifier decides 1.0 or 0.0 — the exact signal GRPO trains on.

Run:  cd <repo root> && .venv/bin/python week23/10_nemo_gym_rl/labs/lab01_build_verifier.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  — inherits DGX_CONN / DGX_BASE_URL / DGX_API_KEY

TIMEOUT_S = 25
MAX_TOKENS = 300          # thinking models need headroom before the final number

# ── the environment: two tasks + one deterministic verifier ───────────────────
TASKS = [
    {"prompt": "What is 37 * 24? Reply with the number only.", "gold": "888"},
    {"prompt": "What is 512 - 137? Reply with the number only.", "gold": "375"},
]


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.S)


def reward_fn(task: dict, answer: str) -> float:
    """Deterministic verifier: the LAST number in the reply must equal gold.
    Same answer → same reward, every time. 1.0 pass / 0.0 fail — nothing else."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", _strip_think(answer))
    return 1.0 if nums and nums[-1] == task["gold"] else 0.0


def _rollout(prompt: str, temperature: float = 0.7) -> str:
    from openai import BadRequestError, OpenAI, UnprocessableEntityError
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=TIMEOUT_S)
    kw = dict(model=config.MODEL, temperature=temperature, max_tokens=MAX_TOKENS,
              messages=[{"role": "user", "content": prompt}])
    try:    # ask thinking models to skip the trace — the verifier needs the ANSWER
        r = client.chat.completions.create(
            extra_body={"reasoning_effort": "none"}, **kw)
    except (BadRequestError, UnprocessableEntityError):
        r = client.chat.completions.create(**kw)   # server rejects the hint → plain
    msg = r.choices[0].message
    return (str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")).strip()


def main() -> None:
    print("━" * 64)
    print("  ▣ Lab 01 — build a verifiable-reward environment")
    print("━" * 64)

    print("\n◈ Step 1 — the verifier is DETERMINISTIC (canned answers, no model):")
    for cand in ["888", "The answer is 888.", "It's roughly 890.", ""]:
        r = reward_fn(TASKS[0], cand)
        print(f"  reward_fn(gold='888', answer={cand!r:<24}) → {r:.1f}  "
              f"{'PASS ✓' if r == 1.0 else 'fail ✗'}")
    print("  — same input, same reward, forever. A fuzzy LLM judge can't promise that.")

    if config.MODE != "real":
        print("\n◈ Step 2 — needs a live endpoint, and none is reachable.")
        print("  Point the labs at one, then re-run:")
        print("    ollama pull qwen3.6            # C-path: local Ollama at :11434")
        print("    export DGX_CONN=local")
        print("    # or B-path (build.nvidia.com):")
        print("    export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1")
        print("    export DGX_API_KEY=nvapi-...   DGX_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        print("\n  [no endpoint — showing expected output]")
        print("  » What is 37 * 24? Reply with the number only.")
        print("  · rollout: '888'                          → reward 1.0  PASS ✓")
        print("  » What is 512 - 137? Reply with the number only.")
        print("  · rollout: 'The result is 375'            → reward 1.0  PASS ✓")
        print("\n  ✓ (expected) 2/2 rollouts verified — reward came from code, not a person.")
        return

    print(f"\n◈ Step 2 — REAL rollouts: {config.MODEL} @ {config.safe_base_url()}")
    print(f"  connection: {config.CONN} · {config.cost_note()}")
    passed = 0
    for task in TASKS:
        print(f"\n  » {task['prompt']}")
        try:
            ans = _rollout(task["prompt"])
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ rollout failed ({type(e).__name__}) — check the endpoint & model.")
            continue
        tail = _strip_think(ans).strip().replace("\n", " ")[-70:]
        r = reward_fn(task, ans)
        passed += r == 1.0
        print(f"  · rollout tail: …{tail!r}")
        print(f"  · reward_fn → {r:.1f}  {'PASS ✓' if r == 1.0 else 'fail ✗'}")

    print(f"\n✓ {passed}/{len(TASKS)} rollouts verified. That scalar — 1.0 or 0.0, from")
    print("  code you wrote — is the entire training signal GRPO needs. No labels,")
    print("  no reward model. Next: lab02 samples a GROUP and computes advantages.")
    if passed == 0:
        print("  ◈ 0 passes with a thinking model usually means the reasoning trace ate")
        print("    the token budget — raise MAX_TOKENS, or pick a direct-answer model.")


if __name__ == "__main__":
    main()
