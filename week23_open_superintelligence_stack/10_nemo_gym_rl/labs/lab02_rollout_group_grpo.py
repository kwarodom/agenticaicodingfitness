#!/usr/bin/env python3
"""Lab 02 · Sample a rollout GROUP and compute GRPO advantages — on real rewards.

One GRPO step, minus the weight update: sample G completions for ONE prompt at
temperature 1.0, score each with a deterministic verifier, then compute
advantage = reward − group mean. You'll see exactly which rollouts a policy
update would reinforce (↑) and which it would suppress (↓).

Run:  cd <repo root> && .venv/bin/python week23/10_nemo_gym_rl/labs/lab02_rollout_group_grpo.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GROUP_SIZE = 4            # NeMo RL's grpo.num_rollouts_per_prompt (small, to stay <60s)
TIMEOUT_S = 25
MAX_TOKENS = 300
DEADLINE_S = 50           # stop sampling past this wall-clock budget

TASK = {"prompt": "What is 23 * 19? Reply with the number only.", "gold": "437"}


def reward_fn(task: dict, answer: str) -> float:
    txt = re.sub(r"<think>.*?</think>", " ", answer, flags=re.S)
    nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
    return 1.0 if nums and nums[-1] == task["gold"] else 0.0


def _rollout(prompt: str) -> str:
    from openai import BadRequestError, OpenAI, UnprocessableEntityError
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=TIMEOUT_S)
    kw = dict(model=config.MODEL, temperature=1.0, max_tokens=MAX_TOKENS,  # temp 1.0 = diverse group
              messages=[{"role": "user", "content": prompt}])
    try:    # ask thinking models to skip the trace — keeps 4 rollouts inside the budget
        r = client.chat.completions.create(
            extra_body={"reasoning_effort": "none"}, **kw)
    except (BadRequestError, UnprocessableEntityError):
        r = client.chat.completions.create(**kw)   # server rejects the hint → plain
    msg = r.choices[0].message
    return (str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")).strip()


def show_group(rewards: list[float], note: str) -> None:
    mean = sum(rewards) / len(rewards)
    print(f"\n  rewards    = {rewards}   ({note})")
    print(f"  group mean = {mean:.3f}   ← the GRPO baseline (no critic network)")
    for i, r in enumerate(rewards):
        adv = r - mean
        mark = "↑ reinforce" if adv > 0 else ("↓ suppress" if adv < 0 else "— no signal")
        print(f"    rollout {i}: reward {r:.1f}  advantage {adv:+.3f}   {mark}")
    if len(set(rewards)) == 1:
        print("  ◈ all rewards equal → every advantage is 0 → this prompt teaches NOTHING.")
        print("    GRPO needs MIXED groups; task difficulty must sit near the model's edge.")


def main() -> None:
    print("━" * 64)
    print("  ▣ Lab 02 — rollout group → GRPO group-relative advantage")
    print("━" * 64)
    print(f"\n◈ Task: {TASK['prompt']!r}   verifier: last number == {TASK['gold']}")
    print(f"◈ Group size G={GROUP_SIZE} at temperature 1.0 — diversity IS the exploration.")

    if config.MODE != "real":
        print("\n◈ No live endpoint — the real sampling commands, then expected output:")
        print("    ollama pull qwen3.6 && export DGX_CONN=local     # C-path")
        print("    export DGX_CONN=cloud DGX_API_KEY=nvapi-...      # B-path (build.nvidia.com)")
        print("\n  [no endpoint — showing expected output]")
        show_group([1.0, 0.0, 1.0, 0.0], "expected sample, NOT a real run")
        print("\n  ✓ (expected) mixed group → nonzero advantages → a real learning signal.")
        return

    print(f"\n◈ Sampling {GROUP_SIZE} rollouts from {config.MODEL} @ {config.safe_base_url()}")
    print(f"  connection: {config.CONN} · {config.cost_note()}")
    start = time.time()
    rewards: list[float] = []
    for i in range(GROUP_SIZE):
        if time.time() - start > DEADLINE_S:
            print(f"  ⏱ {DEADLINE_S}s budget reached — scoring the {len(rewards)} sampled so far.")
            break
        try:
            ans = _rollout(TASK["prompt"])
            r = reward_fn(TASK, ans)
            tail = re.sub(r"\s+", " ", ans)[-48:]
            print(f"  rollout {i}: …{tail!r}  → {r:.1f} {'✓' if r == 1.0 else '✗'}")
            rewards.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  rollout {i}: request failed ({type(e).__name__}) — scored 0.0")
            rewards.append(0.0)

    if len(rewards) < 2:
        print("\n✗ fewer than 2 rollouts — can't form a group. Check the endpoint and re-run.")
        return
    show_group(rewards, "real verifier rewards")
    print("\n✓ That table IS one GRPO step's learning signal. NeMo RL would now take a")
    print("  policy-gradient step toward the ↑ rollouts — repeat over rounds and the")
    print("  mean reward climbs (the 0.30→0.79 curve in demos/step03_grpo_training.py).")


if __name__ == "__main__":
    main()
