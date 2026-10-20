# ▶ Hands-on Lab 10 — NeMo Gym + NeMo RL: verifiable rewards, rollouts, GRPO

> Part of Week 23 · The Open Superintelligence Stack. Companion explainer app: `.venv/bin/python week23/10_nemo_gym_rl/tutorial_server.py` → http://127.0.0.1:8109. This lab is the DO-IT side: you type the commands, you see the output.

**What you'll actually do**
- Write a verifiable-reward **environment** (task + deterministic `reward_fn`) and use it to score REAL model rollouts — the exact 1.0/0.0 signal GRPO trains on
- Sample a **GROUP** of rollouts at temperature 1.0 and compute GRPO group-relative advantages from live rewards — see which rollouts a policy update would reinforce (↑) and suppress (↓)
- Measure a base model's **pass-rate across three environments** and apply a promotion gate — the eval that feeds App 11's flywheel
- (Path A) clone `NVIDIA-NeMo/RL` on the Spark and launch a real GRPO run on a small policy
- Learn the fit-math: which policies train on 1 Spark, which need 2 (`cluster.tp=2`), which stay SIM

**Time** ~45 min · **Difficulty** intermediate–advanced · **Cost** $0 on Spark/local; cloud path is usage-billed.

## 0 · Pick your path
| Path | You have | You'll use |
|---|---|---|
| A 🖥️ | a DGX Spark (or any Linux GPU box) | real commands over SSH |
| B ☁️ | an nvapi- key from build.nvidia.com | hosted NIM endpoints, same API |
| C 💻 | just this laptop | local Ollama (localhost:11434) or the app's SIM mode |
(every numbered step below marks which paths it applies to)

## 1 · Launch the explainer app and read Ch 2 (A/B/C)

Get the concept map in your head first: environment = task + verifier; NeMo Gym supplies environments, NeMo RL runs GRPO.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/10_nemo_gym_rl/tutorial_server.py
```

Expected output:
```
  ▣  NeMo Gym + NeMo RL — agents that learn from outcomes
      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.
      open  →  http://127.0.0.1:8109
```

Open http://127.0.0.1:8109, click **Run** on Ch 2 (what is verifiable-reward RL?) and Ch 3 (define an environment). The banner says REAL if an endpoint is already up — either is fine for this step.

✓ Checkpoint: you should now see the GRPO loop diagram (rollout → reward → advantage → update) in the app's output.

## 2 · Point the labs at a live endpoint (A/B/C)

The labs reuse this folder's `config.py`, so one set of env vars aims all three at your endpoint. Verify the endpoint answers BEFORE running anything.

**A — DGX Spark** (Ollama serving on the Spark, reached over your Tailscale/LAN):
```bash
export DGX_CONN=tunnel
export DGX_TUNNEL_URL=http://<your-spark>.<your-tailnet>.ts.net:11434/v1
curl -s $DGX_TUNNEL_URL/models | head -c 300
```

**B — build.nvidia.com** (get a free `nvapi-` key from any model page → "Get API Key"):
```bash
export DGX_CONN=cloud
export DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1
export DGX_API_KEY=nvapi-...
export DGX_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning   # verified live ID
curl -s https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $DGX_API_KEY" | head -c 300
```
List live IDs via `GET /v1/models` and never hardcode — the Nano ID above is verified; exact Super/Ultra suffixes vary, so verify with the `/models` call first (some personal orgs also hit 403 "missing public API endpoints permission" on Super).

**C — this laptop** (local Ollama):
```bash
ollama pull qwen3.6          # any chat model works; qwen3.6 is the course workhorse
export DGX_CONN=local
curl -s http://localhost:11434/v1/models | head -c 300
```

Expected output (any path):
```
{"object":"list","data":[{"id":"qwen3.6:35b-a3b-q8_0","object":"model",...
```

No endpoint at all? Skip the exports — every lab detects SIM and prints the real commands plus a clearly-labeled expected-output sample instead of crashing.

✓ Checkpoint: you should now have a JSON model list from `/v1/models` (or a conscious decision to run SIM).

## 3 · Build an environment and score real rollouts (A/B/C)

An environment is ~15 lines: a task sampler and a `reward_fn` that returns 1.0 on pass else 0.0. Lab 01 first proves the verifier is deterministic on canned answers, then scores LIVE rollouts from your model.

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python week23/10_nemo_gym_rl/labs/lab01_build_verifier.py
```

Expected output (abbreviated):
```
  ▣ Lab 01 — build a verifiable-reward environment
◈ Step 1 — the verifier is DETERMINISTIC (canned answers, no model):
  reward_fn(gold='888', answer='888'          ) → 1.0  PASS ✓
  reward_fn(gold='888', answer="It's roughly 890.") → 0.0  fail ✗
◈ Step 2 — REAL rollouts: qwen3.6:35b-a3b-q8_0 @ http://localhost:11434/v1
  » What is 37 * 24? Reply with the number only.
  · reward_fn → 1.0  PASS ✓
✓ 2/2 rollouts verified. That scalar ... is the entire training signal GRPO needs.
```

✓ Checkpoint: you should now have real model output scored by a verifier YOU can read — no reward model, no human label.

## 4 · Sample a rollout GROUP and compute GRPO advantages (A/B/C)

This is one GRPO step minus the weight update: G=4 completions of ONE prompt at temperature 1.0, each scored, then advantage = reward − group mean.

```bash
.venv/bin/python week23/10_nemo_gym_rl/labs/lab02_rollout_group_grpo.py
```

Expected output (abbreviated — your rewards will differ, that's the point):
```
  ▣ Lab 02 — rollout group → GRPO group-relative advantage
  rollout 0: …'437'  → 1.0 ✓
  rollout 1: …'23 × 19 = 437 ... 437'  → 1.0 ✓
  rollout 2: …'447'  → 0.0 ✗
  rewards    = [1.0, 1.0, 0.0, 1.0]
  group mean = 0.750   ← the GRPO baseline (no critic network)
    rollout 0: reward 1.0  advantage +0.250   ↑ reinforce
    rollout 2: reward 0.0  advantage -0.750   ↓ suppress
```

If every rollout passes (or fails), all advantages are 0 and the lab tells you so — an all-pass group teaches nothing, which is why GRPO curricula keep tasks near the model's edge of ability.

✓ Checkpoint: you should now see either a mixed group (at least one ↑) or the explicit "teaches NOTHING" message — and understand that GRPO's "baseline" is just the group mean: no value network, no separate critic. Strong models often ace this easy task (all-1.0 group); re-run or try the harder-task modification below to see a real ↑/↓ split.

## 5 · Run real GRPO on the Spark (A; C runs SIM instead)

Only Path A hardware can actually update weights. Per the DGX Spark runbook: GRPO on a 1–8B policy fits **1 Spark** (training wants ~4–6× weight memory for optimizer/grads/rollouts); Nemotron Nano 30B LoRA is borderline; anything bigger → **2 Sparks** (`cluster.tp=2`) or SIM.

**A — on the Spark** (via SSH or the app's 🖥️ DGX console):
```bash
git clone https://github.com/NVIDIA-NeMo/RL nemo-rl && cd nemo-rl
uv sync            # ARM64 note: if torch wheels fight aarch64, run inside nvcr.io/nvidia/pytorch:25.11-py3
uv run python examples/run_grpo_math.py policy.model_name=Qwen/Qwen2.5-1.5B cluster.gpus_per_node=1
```

Expected output (abbreviated — first minutes of a run):
```
[train] step 1/...  rollouts=... mean_reward=0.31 ...
[train] step 2/...  mean_reward=0.34 ...
```

Two things the runbook flags: **NeMo RL has no serving port** — it's a batch job, so "is it REAL?" is answered by `ssh <spark> nvidia-smi` showing a training process, not by an HTTP probe. And the environments come from NeMo Gym (`git clone https://github.com/NVIDIA-NeMo/Gym`), which is what the `env=...` flags select. The `uv run nemo_rl.grpo policy.model=...` block in `demos/step03_grpo_training.py` shows the SHAPE of the config; treat the runbook command above as the one to actually type.

**C — no GPU:** run the SIM training curve instead — same loop, honest about being simulated:
```bash
.venv/bin/python week23/10_nemo_gym_rl/demos/step03_grpo_training.py
```

Expected output:
```
  round  0  reward 0.30  ██████████··...
  round 19  reward 0.79  █████████████████████████··...
  base 0.30 → trained 0.79  (mean verifier reward)
```

✓ Checkpoint: you should now have either a real NeMo RL process visible in `nvidia-smi` (A) or the simulated 0.30→0.79 reward curve (C).

## 6 · The eval gate: multi-env pass-rate (A/B/C)

Training on ONE verifier overfits to it. Lab 03 measures pass-rate across three environments with three different verifiers (math exact-match, JSON tool-call check, output constraint) and applies a promotion gate — the measurement that decides base vs trained.

```bash
.venv/bin/python week23/10_nemo_gym_rl/labs/lab03_multienv_passrate.py
```

Expected output (abbreviated):
```
  ▣ Lab 03 — multi-env pass-rate: the promotion gate
  [math_verify ] 'What is 41 * 12? Reply with the number only.'     → 1.0 ✓
  [tool_call   ] 'Book the 2pm meeting room. Reply with ONLY thi'   → 1.0 ✓
  [constraint  ] 'What color is a stop sign? Answer with exactly'   → 1.0 ✓
  Environment    pass-rate
  math_verify         100%  ████████████████
  tool_call            50%  ████████········
  constraint          100%  ████████████████
  OVERALL              83%  gate ≥60% → PROMOTE ✓
```

Tasks run ROUND-ROBIN (task 1 of every env, then task 2, …), so even a slow or cold endpoint scores every verifier at least once before the 50s budget cuts depth — you may see `⏱ 50s budget reached` on a first run while the model loads; that's fine.

✓ Checkpoint: you should now have a per-environment pass-rate table for YOUR model — the baseline a GRPO-trained policy would have to beat.

## Labs (run these)

**labs/lab01_build_verifier.py** — builds a two-task environment with a deterministic `reward_fn`, proves determinism on canned answers, then scores 2 real rollouts from your endpoint.
Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/10_nemo_gym_rl/labs/lab01_build_verifier.py`
Look for: the same canned answer always getting the same reward, then live rollouts collapsing to a bare 1.0/0.0.
Modify it: change one task's `gold` to a wrong value and re-run — watch a "correct" model answer score 0.0. A verifier bug IS a reward-hacking surface; this is why verifiers get code review.

**labs/lab02_rollout_group_grpo.py** — samples G=4 rollouts of one prompt at temperature 1.0, scores each, prints reward − mean advantages with ↑/↓ marks.
Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/10_nemo_gym_rl/labs/lab02_rollout_group_grpo.py`
Look for: whether your group is MIXED (nonzero advantages = a learning signal) or uniform (all-zero = nothing to learn).
Modify it: swap `TASK` for a much harder multiplication (e.g. 4-digit × 4-digit) and re-run — an all-fail group shows why task difficulty must track model ability.

**labs/lab03_multienv_passrate.py** — a mini eval harness: 3 environments × 2 tasks, per-env pass-rate, overall score, PROMOTE/HOLD gate at 60%.
Run: `cd /Users/altodev/Desktop/agenticaicodingfitness && .venv/bin/python week23/10_nemo_gym_rl/labs/lab03_multienv_passrate.py`
Look for: which environment your model is weakest in — that's where GRPO training data should come from.
Modify it: raise `GATE` to 0.9 and re-run — most base models now HOLD, which is exactly the honest state before post-training.

## Try it yourself

**1. Write a unit-test-style verifier.** In lab01, add a task asking the model for a Python one-liner (e.g. "reply with only a Python expression for the area of a circle of radius r") and a `reward_fn` that checks it with `eval` against a known input.
<details><summary>Solution</summary>

```python
TASKS.append({"prompt": "Reply with ONLY a Python expression for the area of a "
              "circle with radius r (use 3.14159 for pi).", "gold": "area"})

def reward_area(task, answer):
    expr = _strip_think(answer).strip().strip("`").splitlines()[-1]
    try:
        return 1.0 if abs(eval(expr, {"r": 2.0}) - 12.56636) < 1e-3 else 0.0
    except Exception:
        return 0.0
```
Dispatch on `task["gold"] == "area"` inside the loop. Code verifiers are "run it and check" — the same idea as SWE-bench-style unit-test rewards.
</details>

**2. Grow the group to G=8 and plot the advantage spread.** GRPO's real configs use `grpo.num_rollouts_per_prompt=8`.
<details><summary>Solution</summary>

Set `GROUP_SIZE = 8` in lab02 (and bump `DEADLINE_S` if your endpoint is slow). With more samples the group mean is a steadier baseline, so single lucky rollouts get smaller advantages — this variance reduction is the entire reason GRPO uses groups instead of a single sample + critic.
</details>

**3. Add a fourth environment to the gate.** Give lab03 a `json_schema`-style env: the reply must be JSON with keys `city` and `country`.
<details><summary>Solution</summary>

```python
def v_schema(gold, ans):
    blocks = re.findall(r"\{[^{}]*\}", _clean(ans))
    try:
        d = json.loads(blocks[-1]) if blocks else {}
    except Exception:
        return 0.0
    return 1.0 if {"city", "country"} <= set(d) else 0.0

ENVS.append(("json_schema", v_schema, [
    ('Where is the Eiffel Tower? Reply with ONLY JSON: {"city": "...", "country": "..."}', "-")]))
```
Any business check you can automate this way becomes a training signal — ticket resolved, invoice matched, SQL result equals golden.
</details>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 404 model-not-found | wrong model ID for this endpoint | `curl $BASE/models` and use an ID from the list; set `DGX_MODEL` to it |
| 401 unauthorized | missing/stale key | B-path: `export DGX_API_KEY=nvapi-...`; tunnels may need basic-auth creds in the URL |
| `Connection error` from every lab | base URL missing `/v1` | Ollama is `:11434/v1`, NIM/vLLM `:8000/v1` — `config.py` auto-appends `/v1` only via the app's 🔌 Connection panel |
| labs all print `[no endpoint — showing expected output]` | no endpoint reachable, SIM fallback | that's by design; do Step 2, then re-run |
| every reward 0.0 with a thinking model | reasoning trace ate the token budget before the final answer | the labs already send `reasoning_effort: none`; if your server ignores that hint, raise `MAX_TOKENS` or `export DGX_MODEL=` a direct-answer model (e.g. gemma3/llama3) |
| all advantages 0.000 in lab02 | uniform group (all pass or all fail) | not a bug — pick a harder/easier task so the group mixes |
| `uv sync` fails on torch wheels (Spark) | aarch64/ARM64 wheel gap | run inside `nvcr.io/nvidia/pytorch:25.11-py3` (the runbook's safer path); never pull x86 images |
| GRPO run "not detected" by probes | NeMo RL has no serving port — it's a batch job | check `ssh <spark> nvidia-smi` for the training process, not an HTTP endpoint |
| port 8000 already in use on the Spark | vLLM/NIM/TRT/Dynamo all contend for 8000 | only one serves at a time; `curl :8000/v1/models` to see WHICH answered before killing anything |
| 403 "missing public API endpoints permission" (B) | some personal orgs can't call certain hosted models | pick another ID from `GET /v1/models` (Nano and `meta/llama-3.1-8b-instruct` are safe fallbacks) |

## Next
→ ../11_data_flywheel/TUTORIAL.md (NeMo Data Flywheel — logs → Curator → Customizer → Evaluator → distill → promote) — lab03's promotion gate run BY HAND is exactly the loop the flywheel automates from production logs.
