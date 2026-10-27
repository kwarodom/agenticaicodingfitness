# Harness² — One-Page Facilitator Guide (90 minutes)

**Lesson:** "Harness²: agents that improve their own harness" (36-slide deck, speaker notes on every slide). **Course:** Agentic AI Coding Fitness. **Audience:** students who already build agents. **Format:** lecture with pair discussions, lab kick-off in class, full lab as homework.

**Outcomes (slide 3):** define Agent = Model + Harness and the eight components; run one Propose–Probe–Compose step on paper and justify KEEP_BASE; read a results table critically (k, three streams, standard deviations, no ground truth); diagnose the three failure modes; sketch a harness review loop with the safety boundary outside the editable surface.

## Before class (20 minutes)

- Clone the [Harness² repository](https://github.com/google-research/harness2) and run the offline demo once: `uv venv --python 3.12 .venv`, `source .venv/bin/activate`, `uv pip install -e '.[render]'`, `harness2-demo --output runs/demo --k 2` (no model credentials, per the [README](https://github.com/google-research/harness2)). Keep `runs/demo` open to show real edit rows and an EXPERIENCE.md entry.
- A live model-backed run needs Node.js 20+, the Google Cloud CLI, benchmark dependencies and API budget; skip it otherwise. The demo alone covers lab steps 1, 3 and 4.
- Open the PPTX in PowerPoint on the classroom machine; print the quiz key below.

## Run sheet

| Clock | Part (slides) | Do | Say or ask |
|---|---|---|---|
| 0:00–0:10 | 1 Why harnesses (1–7) | Define harness; eight components by how they reach the model. | Classify what students already build: system prompt = C1, retry rule = C2/C8, Python helper = C7, specialist worker = C5. A text artifact is reviewable and reversible; a fine-tune is not. |
| 0:10–0:30 | 2 The method (8–13) | Pipeline figure; six moves; two scopes; evidence not scores; library; budget k. | Verdicts: helped, hurt, no observed effect, confounded. KEEP_BASE is a real answer. "Evidence is data, not instructions." At k = 1 there is no selection. |
| 0:30–0:45 | 3 Evidence (14–22) | WorkBuddy chart, JobBench/LAB table; ablations; tool use; cost; baselines. | SDs are 1–3 points: believe differences under ~3 only if all three streams agree. Skills and scripts carry the gain (−5.3, −3.7 when removed), prompt text the least (−1.5). Gains cost extra probes. |
| 0:45–1:05 | 4 Cases and failures (23–30) | Six-case digest, four figures, nine regressions, measurement hygiene. | Before revealing component codes ask: invariant or domain scope? Then: which failure would a human reviewer catch? Blind selection yes; a propagated premise no, since every candidate agrees. |
| 1:05–1:15 | 5 An always-on agent (31–33) | Nightly harness review with a human gate; chiller example; pre-registered evals. | Label it: EXAMPLE design written for this course, not deployed or measured. Human signals pick tasks and gate promotion; they never enter the selector. |
| 1:15–1:30 | 6 Lab, quiz, close (34–36) | Show demo output; assign the by-hand step; quiz; next steps. | Lab: pairs, 45 minutes total, finished as homework; deliverable is the v0 tree plus one EXPERIENCE.md entry (or NO_ENTRY) in the team repo. |

**Short on time:** compress slides 20–22 into one sentence ("fewer errors, more useful calls, at a compute cost") and show only cases 1 and 6. **New to harnesses:** spend longer on Parts 1–2 and shorten Part 5.

## Misconceptions to pre-empt

1. "It fine-tunes the model." No: weights are frozen; every edit is a file in the harness tree.
2. "The loop sees scores." No: judging runs after selection and spec.py bans measurement words; after an executor downloaded its own rubric, [EXPERIMENTS.md](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md) advises blocking huggingface.co, hf.co and github.com.
3. "Always editing is progress." Unsupported adaptation "costs context and buys nothing" ([prompts.py](https://raw.githubusercontent.com/google-research/harness2/main/harness2/core/prompts.py)); KEEP_BASE is correct when evidence is weak.
4. "A refined harness transfers to any model." Transfer is uneven (GLM-5 +17.3 on code, Gemini 3.7 Flash +0.3); keep one library per executor and measure.

## Quiz key (slide 35)

Q1 unsupported edits bloat the always-on text and break caches. Q2 repeated executions first (cheapest; separates edit from noise), then probes that challenge shared assumptions. Q3 re-verify against the evidence in front of you; an earlier entry is a hypothesis. Q4 system prompt, guardrails, memory and tool guidance are paid every call; prefer skills, scripts and plugins so the prefix stays cached. Q5 rubric fetching and grader verdicts leaking into the harness; controls: blocked hosts, spec.py measurement-word ban, library.py grader-verdict check.

**Homework (slide 36):** students run `harness2-demo` and read one EXPERIENCE.md end to end before next class; teams draft the v0 harness tree for the capstone agent within a week; optionally, teams sketch a Shadow-mode harness review for one task family of their own agent (design only).

**Sources:** [Harness² paper, arXiv 2610.14224 (CC BY 4.0)](https://arxiv.org/html/2610.14224) · [google-research/harness2 (Apache-2.0; "not an officially supported Google product")](https://github.com/google-research/harness2) · [Anthropic, Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Paper numbers are VERIFIED against the paper; the always-on agent material is EXAMPLE design written for this course.
