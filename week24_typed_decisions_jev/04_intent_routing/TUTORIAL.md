# ▶ Jev Lab 04 — Intent routing for Alto Copilot

> Part of Week 24 · Typed AI decisions with Jev. Your first real workflow: classify each Copilot request, decide which specialist (or team of specialists) should answer it, and pick an approved prompt — without Jev ever writing the answer or granting a permission.

**In plain words:** people type all kinds of requests into a building copilot: *"Why is AHU-3 not cooling?"*, *"Calculate last month's savings"*, *"Write me a Python function"*. Before any AI writes an answer, something has to decide **who should answer**: the HVAC expert, the energy-savings analyst, a coding assistant, or a person, because the request is too vague. Jev makes that decision in under a second. Plain code then applies the rules and picks the right pre-approved instructions (the *prompt*) for that expert.

**New words:**

| Word | Meaning |
|---|---|
| intent / route | which kind of request this is, and so which handler gets it |
| MAS | *multi-agent system*: several AI specialists working on one request together |
| confidence gate | a rule like "only route automatically when Jev is clearly sure; otherwise ask a person" |
| compute tier | a fast, cheap model for simple requests vs a slower reasoning model for hard ones |
| prompt bundle | the approved system instructions for one specialist, stored in code, never invented at runtime |

**What you'll actually do**
- Route nine English/Thai requests with one 7-question call each, using the reference `jev_lab.py`.
- Read a deterministic policy: confidence gate → route, `needs_mas` → dispatch, complexity → compute tier.
- Turn a route into an approved **prompt bundle** from a registry in code.
- Measure English vs Thai on paired requests instead of assuming.
- Add a brand-new route (`iot_connectivity`) and prove it works.

**Time** ~40 min · **Difficulty** intermediate · **Cost** ≈ $0.0008 live · $0 dry

## 0 · The pattern: a typed classifier in front of specialists

TypeSafe calls this **intent routing**: a fast typed classifier decides *where* a request goes; deterministic handlers, specialist LLMs or humans do the work.

```text
user request (EN / TH)
  → code: authenticate user, resolve tenant/project permissions, strip secrets
  → Jev: topic + needs_retrieval + needs_live_data + needs_mas + expertise + complexity   (ONE call)
  → code: confidence gate + policy  → route · dispatch · compute tier
  → registry: approved system prompt + allowed tools for that route
  → specialist LLM / multi-agent dispatcher / human — each re-checks authorization
```

Two rules make this safe:

1. **Jev proposes, code disposes.** Every output of this lab has `execute: false` and `authorization: must_be_checked_separately`.
2. **A label never grants a permission.** Being routed to `energy_mv` does not let a user read another tenant's meters.

The topic labels below are the tutorial's proposal — not necessarily the production Alto Copilot schema:

| Label | Means |
|---|---|
| `hvac` | HVAC, chiller, AHU, VRF, cooling performance or diagnosis |
| `energy_mv` | Energy baselines, savings measurement or verification (M&V) |
| `facility_ops` | Maintenance workflow, site operations, work orders |
| `sustainability` | Carbon, emissions, sustainability evidence or reporting |
| `coding` | Software implementation, debugging or API integration |
| `general` | Ordinary non-building questions, writing, conversation |
| `unknown` | Too vague, unsupported, or no identifiable request |

✓ Checkpoint: you can say which component decides the route (code) and which supplies the judgments (Jev).

## 1 · Route nine requests

The reference implementation from the source guide lives in `week24/jev_lab/jev_lab.py` (standard library only). Lab 01 imports its `QUESTIONS["intent"]` (one choice + five nouls + one score), its `prepare()` (sends only the `message` field — never the teaching label) and its `policy()`.

```bash
.venv/bin/python week24/04_intent_routing/labs/lab01_route_requests.py
```

**Expected output**

```
▣ STEP 1 · look at ONE request in full — 7 questions, 1 call
» intent                 choice  → hvac   (confidence 1.00)
» needs_retrieval        noul    0.91  ██████████████████████░░  likely YES
» needs_live_data        noul    0.92  ██████████████████████░░  likely YES
» needs_mas              noul    0.04  █░░░░░░░░░░░░░░░░░░░░░░░  likely NO
» hvac_expertise         noul    0.80  ███████████████████░░░░░  likely YES
» mv_expertise           noul    0.42  ██████████░░░░░░░░░░░░░░  uncertain
» complexity             score   1.09 on 0…2   (confidence 0.46)
◆ jev-1.13.0 · LIVE · 858 input tok · $0.000036 · 450 ms

▣ STEP 2 · route all nine requests (9 calls)
│ id  message                                           teaching label  jev             conf  route              dispatch         tier
│ i1  Why is AHU-3 not cooling the hotel lobby? Check…  hvac            hvac            1.00  hvac               single_agent     reasoning_candidate
│ i2  ช่วยตรวจสอบว่า AHU-3 ไม่เย็นเพราะอะไร             hvac            hvac            1.00  hvac               single_agent     reasoning_candidate
│ i3  Compare an HVAC engineer's diagnosis with an M&…  hvac            energy_mv       0.83  energy_mv          mas_candidate    reasoning_candidate
│ i4  Write a Python function to validate a JSON payl…  coding          coding          1.00  coding             single_agent     reasoning_candidate
│ i5  Tell me a short story about a cat.                general         general         1.00  general            single_agent     fast_candidate
│ …
│ i9  Please do that thing.                             unknown         unknown         0.96  clarify_or_review  review_dispatch  fast_candidate
◆ Jev agreed with the teaching label on 8/9 rows — read the disagreements, don't just count them
```

The same run is available as a raw JSON-lines CLI — the exact script from the source guide:

```bash
.venv/bin/python week24/jev_lab/jev_lab.py run intent --live --limit 9
```

### The i3 story — when the "gold" label is debatable

Request i3 reads: *"Compare an HVAC engineer's diagnosis with an M&V analyst's savings assessment and resolve their disagreement."* The teaching file labels it `hvac`. Jev said **`energy_mv`** (probability 0.87, confidence 0.83), with `hvac` at 0.12, `needs_mas` = **0.98**, and *both* `hvac_expertise` (0.82) and `mv_expertise` (0.92) high. (Your numbers may differ by a point or two between live calls.)

Who is right? Arguably neither topic alone — it is a two-specialist task, and the policy handled it well: `dispatch = mas_candidate`. The source guide warns about exactly this: *even gold labels can require adjudication*. When an evaluation says "wrong", look before you "fix" the model.

✓ Checkpoint: you ran lab 01 and can explain why i3 went to `mas_candidate`.

## 2 · Read the policy — plain, testable code

Everything after the Jev call is deterministic Python in `jev_lab.py`. No model can change it.

| Decision | Rule (illustrative, uncalibrated) | Why |
|---|---|---|
| **route** | the topic, only if `accepted()`: confidence ≥ .75, peak ≥ .80, margin ≥ .20, and not `unknown` — else `clarify_or_review` | Uncertain requests must not guess a specialist |
| **dispatch** | `needs_mas` ≤ .20 → `single_agent` · ≥ .80 → `mas_candidate` · otherwise `review_dispatch` | 0.5 is uncertainty, not "half a team" |
| **specialists** | each expertise noul ≥ .80 | Suggestions for the multi-agent dispatcher |
| **compute tier** | complexity ≥ 1.5 **or** its confidence < .75 → `reasoning_candidate`, else `fast_candidate` | When unsure how hard it is, pay for the bigger model |

Notice in the table above that most requests got `reasoning_candidate` (only i5 and i9 got `fast_candidate`) — not because they are hard, but because the complexity **score's confidence** was below .75 (i1: 0.46). That is a policy choice you might revisit: it spends money to be safe. This is exactly the kind of threshold you tune on real traffic, *without re-asking the model*.

Try the full router on a request of your own:

```jev
{
  "state": {"message": "Compare an HVAC engineer's diagnosis with an M&V analyst's savings assessment and resolve their disagreement."},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the primary topic of `message`.",
      "criteria": {
        "hvac": "HVAC, chiller, AHU, VRF, cooling performance or diagnosis.",
        "energy_mv": "Energy baselines, savings measurement or verification.",
        "facility_ops": "Maintenance workflow, site operations, work orders.",
        "sustainability": "Carbon, emissions, sustainability evidence or reporting.",
        "coding": "Software implementation, debugging or API integration.",
        "general": "Ordinary non-building questions, writing or conversation.",
        "unknown": "Too vague, unsupported topic, or no identifiable request."
      }
    },
    "needs_mas": {"type": "noul", "instructions": "Does `message` explicitly require combining distinct expert analyses, comparing independent judgments, or resolving their disagreement? Merely having multiple steps is not enough."},
    "needs_live_data": {"type": "noul", "instructions": "Does answering `message` require current or historical operational records from connected systems?"},
    "complexity": {"type": "score", "instructions": "How much reasoning does the request require?", "criteria": ["Simple answer or lookup.", "One specialist with a few steps.", "Multiple specialist analyses or substantial investigation."]}
  }
}
```

Experiments:

1. Replace the message with `"Schedule preventive maintenance for the chiller next week."` In our run this went to **`hvac` (0.73)**, not `facility_ops` — the word *chiller* pulls it. Is that wrong? Decide, then write the boundary into the `facility_ops` description ("maintenance *scheduling* for any equipment, including chillers").
2. Try `"Please do that thing."` — `unknown` should win, so the policy asks the user to clarify.

✓ Checkpoint: you can point to the line of policy that sends a vague request to `clarify_or_review`.

## 3 · From route to an approved prompt bundle

Jev picks the **key**; your registry supplies the **prompt**. The registry is code you review like any other code — the classifier can never inject a new prompt or tool.

```bash
.venv/bin/python week24/04_intent_routing/labs/lab02_prompt_bundle.py
```

**Expected output**

```
▣ STEP 1 · “Why is AHU-3 not cooling the hotel lobby? Check yesterday's trends.”
» intent hvac (confidence 1.00) → route hvac
{
  "agent_candidate": "hvac_expert",
  "system_prompt_candidate": "Separate observed symptoms from hypotheses. Use authorized building data. State missing evidence. Propose inspections only; do not change controls.",
  "compute_tier_candidate": "reasoning_candidate",
  "dispatch_candidate": "single_agent",
  "specialists_suggested": ["hvac_expert"],
  "tools": "resolve separately from the authorized registry",
  "execute": false
}
…
▣ STEP 3 · “Please do that thing.”
» intent unknown (confidence 0.95) → route clarify_or_review
→ no prompt selected automatically — ask the user to clarify, or send to review
```

(The vague request got confidence 0.96 in lab 01 and 0.95 here: two live calls, the same request. Jev is very consistent but not bit-for-bit deterministic — another reason thresholds should not sit on a knife edge.)

The proposed Alto Copilot integration sequence, from the source guide:

1. Authenticate the user and resolve tenant/project permissions in server code.
2. Strip irrelevant context and secrets before sending classifier input.
3. Obtain topic, retrieval, live-data, complexity and MAS signals.
4. Apply deterministic policy and a tested uncertainty gate.
5. Resolve logical model tiers against the allowed model registry, budget and data-residency constraints.
6. Dispatch a single specialist, or send a MAS candidate to a separate multi-agent coordinator that has its own permissions and approvals.
7. Re-check authorization at every retrieval and tool call; a classification label never grants a permission.
8. Log the route, model version, rubric version, latency and corrected outcome.

✓ Checkpoint: you can explain why the prompt text lives in a registry and not in Jev's answer.

## 4 · Thai vs English — measure, don't assume

English is Jev's primary training language. Alto Copilot users write Thai, English and a mix. Lab 03 sends the same request in both languages and compares the topic, the confidence, and the **total variation distance** (TVD) between the two probability distributions (0 = identical, 1 = no overlap).

```bash
.venv/bin/python week24/04_intent_routing/labs/lab03_thai_vs_english.py
```

**Expected output**

```
│ Thai / mixed message                              EN topic        TH topic        confidence EN→TH  TVD   live-data EN→TH
│ ทำไม AHU-3 ไม่เย็นที่ล็อบบี้โรงแรม                hvac            hvac            1.00 → 1.00       0.00  0.84 → 0.75
│ คำนวณผลการประหยัดไฟฟ้าที่ผ่านการตรวจสอบ เทียบกั…  energy_mv       energy_mv       1.00 → 1.00       0.00  0.87 → 0.85
│ นัดบำรุงรักษาเชิงป้องกันชิลเลอร์สัปดาห์หน้า       hvac            hvac            0.73 → 0.84       0.09  0.52 → 0.43
│ สรุปการปล่อยคาร์บอนของโรงแรมสำหรับรายงาน ESG      sustainability  sustainability  1.00 → 1.00       0.00  0.91 → 0.85
│ ช่วย check chiller plant kW/ton ของเมื่อวานหน่อย  hvac            hvac            1.00 → 0.99       0.00  0.94 → 0.93
◆ same topic in 5/5 pairs
```

Encouraging: topics matched 5/5, and even the mixed Thai/English message was stable. But notice the **nouls drift a little** in Thai (`needs_live_data` 0.84 → 0.75 for the AHU question). With a threshold at 0.80, that drift would flip a decision. Five pairs prove nothing statistically — they tell you *what to measure* on hundreds of real, de-identified messages.

```jev
{
  "state": {"message": "ช่วยตรวจสอบว่า AHU-3 ไม่เย็นเพราะอะไร ดูเทรนด์เมื่อวานด้วย"},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the primary topic of `message`.",
      "criteria": {
        "hvac": "HVAC, chiller, AHU, VRF, cooling performance or diagnosis.",
        "energy_mv": "Energy baselines, savings measurement or verification.",
        "facility_ops": "Maintenance workflow, site operations, work orders.",
        "unknown": "Too vague, unsupported topic, or no identifiable request."
      }
    },
    "needs_live_data": {"type": "noul", "instructions": "Does answering `message` require current or historical operational records from connected systems?"}
  }
}
```

Experiments: remove "ดูเทรนด์เมื่อวานด้วย" ("check yesterday's trends too"). Does `needs_live_data` drop? Then write the English version and compare.

✓ Checkpoint: you can name the metric that tells you two distributions differ (TVD), and why a small drift matters near a threshold.

## 5 · Shadow mode — how this reaches production

Never switch routing on because a notebook looked good. The source guide's recommended first deployment is **shadow mode**: run the Jev router beside the existing router, log both proposals, change nothing live, and compare per class and per language.

| Phase | What changes for users | Exit condition |
|---|---|---|
| Lab (this module) | nothing | everyone can explain fields, uncertainty and permissions |
| Shadow | nothing — both routers log | stable per-class EN/TH measurements; all errors review-safe |
| Assisted | suggestions shown to operators | human override data and audit trail |
| Router integration | Jev runs beside the existing router and the multi-agent coordinator | held-out quality, cost and latency targets met |

✓ Checkpoint: you can explain why shadow mode changes nothing for users yet still produces the evidence you need.

## Labs — run them here

**labs/lab01_route_requests.py** — Nine EN/TH requests through the reference 7-question intent schema and policy.

**labs/lab02_prompt_bundle.py** — Route → approved system prompt from a registry in code; vague requests fall through to clarify.

**labs/lab03_thai_vs_english.py** — Paired English/Thai/mixed requests: topic agreement, confidence and distribution distance.

## Try it yourself

**Exercise 04 — add a route.** Alto Copilot gets many "gateway offline / sensor not reporting" questions. Open `week24/04_intent_routing/exercises/ex04_add_a_route.py`:

1. **Part A (offline, free):** implement `accepted(a)` — the confidence gate — and pass five fixed fixtures.
2. **Part B:** write the description for the new `iot_connectivity` option (what belongs, and what does **not**), plus three test messages of your own — try one in Thai.
3. **Part C (automatic):** the checker routes your messages, two hidden gateway messages, and two controls that must stay `hvac` and `coding`.

```bash
.venv/bin/python week24/04_intent_routing/exercises/ex04_add_a_route.py
```

**Expected output**

```
▣ STEP A · your accepted() gate against 5 fixed fixtures (offline, $0)
✓ case 1: accepted → True
…
✓ case 5: accepted → True
▣ STEP C · route 7 messages (one call each)
│ kind     message                                           want              jev               conf  accepted?
│ yours    The Modbus gateway on the chiller plant keeps d…  iot_connectivity  iot_connectivity  0.98  yes        ✓
│ yours    เกตเวย์ที่โรงแรมสาขาภูเก็ตออฟไลน์ ข้อมูลไม่เข้า…  iot_connectivity  iot_connectivity  1.00  yes        ✓
│ hidden   The LoRaWAN gateway at Siam Tower has been offl…  iot_connectivity  iot_connectivity  1.00  yes        ✓
│ control  Why is AHU-3 not cooling the hotel lobby?         hvac              hvac              1.00  yes        ✓
✓ hidden gateway messages route to iot_connectivity, controls stay put
```

<details><summary>Hint — a description that does not steal HVAC problems</summary>

"Gateways, IoT sensors or meters that are offline, not reporting, or sending stale or missing data. **Not** for equipment that is reporting normally but performing badly (that is hvac)." The second sentence is what keeps "AHU-3 is not cooling" in `hvac`.

</details>

<details><summary>Hint — the margin rule</summary>

If the probabilities sum to 1 and the peak is ≥ 0.80, the second option is at most 0.20, so the margin is at least 0.60. The margin rule only starts to matter if you *lower* the peak threshold. Keep it anyway — it documents intent and protects you when thresholds change.

</details>

✓ Checkpoint: all five gate cases pass and the hidden + control rows are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: jev_lab` | Run from the repo root; the labs add `week24/jev_lab` to `sys.path` themselves |
| Everything routes to `clarify_or_review` | You are in DRY mode with a changed request (placeholder answers have confidence 0) — switch to ⚡ Live |
| `jev_lab.py run … --live` says "Set TYPESAFE_API_KEY" | Source the key: `set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a` (the built-in terminal already has it) |
| `Output file already exists` | The reference CLI refuses to overwrite result files — pick a new `--out` name |
| Thai results look weaker on your own data | Measure per class, revise the rubric, and compare another classifier (Lab 10 covers Laya) |

## Next

Continue to [Lab 05 — email triage without giving AI control of the inbox](../05_email_triage/TUTORIAL.md).
