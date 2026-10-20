# ▶ Jev Lab 06 — HR evidence checklists, responsibly

> Part of Week 24 · Typed AI decisions with Jev. Recruiting is where careless AI does real harm. This lab uses Jev for one narrow, auditable job: **finding explicit evidence in professional text** for a recruiter to verify — never ranking, scoring or rejecting people.

**What you'll actually do**
- Learn the ground rules first: where the data may come from, and what the model may never decide.
- Build an evidence checklist: for each job criterion, `evidenced` / `not_stated` / `unclear`.
- Go one step further: have Jev **select the exact sentence** that is the evidence, and quote it verbatim from code.
- Add your own criterion and a checklist policy that cannot rank or reject.

**Time** ~35 min · **Difficulty** intermediate · **Cost** ≈ $0.0003 live · $0 dry

## 0 · Ground rules before any code

Read these first. They are design rules for this lab, not model features — and they matter more than the code.

**Where the text comes from**

- ✓ Candidates submit their resume or professional summary **directly** to you.
- ✓ Records your organisation already lawfully holds in its ATS, under its processing terms.
- ✓ LinkedIn-origin data **only** through a LinkedIn-approved integration, for operations your agreement allows.
- ✗ **No scraping.** LinkedIn prohibits scraping tools, bots and browser extensions. Jev does not browse or scrape — and this course will not show you how. "Publicly visible" is not the same as "permitted to process".

**What the text may contain**

Only minimised, job-related professional text. Remove names, contact details, photos, age, gender, nationality and anything unrelated to the job **before** it reaches any model. The sample data in `data/profiles.csv` is fictional and already minimised — it has only a `candidate_id` and `professional_text`.

**What the model may decide**

| Allowed (a proposal a recruiter verifies) | Never |
|---|---|
| "Does this text explicitly describe Python implementation work?" | "Is this a good candidate?" |
| "Which sentence is the evidence?" | A total score, a ranking, a shortlist |
| "Is this criterion unclear?" | Automatic rejection |
| | Personality, "culture fit", or anything not in the job criteria |

> ⚠ **`not_stated` means "this text does not show it" — never "this person cannot do it".** People describe their work in very different ways. The right response to `not_stated` is to *ask the candidate*.

✓ Checkpoint: you can name two lawful sources of professional text and explain why `not_stated` must never become a rejection.

## 1 · Design narrow evidence questions

The example role is an **AI/IoT integration engineer**. Instead of one vague "is this person qualified?", we ask three literal, job-related questions — each with the same three answers:

| Label | Meaning |
|---|---|
| `evidenced` | A specific project, implementation or work responsibility is described |
| `not_stated` | No such work is described in the text |
| `unclear` | The technology or domain is named, but hands-on work is ambiguous |

Try the literal-reading effect yourself. The text below mentions Python twice — but only as an interest and a course:

```jev
{
  "state": {"professional_text": "Worked six years as a facility technician on chiller plants and AHUs. Completed an online Python course last year."},
  "questions": {
    "python_evidence": {
      "type": "choice",
      "instructions": "Does `professional_text` provide explicit evidence of hands-on Python implementation work? A course or an interest is not hands-on work.",
      "criteria": {
        "evidenced": "A specific Python project, implementation or work responsibility is described.",
        "not_stated": "No concrete Python implementation work is described.",
        "unclear": "Python is named but practical work is ambiguous."
      }
    },
    "building_evidence": {
      "type": "choice",
      "instructions": "Does `professional_text` state work with HVAC, BMS or building energy systems?",
      "criteria": {
        "evidenced": "Specific relevant building-system work is described.",
        "not_stated": "No such work is described.",
        "unclear": "A possible connection is mentioned without concrete work."
      }
    }
  }
}
```

Now experiment:

1. Delete `"A course or an interest is not hands-on work."` from the instruction. Does `python_evidence` shift toward `unclear` or `evidenced`? (When we ran it: **with** the sentence, `not_stated` at confidence **1.00**; lab 01 asks the same question *without* it and gets `not_stated` at only **0.75** with 14% on `unclear`. One sentence of rubric removed the ambiguity.)
2. Change the second sentence to `"Wrote Python scripts to export chiller trend logs every night."` — what should happen, and does it?
3. Add a sentence that is irrelevant to the job, e.g. `"Enjoys marathon running."` Nothing should change. (If a hobby moved a job criterion, that would be a bias problem to investigate.)

✓ Checkpoint: you saw that the exact wording of the instruction decides whether a course counts as evidence.

## 2 · Build the checklist — lab 01

Lab 01 reads four fictional profiles from `data/profiles.csv` and asks the three evidence questions about each. Every record, whatever the labels, goes to `recruiter_review`.

```bash
.venv/bin/python week24/06_hr_evidence/labs/lab01_evidence_checklist.py
```

**Expected output**

```
▣ STEP 2 · the checklist for all 4 profiles
│ id             python             integration        building           route
│ candidate-001  evidenced (1.00)   evidenced (1.00)   evidenced (1.00)   recruiter_review
│ candidate-002  not_stated (0.88)  not_stated (1.00)  not_stated (1.00)  recruiter_review
│ candidate-003  not_stated (0.75)  not_stated (1.00)  evidenced (0.97)   recruiter_review
│ candidate-004  not_stated (0.46)  evidenced (0.99)   evidenced (0.97)   recruiter_review
▣ STEP 3 · what the recruiter does next
→ 'not_stated' = missing from this text. Ask the candidate; do not assume they lack it
═ execute: False · route: recruiter_review · no_ranking: True · no_auto_rejection: True
```

Look closely at **candidate-004**: *"Reviewed Python pull requests for the analytics team."* Jev said `not_stated` — but with **confidence 0.46**. Is reviewing Python code "hands-on Python implementation work"? Reasonable people disagree. The low confidence is the model telling you *this one needs a human*. That is exactly what the checklist is for.

Notice also what is **not** in the table: no total, no rank, no "top candidate". Adding the three columns together would silently turn an evidence checklist into a ranking — so the code never does it.

✓ Checkpoint: you can explain why candidate-004's Python row has low confidence, and why the table has no total column.

## 3 · Select, don't generate — lab 02

A label like `evidenced` is still hard to audit: *where* is the evidence? You might be tempted to ask a generative model to "quote the evidence". Don't — it can invent a quotation that sounds right.

Instead, use Jev's `choice` as a **selector**:

1. **Code** splits the text into numbered sentences: `s1`, `s2`, `s3`.
2. **Jev** chooses which sentence id is the clearest evidence — or `not_stated`.
3. **Code** prints that sentence **verbatim** from the original text.

Because the options are the sentences themselves, Jev *cannot* return words that are not in the text.

```jev
{
  "state": {"sentences": {
    "s1": "Led a team that wrote BACnet drivers in C++.",
    "s2": "Reviewed Python pull requests for the analytics team.",
    "s3": "Presented at a smart-building meetup."
  }},
  "questions": {
    "integration": {
      "type": "choice",
      "instructions": "Which ONE sentence in `sentences` is the clearest explicit evidence of hands-on REST API, BACnet or Modbus integration work?",
      "criteria": {
        "s1": "Sentence s1: Led a team that wrote BACnet drivers in C++.",
        "s2": "Sentence s2: Reviewed Python pull requests for the analytics team.",
        "s3": "Sentence s3: Presented at a smart-building meetup.",
        "not_stated": "No sentence describes integration work."
      }
    }
  }
}
```

Experiment: change the question to *building-systems* work. Does it pick `s1` (BACnet drivers) or `s3` (a smart-building meetup)? Which one would *you* call evidence of hands-on work?

```bash
.venv/bin/python week24/06_hr_evidence/labs/lab02_sentence_evidence.py
```

**Expected output**

```
▣ STEP 1 · candidate-001 — 3 sentences
   s1: Implemented Python ETL services for hotel energy meters.
   s2: Integrated Modbus gateways with REST APIs.
   s3: Maintained BMS telemetry pipelines for 40 buildings.
» python       → s1         conf 1.00   “Implemented Python ETL services for hotel energy meters.”
» integration  → s2         conf 1.00   “Integrated Modbus gateways with REST APIs.”
» building     → s3         conf 1.00   “Maintained BMS telemetry pipelines for 40 buildings.”
…
▣ STEP 4 · candidate-004 — 3 sentences
» python       → not_stated conf 0.64   (ask the candidate)
» integration  → s1         conf 0.99   “Led a team that wrote BACnet drivers in C++.”
» building     → s1         conf 1.00   “Led a team that wrote BACnet drivers in C++.”
◆ every quoted sentence above was copied from the original text by code — Jev only chose an id
═ execute: False · route: recruiter_review · no_ranking: True · no_auto_rejection: True
```

The recruiter now sees **real words they can check in seconds**. This "select, don't generate" pattern is one of the most useful things Jev does — you will see it again for RAG passages (Lab 08).

✓ Checkpoint: you can explain why a Choice over sentence ids cannot fabricate a quotation, while "quote the evidence" to a generative model can.

## 4 · Before any real candidate data

- Get an explicit decision on **approved processing, retention and cross-border transfer** before sending real candidate text to a hosted API. A candidate's consent to apply is not automatically consent to every third-party service.
- Keep **provenance** (where the text came from), processing authority and a retention deadline in the ATS.
- **Audit** the rubric: run the same questions on paired texts that differ only in irrelevant ways (writing style, a hobby, a non-English phrasing) and check the labels do not move.
- Apply the **same job-related criteria to everyone**, and log model version + rubric version with every checklist.

The full reference implementation (same three questions, same policy) is in the original script:

```bash
.venv/bin/python week24/jev_lab/jev_lab.py run hr --limit 2 --live
```

✓ Checkpoint: you can list the four checks that must happen before real candidate text is processed.

## Labs — run them here

**labs/lab01_evidence_checklist.py** — Four fictional profiles → evidenced / not_stated / unclear per criterion → recruiter review, no ranking.

**labs/lab02_sentence_evidence.py** — Code numbers the sentences, Jev selects the evidence id, code quotes it verbatim.

## Try it yourself

**Exercise 06 — add a criterion and a fair policy.** Open `week24/06_hr_evidence/exercises/ex06_new_criterion.py`.

1. **TODO 1** — write `cloud_evidence`: a `choice` with exactly `evidenced` / `not_stated` / `unclear` about hands-on cloud deployment (AWS, Azure, GCP…). Say in the instruction that an interest or a course is not hands-on work.
2. **TODO 2** — write `checklist_row(answers)` returning `{"cloud": …, "follow_up": [criteria not evidenced], "route": "recruiter_review"}`. It must never contain a score, rank, total or rejection.

The checker runs the shape and policy tests offline first, then asks Jev about three test texts.

```bash
.venv/bin/python week24/06_hr_evidence/exercises/ex06_new_criterion.py
```

**Expected output**

```
✓ cloud_evidence is a choice
✓ labels are exactly evidenced / not_stated / unclear
✓ every record goes to recruiter_review
✓ follow_up lists the criteria that are not evidenced
✓ no score, rank or rejection in the output
▣ STEP 3 · live — does Jev agree with the expected labels?
│ ✓  Deployed Python microservices to AWS ECS with T…  evidenced           evidenced   1.00
│ ✓  Interested in learning Azure. Completed a cloud…  not_stated|unclear  not_stated  1.00
│ ✓  Built Excel reports for the finance team.         not_stated          not_stated  1.00
```

<details><summary>Hint — why call it "follow_up" and not "missing"?</summary>

Words shape behaviour. A column called `missing` or `gaps` invites a recruiter to treat it as a deficit. `follow_up` says what to actually do: ask the candidate about it.

</details>

<details><summary>Stretch — sentence-level cloud evidence</summary>

Copy `evidence_questions()` from lab 02 and add `"cloud": "hands-on cloud deployment"` to `CRITERIA`. Run it on the three test texts and print the verbatim sentence for each.

</details>

✓ Checkpoint: all offline checks are ✓ and the live table shows three ✓ rows (or a ⚠ you can explain).

## Troubleshooting

| Symptom | Fix |
|---|---|
| A course or interest is marked `evidenced` | Say explicitly in the instruction that interests and courses are not hands-on work — Jev reads literally |
| Low confidence on one row | That is useful information — it marks a borderline case for the recruiter. Do not "fix" it by lowering a threshold |
| Lab 02 picks a sentence you disagree with | Discuss it and sharpen the criterion wording. Keep a note — it becomes an evaluation example |
| `every row needs candidate_id and professional_text` | A CSV row is empty. Fix the data; never silently skip a candidate |
| Someone asks you to "just add a total score" | Don't. That turns an evidence checklist into an automated ranking — a different, much riskier system needing its own legal and fairness review |

## Next

Continue to [Lab 07 — HVAC fault triage: code computes, Jev judges](../07_afdd_triage/TUTORIAL.md).
