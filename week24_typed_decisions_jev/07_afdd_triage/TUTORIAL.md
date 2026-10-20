# ▶ Jev Lab 07 — HVAC fault triage: code computes, Jev judges

> Part of Week 24 · Typed AI decisions with Jev. Automated fault detection and diagnostics (AFDD) produces a stream of operator notes and telemetry. Jev's job here is narrow: put each incident in the right **investigation queue** for an engineer. It never diagnoses a root cause, and it never touches the building.

**What you'll actually do**
- Compute the numeric facts in Python first — staleness, data quality, deviation from setpoint.
- Give Jev those facts plus the operator's note, and get a queue, a safety signal and a severity.
- Watch a deterministic policy **override** Jev when the data cannot be trusted.
- Hit the classic literal-reading trap: *"No smoke reported"* vs *"smoke reported"*.
- Write the flag computation yourself, tested offline to the decimal.

**Time** ~30 min · **Difficulty** intermediate · **Cost** ≈ $0.0003 live · $0 dry

## 0 · Why code does the numbers

TypeSafe's own documentation lists this as a known weakness of Jev 1.13: *it is not a calculator*. It reads numbers and dates as text. So we split the work:

| Job | Who does it | Why |
|---|---|---|
| Is the reading older than 10 minutes? | **code** | exact comparison of trusted timestamps |
| Is the zone > 2 °C above setpoint? | **code** | arithmetic |
| Is the sensor quality flag bad? | **code** | it is a boolean from the BMS |
| Which investigation queue fits this note + these facts? | **Jev** | a judgment about language and context |
| Does the note report a safety concern? | **Jev + a code backstop** | never rely on one signal for safety |
| Change a setpoint, write BACnet, open a work order | **nobody here** | a separately approved system, with a human |

```text
BMS reading ──► compute flags (code) ──┐
                                        ├──► Jev: queue · safety · severity ──► policy (code) ──► engineer_review
operator note ─────────────────────────┘                                           │
                                                   stale / bad data overrides the queue · safety is OR'd with a keyword check
```

The thresholds in this lab (**600 s** stale, **2 °C** warm deviation) are **teaching constants**, not validated AltoTech limits or comfort standards. A real site loads approved thresholds per site, equipment, occupancy mode and schedule.

✓ Checkpoint: you can say which three facts code computes before Jev sees anything, and why.

## 1 · Give Jev facts, not arithmetic

Here is the state Jev receives for one incident — the note, plus a `computed` block that code filled in. Try it live:

```jev
{
  "state": {
    "operator_note": "Guests report warm rooms on floor 3. AHU-3 is running.",
    "computed": {"stale": false, "bad_quality": false, "warm_deviation": true, "deviation_c": 5.1}
  },
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Using `operator_note` and the `computed` flags, select the next investigation queue. Do not claim a proven root cause.",
      "criteria": {
        "cooling": "Comfort or cooling-performance investigation.",
        "sensor": "Sensor plausibility or calibration investigation.",
        "connectivity": "Offline gateway, missing or stale telemetry investigation.",
        "other": "Insufficient or conflicting evidence; engineer triage."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the reported operational impact?",
      "criteria": ["Informational or no impact stated.", "Localized discomfort or limited degradation.", "Significant service disruption.", "Possible immediate safety incident."]
    }
  }
}
```

Experiment:

1. Set `"stale": true`. Does the queue move toward `connectivity`? (It might — but in step 2 you will see the policy does not *depend* on that.)
2. Set `"bad_quality": true` and change the note to `"Room 214 sensor reads 35 C but the guest says the room feels cold."` Which queue now?
3. Change the note to `"Water is dripping from the ceiling near the AHU."` Watch `severity`.

✓ Checkpoint: you changed a computed flag and saw Jev use it — without Jev computing it.

## 2 · Triage five incidents — lab 01

```bash
.venv/bin/python week24/07_afdd_triage/labs/lab01_triage_incidents.py
```

**Expected output**

```
▣ STEP 1 · what code computes BEFORE the model sees anything
   raw reading: zone 29.1 °C, setpoint 24.0 °C, age 120 s, quality_ok True
✓ computed:   {"stale": false, "bad_quality": false, "warm_deviation": true, "deviation_c": 5.1}
…
▣ STEP 3 · triage all 5 incidents
│ id       Δ°C    code flags                  jev queue     safety  sev   proposed queue                safety rev
│ fault-1  +5.1   warm_deviation              cooling       0.03    1.01  cooling                       -
│ fault-2  -1.0   stale                       connectivity  0.03    1.82  connectivity_or_data_quality  -
│ fault-3  +0.4   -                           other         0.73    2.79  other                         YES
│ fault-4  +12.0  bad_quality,warm_deviation  sensor        0.03    1.03  connectivity_or_data_quality  -
│ fault-5  +1.2   -                           other         0.02    0.60  other                         -
═ execute: False · route: engineer_review · no_bacnet_write: True · no_setpoint_change: True
```

The policy is ten lines of plain Python:

```python
def policy(state, a):
    flags = state["computed"]
    data_problem = flags["stale"] or flags["bad_quality"]
    keyword_hit = bool(SAFETY_WORDS.search(state["operator_note"]))
    return {
        "route": "engineer_review",
        "proposed_queue": "connectivity_or_data_quality" if data_problem else a["queue"]["choice"],
        "safety_review": a["safety_concern"]["noul"] >= 0.20 or keyword_hit,   # OR — never AND
        "execute": False, "no_bacnet_write": True, "no_setpoint_change": True,
    }
```

Read the interesting rows:

- **fault-4** — the sensor reads 35 °C (+12 °C!) but the guest says it is *cold* and `quality_ok` is False. Jev sensibly picked `sensor`. The policy still overrides to the data-quality queue: **never build a comfort diagnosis on data you know is bad.**
- **fault-3** — *"Burning smell reported near the AHU-2 plant room."* Jev's safety probability is only **0.73**, not 0.99. The policy flags safety review anyway — both because 0.73 ≥ 0.20 and because the keyword backstop hit `burning`. **A low model probability must never suppress a safety response.**
- **fault-2** — Jev said `connectivity` on its own. Good, but the override does not rely on it.

> ⚠ Existing fire, safety and emergency alarm procedures must run **independently** of this classifier. An API timeout or a low probability must never delay them.

✓ Checkpoint: you can explain why fault-4 was overridden even though Jev's `sensor` answer looked sensible.

## 3 · What if the numbers change? — lab 02

Lab 02 asks Jev about one operator note **once**, then runs four different telemetry situations through the same policy. One model call, four outcomes.

```bash
.venv/bin/python week24/07_afdd_triage/labs/lab02_what_if.py
```

**Expected output**

```
▣ STEP 2 · same answers, four telemetry situations → the policy decides
│ situation           Δ°C   stale  bad_q  warm   proposed queue
│ fresh, 4.2 °C warm  +4.2  False  False  True   cooling
│ stale (2 h old)     +4.2  True   False  True   connectivity_or_data_quality
│ bad quality flag    +4.2  False  True   True   connectivity_or_data_quality
│ fresh, on setpoint  +0.3  False  False  False  cooling
◆ 1 model call · 4 outcomes — the override is plain Python you can unit-test
▣ STEP 3 · a literal-reading trap: 'No smoke' vs 'smoke'
» no_smoke  jev safety 0.02 · keyword backstop HIT   “Lobby is warm. AHU fan is running. No smoke or unusual noise reported.”
» smoke     jev safety 0.89 · keyword backstop HIT   “Lobby is warm. AHU fan is running. Smoke smell reported near the AHU.”
◆ compare STEP 1: lab 01 asks whether the note 'explicitly MENTIONS smoke…' → 0.61 on the SAME 'No smoke' note.
```

Why does the keyword backstop say **HIT** on *"No smoke…"*? Because it is a deliberately dumb regex: it sees the word `smoke` and fires, negation or not. That is the point. The backstop is allowed to over-trigger, because a false alarm costs one engineer's glance, while a missed fire costs far more. Jev's 0.02 shows it read the negation correctly; the policy still sends the note to safety review because *either* signal is enough.

This is the most important finding of the lab. The note says **"No smoke or unusual noise reported."**

- Lab 01's question asks whether the note *"explicitly **mention**s smoke, electrical burning, fire…"* → **0.61**. Jev read it literally: "No smoke" *does* mention smoke.
- Lab 02 asks whether a concern is *"**currently present**"* → **0.02**.

Same note, same model, different wording, completely different answer. **The question text is part of your program** — test negations explicitly ("do not turn off the chiller" vs "turn off the chiller").

Try it:

```jev
{
  "state": {"note": "Lobby is warm. AHU fan is running. No smoke or unusual noise reported."},
  "questions": {
    "mentions": {"type": "noul", "instructions": "Does `note` explicitly mention smoke, burning, fire or another safety concern?"},
    "present": {"type": "noul", "instructions": "Does `note` report smoke, burning, fire or another safety concern that is currently present?"}
  }
}
```

When we ran this block: `mentions` **0.72**, `present` **0.02**. Experiment: change the note to `"Smoke smell near the AHU."` Both should jump. Then try `"The smoke alarm test passed this morning."` — which wording handles it better?

✓ Checkpoint: you saw the same "No smoke" note score high on one wording and low on another, and you can explain why the keyword backstop still fires.

## 4 · Connect it to a real building (the safe way)

The proposed production flow keeps every boundary from this lab:

1. **Authorized** time-series query for the site the user may see.
2. Timestamp and quality checks in **server code** — never let a user or an LLM assert that data is fresh.
3. AFDD feature computation in code (deviations, run-hours, cycling counts).
4. Jev triage → investigation queue proposal.
5. An HVAC specialist (person, or a generative model with evidence) writes the explanation.
6. A **human-reviewed** maintenance proposal. The Copilot stays propose-only.

The reference implementation in the original script uses the same questions and override:

```bash
.venv/bin/python week24/jev_lab/jev_lab.py run afdd --limit 2 --live
```

✓ Checkpoint: you can point to the step in this flow where a person approves before anything changes on site.

## Labs — run them here

**labs/lab01_triage_incidents.py** — Five incidents: code computes flags, Jev picks a queue, the policy overrides on bad data and ORs the safety signals.

**labs/lab02_what_if.py** — One Jev call, four telemetry situations, and the "No smoke" literal-reading trap.

## Try it yourself

**Exercise 07 — compute the flags.** Open `week24/07_afdd_triage/exercises/ex07_compute_flags.py` and implement `compute_flags(reading)` returning exactly:

| key | rule |
|---|---|
| `stale` | `age_seconds > 600` |
| `bad_quality` | `quality_ok is False` |
| `warm_deviation` | `zone_c − setpoint_c > 2.0` |
| `cold_deviation` | `setpoint_c − zone_c > 2.0` *(new)* |
| `humidity_high` | `humidity_pct > 70` *(new)* |
| `deviation_c` | `zone_c − setpoint_c`, rounded to 1 decimal |

Five exact offline asserts run first — including the edge case "exactly 2.0 °C is **not** a deviation". Then your flags feed a live triage of the five incidents.

```bash
.venv/bin/python week24/07_afdd_triage/exercises/ex07_compute_flags.py
```

**Expected output**

```
✓ warm and fresh
✓ cold room
✓ exactly 2.0 warm is NOT a deviation
✓ stale + humid
✓ bad quality
▣ STEP 2 · your flags feed a live Jev triage
│ fault-4  bad_quality,warm_deviation  sensor        0.03
```

<details><summary>Hint — the 2.0 °C edge case</summary>

The rule is "more than 2.0", so use `>` not `>=`. This is exactly the kind of detail Jev would be unreliable at — and exactly why it lives in code with a test.

</details>

<details><summary>Stretch — a humidity queue</summary>

Add a `humidity` option to the `queue` choice ("Dehumidification or latent-load investigation") in a copy of lab 01's `QUESTIONS`, and pass your `humidity_high` flag. Which incidents move?

</details>

✓ Checkpoint: 5/5 asserts pass and the live table shows your flags beside Jev's queue.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `invalid numeric field` | A reading has a missing or non-numeric value. Fix the data upstream; never let the model guess a number |
| Safety probability seems low for an obvious hazard | Expected sometimes (0.73 for "burning smell"). That is why the policy ORs it with a keyword backstop and uses a low 0.20 threshold |
| "No smoke" flagged as a safety concern | Your instruction says "mentions". Ask whether a concern is "currently present" — and keep the backstop anyway |
| Queue looks right but the policy overrides it | Working as designed: stale or bad-quality data always goes to the data-quality queue first |
| Exercise `0/5 pass` | `compute_flags()` still returns `{}` — implement all six keys |

## Next

Continue to [Lab 08 — sales leads, RAG passage filtering and BMS point mapping](../08_leads_rag_points/TUTORIAL.md).
