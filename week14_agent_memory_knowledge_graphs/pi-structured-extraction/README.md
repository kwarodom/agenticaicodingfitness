# Example for Structured Data Extraction — pi extension

A working pi extension demonstrating every pattern from Exercise 3 of the
**Claude Certified Architect — Foundations** exam (Scenario 6: Structured
Data Extraction).

The extension turns pi into an extraction agent: register a structured tool,
have the LLM populate it from source documents, validate semantically, and
route by confidence.

## Scope

This extension implements all 5 steps from Exercise 3:

| Step | What | Where in `extension.ts` |
|---|---|---|
| 1 | JSON schema with required, optional, nullable, enum + "other" + detail | `ExtractedInvoice` (lines ~50–95) |
| 2 | Validation-retry loop with feedback | `validate()` + `execute()` retry path |
| 3 | Few-shot examples for varied document formats | `FEW_SHOT_BLOCK` injected via `before_agent_start` hook |
| 4 | Batch processing with `custom_id` correlation | `/batch-extract` command |
| 5 | Field-level confidence + human-review routing | `confidence` field + routing logic in `execute()` |

## Why this works without the Anthropic SDK

Scenario 6 tests *patterns*, not specific SDK calls. JSON Schema, function
calling, validation-retry, batch correlation, and confidence routing are all
provider-agnostic. Pi handles the LLM transport (you're authed to codex via
pi.dev); the extension demonstrates the patterns.

The slides (`../slides/scenario-6.md`) teach the Anthropic-specific
terminology the exam uses (`tool_use`, `tool_choice: "any"`, Message Batches
API, `custom_id`); the extension is the working reference for those concepts.

## Quick start

```bash
# Verify the extension loads (no LLM call, no cost)
pi -e ./pi-structured-extraction/extension.ts --list-models openai-codex

# Run interactively
pi -e ./pi-structured-extraction/extension.ts

# Then in pi:
> /extract samples/invoice-clean.txt
> /extract samples/invoice-missing-fields.txt
> /extract samples/receipt-narrative.txt
> /extract samples/invoice-arithmetic-conflict.txt
> /batch-extract samples/*.txt
> /review-queue
```

## Project-local auto-discovery (optional)

For the extension to load automatically without `-e`, symlink it into the
project-local extensions directory:

```bash
mkdir -p .pi/extensions
ln -s ../../pi-structured-extraction/extension.ts .pi/extensions/extraction.ts
```

Then plain `pi` picks it up, and `/reload` hot-reloads after edits.

## What each sample document tests

| Sample | What it stresses |
|---|---|
| `invoice-clean.txt` | Happy path. All fields present. High confidence → auto route. |
| `invoice-missing-fields.txt` | Nullable behavior. Customer ID, due date, tax breakdown are absent → must return `null`, not fabricate. |
| `receipt-narrative.txt` | Few-shot generalization. Informal prose ("forty-two dollars and change"), no invoice ID, narrative format. Lower confidence → human review. |
| `invoice-arithmetic-conflict.txt` | Self-correction. Stated total ($150.00) doesn't match line-item sum ($84.44). Must surface `conflict_detected: true` and route to review. |

## Verifying validation-retry actually fires

The validation loop is hard to see by eye. Drop in a deliberately broken
extraction by editing `validate()` to log on failure, then watch the LLM
correct itself across turns.

A clean way to force a failure for demo purposes: add this temporary line
near the top of `validate()`:

```ts
if (!e.due_date && e.document_type === "invoice") {
  errors.push("Demo: forcing a retry — invoice should have a due_date.");
}
```

Run `/extract samples/invoice-missing-fields.txt` and you'll see the LLM
retry with corrections (or argue back that the source has no due_date,
which is also valid behavior — the *retry can't fix what isn't there* point
the exam tests).

## How the patterns map to the exam

| Concept the exam uses | What the extension shows |
|---|---|
| `tool_use` with strict JSON schema | `pi.registerTool({ parameters: ExtractedInvoice })` — TypeBox compiles to JSON Schema |
| `tool_choice: "any"` | Pi forces tool calls for registered tools when the model decides extraction is the right action |
| Required vs optional vs nullable | Direct in the schema definition |
| Enum + "other" + detail | `document_type` literal union + `document_type_detail` string |
| Validation-retry with feedback | `validate()` returns errors → tool result content is the feedback the LLM sees |
| Self-correction (`stated_total` + `calculated_total` + `conflict_detected`) | Built into the schema; semantic check verifies internal consistency |
| Few-shot for varied formats | `FEW_SHOT_BLOCK` injected via `before_agent_start` |
| Message Batches API + `custom_id` | `/batch-extract` demonstrates the pattern (sequential demo; real Batch API would be parallel + 50% cheaper) |
| Field-level confidence | `confidence` object in the schema with per-field + overall scores |
| Human review routing | `route === "human_review"` branch in `execute()`; `/review-queue` inspects the queue |

## File layout

```
pi-structured-extraction/
├── extension.ts          # The extension itself
├── samples/              # Test documents
│   ├── invoice-clean.txt
│   ├── invoice-missing-fields.txt
│   ├── receipt-narrative.txt
│   └── invoice-arithmetic-conflict.txt
└── README.md             # This file
```
