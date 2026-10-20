/**
 * Structured Data Extraction — pi extension entry point.
 *
 * This file is the WIRING layer. It composes the five Exercise 3 patterns
 * but contains none of the pattern code itself — each step lives in its own
 * file under pipeline/ for one-file-per-task teaching.
 *
 *   pipeline/step1-schema.ts       — slide 4   the JSON schema
 *   pipeline/step2-validation.ts   — slide 5   semantic validation
 *   pipeline/step3-few-shot.ts     — slide 6   example injection
 *   pipeline/step4-batch.ts        — slide 7   /batch-extract command
 *   pipeline/step5-routing.ts      — slide 8   confidence routing + review queue
 *
 * Loading:
 *   pi -e ./pi-structured-extraction/extension.ts          # explicit
 *   .pi/extensions/extraction.ts                # auto-discovery (project-local)
 *
 * Usage (interactive):
 *   /extract samples/invoice-clean.txt
 *   /extract samples/invoice-arithmetic-conflict.txt
 *   /batch-extract samples/*.txt
 *   /review-queue
 */

import { dirname, isAbsolute, resolve } from "node:path";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

import { ExtractedInvoice } from "./data-extraction-agent/step1-schema.js";
import { validate, formatRetryFeedback } from "./data-extraction-agent/step2-validation.js";
import { installFewShot } from "./data-extraction-agent/step3-few-shot.js";
import { registerBatchCommand } from "./data-extraction-agent/step4-batch.js";
import {
  routeExtraction,
  recordExtraction,
  registerReviewQueueCommand,
  registerShowCommand,
  registerExportReviewCommand,
} from "./data-extraction-agent/step5-routing.js";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));

// Try cwd first (so the user can extract their own documents); fall back to
// the extension's own dir (so bundled samples work from any cwd).
async function loadDoc(cwd: string, path: string) {
  if (isAbsolute(path)) {
    return { path, content: await readFile(path, "utf8") };
  }
  const cwdPath = resolve(cwd, path);
  try {
    return { path: cwdPath, content: await readFile(cwdPath, "utf8") };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
  }
  const extPath = resolve(EXTENSION_DIR, path);
  return { path: extPath, content: await readFile(extPath, "utf8") };
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify(
      "Structured Extraction extension loaded. Try /extract <path>",
      "info",
    );
  });

  // STEP 3 — inject few-shot examples into every turn's system prompt.
  installFewShot(pi);

  // STEPS 1, 2, 5 — register the extraction tool. Schema = step 1, validation
  // = step 2, routing = step 5. The execute() handler composes them.
  pi.registerTool({
    name: "extract_invoice",
    label: "Extract invoice",
    description:
      "Submit a structured extraction of an invoice/receipt/purchase order. " +
      "Read the source first using 'read', then call this with the FULL " +
      "structured object populated. Use null for genuinely absent fields. " +
      "NEVER fabricate values to satisfy required fields. Compute calculated_total " +
      "yourself from line_items.",
    parameters: ExtractedInvoice,
    promptSnippet:
      "extract_invoice — submit a structured invoice/receipt extraction",

    async execute(_toolCallId, extraction, _signal, _onUpdate, ctx) {
      // Step 2 — semantic validation; failure becomes retry feedback.
      const errors = validate(extraction);
      if (errors.length > 0) {
        return {
          content: [{ type: "text", text: formatRetryFeedback(errors) }],
          details: { accepted: false, errors, attempted: extraction },
        };
      }

      // Step 5 — confidence-based routing. Record EVERY extraction in the
      // unified log (auto + review). /review-queue filters; /export-review
      // shows both groups in separate sections.
      const decision = routeExtraction(extraction);
      recordExtraction(extraction, decision);
      if (decision.route === "human_review") {
        ctx.ui.notify(
          `→ Routed ${extraction.invoice_id ?? "(unknown id)"} to human review: ${decision.reason}`,
          "warning",
        );
      }

      const summary =
        `✓ Extraction accepted (${decision.route}).\n` +
        `  vendor=${extraction.vendor ?? "(null)"}\n` +
        `  invoice_id=${extraction.invoice_id ?? "(null)"}\n` +
        `  line_items=${extraction.line_items.length}\n` +
        `  total=$${extraction.calculated_total.toFixed(2)} ` +
        `(stated $${extraction.stated_total.toFixed(2)})\n` +
        `  confidence.overall=${extraction.confidence.overall.toFixed(2)}\n` +
        (extraction.conflict_detected
          ? `  ⚠ conflict_detected=true — total mismatch surfaced for review\n`
          : "") +
        `  routed=${decision.route}`;
      return {
        content: [{ type: "text", text: summary }],
        details: { accepted: true, route: decision.route, extraction },
      };
    },
  });

  // /extract <path> — convenience command for single-document extraction.
  // Not strictly part of any Exercise 3 step; it's how you invoke the pipeline.
  pi.registerCommand("extract", {
    description: "Run structured extraction on a document. Usage: /extract <path>",
    handler: async (args, ctx) => {
      const path = args.trim();
      if (!path) {
        ctx.ui.notify(
          "Usage: /extract <path>  (e.g. /extract samples/invoice-clean.txt)",
          "warning",
        );
        return;
      }

      let loaded: { path: string; content: string };
      try {
        loaded = await loadDoc(ctx.cwd, path);
      } catch (err) {
        ctx.ui.notify(`Failed to read ${path}: ${(err as Error).message}`, "error");
        return;
      }

      const prompt =
        `Extract the following document using extract_invoice. Read carefully, ` +
        `populate every field, use null where genuinely absent, and compute ` +
        `calculated_total yourself.\n\n` +
        `Source path: ${loaded.path}\n` +
        `--- BEGIN DOCUMENT ---\n${loaded.content}\n--- END DOCUMENT ---`;
      pi.sendUserMessage(prompt);
    },
  });

  // STEP 4 — /batch-extract command (registered from its own file).
  registerBatchCommand(pi, EXTENSION_DIR);

  // STEP 5 — review UX commands (registered from their own file).
  registerReviewQueueCommand(pi);  // /review-queue   one-line summary
  registerShowCommand(pi);         // /show <id>      full inline detail
  registerExportReviewCommand(pi); // /export-review  HTML report in browser
}
