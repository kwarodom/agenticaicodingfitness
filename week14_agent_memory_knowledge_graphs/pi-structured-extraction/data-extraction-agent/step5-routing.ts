/**
 * Step 5 of Exercise 3 — confidence-based human review routing.
 *
 * Demonstrates: emit field-level confidence scores, route low-confidence
 * extractions to human review, allow operators to inspect the queue both
 * inline (terminal) and as a polished HTML report (browser).
 *
 * Maps to slide 8 ("How do you decide what to automate?").
 *
 *   routeExtraction(extraction)              — pure function. Returns the
 *                                              route ("auto" | "human_review")
 *                                              and a reason string.
 *
 *   enqueueForReview(extraction, reason)     — push into the in-memory queue.
 *
 *   registerReviewQueueCommand(pi)           — /review-queue: one-line summary
 *   registerShowCommand(pi)                  — /show <id>: full inline detail
 *   registerExportReviewCommand(pi)          — /export-review: HTML report,
 *                                              opens in browser. Source-vs-output
 *                                              side-by-side, confidence bars,
 *                                              Anthropic palette.
 *
 * The 0.7 confidence threshold is ILLUSTRATIVE. In production you'd calibrate
 * it against a labeled validation set (the exam-tested point: self-reported
 * model confidence is uncalibrated by default).
 *
 * Two routing triggers:
 *   - overall confidence below threshold
 *   - conflict_detected from the dual-extraction pattern (slide 4)
 */

import { exec } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import type { Extraction } from "./step1-schema.js";

export type Route = "auto" | "human_review";

export interface RouteDecision {
  route: Route;
  reason: string;
}

interface LogEntry {
  custom_id: string;
  extraction: Extraction;
  route: Route;
  reason: string;        // empty string for auto-routed
  recorded_at: string;
}

// Backwards-compat alias for the HTML renderer below — same shape, plus a
// `routed_at` field for older signatures that read it.
interface QueueEntry extends LogEntry {
  routed_at: string;
}

// Unified log of every extraction the pipeline accepted, regardless of route.
// Filtered views: /review-queue shows route==="human_review" only;
// /export-review shows BOTH groups in separate sections of the HTML report.
const extractionLog: LogEntry[] = [];

const CONFIDENCE_THRESHOLD = 0.7;

export function routeExtraction(extraction: Extraction): RouteDecision {
  if (extraction.conflict_detected) {
    return {
      route: "human_review",
      reason:
        `arithmetic conflict ` +
        `(stated $${extraction.stated_total.toFixed(2)} vs ` +
        `calculated $${extraction.calculated_total.toFixed(2)})`,
    };
  }

  if (extraction.confidence.overall < CONFIDENCE_THRESHOLD) {
    return {
      route: "human_review",
      reason: `low confidence (overall=${extraction.confidence.overall.toFixed(2)})`,
    };
  }

  return { route: "auto", reason: "" };
}

/**
 * Record EVERY extraction in the unified log, regardless of route.
 * extension.ts calls this once per accepted extraction.
 */
export function recordExtraction(
  extraction: Extraction,
  decision: RouteDecision,
): void {
  // Prefer invoice_id, fall back to source_path, then to a synthesized ID,
  // since both invoice_id and vendor can be null for peer/IOU documents.
  const id =
    extraction.invoice_id ??
    extraction.source_path ??
    `UNKNOWN-${new Date().toISOString().slice(0, 10)}-${extractionLog.length + 1}`;
  extractionLog.push({
    custom_id: id,
    extraction,
    route: decision.route,
    reason: decision.reason,
    recorded_at: new Date().toISOString(),
  });
}

/** @deprecated retained for any external caller — prefer recordExtraction. */
export function enqueueForReview(extraction: Extraction, reason: string): void {
  recordExtraction(extraction, { route: "human_review", reason });
}

// ---------------------------------------------------------------------------
// /review-queue — quick one-line summary
// ---------------------------------------------------------------------------
export function registerReviewQueueCommand(pi: ExtensionAPI): void {
  pi.registerCommand("review-queue", {
    description: "Show extractions routed to human review (one line each)",
    handler: async (_args, ctx) => {
      const review = extractionLog.filter((e) => e.route === "human_review");
      const auto = extractionLog.filter((e) => e.route === "auto");
      if (review.length === 0 && auto.length === 0) {
        ctx.ui.notify("No extractions yet.", "info");
        return;
      }
      const lines = review.map(
        (q, i) =>
          `${i + 1}. ${q.custom_id}  (${q.reason})  ${q.recorded_at.slice(0, 19)}`,
      );
      const reviewBlock =
        review.length === 0
          ? "Review queue is empty (everything auto-routed)."
          : `Review queue (${review.length} items):\n` + lines.join("\n");
      ctx.ui.notify(
        reviewBlock +
          `\n\n${auto.length} extraction${auto.length === 1 ? "" : "s"} auto-routed (not shown here). ` +
          `Use /export-review to see ALL ${extractionLog.length} extractions in HTML.`,
        "info",
      );
    },
  });
}

// ---------------------------------------------------------------------------
// /show <invoice_id> — full inline detail of one queued extraction
// ---------------------------------------------------------------------------
export function registerShowCommand(pi: ExtensionAPI): void {
  pi.registerCommand("show", {
    description: "Show full details of a queued extraction. Usage: /show <invoice_id>",
    handler: async (args, ctx) => {
      const id = args.trim();
      if (!id) {
        ctx.ui.notify("Usage: /show <invoice_id>", "warning");
        return;
      }
      // /show searches the FULL log (auto + review), not just the review queue.
      const entry = extractionLog.find((q) => q.custom_id === id);
      if (!entry) {
        ctx.ui.notify(
          `No extraction with id="${id}". Try /review-queue or /export-review.`,
          "warning",
        );
        return;
      }
      const e = entry.extraction;
      const conf = e.confidence;
      const fmtMoney = (n: number | null) =>
        n === null ? "(null)" : `$${n.toFixed(2)}`;
      const fmtPct = (n: number | null) =>
        n === null ? "(null)" : `${(n * 100).toFixed(1)}%`;
      const reasonOrAuto = entry.route === "auto" ? "auto-routed" : entry.reason;

      const lines = [
        `─── ${e.invoice_id ?? entry.custom_id}  (${reasonOrAuto}) ───`,
        ``,
        `vendor:          ${e.vendor ?? "(null)"}`,
        `document_type:   ${e.document_type}${e.document_type_detail ? ` (${e.document_type_detail})` : ""}`,
        `customer_id:     ${e.customer_id ?? "(null)"}`,
        `due_date:        ${e.due_date ?? "(null)"}`,
        `payment_method:  ${e.payment_method ?? "(null)"}`,
        `paid_by:         ${e.paid_by ?? "(null)"}`,
        `source_path:     ${e.source_path ?? "(not provided)"}`,
        `recorded_at:     ${entry.recorded_at}`,
        ``,
        `line_items (${e.line_items.length}):`,
        ...e.line_items.map(
          (li) =>
            `  ${li.quantity}× ${li.description.padEnd(36)} @ $${li.unit_price.toFixed(2)}  = $${li.line_total.toFixed(2)}`,
        ),
        ``,
        `subtotal:          ${fmtMoney(e.subtotal)}`,
        `tax_amount:        ${fmtMoney(e.tax_amount)}`,
        `tip_amount:        ${fmtMoney(e.tip_amount)}    (tip_percentage: ${fmtPct(e.tip_percentage)})`,
        `stated_total:      ${fmtMoney(e.stated_total)}`,
        `calculated_total:  ${fmtMoney(e.calculated_total)}`,
        `conflict_detected: ${e.conflict_detected}`,
        ``,
        `confidence:`,
        `  vendor      ${confBar(conf.vendor)}  ${conf.vendor.toFixed(2)}`,
        `  invoice_id  ${confBar(conf.invoice_id)}  ${conf.invoice_id.toFixed(2)}`,
        `  line_items  ${confBar(conf.line_items)}  ${conf.line_items.toFixed(2)}`,
        `  overall     ${confBar(conf.overall)}  ${conf.overall.toFixed(2)}`,
      ];
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}

function confBar(v: number): string {
  const filled = Math.round(v * 20);
  return "█".repeat(filled) + "░".repeat(20 - filled);
}

// ---------------------------------------------------------------------------
// /export-review — HTML report opened in browser
// ---------------------------------------------------------------------------
export function registerExportReviewCommand(pi: ExtensionAPI): void {
  pi.registerCommand("export-review", {
    description: "Generate an HTML review report and open it in the browser",
    handler: async (_args, ctx) => {
      if (extractionLog.length === 0) {
        ctx.ui.notify(
          "No extractions yet — nothing to export.",
          "warning",
        );
        return;
      }

      // Try to read each source file so the HTML can show side-by-side.
      const enrich = async (entries: LogEntry[]) =>
        Promise.all(
          entries.map(async (q) => {
            let source: string | null = null;
            if (q.extraction.source_path) {
              try {
                source = await readFile(q.extraction.source_path, "utf8");
              } catch {
                // best-effort: HTML will show "(source not readable)" instead
              }
            }
            return { ...q, routed_at: q.recorded_at, source };
          }),
        );

      const review = await enrich(extractionLog.filter((e) => e.route === "human_review"));
      const auto = await enrich(extractionLog.filter((e) => e.route === "auto"));
      const html = renderHtml(review, auto);
      // Write to pi's current working directory (where the user invoked pi)
      // using a stable filename — re-running overwrites, browser refresh works.
      const outPath = join(ctx.cwd, "review-queue.html");
      await writeFile(outPath, html, "utf8");

      ctx.ui.notify(`Wrote ${outPath} — opening in browser…`, "info");

      // macOS / Linux / Windows
      const opener =
        process.platform === "darwin"
          ? "open"
          : process.platform === "win32"
            ? "start"
            : "xdg-open";
      exec(`${opener} "${outPath}"`, (err) => {
        if (err) ctx.ui.notify(`Couldn't auto-open: ${err.message}`, "warning");
      });
    },
  });
}

// ---------------------------------------------------------------------------
// HTML renderer — Anthropic palette, source-vs-output side-by-side,
// confidence bars. Self-contained, no external resources.
// ---------------------------------------------------------------------------
type EnrichedEntry = QueueEntry & { source: string | null };

function renderHtml(review: EnrichedEntry[], auto: EnrichedEntry[]): string {
  const css = `
    :root {
      --dark: #141413; --cream: #faf9f5;
      --mid-gray: #b0aea5; --light-gray: #e8e6dc;
      --orange: #d97757; --blue: #6a9bcc; --green: #788c5d;
    }
    * { box-sizing: border-box; }
    body { background: var(--cream); color: var(--dark);
           font-family: 'Lora', Georgia, serif;
           max-width: 1200px; margin: 2em auto; padding: 0 1.5em; line-height: 1.5; }
    h1 { font-family: 'Poppins', Arial, sans-serif; font-weight: 600;
         border-bottom: 4px solid var(--orange); padding-bottom: 0.4em; margin-bottom: 0.3em; }
    .meta { color: var(--mid-gray); font-size: 0.85em; margin-top: 0; }
    .item { background: white; padding: 1.4em 1.5em; margin: 1.2em 0;
            border-left: 6px solid var(--orange);
            box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-radius: 4px; }
    .item.low-conf { border-left-color: var(--blue); }
    .item h2 { font-family: 'Poppins', Arial, sans-serif; font-weight: 600;
               margin: 0 0 0.2em 0; color: var(--dark); font-size: 1.25em; }
    .reason { color: var(--orange); font-style: italic; margin: 0 0 0.3em 0; font-size: 0.95em; }
    .item.low-conf .reason { color: var(--blue); }
    .item-meta { color: var(--mid-gray); font-size: 0.82em; margin: 0 0 1em 0;
                 font-family: Menlo, Consolas, monospace; }
    .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
    @media (max-width: 800px) { .columns { grid-template-columns: 1fr; } }
    .panel { background: var(--light-gray); padding: 0.9em 1em; border-radius: 4px; }
    .panel h3 { font-family: 'Poppins', Arial, sans-serif; font-weight: 600;
                font-size: 0.7em; text-transform: uppercase; letter-spacing: 1.5px;
                color: var(--mid-gray); margin: 0 0 0.6em 0; }
    pre { font-family: Menlo, Consolas, monospace; font-size: 0.82em; margin: 0;
          white-space: pre-wrap; word-wrap: break-word; color: var(--dark); }
    .conf-grid { display: grid; grid-template-columns: 110px 1fr 50px;
                 gap: 0.6em; align-items: center; margin: 0.3em 0;
                 font-size: 0.85em; font-family: Menlo, Consolas, monospace; }
    .bar { background: var(--light-gray); height: 6px; border-radius: 3px; overflow: hidden; }
    .fill { height: 100%; background: var(--green); transition: width 0.3s; }
    .fill.med { background: var(--blue); }
    .fill.low { background: var(--orange); }
    .conf-label { color: var(--mid-gray); }
    .confidence-section { margin-top: 1em; }
    .confidence-section h3 { font-family: 'Poppins', Arial, sans-serif; font-weight: 600;
                              font-size: 0.7em; text-transform: uppercase; letter-spacing: 1.5px;
                              color: var(--mid-gray); margin: 1em 0 0.4em 0; }
    .section-h { font-family: 'Poppins', Arial, sans-serif; font-weight: 600;
                 color: var(--dark); margin: 2em 0 0.5em 0;
                 padding-bottom: 0.3em; border-bottom: 2px solid var(--light-gray); }
    .section-h.review { border-bottom-color: var(--orange); }
    .section-h.auto { border-bottom-color: var(--green); }
    .item.auto { border-left-color: var(--green); opacity: 0.85; }
    .item.auto .reason { color: var(--green); }
    .footer { color: var(--mid-gray); font-size: 0.8em; text-align: center;
              margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--light-gray); }
  `;

  // Renders a single extraction card. The kind ("review" | "auto") changes
  // the left-border color (orange = needs attention, green = auto-accepted).
  const renderItem = (item: EnrichedEntry, i: number, kind: "review" | "auto") => {
    const e = item.extraction;
    const isLowConf = kind === "review" && !e.conflict_detected;
    const conf = e.confidence;
    const confRow = (label: string, v: number) => {
      const cls = v >= 0.8 ? "" : v >= 0.6 ? "med" : "low";
      return `<div class="conf-grid">
        <span class="conf-label">${label}</span>
        <div class="bar"><div class="fill ${cls}" style="width: ${(v * 100).toFixed(0)}%"></div></div>
        <span>${v.toFixed(2)}</span>
      </div>`;
    };
    const sourceContent =
      item.source ?? `(source not readable: ${e.source_path ?? "no path provided"})`;
    const docTypeLine = e.document_type_detail
      ? `${e.document_type} — ${e.document_type_detail}`
      : e.document_type;
    const reasonText = kind === "auto" ? "auto-accepted (confidence cleared threshold)" : item.reason;
    const titleId = e.invoice_id ?? item.custom_id;
    const titleVendor = e.vendor ?? "(no vendor in source)";
    const cssClass = `item${kind === "auto" ? " auto" : isLowConf ? " low-conf" : ""}`;
    return `<div class="${cssClass}">
      <h2>${i + 1}. ${escape(titleId)} — ${escape(titleVendor)}</h2>
      <p class="reason">→ ${escape(reasonText)}</p>
      <p class="item-meta">${escape(docTypeLine)} · ${escape(item.routed_at.slice(0, 19))} · source ${escape(e.source_path ?? "(not provided)")}</p>
      <div class="columns">
        <div class="panel">
          <h3>Source document</h3>
          <pre>${escape(sourceContent)}</pre>
        </div>
        <div class="panel">
          <h3>Extracted JSON</h3>
          <pre>${escape(JSON.stringify(e, null, 2))}</pre>
        </div>
      </div>
      <div class="confidence-section">
        <h3>Field-level confidence</h3>
        ${confRow("vendor", conf.vendor)}
        ${confRow("invoice_id", conf.invoice_id)}
        ${confRow("line_items", conf.line_items)}
        ${confRow("overall", conf.overall)}
      </div>
    </div>`;
  };

  const reviewSection =
    review.length === 0
      ? `<p class="meta">No items routed to human review (everything auto-cleared the confidence threshold).</p>`
      : review.map((item, i) => renderItem(item, i, "review")).join("\n");

  const autoSection =
    auto.length === 0
      ? `<p class="meta">No items auto-routed.</p>`
      : auto.map((item, i) => renderItem(item, i, "auto")).join("\n");

  const total = review.length + auto.length;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Structured Extraction — All extractions (${total})</title>
<style>${css}</style>
</head>
<body>
<h1>Extraction Report</h1>
<p class="meta">${total} extraction${total === 1 ? "" : "s"} total — ${review.length} routed to human review, ${auto.length} auto-accepted · generated ${new Date().toISOString()}</p>

<h2 class="section-h review">Routed to human review (${review.length})</h2>
${reviewSection}

<h2 class="section-h auto">Auto-accepted (${auto.length})</h2>
${autoSection}

<p class="footer">Generated by pi-structured-extraction · Scenario 6 study material</p>
</body>
</html>`;
}

function escape(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!,
  );
}
