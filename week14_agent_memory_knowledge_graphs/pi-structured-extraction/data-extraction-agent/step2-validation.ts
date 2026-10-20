/**
 * Step 2 of Exercise 3 — semantic validation + retry-with-feedback.
 *
 * Demonstrates: schema checks SHAPE, validation checks MEANING.
 * Maps to slide 5 ("Schema checks shape. Validation checks meaning.").
 *
 *   validate(extraction)        — pure function. Returns string[] of errors.
 *                                 Empty array means it passes.
 *
 *   formatRetryFeedback(errors) — turns errors into the message the LLM sees
 *                                 as its tool result, prompting self-correction.
 *
 * Five failure modes the schema can't catch (one check each below):
 *   1. Internal contradiction in enum + detail pattern
 *   2. Format violation on dates
 *   3. Cross-field arithmetic (sum of line items vs calculated_total)
 *   4. Self-consistency on the conflict_detected flag
 *   5. Confidence sanity (don't claim 0.9+ on empty fields)
 *
 * When retry helps:  format errors, arithmetic, structural mistakes.
 * When retry can't:  the information is genuinely absent — slide 4's
 *                    nullable fields handle that, not retries.
 */

import type { Extraction } from "./step1-schema.js";

export function validate(e: Extraction): string[] {
  const errors: string[] = [];

  // 1. Internal contradiction in enum + detail pattern.
  if (e.document_type === "other" && !e.document_type_detail) {
    errors.push(
      `document_type='other' requires document_type_detail (a string naming the actual type). ` +
        `Got null. Provide the type name or change document_type to invoice/receipt/purchase_order.`,
    );
  }

  // 2. Format violation on date.
  if (e.due_date && !/^\d{4}-\d{2}-\d{2}$/.test(e.due_date)) {
    errors.push(
      `due_date "${e.due_date}" is not ISO 8601 (YYYY-MM-DD). Re-format. ` +
        `If the source has no due date, return null instead.`,
    );
  }

  // 3. Cross-field arithmetic — schema can't enforce sums across fields.
  const sum = e.line_items.reduce((acc, li) => acc + li.line_total, 0);
  if (Math.abs(sum - e.calculated_total) > 0.01) {
    errors.push(
      `calculated_total=${e.calculated_total.toFixed(2)} but ` +
        `sum(line_items.line_total)=${sum.toFixed(2)}. Recompute calculated_total ` +
        `as the sum of all line_item.line_total values.`,
    );
  }

  // 4. Self-consistency on the conflict flag.
  const divergence = Math.abs(e.stated_total - e.calculated_total);
  const expected = divergence > 0.01;
  if (expected !== e.conflict_detected) {
    errors.push(
      `conflict_detected=${e.conflict_detected} but |stated_total - calculated_total|=` +
        `${divergence.toFixed(2)}. Set conflict_detected=${expected}.`,
    );
  }

  // 5. Confidence sanity — don't claim high confidence on absent/empty vendor.
  const vendorAbsent = e.vendor === null || !e.vendor.trim();
  if (vendorAbsent && e.confidence.vendor > 0.5) {
    errors.push(
      `vendor is null/empty but confidence.vendor=${e.confidence.vendor}. ` +
        `Either re-extract vendor or lower confidence.vendor below 0.5.`,
    );
  }

  // 6. Tip percentage unit sanity — must be a decimal fraction, not a whole percent.
  if (e.tip_percentage !== null && (e.tip_percentage < 0 || e.tip_percentage > 1)) {
    errors.push(
      `tip_percentage=${e.tip_percentage} is out of range [0, 1]. ` +
        `Use a decimal fraction: 0.15 for 15%, NOT 15.`,
    );
  }

  // 7. Cross-check breakdown against stated_total — when subtotal + tax + tip
  //    are all populated, they should sum to stated_total. Catches arithmetic
  //    drift in receipts that broke things out.
  if (
    e.subtotal !== null &&
    e.tax_amount !== null &&
    e.tip_amount !== null
  ) {
    const sum = e.subtotal + e.tax_amount + e.tip_amount;
    if (Math.abs(sum - e.stated_total) > 0.01) {
      errors.push(
        `subtotal + tax_amount + tip_amount = ${sum.toFixed(2)} but ` +
          `stated_total = ${e.stated_total.toFixed(2)}. ` +
          `Recheck the breakdown or set conflict_detected=true.`,
      );
    }
  }

  // 8. Cross-check tip_amount and tip_percentage if both given.
  if (
    e.tip_amount !== null &&
    e.tip_percentage !== null &&
    e.subtotal !== null
  ) {
    const expectedTip = e.subtotal * e.tip_percentage;
    if (Math.abs(expectedTip - e.tip_amount) > 0.05) {
      errors.push(
        `tip_amount=${e.tip_amount.toFixed(2)} but ` +
          `subtotal × tip_percentage = ${e.subtotal.toFixed(2)} × ${e.tip_percentage} ` +
          `= ${expectedTip.toFixed(2)}. Reconcile.`,
      );
    }
  }

  return errors;
}

export function formatRetryFeedback(errors: string[]): string {
  return (
    "VALIDATION_FAILED. Re-call extract_invoice with corrections:\n\n" +
    errors.map((e, i) => `${i + 1}. ${e}`).join("\n\n") +
    "\n\nThe schema accepted your output but semantic checks failed. " +
    "Fix the issues above and submit again."
  );
}
