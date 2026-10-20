/**
 * Step 3 of Exercise 3 — few-shot examples for varied document formats.
 *
 * Demonstrates: teach by example, not by rule. When prose instructions don't
 * generalize (informal currency, narrative receipts, missing fields), show
 * 2-4 examples of the right answer and the model picks up the pattern.
 *
 * Maps to slide 6 ("Few-shot — teach by example, not by rule").
 *
 * The block is appended to the system prompt via pi's before_agent_start
 * hook, so every turn includes these examples in the model's context. Few-
 * shot teaches PATTERNS; format normalization rules teach FORMATTING. We
 * use both in the block below.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export const FEW_SHOT_BLOCK = `
## Structured Extraction — Format Variety Examples

When the user asks you to extract a document, ALWAYS:
1. Read the file using the built-in 'read' tool first.
2. Call 'extract_invoice' with the FULL structured object populated.
3. Set 'source_path' to the value from 'Source path:' or 'custom_id=' in the prompt
   (so a human can review the original document later).
4. Use null for any field genuinely absent from the source. NEVER fabricate.
5. Compute calculated_total yourself from the line items. Do not just copy stated_total.
6. Set conflict_detected=true iff |stated_total - calculated_total| > 0.01.
7. Lower confidence scores when the source is informal, ambiguous, or partial.
8. For documents that are not clearly invoice/receipt/PO (handwritten notes, bank
   statement lines, anything ambiguous), use document_type="other" and put a brief
   description in document_type_detail.

### Example A — narrative reimbursement note (RICH extraction — read carefully)
Source: "Quick reimbursement request. Picked up coffee and pastries at Linden Cafe
yesterday (Feb 14). Came out to forty-two dollars and change. Card on file is the
corp Amex. Vendor is 'Linden Cafe LLC' if you need it for the books. About a
fifteen percent tip included in the total."

Right answer (note every field the source mentions, even informally):
  vendor="Linden Cafe LLC"          # use the explicit form when given
  document_type="receipt"
  invoice_id="UNKNOWN-LINDEN-2026-02-14"
  payment_method="corp Amex"        # captured verbatim from source
  line_items=[{description: "coffee and pastries (combined; per-item prices not in source)",
               quantity: 1, unit_price: 36.52, line_total: 36.52}]
  tip_percentage=0.15               # DECIMAL — 0.15, never 15
  tip_amount=5.48                   # derived: stated_total - subtotal
  subtotal=36.52                    # derived: stated_total / (1 + tip_percentage) = 42 / 1.15
  tax_amount=null                   # not stated
  stated_total=42.00
  calculated_total=42.00            # subtotal + tax + tip = 36.52 + 0 + 5.48
  conflict_detected=false
  confidence: { vendor: 0.95, invoice_id: 0.20, line_items: 0.45, overall: 0.65 }

Key extraction lessons from this example:
- payment_method is captured even when informal ("corp Amex" — leave as-is).
- tip_percentage is a DECIMAL (0.15), never a whole number (15).
- When the source gives tip percentage AND total, DERIVE subtotal and tip_amount.
- Don't split "coffee and pastries" into separate line_items unless per-item
  prices are in the source — splitting without prices is fabrication.
- Lower line_items confidence reflects that the breakdown is approximate.

### Example B — invoice where TOTAL contradicts line items
Source line items sum to $84.44 but TOTAL DUE: $150.00
Right answer:
  stated_total=150.00, calculated_total=84.44, conflict_detected=true
DO NOT silently pick one figure. Surface both. The downstream system will decide.

### Example C — missing fields
If the source omits customer_id, due_date, or invoice_id, return null for those fields
(or a clearly-marked synthesized id like "UNKNOWN-..." with low confidence).
Returning fabricated values is a critical failure mode.

### Example D — peer reimbursement note (3-party transaction)
Source: "sam paid for team lunch yesterday. total ~$87 (might've been 88, hard
to read). my share is around 22 bucks. will venmo by friday — j"

Right answer (note paid_by AND requester_share_amount, both used):
  vendor=null                                 # no business name in source
  document_type="other"
  document_type_detail="peer reimbursement IOU (no receipt attached)"
  invoice_id="UNKNOWN-IOU-team-lunch-2026-02-14"
  payment_method="venmo (settlement; not yet sent)"
  paid_by="Sam"                               # Sam fronted the money, not the requester
  requester_share_amount=22.00                # the partial share being requested
  line_items=[{description: "team lunch (no per-item details)",
               quantity: 1, unit_price: 87.00, line_total: 87.00}]
  subtotal=null, tax_amount=null, tip_amount=null, tip_percentage=null
  stated_total=87.00                          # take the firmer figure; note uncertainty in confidence
  calculated_total=87.00
  conflict_detected=false
  source_path="samples/handwritten-note.txt"
  confidence: { vendor: 0.05, invoice_id: 0.10, line_items: 0.20, overall: 0.20 }

Why every confidence is so low:
- This is a PEER reimbursement, not a VENDOR transaction. Schema partially fits.
- Source itself is uncertain ("~$87 might've been 88") — record 87, low confidence.
- No business name → vendor=null.
- This document SHOULD route to human review (overall < 0.7).
- The teaching point: when paid_by is set, you're outside the vendor-centric
  schema's comfort zone. Surface what you can; let humans handle the rest.
`;

/** Wire the few-shot block into every turn's system prompt. */
export function installFewShot(pi: ExtensionAPI): void {
  pi.on("before_agent_start", async (event, _ctx) => {
    return { systemPrompt: event.systemPrompt + "\n\n" + FEW_SHOT_BLOCK };
  });
}
