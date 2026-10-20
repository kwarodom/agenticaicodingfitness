/**
 * Step 1 of Exercise 3 — JSON schema design.
 *
 * Demonstrates: required vs nullable, enum + "other" + detail, dual extraction.
 * Maps to slide 4 ("A schema is a SQL CREATE TABLE for the LLM").
 *
 * The shape defined here is what the LLM sees in the tool definition.
 * Schema enforcement eliminates JSON syntax errors but NOT semantic errors —
 * that's what step2-validation.ts is for.
 *
 * Three patterns to teach from this file:
 *   1. NOT NULL forces fabrication → use sparingly (vendor, invoice_id only).
 *   2. Nullable is the model's "I don't know" escape hatch (customer_id, due_date).
 *   3. Enum + "other" + detail string lets the schema discover new categories
 *      from production data instead of forcing them into a wrong bucket.
 *   4. Dual extraction (stated + calculated + conflict_detected) FORCES the
 *      model to surface arithmetic disagreements it would otherwise hide.
 */

import { Type, type Static } from "typebox";

export const LineItem = Type.Object({
  description: Type.String(),
  quantity:    Type.Number({ minimum: 0 }),
  unit_price:  Type.Number({ minimum: 0 }),
  line_total:  Type.Number({ minimum: 0 }),
});

export const FieldConfidence = Type.Object({
  vendor:     Type.Number({ minimum: 0, maximum: 1 }),
  invoice_id: Type.Number({ minimum: 0, maximum: 1 }),
  line_items: Type.Number({ minimum: 0, maximum: 1 }),
  overall:    Type.Number({ minimum: 0, maximum: 1 }),
});

export const ExtractedInvoice = Type.Object({
  // VENDOR & INVOICE_ID — both NULLABLE. Originally these were required, but
  // the peer-reimbursement case (Example D in few-shot) revealed: real "buying
  // documents" sometimes have no business name (handwritten IOUs) or no
  // invoice number (counter receipts). The slide-4 lesson applies recursively
  // to our own schema: nullable beats required when the field can plausibly
  // be absent. The few-shot examples teach the model to synthesize identifiers
  // ("UNKNOWN-...") with low confidence when useful, or null when not.
  vendor:     Type.Union([Type.String(), Type.Null()], {
    description: "Vendor / supplier name. Null when no business name is in the source.",
  }),
  invoice_id: Type.Union([Type.String(), Type.Null()], {
    description:
      "Invoice number, ID, or document reference. May be a synthesized identifier " +
      "like 'UNKNOWN-LINDEN-2026-02-14' (with low confidence) or null.",
  }),

  // NULLABLE — source documents legitimately omit these. `Union([T, Null])` is
  // the LLM equivalent of "not present in source." Single highest-leverage
  // anti-hallucination control in this whole schema.
  customer_id: Type.Union([Type.String(), Type.Null()], {
    description: "Customer ID if printed. Null if absent.",
  }),
  due_date: Type.Union([Type.String(), Type.Null()], {
    description: "ISO 8601 (YYYY-MM-DD). Null if no due date stated.",
  }),

  // ENUM + "other" + detail — extensible categorization. SQL enums are closed
  // and reject unknowns; this pattern discovers them.
  document_type: Type.Union([
    Type.Literal("invoice"),
    Type.Literal("receipt"),
    Type.Literal("purchase_order"),
    Type.Literal("other"),
  ]),
  document_type_detail: Type.Union([Type.String(), Type.Null()], {
    description: "Required when document_type='other' (the actual type name). Null otherwise.",
  }),

  // PROVENANCE — where this extraction came from. Distinct from invoice_id
  // ("what the document calls itself"). Populated from the prompt context;
  // see step3-few-shot.ts for the instruction. Enables source-vs-output
  // side-by-side review via /export-review (step 5).
  source_path: Type.Union([Type.String(), Type.Null()], {
    description:
      "File path or batch custom_id this extraction came from. Use the value " +
      "from 'Source path:' or 'custom_id=' in the prompt. Null only if no source identifier was given.",
  }),

  line_items: Type.Array(LineItem),

  // PAYMENT METHOD — the INSTRUMENT used to pay. Verbatim from source.
  // ("corp Amex", "Visa ****4421", "cash", "venmo"). Null if not mentioned.
  payment_method: Type.Union([Type.String(), Type.Null()], {
    description:
      "Payment instrument as stated in the source (e.g., 'corp Amex', 'Visa ****1234', " +
      "'cash', 'venmo'). Use the source's wording. Null if not mentioned.",
  }),

  // PAID BY — the PERSON or ENTITY who fronted the money. Distinct from
  // payment_method (instrument). Useful for peer reimbursement notes where
  // someone else paid and the requester is asking to be reimbursed for their
  // share. Null when the requester paid directly or the source doesn't say.
  paid_by: Type.Union([Type.String(), Type.Null()], {
    description:
      "Person or entity who actually paid (e.g., 'Sam', 'the company', " +
      "'Jordan personally'). Different from payment_method. Null when the " +
      "requester paid directly or the source doesn't specify.",
  }),

  // REQUESTER SHARE — when the requester is asking for partial reimbursement
  // (e.g., 'my share is ~$22 of the $87 total'). Captures peer-split cases.
  // Null when the request is for the full stated_total.
  requester_share_amount: Type.Union([Type.Number(), Type.Null()], {
    description:
      "Amount the requester is seeking, when different from stated_total " +
      "(e.g., 'my share is $22'). Null when the request is for the full total.",
  }),

  // BREAKDOWN — subtotal, tax, tip. All nullable: receipts are inconsistent
  // about which they break out. When the source gives a tip percentage and a
  // total, the model can DERIVE tip_amount and subtotal arithmetically.
  subtotal: Type.Union([Type.Number(), Type.Null()], {
    description:
      "Pre-tax, pre-tip amount. Null if not stated AND not derivable. " +
      "Derivable example: stated_total=42 with 15% tip → subtotal=42/1.15≈36.52.",
  }),
  tax_amount: Type.Union([Type.Number(), Type.Null()], {
    description: "Sales tax amount, when separately broken out. Null otherwise.",
  }),
  tip_amount: Type.Union([Type.Number(), Type.Null()], {
    description: "Tip / gratuity in same currency as totals. Null if not present.",
  }),
  tip_percentage: Type.Union(
    [Type.Number({ minimum: 0, maximum: 1 }), Type.Null()],
    {
      description:
        "Tip as a decimal fraction (0.15 for 15%, 0.20 for 20%). " +
        "NOT a whole number. Null if no percentage is stated.",
    },
  ),

  // SELF-CORRECTION via dual extraction. Schema FORCES the comparison the
  // model would otherwise resolve in favor of one figure or the other.
  stated_total:      Type.Number({ description: "Total exactly as printed on the document" }),
  calculated_total:  Type.Number({ description: "Sum of line_total values + tax_amount + tip_amount (your computation)" }),
  conflict_detected: Type.Boolean({
    description: "True iff |stated_total - calculated_total| > 0.01",
  }),

  confidence: FieldConfidence,
});

export type Extraction = Static<typeof ExtractedInvoice>;
