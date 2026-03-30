# Annotation Guidelines: Invoice Field Extraction

## General Rules
- Annotate at span level and keep boundaries as tight as possible.
- Include complete field values when they span multiple tokens or lines.
- Exclude surrounding labels, punctuation, and explanatory text unless they are part of the value.
- When in doubt, flag the sample for review instead of guessing.

## Field Rules

### INVOICE_ID
- Description: Primary invoice reference or identifier.
- Requirement: Required
- Cardinality: Single span
- Examples:
  - INV-2026-001
  - Invoice #12345
  - Facture N 2026-11
- Include:
  - Include the identifier value and business prefix when it is part of the ID.
  - Keep separator characters such as hyphens or slashes when present.
- Exclude:
  - Exclude generic labels like Invoice, Ref, or Number when they are not part of the ID.

### INVOICE_DATE
- Description: Document issue date.
- Requirement: Required
- Cardinality: Single span
- Examples:
  - 2026-03-29
  - 29/03/2026
  - March 29, 2026
- Include:
  - Include the full date span including day, month, and year.
- Exclude:
  - Exclude nearby labels such as Date or Issue Date.

### DUE_DATE
- Description: Invoice payment deadline.
- Requirement: Optional
- Cardinality: Single span
- Examples:
  - Due 15/04/2026
  - Payment before April 15, 2026
- Include:
  - Annotate only the due date value.
- Exclude:
  - Exclude payment terms text that does not belong to the date itself.

### VENDOR_NAME
- Description: Supplier or issuing company name.
- Requirement: Required
- Cardinality: Single span
- Examples:
  - Acme Supplies LLC
  - Globex SARL
- Include:
  - Include the full legal or displayed company name.
  - Include suffixes such as LLC, SARL, or Inc when shown.
- Exclude:
  - Exclude addresses, tax IDs, and contact details unless merged into the visible company name.

### TOTAL_AMOUNT
- Description: Final invoice total payable.
- Requirement: Required
- Cardinality: Single span
- Examples:
  - 1,240.00 USD
  - $1,240.00
  - TND 480.500
- Include:
  - Include the amount and currency symbol or code when present.
- Exclude:
  - Exclude descriptive labels such as Total or Net Amount.
  - Exclude subtotals or tax-only amounts.

### TAX_AMOUNT
- Description: Tax value applied to the invoice total.
- Requirement: Optional
- Cardinality: Single span
- Examples:
  - VAT 19%: 235.60
  - Tax 14.000 TND
- Include:
  - Annotate the numeric tax amount and its currency when present.
- Exclude:
  - Exclude the tax rate unless it is embedded in the same token span and cannot be separated cleanly.

## Quality Control
- Review 10-20 overlapping documents across annotators each week.
- Target Cohen's Kappa of 0.80 or higher for all required fields.
- Update this guideline file whenever a new edge case changes annotation behavior.