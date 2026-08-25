# Verification ledger

The audit record for every tax parameter, benefit rule, and account rule the
engine relies on. This is distinct from the issue tracker: an issue tracks work
to be done, a row here records that a number was checked against an authoritative
source by a human on a specific date.

**Rows are filled in by the human, not by Claude.** An agent may point out that a
row is missing. It may not add one, and it may not fill in a "Date checked" or a
"Verified by".

A parameter is not considered verified until it has a row here with all six
columns populated, and the named test passes.

| Parameter / rule | Source URL | Date checked | Verified by | Test |
|---|---|---|---|---|
| Federal basic personal amount, 2026 (worked example — replace) | https://www.canada.ca/en/revenue-agency/services/tax/individuals/frequently-asked-questions-individuals/canadian-income-tax-rates-individuals-current-previous-years.html | 2026-01-15 | Jan Owoc | `tests/golden/test_federal_tax.py::test_bpa_2026` |

## Conventions for this table

- **Parameter / rule** — name it the way `params/` names it, with the tax year.
  One row per parameter, not one row per file.
- **Source URL** — the primary source (CRA, Service Canada, provincial treasury,
  the Income Tax Act). A secondary source such as TaxTips.ca is acceptable for
  cross-checking a golden value, but say which it is.
- **Date checked** — the date a human opened that URL and compared it to the
  value in `params/`. Not the date the row was written.
- **Verified by** — a person.
- **Test** — the specific test node that would fail if the value were wrong.
  A row with no test is an unverified row.

## Re-verification

Parameters change annually and are sometimes revised mid-year. When a value is
re-checked, update the existing row's date rather than adding a second row for
the same parameter and year. A new tax year gets new rows.
