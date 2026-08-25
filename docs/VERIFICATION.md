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
- **State the period.** A benefit amount is monthly, a tax amount is annual,
  and the engine steps monthly, so a row that does not say which is not a
  verified row. `Maximum OAS monthly pension, 2026` is a row; `Maximum OAS
  pension, 2026` is an invitation to a factor-of-twelve error.
- **Source URL** — the primary source (CRA, Service Canada, provincial treasury,
  the Income Tax Act). A secondary source such as TaxTips.ca is acceptable for
  cross-checking a golden value, but say which it is.
- **Date checked** — the date a human opened that URL and compared it to the
  value in `params/`. Not the date the row was written.
- **Verified by** — a person.
- **Test** — the specific test node that would fail if the value were wrong.
  A row with no test is an unverified row.

## Rules that are not numbers

Some things the engine depends on are calendar rules rather than dollar
amounts, and they need rows here exactly as much as the amounts do. A monthly
timestep makes several of them load-bearing that an annual one could ignore:

- The month a balance owing for a tax year comes due.
- The month each benefit year begins, and the offset from a benefit year to the
  calendar year whose net income governs it.
- How often an amount is indexed — annually for CPP and the tax brackets,
  quarterly for OAS and GIS — and the lag between the close of the CPI window
  and the adjustment taking effect. Together with the scenario's inflation
  assumption these fix the constant amount by which an indexed benefit sits
  below its published real value, which is permanent and is always in the
  optimistic direction. The engine deliberately does not model the oscillation
  around that constant; that choice belongs in a row here too, since it is an
  approximation and not a statutory fact.
- The age, in months, at which each benefit may first start and at which its
  adjustment is neutral.
- The deadline for converting an RRSP, and the date the RRIF minimum's balance
  and age are read from.
- The assumption used to convert an annual mortality rate into a monthly
  hazard. Not statutory, but a modelling choice the results depend on, so it is
  recorded here alongside the table it applies to and the test that pins it.

Cite the rule the same way an amount is cited: primary source, date a human
checked it, and the test that would fail if it were wrong.

## Re-verification

Parameters change annually and are sometimes revised mid-year. When a value is
re-checked, update the existing row's date rather than adding a second row for
the same parameter and year. A new tax year gets new rows.
