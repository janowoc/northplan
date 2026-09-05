<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Outstanding work

Planned work is tracked as GitHub issues, in the order given in
`docs/roadmap.md`. Simplifications that are decided rather than pending are
in `docs/limitations.md`. This file holds only what is neither: things known
to be missing that are not scheduled and not accepted as permanent.

Nothing here is a parameter value. Where a value is needed, the item says
which source to check, never what the answer is.

---

## Deferred beyond the first version

- **A second province.** `params/province-template.yaml` is the starting
  point and `engine/tax/provincial.py` branches on nothing, so it is a copy
  plus sourcing. `test_every_province_file_has_the_same_shape` becomes
  load-bearing at that moment. Ontario first, because of the surtax and
  health premium hooks it would force into the template.
- **Federally regulated pension jurisdiction.** Needs a file of its own with
  the OSFI LIF table. Decide the code before writing it: `federal` collides
  with the tax file; `pbsa` is clearer.
- **The Alberta LIF greater-of rule.** `ab.yaml` records the flag. Modelling
  it needs the prior year's investment earnings carried per account, credited
  growth only, not closing minus opening. See L27.
- **GIS proper.** The schema is drafted at `params/gis_not_implemented.yaml`.
  It is a step table with a top-up, not a line, and it is computed on the
  prior year's income for a July-to-June benefit year. See L2.
- **More policies.** Dynamic spending rules, a glide path per account, RESP
  capital preservation near enrolment, spousal RRSPs. See L44.
- **Stochastic inflation** and mortality improvement scales. See L6, L8.
