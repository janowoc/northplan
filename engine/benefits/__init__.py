"""Government benefits: CPP, OAS, GIS.

Same shape as ``engine/tax/``: array-valued pure functions over ``(n_paths,)``,
every constant loaded from ``params/``.

Two conventions in this package are easy to get wrong and are stated once here:

- **Monthly vs annual.** CPP and OAS are published as *monthly* amounts. Every
  function below returns an *annual* amount unless its name ends in
  ``_monthly``. The conversion happens once, at the boundary, never twice.
- **Benefit year vs calendar year.** The OAS recovery tax for the benefit
  period running July of year Y to June of year Y+1 is assessed on net income
  from the *prior* calendar year. A function that takes a year must document
  which year it means.
"""
