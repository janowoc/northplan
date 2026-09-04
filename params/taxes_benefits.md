# How Canadian Taxes, Registered Accounts and Public Benefits Work — Engine Reference

**Purpose.** This document explains *how* the main Canadian personal-tax rules, registered accounts and public benefits work, in enough detail to (a) know which parameters must exist in the engine's YAML parameter files, (b) implement the engine logic, and (c) write meaningful tests. It is deliberately a *narrative* reference: the exact numbers live in the YAML files, which carry a source URL and a check date for every figure.

**Conventions.**
- Illustrative values are the best available figures as of **September 3, 2026**. They are there so the mechanics are concrete, not so the engine can read them from here.
- 🔄 marks a figure that is **indexed or re-announced every year (or quarter)** and must be refreshed in the YAML.
- 📌 marks a figure that is **fixed by statute** and only changes when Parliament or a legislature amends the law (rare, but it happens).
- Each section ends with a **Parameters** table (what belongs in YAML, how often it changes, which source document publishes it) and a **Sources** list naming the primary document.
- "Prior-year" always means the *calendar* year before the one being simulated. Canada's personal-tax year is the calendar year.

**Suggested reading order for a new contributor:** §1 → §2 → §3 → §5 (RRSP family) → §6 (TFSA) → §9 (CPP) → §10 (OAS/GIS) → §12 (interactions). RESP (§7) and FHSA (§8) matter for the accumulation use case.

---

## Contents

1. How the Canadian personal-tax system is structured
2. Federal income tax
3. Provincial and territorial income tax (including atypical provinces)
4. How different kinds of income are taxed
5. RRSP → RRIF, LIRA → LIF, and workplace pensions
6. TFSA
7. RESP
8. FHSA
9. CPP and QPP
10. OAS, GIS and the Allowances
11. Other income-tested federal benefits
12. How the pieces interact: splitting, sequencing, effective marginal rates
13. Death, spousal rollovers and probate
14. Annual maintenance calendar

---

## 1. How the Canadian personal-tax system is structured

**Two governments, one return (except Quebec).** Every resident pays *federal* income tax to the Canada Revenue Agency (CRA) and *provincial or territorial* income tax to the jurisdiction where they lived on **December 31** of the tax year. Outside Quebec both taxes are computed on the same federal return (the T1) and collected by CRA under Tax Collection Agreements; the province's tax is computed on the provincial "428" schedule (ON428, AB428, …). Quebec administers its own tax and its residents file a **separate Quebec return (TP-1)** with Revenu Québec in addition to the federal T1.

**Individual, not joint, filing.** Each person files their own return and is taxed on their own income. There is no household or joint return. Everything in this document that looks like "income splitting" (spousal RRSPs, pension income splitting, spousal-credit transfers, CPP pension sharing) exists precisely because filing is individual — the engine must therefore compute **two separate tax returns for a couple** and treat *who* reports a dollar of income as a first-class decision variable.

**Progressive brackets, marginal rates.** Both levels use bracket schedules: each rate applies only to the slice of taxable income inside its band. The "combined marginal rate" is just the federal rate plus the provincial rate (plus any provincial surtax effect) at a given income. Most brackets, credits and thresholds are **indexed to CPI once a year** — the federal factor for 2026 is 🔄 **2.0%** (2025: 2.7%); each province publishes its own factor (§3).

**Deductions vs. credits.**
- A **deduction** reduces *taxable income* and is worth your marginal rate (RRSP contributions, pension-splitting deduction, union dues, carrying charges).
- A **non-refundable credit** reduces *tax payable* and is worth a fixed rate — federally the **lowest bracket rate** (🔄 14% for 2026) times the credit amount; provincially the province's lowest rate. Non-refundable credits cannot take tax below zero. The important ones for a life-cycle simulator: the basic personal amount, the age amount, the pension income amount, the spousal amount, the disability amount, medical expenses, charitable donations, and CPP/EI contributions.
- A **refundable credit** is paid even with no tax owing (Canada Workers Benefit, GST/HST credit, Canada Child Benefit, provincial low-income credits) — these behave like benefits and are income-tested (§11).

**Net income vs. taxable income.** Almost every benefit test (OAS recovery tax, GIS, age amount, CCB, GST credit) is run on **net income (line 23600)** or a close variant, *not* on taxable income. Net income is total income minus RRSP/FHSA/pension deductions, child-care and moving expenses, carrying charges, etc.; taxable income is net income minus a few further deductions (capital-gains deduction, net capital losses of other years, northern residents). The engine needs both quantities per person per year.

**Withholding is not tax.** Employers, pension payers and financial institutions withhold at source (payroll tables, flat percentages on RRSP/RRIF lump sums). Withholding is a prepayment reconciled on the return; the *lifetime tax* answer depends only on the return, so the engine can model withholding as a cash-timing effect and, for most purposes, ignore it.

**Instalments.** Retirees whose net tax owing exceeds **📌 $3,000** ($1,800 for Quebec residents) in the current and either of the two prior years must pay quarterly instalments (March 15, June 15, September 15, December 15). Cash-flow only; no effect on total tax.

---

## 2. Federal income tax

### 2.1 Brackets

Five brackets, thresholds indexed annually 🔄. The lowest rate was cut from 15% to **14%** effective July 1, 2025 (blended 14.5% for the 2025 year; 14% for 2026 onward).

| Rate | 2025 taxable income | 2026 taxable income |
|---|---|---|
| 14.5% (2025) / **14%** (2026) | first $57,375 | first $58,523 |
| 20.5% | $57,375 – $114,750 | $58,523 – $117,045 |
| 26% | $114,750 – $177,882 | $117,045 – $181,440 |
| 29% | $177,882 – $253,414 | $181,440 – $258,482 |
| 33% | over $253,414 | over $258,482 |

The statutory 4th-bracket rate is 29%. Because the enhanced basic personal amount (below) is phased out across exactly that bracket, the *effective* marginal rate there is ≈ 29.3%. Model the phase-down explicitly rather than hard-coding 29.3%.

### 2.2 Basic personal amount (BPA)

Everyone gets a credit on the BPA. Since 2020 the BPA has two pieces: a **base amount** and an **enhancement** that is reduced *linearly* to zero as net income rises from the bottom to the top of the 4th bracket.

- 2026 🔄: full BPA **$16,452** at net income ≤ $181,440, falling to **$14,829** at net income ≥ $258,482 (2025: $16,129 → $14,538 over $177,882 → $253,414).
- Credit value = BPA × lowest rate (2026: $16,452 × 14% = $2,303).
- Because it's a credit, not a deduction, income up to the BPA is effectively federally tax-free.

### 2.3 Other credits a retirement/accumulation model needs

| Credit | How it works | 2026 value 🔄 |
|---|---|---|
| **Age amount** | For anyone **65+ at Dec 31**. Reduced by **15%** of net income above a threshold; nil above roughly $107k. Unused portion transferable to a spouse. | max **$9,208**, reduced above **$46,432** |
| **Pension income amount** | 📌 On the first **$2,000** of *eligible pension income* (see §5.4 for what qualifies — RRIF income only counts from age 65). Not indexed. | $2,000 |
| **Spousal / common-law partner amount** | Equal to the BPA, reduced dollar-for-dollar by the spouse's net income. | up to $16,452 |
| **Canada employment amount** | For employment income only (not pensions). | $1,501 |
| **CPP/QPP and EI contributions** | Credit on the base contributions; the *enhanced* CPP portion (first and second additional contributions) is a **deduction** instead. | actual amounts |
| **Disability amount** | Requires an approved T2201. | $10,341 |
| **Medical expenses** | Expenses above the lesser of 3% of net income or a dollar threshold. | threshold $2,890 |
| **Charitable donations** | Two-tier: lowest rate on the first $200, then 29% (33% to the extent of income taxed at 33%). *Budget 2025 kept the first-tier rate at 15% through 2030.* | 15%/29%/33% |

### 2.4 Alternative Minimum Tax (AMT)

A parallel calculation that mostly bites in years with very large capital gains or donations of securities. Reformed in 2024: rate **📌 20.5%** on "adjusted taxable income" above an exemption equal to the start of the 4th bracket (🔄 **$181,440** for 2026); 100% of capital gains included; only 50% of most non-refundable credits allowed. AMT paid is recoverable against regular tax over the following **7 years**. A life-cycle engine can usually ignore AMT unless it models lump-sum property or business sales.

### 2.5 Parameters — federal tax

| Parameter | Illustrative value (2026) | Changes | Source document |
|---|---|---|---|
| Bracket thresholds (4) | $58,523 / $117,045 / $181,440 / $258,482 | 🔄 annually (Nov) | CRA — *Indexation adjustment for personal income tax and benefit amounts* |
| Bracket rates (5) | 14 / 20.5 / 26 / 29 / 33 % | 📌 legislation | CRA — *Canadian income tax rates for individuals – current and previous years* |
| BPA max / min | $16,452 / $14,829 | 🔄 | CRA indexation page; Form TD1 |
| Age amount, threshold | $9,208 / $46,432 | 🔄 | CRA indexation page |
| Pension income amount | $2,000 | 📌 | Income Tax Act s.118(3) |
| Canada employment amount | $1,501 | 🔄 | CRA indexation page |
| Medical threshold | $2,890 | 🔄 | CRA indexation page |
| AMT rate / exemption | 20.5% / $181,440 | 📌 / 🔄 | CRA — *Minimum tax* (Form T691) |
| Indexation factor | 2.0% (2026) | 🔄 | CRA indexation page |
| Instalment threshold | $3,000 ($1,800 QC) | 📌 | CRA — *Paying your income tax by instalments* |

**Sources:** CRA, *Indexation adjustment for personal income tax and benefit amounts* (published each November); CRA, *Canadian income tax rates for individuals – current and previous years*; CRA, *Form TD1 Personal Tax Credits Return* (annual); CRA, *Income Tax Folio S1-F4-C1 (Basic personal and dependant credits)*.

---

## 3. Provincial and territorial income tax

### 3.1 The common pattern

Every province and territory levies its own income tax on the **same taxable income** as the federal return (Quebec computes its own taxable income, which differs slightly). The standard structure is:

1. A bracket schedule (2 to 8 brackets) with its own thresholds, indexed by the province's own factor (usually its provincial CPI; some cap or freeze it).
2. Its own **basic personal amount** and non-refundable credits, valued at the province's *lowest* rate. Provincial age amounts, pension amounts, spousal amounts and disability amounts exist everywhere but with different dollar values and, often, different phase-out thresholds.
3. Its own **dividend tax credit rates** (§4.2) — the federal gross-up is common, the provincial credit is not.
4. Frequently a **low-income tax reduction** that wipes out provincial tax below some income (BC, ON, NB, NS, PEI, NL, MB, SK all have one in some form) — relevant for low-income retirees and for students receiving RESP payments.

The engine should treat a jurisdiction as: `brackets[]`, `bpa`, `credit_rate`, `age_amount + threshold`, `pension_amount`, `dividend_credit_rates`, `indexation_factor`, plus a small set of *jurisdiction-specific hooks* for the exceptions below. An unrecognised province code should fail loudly (or fall back to Ontario with a warning) — never silently to zero provincial tax.

**2026 indexation factors 🔄** (from CRA T4032 tables / KPMG): federal 2.0%; Ontario 1.9%; Quebec 2.05%; Alberta 2.0% (capped); BC 2.2%; SK, NB, YT, NT, NU 2.0%; **Manitoba 0% (paused)**; NS, PEI, NL follow their own CPI. **No province changed a top rate for 2026.**

### 3.2 Combined top marginal rates (ordinary income, 2025/2026)

| Jurisdiction | Combined top rate | Where it starts |
|---|---|---|
| Newfoundland & Labrador | 54.80% | ≈ $1.1M (NL top bracket) |
| Nova Scotia | 54.00% | $258,482 |
| Ontario | 53.53% | $258,482 |
| British Columbia | 53.50% | $258,482 |
| Quebec | 53.31% | $258,482 |
| New Brunswick | 52.50% | $258,482 |
| Prince Edward Island | 51.75% | $258,482 |
| Manitoba | 50.40% | $258,482 |
| Alberta | 48.00% | ≈ $363k (AB top bracket) |
| Yukon | 48.00% | $500,000 (YT top bracket) |
| Saskatchewan | 47.50% | $258,482 |
| Northwest Territories | 47.05% | $258,482 |
| Nunavut | 44.50% | $258,482 |

Use these as **test assertions** (one per jurisdiction) rather than as inputs — recompute from the bracket YAML and the assertion will catch a stale schedule.

### 3.3 Provinces with atypical treatment (beyond BPA and brackets)

**Ontario — surtax, health premium, frozen upper brackets.**
- Five brackets 5.05% / 9.15% / 11.16% / 12.16% / 13.16%. For 2026 🔄 the first two thresholds are $53,891 and $107,785; the **$150,000 and $220,000 thresholds are not indexed** (frozen since 2014).
- **Surtax** is levied on *Ontario tax payable* (after non-refundable credits), not on income: **20%** of Ontario tax above 🔄 **$5,818** plus a further **36%** of Ontario tax above 🔄 **$7,446** (2026; 2025: $5,710 / $7,307). Above the second trigger every dollar of Ontario tax costs $1.56, which turns the 13.16% headline rate into an effective **20.53%**, hence the 53.53% combined top rate. The surtax starts biting around $90k of taxable income — long before the top bracket — so Ontario's *effective* marginal schedule has more steps than its bracket table. Follow the ON428 line order exactly (non-refundable credits → surtax → dividend tax credit → Ontario tax reduction); the ordering changes the answer for dividend-heavy retirees, so it should be a test case.
- **Ontario Health Premium**: an extra levy of $0–$900 keyed to *taxable income* (starts above $20,000, reaches the $900 maximum above $200,600; thresholds 📌 never indexed). Collected on the return, so it is part of the tax bill even for retirees with no employment income.
- Ontario BPA 2026 🔄 **$12,989** (5.05% credit); Ontario age amount reduced above 🔄 $47,210; Ontario has its own **Ontario Tax Reduction** and **LIFT credit** for low incomes.
- No provincial pension-splitting quirks; Ontario follows the federal T1032 election.

**Quebec — a genuinely separate system.**
- Separate TP-1 return; four brackets 🔄 **14% to $54,345 / 19% to $108,680 / 24% to $132,245 / 25.75% above** (2026, indexed 2.05%); BPA 🔄 **$18,952** credited at 14%, with **no** high-income phase-down.
- **Federal abatement 📌 16.5%.** A Quebec resident's *basic federal tax* is reduced by 16.5% (line 44000) because Ottawa vacates that tax room to Quebec, which funds the equivalent programs itself. Without applying the abatement, Quebec combined rates come out ~9 points too high. Top combined rate: 33% × (1 − 0.165) + 25.75% = **53.31%**.
- **QPP instead of CPP**, **QPIP instead of the EI parental portion** (lower EI premium rate for Quebec employees). See §9.
- **RRSP/RRIF lump-sum withholding differs:** federal withholding is only 5% / 10% / 15% (instead of 10/20/30), **plus Quebec withholding of 14%**, so the *combined* prepayment is 19% / 24% / 29% — slightly less than the rest of Canada at the low tier, about the same at the top. Withholding is a prepayment; lifetime tax is unaffected.
- **Health Services Fund (HSF) individual contribution:** 1% of income that is *not* subject to employer HSF (i.e. pension, RRIF, investment and self-employment income — OAS and employment income are excluded), above an exemption (≈ $18,130 for 2025 🔄) with a $150 plateau in the middle band and a **📌 $1,000 annual cap** — the cap is reached at exactly (third threshold + $85,000). A retiree melting down an RRSP pays it. It is reported on the TP-1.
- **Quebec Prescription Drug Insurance premium (RAMQ):** anyone not covered by a private drug plan — most retirees — pays an income-tested premium on the TP-1, up to roughly $750–$800 🔄 per adult per year. Easy to forget and material for a Quebec retiree's tax bill.
- **Senior credits are pooled:** the Quebec age amount (🔄 $3,986), retirement-income amount (🔄 $3,541) and living-alone amount (🔄 $2,172) are added together and reduced by **18.75%** of *family* net income above 🔄 $42,955 (2026). Note that QPP/CPP/OAS do **not** qualify as retirement income for the credit; RRIF and annuity income do.
- **Pension income splitting for Quebec purposes is only allowed if the transferring spouse is 65 or older** (no under-65 RPP exception, unlike the federal rule in §12.1).
- Quebec sets its **own dividend tax credit rates** (≈ 11.7% eligible / ≈ 3.42% non-eligible of the grossed-up amount 🔄) and its own RESP incentive (QESI, §7).
- **Quebec LIFs have no maximum withdrawal from age 55** (since January 1, 2025) — see §5.3.
- Quebec's own **Solidarity Tax Credit** replaces the federal-style provincial low-income credits.

**Manitoba — frozen indexation and a phased-out BPA.** Bracket thresholds and the 🔄 **$15,780** BPA have been frozen since 2025 (no indexation). From 2025 the BPA is **phased out for net income between $200,000 and $400,000** (nil above $400k), so Manitoba's effective marginal rate in that range exceeds the 17.4% headline. Top combined 50.40%.

**Alberta — low, flat-ish, capped indexation.** Six brackets from **8%** (first ≈ $60,000, introduced 2025) to 15%; a high BPA (🔄 $22,769 for 2026); no surtax, no health premium. Brackets and credits are indexed by Alberta CPI **capped at 2.0%**. Top combined 48%.

**Saskatchewan — three brackets, rising BPA.** 10.5% / 12.5% / 14.5%; BPA being increased by $500 a year through 2028 (🔄 $20,381 for 2026). Top combined 47.5%.

**British Columbia — many brackets, top rate 20.5%.** Seven brackets from 5.06% to 20.5% (top starts ≈ $259k); BPA 🔄 $13,216 (2026); a BC **tax reduction credit** for low incomes; no health premium since MSP was abolished in 2020. Top combined 53.5%.

**Nova Scotia — recently began indexing; BPA supplement.** Brackets indexed only since 2025. BPA $11,744 plus an extra up to $3,000 for taxable income ≤ $25,000, phased out by $75,000. Top combined 54%.

**Prince Edward Island — surtax abolished.** PEI's 10% surtax was eliminated for 2024; now a plain five-bracket schedule (9.5% → 19%). Top combined 51.75%.

**New Brunswick / Newfoundland & Labrador.** Standard structure; both have low-income tax reductions. NL has eight brackets with the top (21.8%) starting above ≈ $1.1M, giving Canada's highest combined rate, 54.8%.

**Territories.** Yukon mirrors the federal BPA (including the high-income phase-down) and has a 15% bracket above $500,000. Northwest Territories and Nunavut have the lowest rates (Nunavut 4% to 11.5%; combined top 44.5%) and residents may claim the **northern residents deduction**.

### 3.4 Parameters — provincial

Per jurisdiction (13 YAML blocks): bracket thresholds and rates 🔄; BPA 🔄; lowest-rate credit factor; age amount and threshold 🔄; pension amount; dividend tax credit rates (eligible / non-eligible) 🔄; indexation factor 🔄; low-income reduction parameters 🔄; **hooks:** Ontario surtax triggers 🔄 and health-premium schedule 📌; Quebec abatement 📌, HSF thresholds 🔄, RAMQ premium max 🔄, senior-credit pool and reduction threshold 🔄; Manitoba BPA phase-out range 📌.

**Sources:** CRA, *T4032 Payroll Deductions Tables* (one per province; the January edition states the year's brackets, BPA and indexation factor); CRA, *Form 428* for each province (e.g. ON428, AB428) and the *Provincial and territorial tax and credits for individuals* page; Ontario Ministry of Finance, *Personal Income Tax* page (surtax and health premium); Revenu Québec, *Income tax rates* and *Contribution to the health services fund*; Ministère des Finances du Québec, *Parameters of the Personal Income Tax System* (published each November/December); Régie de l'assurance maladie du Québec, *Premium amounts*; KPMG *Canadian Personal Tax Tables* and TaxTips.ca as convenient secondary cross-checks.

---

## 4. How different kinds of income are taxed

The engine must tag every cash flow with an income *type*, because the tax treatment differs sharply.

### 4.1 Fully taxable (100% inclusion)
Employment income, self-employment income, **interest**, foreign income (grossed up to include any foreign withholding, with a foreign tax credit), rental income, **CPP/QPP**, **OAS**, employer pension income, **RRSP/RRIF/LIF withdrawals**, RESP Educational Assistance Payments (taxed to the student), annuity income from registered funds. Interest and foreign dividends are the least tax-efficient income to hold in a non-registered account.

### 4.2 Canadian dividends — gross-up and dividend tax credit (DTC)
Dividends from Canadian corporations are "grossed up" to approximate the pre-tax corporate profit, tax is computed on the grossed-up amount, then a credit offsets the corporate tax already paid (integration).

| | Gross-up 📌 | Federal DTC (of grossed-up amount) 📌 | Typical source |
|---|---|---|---|
| **Eligible** dividends | 38% | 15.0198% | public companies, most Canadian dividend ETFs |
| **Non-eligible** dividends | 15% | 9.0301% | small-business (CCPC) income |

Each province adds its own DTC (§3). Two consequences the engine must capture: (1) at low incomes the combined effective rate on eligible dividends is **negative** in several provinces; (2) the *grossed-up* amount — 1.38× the cash — is what enters net income for OAS-recovery, GIS and age-amount tests, so $10,000 of eligible dividends costs a GIS recipient $13,800 of test income. This is why dividend income is often *worse* than capital gains for benefit-tested retirees.

### 4.3 Capital gains
- **Inclusion rate 📌 50%**: half the gain is added to income, taxed at the marginal rate. The proposed increase to two-thirds on gains above $250,000 was deferred (January 2025) and **cancelled (March 21, 2025)** — it never became law. Half of capital *losses* offsets taxable gains; net losses carry back 3 years and forward indefinitely against gains only.
- Gains are realised only on **disposition** (sale, gift, deemed disposition on death or emigration), so a non-registered portfolio needs **adjusted cost base (ACB)** tracking per holding; ETF reinvested distributions and return of capital adjust ACB.
- **Superficial-loss rule 📌:** a loss is denied if the same property is repurchased (by you, your spouse, or your RRSP/TFSA) within 30 days before or after the sale.
- **Principal residence exemption**: the gain on a designated principal residence is exempt (one per family unit per year; must be reported since 2016).
- **Lifetime capital gains exemption (LCGE)** on qualified small-business shares and farm/fishing property: 📌 **$1.25 million** since June 25, 2024, indexed from 2026 (🔄 ≈ $1,275,000).
- Only the *included* half enters net income for benefit tests, which makes realised capital gains the most benefit-friendly taxable income.

### 4.4 Return of capital and tax-deferred distributions
ROC distributions (common from REITs and some income ETFs) are not taxed when received; they reduce ACB, deferring tax into a future capital gain.

### 4.5 Parameters — income types

| Parameter | Value | Changes | Source document |
|---|---|---|---|
| Capital-gains inclusion rate | 50% | 📌 | Income Tax Act s.38; Finance Canada news releases |
| Eligible / non-eligible gross-up | 38% / 15% | 📌 | ITA s.82(1) |
| Federal DTC rates | 15.0198% / 9.0301% | 📌 | CRA — *Line 40425 – Federal dividend tax credit* |
| Provincial DTC rates | per province | 🔄 | each province's Form 428 / provincial finance page |
| LCGE | ≈ $1,275,000 (2026) | 🔄 | CRA — *Line 25400 – Capital gains deduction* |
| Superficial-loss window | 30 days | 📌 | ITA s.54 |

**Sources:** CRA, *Line 12700 – Taxable capital gains* and *Guide T4037 Capital Gains*; CRA, *Line 40425 – Federal dividend tax credit*; Department of Finance Canada, *Government of Canada announces deferral in implementation of change to capital gains inclusion rate* (Jan 31, 2025) and the March 21, 2025 cancellation release; CRA, *Line 25400 – Capital gains deduction*.

---

## 5. RRSP → RRIF, LIRA → LIF, and workplace pensions

These accounts are one family: money goes in pre-tax (or is transferred from a pension), grows untaxed, and **every dollar that comes out is fully taxable ordinary income** in the year it is withdrawn. The only questions are *how much room* goes in, *when* it must come out, and *who* is taxed on it.

### 5.1 RRSP (Registered Retirement Savings Plan)

**Room.** New room each year = **📌 18%** of the *prior year's* **earned income** (employment, self-employment, net rental income — not investment income, not pension income), capped at the annual **RRSP dollar limit** 🔄 (**$32,490** for 2025, **$33,810** for 2026; the 2027 limit equals the 2026 money-purchase limit, $35,390), **minus** the prior year's **pension adjustment (PA)** from any workplace pension (§5.5), plus any pension adjustment reversal. Unused room **carries forward indefinitely**. Room can be earned from age 0 (any earned income) and used until **December 31 of the year the annuitant turns 71**; after 71 a person with room can still contribute to a **spousal** RRSP if the spouse is 71 or younger.

**Contributions and deductions.** Contributions made in the year or in the **first 60 days of the following year** may be deducted for that year (deadline for the 2026 tax year: March 1, 2027). A contribution and its deduction are separable: you may contribute now and **carry the deduction forward** to a higher-income year — a useful lever for someone expecting a big income year. A lifetime **$2,000 over-contribution cushion 📌** exists; beyond it, excess is penalised at **1% per month**.

**Withdrawals.** Any amount, any time, fully taxable. Financial institutions withhold at source on lump sums (a prepayment):

| Lump sum | Outside Quebec | Quebec resident (federal + Quebec) |
|---|---|---|
| ≤ $5,000 | 10% | 5% + 14% = 19% |
| $5,001 – $15,000 | 20% | 10% + 14% = 24% |
| > $15,000 | 30% | 15% + 14% = 29% |
| Non-resident | 25% flat (treaty may reduce) | — |

Two exceptions let money out *without* immediate tax, as loans to yourself:
- **Home Buyers' Plan (HBP)**: up to **📌 $60,000** per person (raised from $35,000 in April 2024) for a first home; repaid over **15 years** starting the *second* year after withdrawal (withdrawals made 2022–2025 got a temporary five-year deferral of the first repayment). Missed repayments are added to income.
- **Lifelong Learning Plan (LLP)**: up to **📌 $10,000/year, $20,000 total** for full-time education of the annuitant or spouse; repaid over 10 years.

**Spousal RRSP.** The *contributor* uses their own room and takes the deduction, but the plan belongs to the *annuitant* spouse, who is taxed on withdrawals. This shifts future income to the lower-income spouse. **Attribution rule 📌:** if the annuitant withdraws in the calendar year of a contribution **or the two following years**, the withdrawal is taxed to the contributor to the extent of contributions made in that window. After age 65 pension income splitting (§12.1) makes spousal RRSPs less necessary, but they remain the only way to split RRIF income *before* 65.

**Maturity.** By **December 31 of the year the annuitant turns 71** the RRSP must be (a) transferred to a RRIF, (b) used to buy a life or term-to-90 annuity, or (c) cashed out (fully taxable — almost never sensible). Conversion can happen earlier, and *partial* conversion at 65 is common so that RRIF income qualifies for the $2,000 pension credit and for splitting.

**Death.** See §13 — the whole balance is income of the deceased in the year of death unless rolled to a spouse or financially dependent child/grandchild.

### 5.2 RRIF (Registered Retirement Income Fund)

A RRIF is the same tax shelter with **mandatory minimum withdrawals** and no new contributions (transfers from RRSPs, other RRIFs and pensions are allowed).

**Minimum withdrawal.** Each year (starting the calendar year *after* the RRIF is opened — the opening year has no minimum) the annuitant must withdraw at least the **January 1 fair market value × a prescribed factor** based on age on January 1:

- Under 71: factor = **1 / (90 − age)** (e.g. 4.00% at 65, 5.00% at 70).
- 71 and over: the prescribed table 📌 (unchanged since 2015):

| Age | % | Age | % | Age | % |
|---|---|---|---|---|---|
| 71 | 5.28 | 80 | 6.82 | 89 | 10.99 |
| 72 | 5.40 | 81 | 7.08 | 90 | 11.92 |
| 73 | 5.53 | 82 | 7.38 | 91 | 13.06 |
| 74 | 5.67 | 83 | 7.71 | 92 | 14.49 |
| 75 | 5.82 | 84 | 8.08 | 93 | 16.34 |
| 76 | 5.98 | 85 | 8.51 | 94 | 18.79 |
| 77 | 6.17 | 86 | 8.99 | 95+ | 20.00 |
| 78 | 6.36 | 87 | 9.55 | | |
| 79 | 6.58 | 88 | 10.21 | | |

- **Younger-spouse election:** at RRIF setup the annuitant may irrevocably elect to use the *spouse's* age, permanently lowering minimums.
- Governments have twice cut the minimum by 25% for one year in market crises (2008, 2020). A similar one-year cut was promised in April 2025 but **was not included in Budget 2025 and has not been legislated** — do not apply one for 2025 or 2026. Keep a `rrif_min_reduction_factor` hook (default 1.0) for the next time it happens.

**Withholding.** The **minimum is paid with no withholding**; amounts above the minimum are withheld at the RRSP lump-sum rates above (applied to the excess only). Everything is taxable regardless.

**Why it matters for planning.** Minimums are the mechanism that forces RRSP money into income in the 70s and 80s, when it stacks on CPP and OAS and can trigger the OAS recovery tax (§10.2). This is the whole rationale for the "RRSP meltdown" — drawing RRSP/RRIF money deliberately in low-income years (early retirement, before CPP/OAS start) to fill low brackets, and parking the after-tax surplus in a TFSA. The engine's decumulation optimiser is largely about this trade-off.

### 5.3 LIRA and LIF (locked-in accounts)

**Origin.** When you leave an employer with a registered pension plan before retirement, the *commuted value* of your pension is transferred to a **LIRA** (Locked-in Retirement Account; a "locked-in RRSP" under federal rules). Pension law — not tax law — governs it, and the applicable law is that of the **pension's jurisdiction** (federal/OSFI for federally regulated employers such as banks, telecoms, airlines, federal public service; otherwise the province where the member worked), *not* where the person lives now. The engine therefore needs a `locked_in_jurisdiction` field distinct from the residence province.

**Rules in common with the RRSP/RRIF:** same tax treatment; must be converted to a **LIF** (or annuity) by December 31 of the year the holder turns 71; LIF minimum withdrawal = the RRIF minimum; LIF income qualifies for pension splitting at 65.

**The locked-in differences:**
- Generally **no withdrawals before age 55** (some jurisdictions 50).
- A LIF has a **maximum annual withdrawal** as well as a minimum, computed from age and balance with a jurisdiction-specific formula (federal and most provinces use a factor table driven by a reference long-bond rate published each year 🔄; Ontario publishes its own maximum table each year). The maximum is what distinguishes a LIF from a RRIF.
- **Unlocking provisions** (jurisdiction-specific — the most common patterns): a **one-time 50% transfer to an RRSP/RRIF** at conversion (federal: age 55+, to a Restricted LIF; Ontario: within 60 days of opening a LIF at 55+; Alberta: 50+; Manitoba: 50% to a Prescribed RRIF at 55+, 100% at 65+; Saskatchewan: full transfer to a PRIF at 55+); **small-balance** unlocking when total locked-in assets are below a fraction of the YMPE (e.g. federally 50% of YMPE at 55+); **financial hardship**, **shortened life expectancy** and **non-residency (2 years)** unlocking.
- **Quebec (from January 1, 2025): no LIF maximum at all for holders aged 55+.** Any amount above the minimum may be withdrawn, in one or more instalments, without spousal consent. Under 55 a maximum remains (2026 prescribed rate 6.25% 🔄; maximum "temporary income" $37,300 🔄). Direct LIF→RRSP/RRIF transfers were abolished at the same time. For a Quebec-jurisdiction LIF the engine should model minimum-only constraints from 55.

### 5.4 What counts as "eligible pension income" (for the $2,000 credit and for splitting)
- **Any age:** lifetime annuity payments from a registered pension plan (defined benefit or a DC plan's annuity), and (on death of a spouse) survivor benefits from a RRIF/annuity.
- **Age 65 and over only:** **RRIF and LIF withdrawals**, RRSP annuity payments, DPSP annuity payments, and the interest portion of non-registered annuities.
- **Never:** CPP/QPP, OAS, RRSP lump-sum withdrawals, salary. (Quebec's provincial rule: 65+ for everything — §3.3.)

### 5.5 Workplace pensions (RPPs) — what the engine needs to know

- **Defined benefit (DB):** the plan promises a formula pension (e.g. 2% × years × best-5 average). Each year the **pension adjustment** PA = 9 × benefit accrued − $600 is reported on the T4 and **reduces next year's RRSP room** almost to nothing for a 2% plan. Many DB plans pay a **bridge benefit** from retirement to 65 that stops when CPP is assumed to start; integrated plans also reduce the lifetime pension at 65. Indexation varies (full CPI, partial, or none) and is a key input. On leaving before retirement the member may take the **commuted value**: the portion within the *maximum transfer value* (ITR 8517) goes to a LIRA tax-free, the excess is **taxable cash** in the year received (a common surprise).
- **Defined contribution (DC):** PA = total employer + employee contributions; the balance behaves like a locked-in RRSP and becomes a LIF (or, in some jurisdictions, stays in the plan paying "variable benefits").
- **Group RRSPs / DPSPs / PRPPs:** not locked in (RRSP) or partly locked (DPSP vesting), room consumed via contributions/PA.
- The annual **money purchase (MP) limit** 🔄 ($33,810 for 2025, ≈ $34,860 for 2026) caps DC contributions and sets the next year's RRSP dollar limit; the **DB limit** 🔄 (1/9 of the MP limit per year of service) caps DB accrual.

### 5.6 Parameters — RRSP family

| Parameter | Illustrative value | Changes | Source document |
|---|---|---|---|
| RRSP earned-income percentage | 18% | 📌 | ITA s.146(1) |
| RRSP dollar limit | $32,490 (2025) / $33,810 (2026) | 🔄 | CRA — *MP, DB, RRSP, DPSP, ALDA, TFSA limits and the YMPE* |
| Money-purchase limit | $33,810 (2025) / ≈ $34,860 (2026) | 🔄 | same page |
| Over-contribution cushion; penalty | $2,000; 1%/month | 📌 | CRA — *Excess contributions* |
| Maturity age | 71 | 📌 | ITA s.146(2)(b.4) |
| Withholding tiers | 10/20/30 % (5/10/15 + 14% QC); 25% non-resident | 📌 (Quebec rate 🔄) | CRA — *Tax rates on withdrawals*; Revenu Québec — *Source deductions on RRSP/RRIF withdrawals* |
| HBP / LLP limits | $60,000 / $10,000 & $20,000 | 📌 | CRA — *Home Buyers' Plan*; *Lifelong Learning Plan* |
| Spousal attribution window | year + 2 following | 📌 | ITA s.146(8.3) |
| RRIF factors | table above; 1/(90−age) under 71 | 📌 | CRA — *Chart – Prescribed factors* (ITR 7308) |
| RRIF one-year reduction factor | 1.0 (none in force) | 📌 hook | Finance Canada announcements |
| LIF maximum factors | per jurisdiction, per year | 🔄 | OSFI — *Life Income Fund (LIF) maximum withdrawal factors*; FSRA (Ontario) — *LIF maximum annual income payment*; Retraite Québec — *Life income fund (LIF)*; other provincial pension regulators |
| Unlocking thresholds (% of YMPE) | per jurisdiction | 📌 | same regulators |
| PA formula constants | 9×, −$600 | 📌 | CRA — *Pension Adjustment Guide (T4084)* |

**Sources:** CRA, *RRSPs and related plans* hub; CRA, *MP, DB, RRSP, DPSP, ALDA, TFSA limits and the YMPE* (annual table); CRA, *Tax rates on withdrawals* (RRSP withholding); CRA, *Chart – Prescribed factors* (RRIF minimums); CRA, *Home Buyers' Plan (HBP)*; CRA, *Pension income splitting*; OSFI, *Locked-in registered retirement savings plans and LIFs*; FSRA, *Life income funds – maximum annual income payment*; Retraite Québec, *Amendments related to life income funds (LIFs) as of 2025* and *Flash Retirement – LIF capsule* (annual prescribed rate); CRA, *T4084 Pension Adjustment Guide*.

---

## 6. TFSA (Tax-Free Savings Account)

**Concept.** After-tax money goes in (no deduction); growth and withdrawals are **completely tax-free** and — critically — **do not count as income for any purpose**: not for tax, not for the OAS recovery tax, not for GIS, not for the age amount, not for CCB. That makes the TFSA the most valuable account for anyone who will face a benefit clawback, and the natural destination for RRSP-meltdown proceeds.

**Room.** Accrues every year from the year a person turns **18** while resident in Canada, whether or not they have income or have opened an account. Annual limit 🔄 **$7,000** for 2024, 2025 and 2026; the limit is the indexed dollar amount **rounded to the nearest $500**, so it moves in steps (indexed value $7,185 for 2026; the next step to $7,500 arrives when the indexed value crosses $7,250 — possibly 2027). Cumulative room for someone 18+ in 2009 who never contributed: **$102,000 through 2025, $109,000 through 2026.**

| Years | Annual limit 📌 (history) |
|---|---|
| 2009–2012 | $5,000 |
| 2013–2014 | $5,500 |
| 2015 | $10,000 |
| 2016–2018 | $5,500 |
| 2019–2022 | $6,000 |
| 2023 | $6,500 |
| 2024–2026 | $7,000 |

**Withdrawals restore room — the following January 1.** A withdrawal in 2026 is added back to room on January 1, 2027. Re-contributing in the same year without room triggers a **1%-per-month penalty 📌** on the excess. The engine must therefore track room as: prior room − contributions + (prior-year withdrawals) + new annual limit, with the withdrawal add-back lagged one year.

**Other rules the engine can rely on:**
- No age limit for contributing (unlike the RRSP's 71) — TFSA room keeps accruing for life.
- Spousal attribution rules do **not** apply: one spouse may give the other money to contribute to their own TFSA. This makes "fill both spouses' TFSAs" a standard household policy.
- Foreign dividends inside a TFSA suffer non-recoverable foreign withholding tax; Canadian dividends and capital gains are completely sheltered.
- On death, a spouse named **successor holder** takes over the account intact; any other beneficiary receives the date-of-death value tax-free but growth after death is taxable (§13).
- Non-residents accrue no room and pay 1%/month on contributions made while non-resident.

**Parameters:**

| Parameter | Illustrative value | Changes | Source document |
|---|---|---|---|
| Annual dollar limit | $7,000 (2026) | 🔄 (step function) | CRA — *TFSA contribution room* / *MP, DB, RRSP, DPSP, ALDA, TFSA limits and the YMPE* |
| Limit history 2009–present | table above | 📌 | same |
| Eligibility age | 18 | 📌 | ITA s.146.2 |
| Over-contribution penalty | 1%/month | 📌 | CRA — *Tax payable on excess TFSA amount* |

**Sources:** CRA, *Tax-Free Savings Account (TFSA), Guide for Individuals (RC4466)*; CRA, *TFSA contribution room* page; CRA annual limits table.

---

## 7. RESP (Registered Education Savings Plan)

**Concept.** A subscriber (usually a parent or grandparent) contributes **after-tax** money for a beneficiary's post-secondary education. There is no deduction; the value is (a) government grants and bonds paid *into* the plan and (b) tax-deferred growth that is eventually taxed in the *student's* hands, typically at a zero or very low rate.

**Contribution limits 📌.** No annual limit; a **$50,000 lifetime limit per beneficiary** (across all plans for that child). Excess is penalised at 1% per month until withdrawn. Contributions are allowed for **31 years** after the plan is opened; the plan must be wound up by the end of its **35th** year. Individual plans (one beneficiary) and family plans (several related beneficiaries sharing contributions and growth) exist; group scholarship plans are a third, fee-heavy type.

**Canada Education Savings Grant (CESG) 📌.**
- **Basic CESG = 20%** of contributions, up to **$500 per beneficiary per year** (i.e. on the first $2,500 contributed), **lifetime maximum $7,200**.
- Unused grant room carries forward, but the catch-up is capped: at most **$1,000** of basic CESG in any one year (on $5,000 of contributions). Ten missed years cannot be fully recovered.
- Paid until the end of the calendar year the beneficiary turns **17**, subject to the **16/17 rule**: grants at 16 and 17 are paid only if at least $2,000 had been contributed (and not withdrawn) before the year the child turned 16, or at least $100/year in any four earlier years. Practically: open the plan by the year the child turns 15.
- **Additional CESG (A-CESG)** for lower-income families: an extra **20%** or **10%** on the *first $500* contributed each year (max $100/$50), based on the primary caregiver's adjusted family net income against thresholds that track the first two federal bracket thresholds 🔄 (2026: ≈ $58,523 / $117,045). A-CESG counts toward the $7,200 lifetime cap.

**Canada Learning Bond (CLB) 📌.** For children born **2004 or later** in low-income families (income test tied to the CCB thresholds, indexed 🔄; roughly ≤ $58,500 for families with up to three children for the July 2026–June 2027 year). **No contribution required.** $500 in the first year of eligibility plus $100 for each subsequent eligible year to age 15, **maximum $2,000**, paid retroactively when the plan is opened. Currently a caregiver can claim it until the child turns 18 and the beneficiary can self-claim from 18 until the day before turning 21. **From April 2028** (legislated in 2024) the government will automatically open an RESP and deposit the CLB for eligible children born 2024+ who have no RESP by age 4, and the self-claim age will be extended to **30**.

**Provincial add-ons 🔄:** Quebec **QESI** (10% of contributions, max $250/year plus a low-income increase, lifetime $3,600); British Columbia **BCTESG** (one-time $1,200 when the child is 6–8, no contribution required). Encode as jurisdiction hooks keyed to the *beneficiary's* residence.

**Withdrawals — three distinct buckets** (the engine must track them separately):

| Bucket | What it is | Tax | Limits |
|---|---|---|---|
| **PSE withdrawal** (refund of contributions) | the subscriber's own contributions | **tax-free**, to subscriber or student | none, any time the student is enrolled (if not enrolled, withdrawing contributions triggers **repayment of grants** at 20% of the amount) |
| **EAP** (Educational Assistance Payment) | grants + bonds + all growth | **taxable to the student** (T4A box 042); usually little or no tax after the student's BPA and tuition credit | **$8,000** in the first **13 consecutive weeks** of full-time study (**$4,000** per 13-week period part-time); unlimited after that while enrolled; cap resets after a 12-month gap. Above an annual threshold (🔄 $29,459 for 2026) the promoter must vet expenses. EAPs may continue for **6 months** after enrolment ends |
| **AIP** (Accumulated Income Payment) | growth taken back by the subscriber because no beneficiary attends | taxed as the subscriber's income **plus a 20% penalty tax** (12% federal + 8% Quebec for Quebec residents); up to **$50,000** can be rolled to the subscriber's or spouse's **RRSP** if they have room, avoiding both taxes | plan must have been open 10 years and all beneficiaries be 21+ (or deceased), or the plan be in its 35th year; grants and bonds are returned to the government; plan must close by the end of February of the following year |

**Sequencing advice the engine can encode:** draw **EAPs first** (they're taxable, so use the student's low-income years) and PSE last; in the first term, combine the $8,000 EAP cap with unlimited PSE. In a family plan, CESG can be paid out to any beneficiary, but each beneficiary's lifetime CESG (from any plan) is capped at $7,200 — excess must be repaid.

**Near-withdrawal risk.** Because a defined amount is needed on a defined date, the RESP is the one account where a glide path into capital-preservation assets (GICs, or short-duration/money-market ETFs where the promoter doesn't offer GICs) in the last few years matters more than expected return. The engine's accumulation module should expose this as a per-beneficiary policy.

**Parameters:**

| Parameter | Value | Changes | Source document |
|---|---|---|---|
| Lifetime contribution limit | $50,000 | 📌 | CRA — *RESP contribution limits* |
| CESG rate / annual max / catch-up max / lifetime max | 20% / $500 / $1,000 / $7,200 | 📌 | ESDC — *Canada Education Savings Grant* |
| A-CESG rates and income thresholds | 20%/10% on first $500; ≈ $58,523 / $117,045 (2026) | 🔄 | ESDC — *CESG* (thresholds published each July / CRA indexation) |
| CLB amounts / income thresholds | $500 + $100/yr, max $2,000; thresholds by family size | 📌 / 🔄 | ESDC — *Canada Learning Bond*; ESDC *Notice – Revised income brackets for the CLB* (annual, July) |
| EAP caps / annual threshold | $8,000 / $4,000; $29,459 (2026) | 📌 / 🔄 | CRA — *RESP Bulletin No. 1R3* |
| AIP penalty / RRSP rollover cap | 20% (12% + 8% QC); $50,000 | 📌 | CRA — *RESP payments* (T1172) |
| Plan life | 31 years contributions / 35 years | 📌 | ITA s.146.1 |
| QESI / BCTESG | 10% to $250; $1,200 | 🔄 / 📌 | Revenu Québec — *QESI*; BC Ministry of Education — *BCTESG* |

**Sources:** ESDC / canada.ca, *Education savings* hub (*How much money benefits could add to the RESP*; *Canada Education Savings Grant*; *Canada Learning Bond*; *Paying for education using the RESP*); CRA, *Registered Education Savings Plans (RESPs)* hub and *RESP Bulletin No. 1R3*; CRA, *Information Sheet RC4092*; Revenu Québec, *Québec education savings incentive*; Government of British Columbia, *B.C. Training and Education Savings Grant*.

---

## 8. FHSA (First Home Savings Account)

**Concept.** Introduced April 2023. The only account that combines an RRSP-style **deduction** on the way in with a TFSA-style **tax-free** withdrawal on the way out, provided the money buys a first home. If it doesn't, it collapses into an RRSP with no penalty — so for anyone eligible it dominates an RRSP contribution.

**Eligibility 📌.** Resident, age **18 to 71**, and a **first-time home buyer**: neither the individual nor their spouse owned (and lived in) a home in the current calendar year or the previous four calendar years.

**Room 📌.** **$8,000 per year, $40,000 lifetime.** Room begins only once an account is **opened** (unlike TFSA/RRSP room, which accrues automatically), so open one early with $1 if eligible. Unused annual room carries forward **one year only**, to a maximum of $8,000 carried (so at most $16,000 in one year). Over-contributions: 1% per month.

**Tax treatment.**
- Contributions are deductible (deduction may be carried forward to a later year, like an RRSP's). Contributions in the first 60 days of a year do **not** count for the prior year (unlike RRSPs).
- **Qualifying withdrawal** (written agreement to buy or build a qualifying home in Canada before October 1 of the following year, intend to occupy within a year, first-time buyer at the time): **tax-free, unlimited**, and the whole balance can be taken.
- Can be **combined with the RRSP Home Buyers' Plan** ($60,000) for the same purchase.
- **Transfers to an RRSP/RRIF** are tax-free and **do not use RRSP room** — the escape hatch if no home is bought.
- Non-qualifying withdrawals are fully taxable (with withholding).

**Lifespan 📌.** The account must be closed by December 31 of the earliest of: the **15th anniversary** of opening, the year the holder turns **71**, or the year **following the first qualifying withdrawal**. Remaining funds must be transferred to an RRSP/RRIF or withdrawn (taxable).

**Parameters:**

| Parameter | Value | Changes | Source document |
|---|---|---|---|
| Annual / lifetime limits | $8,000 / $40,000 | 📌 | CRA — *Contributing to your FHSA* |
| Carry-forward cap | $8,000 (one year) | 📌 | same |
| Age window; first-time-buyer look-back | 18–71; current + 4 prior years | 📌 | CRA — *Opening your FHSA* |
| Lifespan triggers | 15 years / 71 / year after first qualifying withdrawal | 📌 | CRA — *Closing your FHSA* |

**Sources:** CRA, *First Home Savings Account (FHSA)* hub (*Opening*, *Contributing*, *Making withdrawals*, *Closing*); CRA, *Guide RC722*.

---

## 9. CPP and QPP (Canada / Québec Pension Plan)

**Concept.** A contributory, earnings-related public pension. You earn it by contributing on employment or self-employment income between 18 and the year you start the pension; the amount depends on *how much* and *for how long* you contributed relative to the ceiling. It is **fully taxable**, **indexed to CPI every January**, and paid for life. Quebec workers contribute to the **QPP** instead; the two plans are coordinated so a career split between them yields one combined pension.

### 9.1 Contributions (the accumulation side)

| Item (2026 🔄) | CPP | QPP |
|---|---|---|
| Year's Basic Exemption (YBE) | $3,500 📌 | $3,500 |
| Year's Maximum Pensionable Earnings (YMPE, "first ceiling") | **$74,600** (2025: $71,300) | $74,600 |
| Year's Additional Maximum Pensionable Earnings (YAMPE, "second ceiling", = 114% of YMPE) | **$85,000** (2025: $81,200) | $85,000 |
| Employee/employer rate on YBE→YMPE (base 4.95% + first additional 1%) | **5.95%** each (11.9% self-employed) | **6.3%** each in 2026 (base 5.3% — temporarily cut from 5.4% for 2026 only — + 1%) |
| Rate on YMPE→YAMPE (second additional, "CPP2") | **4%** each (8% self-employed) | 4% each |
| Max employee contribution | $4,230.45 + $416.00 | $4,479.30 + $416.00 |
| Contributory period | 18 to pension start (or 70); at 65+ a working pensioner may opt out (CPT30); contributions stop at 70 | 18 to 72; may opt out at 65+ if receiving a pension; automatically stop Jan 1 after turning 72 |

The base contribution earns a *credit*; the enhanced (first and second additional) contributions are a *deduction*. Self-employed people deduct the employer half.

### 9.2 How the retirement pension is calculated (enough to model it)

- **Base CPP** (pre-2019 design) replaces **25%** of average lifetime pensionable earnings, where each year's earnings are capped at that year's YMPE, expressed as a fraction of YMPE, and the pension is scaled by the average YMPE of the last five years. **Dropouts:** the lowest **17%** of months (up to ~8 years) are dropped (general dropout); months while a child was under 7 with low earnings (child-rearing provision), months on CPP disability, and months after 65 with low earnings may also be dropped. Roughly 39 years at or above the YMPE are needed for the maximum.
- **Enhanced CPP** (phased in 2019–2025) adds a second layer that will eventually raise replacement to **33.33%** of earnings up to the YAMPE. It only counts contributions made since 2019, so its effect on someone retiring in 2026 is small (a few percent) and grows for younger cohorts; full maturity is ~2065. A cohort-aware engine should model the enhancement as a separate accrual on post-2018 earnings.
- **Start age adjustment 📌:** pension may start any month from **60 to 70** (QPP: **60 to 72**). Before 65: **−0.6% per month** (−36% at 60); after 65: **+0.7% per month** (+42% at 70; QPP +58.8% at 72). QPP's early reduction is graduated **0.5%–0.6%/month**, reaching 0.6% only for a maximum-pension claimant. There is no benefit to delaying past 70 (72 QPP). Deferral is applied *after* indexing, so the deferred amount is also inflation-protected.
- **2026 maxima 🔄** (new pensions starting January 2026): **$1,507.65/month at 65** (2025: $1,433.00); $964.90 at 60; $2,140.86 at 70 (QPP $2,394.15 at 72). The **average** new age-65 pension in January 2026 was **$925.35** (≈ 61% of max) — the engine should default to a user-supplied Statement of Contributions estimate, then the average, never the maximum.
- **Indexation 🔄:** benefits in pay rise every January by the 12-month average CPI change (**+2.0%** for 2026; +2.6% for 2025). Note the enhancement makes the *maximum for new beneficiaries* creep up month by month; canada.ca publishes the January figure.
- **Post-Retirement Benefit (PRB):** contributions made while already receiving CPP (ages 60–70) buy a small additional pension each following January (max 2026 🔄 $54.69/month per year of max contributions). QPP calls this the retirement pension supplement.

### 9.3 Other CPP/QPP benefits a household model may need
- **Survivor's pension 🔄 (2026 max):** $904.59/month if the survivor is 65+ (60% of the deceased's calculated retirement pension), $803.54 under 65 (flat + 37.5%). **Combined survivor + own retirement pension is capped** at the maximum retirement pension ($1,531.56 combined max in 2026) — a widow(er) who already has a large CPP receives little or nothing extra. Important for two-person simulations at the first death.
- **Death benefit 📌:** one-time $2,500 to the estate (an additional $2,500 in certain no-survivor cases from 2025).
- **Disability pension 🔄:** max $1,741.20/month (2026); converts to retirement pension at 65.
- **Pension sharing (assignment) between spouses:** while both are 60+ the portion of CPP earned during cohabitation can be split equally for *tax* purposes — a modest income-splitting tool (Form ISP1002). **Credit splitting** divides contributory credits on separation/divorce.

### 9.4 Parameters — CPP/QPP

| Parameter | Illustrative value (2026) | Changes | Source document |
|---|---|---|---|
| YMPE / YAMPE / YBE | $74,600 / $85,000 / $3,500 | 🔄 (announced ~Nov 1) / 📌 | CRA — *CPP contribution rates, maximums and exemptions*; Retraite Québec — *Québec Pension Plan figures* |
| Contribution rates (base+1st, 2nd) | 5.95% / 4% (QPP 6.3% / 4%) | 📌 (QPP 2026 temporary cut 🔄) | same |
| Max retirement pension at 65 (new) | $1,507.65 | 🔄 (January) | Service Canada — *CPP retirement pension: How much you could receive*; ESDC *quarterly benefit amounts and related figures* (rate card) |
| Average new pension at 65 | $925.35 | 🔄 (monthly) | same |
| Annual indexation of benefits in pay | 2.0% | 🔄 | ESDC rate card |
| Early/late adjustment factors | −0.6% / +0.7% per month; 60–70 (QPP 0.5–0.6% / +0.7%; 60–72) | 📌 | Service Canada — *When to start your retirement pension*; Retraite Québec — *Calculation of your retirement pension* |
| Replacement rates | 25% base / 33.33% enhanced | 📌 | Service Canada — *CPP enhancement* |
| General dropout | 17% of months | 📌 | CPP Act |
| Survivor / death / disability maxima | $904.59 & $803.54 / $2,500 / $1,741.20 | 🔄 | canada.ca — *CPP pensions and benefits monthly amounts* |

**Sources:** Service Canada, *Canada Pension Plan – Retirement pension* pages (*How much you could receive*; *When to start*; *CPP enhancement*); canada.ca, *Canada Pension Plan: Pensions and benefits monthly amounts*; ESDC, *Maximum benefit amounts and related figures – CPP and OAS* (quarterly rate card); CRA, *CPP contribution rates, maximums and exemptions*; Retraite Québec, *Québec Pension Plan figures* and *Calculation of your retirement pension*; Revenu Québec, *Maximum pensionable earnings and QPP contribution rate*.

---

## 10. OAS, GIS and the Allowances

### 10.1 Old Age Security (OAS) pension

**Concept.** A **residence-based** (not contribution-based) pension paid from general revenue to almost everyone **65+**. Full pension requires **40 years** of residence in Canada after age 18; 10–39 years earns a **partial pension of 1/40 per year** (minimum 10 years, or 20 to be paid outside Canada). Work history is irrelevant. Most people are **auto-enrolled** (letter at 64).

**Amount 🔄.** Adjusted **quarterly** (January, April, July, October) to CPI and **never decreases**. Current quarter (**July–September 2026**): **$751.97/month** at 65–74 and **$827.17** at 75+ — the 75+ figure reflects a **permanent 10% increase** that applies automatically from the month after the 75th birthday (introduced July 2022). Recent quarters for back-testing: $727.67 (Jan–Jun 2025), $734.95 (Jul–Sep 2025), $740.09 (Oct–Dec 2025), $742.31 (Jan–Mar 2026), $743.05 (Apr–Jun 2026). The October–December 2026 rate is due late September 2026.

**Deferral 📌.** May be deferred up to 60 months past 65: **+0.6% per month, +36% at 70**. No further gain after 70. Deferral does not extend the partial-pension residence count; someone who defers also forgoes GIS for those months (GIS requires OAS in pay).

**Taxable** (T4A(OAS)), included in net income — which means OAS itself counts toward its own recovery test.

### 10.2 OAS recovery tax ("clawback")

The single most important income threshold in Canadian retirement planning.

- If **individual net income before adjustments (line 23400)** for a tax year exceeds a threshold, the pensioner repays **📌 15%** of the excess, up to the full OAS received. It is assessed on the **individual**, not the household, and includes OAS itself, CPP, pensions, RRIF/LIF withdrawals, interest, *grossed-up* dividends and the taxable half of capital gains. TFSA withdrawals and GIS are excluded.
- **Timing — the July-to-June lag.** Income from tax year *Y* is reported in spring *Y+1* and drives a **monthly withholding from July Y+1 to June Y+2**; the actual liability is reconciled on the *Y+1* return. The engine must model both: the recovery tax as a line on year-*Y*'s return (reduces after-tax income in *Y*) and the cash-flow effect of reduced deposits from July *Y+1*.
- **Thresholds 🔄** (indexed with the federal factor):

| Income year | Threshold | Full clawback 65–74 | Full clawback 75+ | OAS payment period affected |
|---|---|---|---|---|
| 2024 | $90,997 | $148,451 | $154,196 | Jul 2025 – Jun 2026 |
| **2025** | **$93,454** | **$152,062** | **$157,923** | **Jul 2026 – Jun 2027 (current)** |
| 2026 | $95,323 | ≈ $155,109 | ≈ $161,088 | Jul 2027 – Jun 2028 (estimates until Oct 2026) |

The full-clawback ceiling is simply threshold + (annual OAS ÷ 0.15), so **compute it** from the threshold and the OAS rate in force rather than storing it. Deferring OAS to 70 both raises the pension 36% and (because the ceiling scales with the pension) raises the income at which it is fully lost.

### 10.3 Guaranteed Income Supplement (GIS)

**Concept.** A **non-taxable**, income-tested top-up for low-income OAS pensioners (must be 65+, receiving OAS, resident in Canada). Recalculated every **July** from the **prior calendar year's income** (July 2026–June 2027 uses 2025 income) and adjusted quarterly with CPI. Filing a tax return each year renews it automatically.

**Income test 📌.** Countable income = net income *minus* OAS *minus* an employment exemption (first **$5,000** of employment/self-employment income fully exempt, **50%** of the next **$10,000** exempt). CPP/QPP, RRIF/LIF withdrawals, pensions, interest, grossed-up dividends and taxable gains all count. For couples the test uses **combined** income. GIS is then reduced by **$0.50 per $1** of countable income (**$0.75 per $1** in the low band where the GIS *top-up* is being phased out), so a GIS recipient's **effective marginal rate on a RRIF withdrawal is 50%+ before income tax** — often the highest marginal rate in the whole system, and the reason low-income households should usually favour TFSAs over RRSPs and draw down RRSPs *before* 65.

**Current maximums 🔄 (July–September 2026):**

| Situation | Max monthly GIS | Combined with max OAS (65–74) | Countable-income cutoff |
|---|---|---|---|
| Single, widowed or divorced | **$1,123.17** | $1,875.14 | **$22,800** |
| Spouse receives the full OAS pension | **$676.09** each | — | **$30,096** (couple) |
| Spouse receives the Allowance | **$676.09** | — | **$42,144** (couple) |
| Spouse receives neither OAS nor the Allowance | **$1,123.17** | — | **$54,624** (couple) |

(Apr–Jun 2026: $1,109.85 / $668.08; cutoffs $22,512 / $29,760 / $41,664 / $53,952.)

Partial-OAS pensioners receive a correspondingly *higher* GIS so that the OAS + GIS floor is preserved. A few provinces add small top-ups (e.g. Ontario GAINS, BC Senior's Supplement, Alberta Seniors Benefit) 🔄 — worth a jurisdiction hook but second-order.

### 10.4 The Allowance and Allowance for the Survivor

For low-income people **aged 60–64**: the **Allowance** if their spouse receives OAS+GIS (max 🔄 **$1,428.06**/month, Jul–Sep 2026; couple income < $42,144), and the **Allowance for the Survivor** for widowed persons who have not re-partnered (max 🔄 **$1,702.34**; income < ≈ $30,700). Both are non-taxable, must be applied for, and stop at 65 when OAS/GIS take over. They matter for couples with an age gap.

### 10.5 Parameters — OAS/GIS

| Parameter | Illustrative value | Changes | Source document |
|---|---|---|---|
| OAS max monthly, 65–74 / 75+ | $751.97 / $827.17 (Jul–Sep 2026) | 🔄 quarterly | Service Canada — *Old Age Security payment amounts*; ESDC quarterly rate card |
| Deferral factor / max | 0.6%/month / 36% at 70 | 📌 | Service Canada — *OAS: How much you could receive* |
| 75+ uplift | 10% | 📌 | same |
| Residence rule | 40 years full; 1/40 per year; 10-year minimum | 📌 | Service Canada — *OAS eligibility* |
| Recovery-tax rate | 15% | 📌 | Service Canada — *OAS pension recovery tax* |
| Recovery threshold by income year | $93,454 (2025); $95,323 (2026) | 🔄 | same page (three-year table) |
| GIS maxima and cutoffs (4 situations) | table above | 🔄 quarterly (July reset) | Service Canada — *GIS: How much you could receive* |
| GIS employment exemption | $5,000 + 50% of next $10,000 | 📌 | Service Canada — *GIS: Do you qualify* |
| GIS reduction rates | 50% / 75% (top-up band) | 📌 | OAS Act & Regulations; ESDC rate card (top-up cutoffs) |
| Allowance / Allowance for Survivor maxima | $1,428.06 / $1,702.34 | 🔄 quarterly | Service Canada — *Allowance* pages |

**Sources:** Service Canada, *Old Age Security payment amounts* (quarterly table with all OAS/GIS/Allowance maxima and income cutoffs); Service Canada, *Old Age Security pension recovery tax*; Service Canada, *Guaranteed Income Supplement – How much you could receive* and *Do you qualify*; ESDC, *Maximum benefit amounts and related figures – CPP and OAS* (quarterly rate card, includes GIS top-up cutoffs); CRA, *Line 23500 – Social benefits repayment*.

---

## 11. Other income-tested federal benefits

These are outside the retirement core but drive the *effective* marginal tax rate of accumulation-phase households (which affects the RRSP-vs-TFSA decision and RESP affordability). All are computed by CRA from the prior year's return, paid **July–June**, and reduced above income thresholds indexed annually 🔄.

- **Canada Child Benefit (CCB):** tax-free monthly payment per child under 18. July 2026–June 2027 maximums ≈ **$8,157** per child under 6 and ≈ **$6,883** per child 6–17 (indexed 2%; verify against CRA), reduced by 7%–23% of *adjusted family net income* above ≈ $38,200 and again above ≈ $82,800 (rates depend on number of children). For a two-child family the CCB phase-out adds ~13.5 points to the marginal rate between those thresholds — RRSP contributions that lower family net income *increase* CCB, a strong accumulation-phase incentive.
- **GST/HST credit:** quarterly, small (≈ $350/adult), phased out above ≈ $46,000 family net income.
- **Canada Workers Benefit:** refundable credit for low-income workers; irrelevant once retired.
- **Canada Dental Care Plan** (family net income < $90,000, no private dental coverage) and **Canada Disability Benefit** (up to $200/month, income-tested, from July 2025): eligibility depends on filed returns; second-order for the engine but worth noting as reasons retirees keep filing.
- Provincial equivalents (Ontario Trillium Benefit, BC Family Benefit, Quebec Family Allowance and Solidarity Tax Credit, Alberta Child and Family Benefit, etc.) follow the same July–June, prior-year-income pattern.

**Sources:** CRA, *Canada child benefit – How much you can get* (and the annual *CCB payment amounts* update each July); CRA, *GST/HST credit – How much you can expect to receive*; CRA, *Canada workers benefit*; Health Canada, *Canada Dental Care Plan*; Service Canada, *Canada Disability Benefit*.

---

## 12. How the pieces interact: splitting, sequencing, effective marginal rates

### 12.1 Pension income splitting (federal Form T1032)
A couple may jointly elect each year to move **up to 50%** of one spouse's *eligible pension income* (§5.4 — RRIF/LIF only from 65, RPP annuity at any age) onto the other spouse's return. No money moves; the transferee also gets a proportional share of the tax withheld and can claim their own **$2,000 pension credit** on the split amount. The election is re-optimised every year (the optimal percentage is rarely exactly 50%) and interacts with the age amount, the OAS recovery tax and the spousal credit of *both* returns — the engine should solve for the split that minimises **combined** household tax plus clawbacks. Both spouses must be Canadian residents at December 31. Quebec allows the provincial split only when the transferor is 65+ (§3.3).

### 12.2 Other legitimate income-shifting tools
- **Spousal RRSP** (§5.1) — the only way to shift RRIF income before 65.
- **CPP pension sharing** (§9.3).
- **TFSA gifting** — no attribution.
- **Non-registered accounts: attribution rules 📌 apply.** Income and capital gains on money *gifted* between spouses (or to minor children — income only) are taxed back to the giver. The standard workaround is a **prescribed-rate spousal loan** at CRA's prescribed rate (🔄 3% through Q3 2026) with interest actually paid by January 30. The engine can assume the higher-income spouse funds household spending so the lower-income spouse's own income is invested in their name ("pay the bills from the high earner's account").
- **Ordering of withdrawals** — the engine's core policy space. The classic levers, in order of leverage: (1) which spouse's account is drawn; (2) RRSP/RRIF draw-down in low-income years before CPP/OAS start; (3) CPP and OAS start ages (deferral to 70 buys indexed, longevity-insured income and raises the OAS clawback ceiling); (4) TFSA as the last-resort and clawback-free source; (5) realising capital gains (50% inclusion) rather than dividends (138% inclusion) in benefit-tested years.

### 12.3 Effective marginal rates (METRs) the engine should surface
The statutory brackets understate the true marginal rate wherever a benefit phase-out overlaps. Common stacks for a retiree:
- Base: federal + provincial marginal rate (e.g. 20.05% for an Ontario retiree in the lowest bracket in 2026: 14% + 5.05%).
- **+ GIS clawback 50% (or 75%)** for income under ≈ $23k (single) → METR ≈ 70%+.
- **+ Age amount phase-out 15%** federally (plus the provincial age amount, typically 15% too) for net income between ≈ $46k and ≈ $107k.
- **+ OAS recovery 15%** for net income between $93,454 and ≈ $152k (2025 income).
- **+ Ontario surtax** effect above ≈ $90k, **+ Ontario Health Premium** steps.
- **+ Quebec senior-credit pool 18.75%** above ≈ $43k family income, **+ HSF 1%**.
A good regression test: METR plots for a single Ontario retiree and a single Quebec retiree should show these plateaus at the right incomes.

### 12.4 Age 71 as the universal wind-up age
RRSP → RRIF, LIRA → LIF, FHSA closure, last RRSP contribution: all by December 31 of the year the holder turns 71. CPP must start by 70 (QPP 72); OAS by 70. The engine should treat these as hard constraints, not policy choices.

**Sources:** CRA, *Pension income splitting* and *Form T1032 Joint Election to Split Pension Income*; CRA, *Prescribed interest rates* (quarterly); CRA, *Income Tax Folio S1-F5-C1 (Related persons and dealing at arm's length)*, *S3-F2-C1 (Capital dividends)* and the attribution-rule folios under s.74.1–74.5; Service Canada, *CPP pension sharing*.

---

## 13. Death, spousal rollovers and probate

A two-person household simulation needs a first-death and second-death treatment; taxes at death are often the largest single tax event in a lifetime.

- **Deemed disposition 📌.** At death every capital property is deemed sold at fair market value on the *terminal return*; accrued gains are taxed (50% inclusion) in the deceased's final year, stacked on all other income of that year. The principal residence remains exempt.
- **Spousal rollover 📌.** Property left to a spouse/common-law partner (or a qualifying spousal trust) passes at cost — no tax until the survivor sells or dies. This is automatic unless the executor elects out (sometimes done to use the deceased's low bracket or capital losses).
- **RRSP/RRIF/LIF.** The full fair market value is income on the deceased's terminal return **unless** it passes to a spouse (as successor annuitant of a RRIF or via transfer to the spouse's RRSP/RRIF) or to a financially dependent child/grandchild (minor or disabled), in which case it rolls over and is taxed to the recipient later. At the **second death** the entire remaining registered balance is taxed at once — frequently at the top bracket — which is a core argument for melting down RRSPs earlier and why the engine's objective should include after-tax estate value, not just spending.
- **TFSA.** A spouse named **successor holder** simply becomes the holder (room and tax-free status preserved). A **beneficiary** receives the date-of-death value tax-free; growth between death and distribution is taxable to them. Naming a successor holder is strictly better for a spouse.
- **RESP.** Belongs to the subscriber, not the child; it should be dealt with in the will or joint-subscribed, otherwise it forms part of the estate.
- **Pensions and annuities.** DB survivor pensions are typically 60% (jointly-elected; a reduction applies to the original pension); CPP survivor rules in §9.3; OAS/GIS simply stop, and the survivor is reassessed as single for GIS (often *higher* GIS).
- **Probate (estate administration tax) 🔄 — provincial and highly variable:** Ontario 1.5% of estate value above $50,000; BC ≈ 1.4% above $50,000; Alberta a flat fee capped at $525; Quebec none for notarial wills. Assets passing by beneficiary designation (RRSP, RRIF, TFSA, insurance) or joint ownership bypass probate. Second-order for a tax engine but material for an estate-value output.
- **Final-year credits.** The deceased still gets full personal credits for the year; medical expenses for the last 24 months and charitable gifts (up to 100% of income) can be claimed on the terminal return or by the estate (graduated-rate estate rules).

**Sources:** CRA, *Guide T4011 Preparing Returns for Deceased Persons*; CRA, *Death of an RRSP annuitant* and *Death of a RRIF annuitant*; CRA, *Death of a TFSA holder*; Ontario Ministry of Finance, *Estate Administration Tax*; provincial probate-fee schedules.

---

## 14. Annual maintenance calendar (what changes when, and where to look)

| When | What is published | Feeds which YAML | Source document |
|---|---|---|---|
| **Late Sept / late Dec / late Mar / late Jun** | Next quarter's OAS, GIS, Allowance maxima and GIS cutoffs | `oas`, `gis` | Service Canada — *Old Age Security payment amounts*; ESDC quarterly rate card |
| **~Nov 1** | YMPE, YAMPE, CPP/QPP contribution rates and maxima for next year | `cpp`, `qpp` | CRA — *CPP contribution rates, maximums and exemptions*; Retraite Québec — *QPP figures* |
| **Mid/late Nov** | Federal indexation factor; brackets; BPA; age amount; all indexed credits; TFSA limit; RRSP/MP/DB limits; OAS recovery threshold for the coming income year | `federal_tax`, `tfsa`, `rrsp`, `oas` | CRA — *Indexation adjustment for personal income tax and benefit amounts*; CRA — *MP, DB, RRSP, DPSP, ALDA, TFSA limits and the YMPE* |
| **Nov–Dec** | Quebec indexation factor, brackets, credits, HSF thresholds, RAMQ premium | `quebec_tax` | Ministère des Finances du Québec — *Parameters of the Personal Income Tax System*; Revenu Québec |
| **Dec–Jan** | Provincial T4032 payroll tables (each province's brackets, BPA, factor); Ontario surtax triggers | `provincial_tax/*` | CRA — *T4032 Payroll Deductions Tables* (January edition) |
| **Late Dec / Jan** | CPP/QPP annual benefit indexation; new-year maximum and average pensions; survivor/disability maxima | `cpp`, `qpp` | canada.ca — *CPP pensions and benefits monthly amounts*; Retraite Québec |
| **Jan** | LIF maximum factors for the year (federal reference rate; Ontario table; Quebec prescribed rate) | `lif/*` | OSFI; FSRA; Retraite Québec — *Flash Retirement* |
| **Feb–Apr** | Provincial budgets (rate changes usually take effect the *following* January, sometimes mid-year) | `provincial_tax/*` | each provincial Ministry of Finance budget |
| **Spring / Fall** | Federal budget (Budget 2025 was Nov 4, 2025; budgets are now expected in the fall) — watch for RRIF, capital-gains, credit-rate and benefit changes | any | Department of Finance Canada |
| **July 1** | New benefit year: GIS/Allowance reset to prior-year income; CCB, GST credit and A-CESG/CLB income thresholds; OAS recovery period rolls to prior-year income | `gis`, `benefits`, `resp` | Service Canada; CRA; ESDC *CLB income brackets notice* |
| **Quarterly** | CRA prescribed interest rate (spousal loans, refund/arrears interest) | `misc` | CRA — *Prescribed interest rates* |

**Test-suite implications.** Every YAML figure marked 🔄 should carry a `valid_from` / `valid_to` and a `checked_on` date, and the test suite should (a) assert the combined top marginal rate per jurisdiction (§3.2), (b) assert the OAS full-clawback ceiling is *computed*, not stored, (c) reproduce the canada.ca GIS table for a zero-income single and a zero-income couple, (d) reproduce the 50%-inclusion vs. 138%-gross-up asymmetry between capital gains and eligible dividends in a benefit test, and (e) fail loudly when the simulation date passes the `valid_to` of any quarterly figure.

---

*Document prepared September 3, 2026 for the northplan engine. Illustrative figures are believed current as of that date; the YAML parameter files are authoritative.*