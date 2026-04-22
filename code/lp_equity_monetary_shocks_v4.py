"""
Please implement a v4 extension of the current model that introduces additional regime distinctions in a structured and econometrically robust way.

IMPORTANT:
- Do NOT replace or break existing v2/v3 functionality
- Build v4 as an extension (new outputs / new specs)
- Minimal refactoring only where necessary
- Preserve all existing plots and subsample figures

--------------------------------------------------
1) OBJECTIVE
--------------------------------------------------

Move from simple subsample splits to a structured regime framework that distinguishes:

1) Policy regime changes
2) Communication regime changes
3) Economic environment regimes

--------------------------------------------------
2) DEFINE REGIME DATES (CONSTANTS)
--------------------------------------------------

Define once and reuse:

QE_START_DATE = "2009-03-05"
SUPER_THURSDAY_START_DATE = "2015-08-06"
COVID_START_DATE = "2020-03-01"
QT_START_DATE = "2022-01-01"   # approximate start of tightening cycle

--------------------------------------------------
3) CREATE REGIME DUMMIES
--------------------------------------------------

Add boolean/dummy variables to the dataset:

- post_qe = date >= 2009-03-05
- post_super_thursday = date >= 2015-08-06
- covid_period = date >= 2020-03-01
- qt_period = date >= 2022-01-01

Ensure these are computed once and reused everywhere.

--------------------------------------------------
4) ADD INTERACTION-BASED LOCAL PROJECTIONS
--------------------------------------------------

Extend the existing LP specification by adding interaction terms:

For each shock (TARGET, PATH, QE where applicable):

Estimate models including:

- Shock
- Shock × post_qe
- Shock × post_super_thursday
- Shock × covid_period
- Shock × qt_period

Do NOT remove the baseline specification — add this as a new v4 specification.

--------------------------------------------------
5) OUTPUTS
--------------------------------------------------

Generate new IRF-style plots showing:

- baseline response
- regime-adjusted responses using interactions

Options:
- either plot separate IRFs per regime
- or plot differences (interaction effects)

Keep plotting style consistent with v3.

--------------------------------------------------
6) QE HANDLING
--------------------------------------------------

Maintain previous rule:

- QE sample restricted to date >= 2009-03-05
- Apply regime dummies within that restricted sample

--------------------------------------------------
7) DO NOT ADD 2013 SPLIT
--------------------------------------------------

Do NOT create a separate forward-guidance subsample split.

--------------------------------------------------
8) SUMMARY OUTPUT
--------------------------------------------------

After implementation, report:

- where regime dummies were added
- how interaction models were implemented
- which new figures were created
- any small refactors required

--------------------------------------------------

Goal:
Upgrade the analysis from simple time splits to a structured regime-dependent framework, while preserving clarity and statistical power.
"""
