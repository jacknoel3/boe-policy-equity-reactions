# UK Equity Monetary Shocks

This repository contains code and supporting materials for a thesis project on the dynamic response of UK equity markets to Bank of England monetary policy shocks.

Main research question: "Do UK equity markets react asymmetrically to BoE monetary tightening vs easing?"

The analysis focuses on high-frequency monetary policy shocks decomposed into `Target`, `Path`, and `QE` components, with local projection specifications used to trace equity-market responses over time.

Project status: work in progress.

## Repository Structure

```text
research-project/
├── README.md
├── .gitignore
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx
│   ├── uk_equity_indices_PX_LAST_19970101_20260312.csv
│   └── diagnostics/
├── code/
├── output*/
│   ├── figures/
│   └── tables/
└── notebooks/
```

## Data

- `data/`: original input files used by the analysis scripts.
- `data/diagnostics/`: pre-analysis audit outputs created by `code/data_inspection.py`.

Generated model tables and figures are written to the versioned `output*` folders.

## Code

- `code/data_inspection.py`: pre-analysis diagnostics for monetary-shock and equity inputs.
- `code/lp_equity_monetary_shocks_v1.py`: baseline local projection model for FTSE 100 responses.
- `code/lp_equity_monetary_shocks_v2.py`: extensions and robustness checks for the baseline specification.
- `code/lp_equity_monetary_shocks_v3.py`: FTSE 250 robustness and subsample analysis.
- `code/lp_equity_monetary_shocks_v4.py`: regime-interaction extension.
- `code/lp_sector_analysis.py`: planned sector-level equity analysis.
- `code/utils.py`: shared helper functions.

## Output

- `output*/figures/`: saved impulse-response plots and related visual outputs.
- `output*/tables/`: regression tables and exported result files.

## Reproduction

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Add required data files to `data/`.
4. Optionally run `python code/data_inspection.py` to create pre-analysis diagnostics under `data/diagnostics/`.
5. Run scripts from the repository root or from within `code/`, depending on the script documentation.
6. Review generated tables and figures under `output*/`.

## Future Extensions

- FTSE 250 responses to monetary policy shocks.
- Sector analysis across UK equity indices.
- Robustness checks with alternative controls and specifications.
