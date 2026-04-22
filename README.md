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
│   ├── raw/
│   ├── interim/
│   └── processed/
├── code/
├── output/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── docs/
└── notebooks/
```

## Data

- `data/raw/`: original source files as obtained from vendors, spreadsheets, or manual downloads.
- `data/interim/`: cleaned intermediate files produced during inspection and preparation.
- `data/processed/`: analysis-ready datasets used in estimation and plotting.

Some large or proprietary data may not be redistributed through this repository and may need to be added manually to `data/raw/`.

## Code

- `code/data_inspection.py`: initial inspection of raw monetary-shock and equity files.
- `code/lp_equity_monetary_shocks_v1.py`: baseline local projection model for FTSE 100 responses.
- `code/lp_equity_monetary_shocks_v2.py`: planned extensions and robustness checks for the baseline specification.
- `code/lp_ftse250.py`: planned FTSE 250 analysis.
- `code/lp_sector_analysis.py`: planned sector-level equity analysis.
- `code/utils.py`: shared helper functions for future scripts.

## Output

- `output/figures/`: saved impulse-response plots and related visual outputs.
- `output/tables/`: regression tables and exported result files.
- `output/logs/`: run logs and execution notes.

## Reproduction

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Add required raw data files to `data/raw/`.
4. Run scripts from the repository root or from within `code/`, depending on the script documentation.
5. Review generated tables and figures under `output/`.

## Future Extensions

- FTSE 250 responses to monetary policy shocks.
- Sector analysis across UK equity indices.
- Robustness checks with alternative controls and specifications.
