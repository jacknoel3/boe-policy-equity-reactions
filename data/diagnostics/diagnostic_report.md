# Data Diagnostics

This report audits the source files before the local-projection analysis. Outlier tables flag notable observations; they do not imply deletion or winsorisation.

## Files
- `data/measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx`: monetary policy shocks (196,126 bytes)
- `data/uk_equity_indices_PX_LAST_19970101_20260312.csv`: consolidated equity prices (2,936,551 bytes)

## Monetary Shocks
- `target`: 419 nonzero events, range 1997-06-06 to 2025-11-06, std 0.0467332, min -0.421088, max 0.226287
- `path`: 419 nonzero events, range 1997-06-06 to 2025-11-06, std 0.0324032, min -0.143419, max 0.166905
- `qe`: 419 nonzero events, range 1997-06-06 to 2025-11-06, std 0.0287187, min -0.162213, max 0.125745

## Equity Series
- `ftse100` matched on Label=`FTSE 100`: 7,376 observations from 1997-01-02 to 2026-03-12; return min -11.5124%, max 9.3843%
- `ftse250` matched on Label=`FTSE 250`: 7,376 observations from 1997-01-02 to 2026-03-12; return min -9.8202%, max 8.0388%
- `ftse_all_share` matched on Label=`FTSE All Share`: 7,376 observations from 1997-01-02 to 2026-03-12; return min -11.0836%, max 8.8107%

## Event-Date Merge
- `ftse100`: 376/376 shock event dates matched to equity trading dates (100.0%).
- `ftse250`: 376/376 shock event dates matched to equity trading dates (100.0%).
- `ftse_all_share`: 376/376 shock event dates matched to equity trading dates (100.0%).

## Output Files
- `data_file_inventory.csv`
- `shock_summary.csv`
- `shock_extreme_observations.csv`
- `equity_series_summary.csv`
- `equity_return_extremes.csv`
- `merge_diagnostics.csv`

Largest observations are intentionally reported for transparency. In this application, large shocks or returns may be economically meaningful event observations rather than data errors.
