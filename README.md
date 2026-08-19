# Actual vs Predicted Food Crisis Dashboard

This standalone repository publishes an interactive companion to the current
`Food_Crisis_Cluster/final_artifacts_in_paper_updated` release. The left map is
always the observed binary FEWSNET crisis label (`y_true`); the right map is
always the corresponding fixed-partition prediction (`y_pred_partitioned`).

The dashboard contains:

- GeoRF and GeoDT;
- 4-, 8-, and 12-month horizons (`fs1`, `fs2`, and `fs3`);
- the 12 evaluated February/June/October target months from 2021 through 2024;
- embedded SVG geometry and data, with no runtime API or CDN dependency.

GitHub Pages entry point: `index.html` redirects to
`actual_predicted_dashboard.html`.

## Data contract and provenance

`refresh_dashboard.py` reads the six frozen Stage 3
`predictions_monthly.csv` providers recorded as `clean` in:

```text
final_artifacts_in_paper_updated/artifact_source_audit.csv
```

Before writing, it requires:

- exactly 62,189 rows per provider;
- unique `(FEWSNET_admin_code, month_start)` keys;
- the same key population and `y_true` values across all six providers;
- binary `y_true` and `y_pred_partitioned` values;
- the expected 12 target months;
- a successful join between the retained SVG geometry and GeoRF fs2 for
  June 2024.

The generated `manifest.json` records source paths, SHA-256 hashes, scope
definitions, label definitions, and validation results. No additional
probability threshold is applied.

## Refresh and verify

Run from this directory while it remains inside a
`Food_Crisis_Cluster` checkout:

```bash
python3 refresh_dashboard.py
python3 refresh_dashboard.py --check
```

When this repository is cloned separately, point the script to a checkout that
contains the frozen providers:

```bash
python3 refresh_dashboard.py --project-root /path/to/Food_Crisis_Cluster
```

## Version history

The local repository begins with an untouched archived-dashboard baseline, so
the original payload and alpha controls remain recoverable from Git history.
The public `main` branch is deployed by `.github/workflows/pages.yml`.
