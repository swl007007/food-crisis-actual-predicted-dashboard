#!/usr/bin/env python3
"""Refresh the standalone actual-vs-predicted dashboard from frozen Stage 3 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HTML_NAME = "actual_predicted_dashboard.html"
MANIFEST_NAME = "manifest.json"
SOURCE_ROOT_RELATIVE = Path("archived/release_20260624_reproducibility_inputs")
AUDIT_RELATIVE = Path("final_artifacts_in_paper_updated/artifact_source_audit.csv")
REPRODUCTION_RELATIVE = Path("CURRENT_RESULTS_REPRODUCTION.md")
REQUIRED_COLUMNS = (
    "FEWSNET_admin_code",
    "month_start",
    "y_true",
    "y_pred_partitioned",
)
MODEL_SOURCES = (
    ("GF", "GeoRF"),
    ("DT", "GeoRF-Single"),
)
SCOPE_HORIZONS = {
    "fs1": 4,
    "fs2": 8,
    "fs3": 12,
}
EXPECTED_ROWS = 62_189
EXPECTED_DATES = (
    "2021-02-01",
    "2021-06-01",
    "2021-10-01",
    "2022-02-01",
    "2022-06-01",
    "2022-10-01",
    "2023-02-01",
    "2023-06-01",
    "2023-10-01",
    "2024-02-01",
    "2024-06-01",
    "2024-10-01",
)
DATA_LINE_RE = re.compile(r"(?m)^const DASHBOARD_DATA = (.*);$")


class DashboardRefreshError(RuntimeError):
    """Raised when a source or dashboard contract is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the embedded dashboard payload from the current frozen no-leak "
            "GeoRF/GeoRF-Single Stage 3 prediction providers."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Food_Crisis_Cluster root. Auto-detected when the script remains in this checkout.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override the frozen Stage 3 provider directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the committed HTML against the providers without rewriting files.",
    )
    return parser.parse_args()


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / AUDIT_RELATIVE).is_file() and (
            candidate / SOURCE_ROOT_RELATIVE
        ).is_dir():
            return candidate
    raise DashboardRefreshError(
        "Could not auto-detect Food_Crisis_Cluster root; pass --project-root."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_display(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_embedded_payload(html_text: str) -> dict[str, Any]:
    match = DATA_LINE_RE.search(html_text)
    if match is None:
        raise DashboardRefreshError("Could not find the embedded DASHBOARD_DATA payload")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise DashboardRefreshError(f"Embedded DASHBOARD_DATA is invalid JSON: {exc}") from exc
    if not isinstance(payload.get("features"), list) or not payload["features"]:
        raise DashboardRefreshError("Embedded geometry features are missing")
    return payload


def load_source_audit(project_root: Path) -> pd.DataFrame:
    audit_path = project_root / AUDIT_RELATIVE
    audit = pd.read_csv(audit_path, dtype="string").fillna("")
    required = {"artifact_group", "artifact_type", "path", "status"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise DashboardRefreshError(f"Artifact source audit is missing columns: {missing}")
    return audit


def validate_audit_provider(
    audit: pd.DataFrame,
    provider_name: str,
    run_manifest: Path,
    project_root: Path,
) -> str:
    rows = audit[
        (audit["artifact_group"] == provider_name)
        & (audit["artifact_type"] == "provider_manifest")
    ]
    if len(rows) != 1:
        raise DashboardRefreshError(
            f"Expected one provider audit row for {provider_name}; found {len(rows)}"
        )
    row = rows.iloc[0]
    expected_path = relative_display(run_manifest, project_root)
    if row["status"] != "clean" or row["path"].replace("\\\\", "/") != expected_path:
        raise DashboardRefreshError(
            f"Provider audit mismatch for {provider_name}: status={row['status']!r}, "
            f"path={row['path']!r}, expected={expected_path!r}"
        )
    return str(row["status"])


def normalize_prediction_frame(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing:
        raise DashboardRefreshError(f"{path} is missing required columns: {missing}")

    frame = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"FEWSNET_admin_code": "string"},
    )
    if len(frame) != EXPECTED_ROWS:
        raise DashboardRefreshError(
            f"{path} has {len(frame):,} rows; expected {EXPECTED_ROWS:,}"
        )

    frame["FEWSNET_admin_code"] = frame["FEWSNET_admin_code"].str.strip()
    if frame["FEWSNET_admin_code"].isna().any() or (
        frame["FEWSNET_admin_code"] == ""
    ).any():
        raise DashboardRefreshError(f"{path} contains missing admin codes")

    parsed_months = pd.to_datetime(frame["month_start"], errors="coerce")
    if parsed_months.isna().any():
        raise DashboardRefreshError(f"{path} contains unparseable month_start values")
    frame["month_start"] = parsed_months.dt.strftime("%Y-%m-%d")

    dates = tuple(sorted(frame["month_start"].unique().tolist()))
    if dates != EXPECTED_DATES:
        raise DashboardRefreshError(
            f"{path} has unexpected dates: {dates}; expected {EXPECTED_DATES}"
        )

    for column in ("y_true", "y_pred_partitioned"):
        if frame[column].isna().any():
            raise DashboardRefreshError(f"{path} contains missing {column} values")
        numeric = pd.to_numeric(frame[column], errors="raise").astype(int)
        values = set(numeric.unique().tolist())
        if not values.issubset({0, 1}):
            raise DashboardRefreshError(
                f"{path} contains non-binary {column} values: {sorted(values)}"
            )
        frame[column] = numeric

    duplicate_count = int(
        frame.duplicated(["FEWSNET_admin_code", "month_start"]).sum()
    )
    if duplicate_count:
        raise DashboardRefreshError(
            f"{path} contains {duplicate_count} duplicate admin-month keys"
        )
    return frame


def frame_key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["FEWSNET_admin_code"], frame["month_start"], strict=True))


def frame_actual_map(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (row.FEWSNET_admin_code, row.month_start): int(row.y_true)
        for row in frame.itertuples(index=False)
    }


def build_datasets(
    project_root: Path,
    input_dir: Path,
) -> tuple[
    dict[str, dict[str, dict[str, int]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    audit = load_source_audit(project_root)
    datasets: dict[str, dict[str, dict[str, int]]] = {}
    included_sources: list[dict[str, Any]] = []
    reference_keys: set[tuple[str, str]] | None = None
    reference_actual: dict[tuple[str, str], int] | None = None

    for token, model in MODEL_SOURCES:
        for scope, horizon_months in SCOPE_HORIZONS.items():
            provider_name = f"result_partition_k40_compare_{token}_{scope}"
            provider_dir = input_dir / provider_name
            predictions_path = provider_dir / "predictions_monthly.csv"
            run_manifest_path = provider_dir / "run_manifest.json"
            if not predictions_path.is_file() or not run_manifest_path.is_file():
                raise DashboardRefreshError(
                    f"Missing frozen provider files under {provider_dir}"
                )

            audit_status = validate_audit_provider(
                audit,
                provider_name,
                run_manifest_path,
                project_root,
            )
            frame = normalize_prediction_frame(predictions_path)
            keys = frame_key_set(frame)
            actual = frame_actual_map(frame)
            if reference_keys is None:
                reference_keys = keys
                reference_actual = actual
            else:
                missing_keys = reference_keys - keys
                extra_keys = keys - reference_keys
                if missing_keys or extra_keys:
                    raise DashboardRefreshError(
                        f"Key population mismatch for {provider_name}: "
                        f"missing={len(missing_keys):,}, extra={len(extra_keys):,}"
                    )
                mismatched_actual = sum(
                    actual[key] != reference_actual[key] for key in reference_keys
                )
                if mismatched_actual:
                    raise DashboardRefreshError(
                        f"Actual-label mismatch for {provider_name}: "
                        f"{mismatched_actual:,} admin-month keys"
                    )

            for month, month_frame in frame.groupby("month_start", sort=True):
                datasets[f"{model}|{scope}|{month}"] = {
                    row.FEWSNET_admin_code: {
                        "actual": int(row.y_true),
                        "predicted": int(row.y_pred_partitioned),
                    }
                    for row in month_frame.itertuples(index=False)
                }

            included_sources.append(
                {
                    "model": model,
                    "model_token": token,
                    "scope": scope,
                    "horizon_months": horizon_months,
                    "path": relative_display(predictions_path, project_root),
                    "sha256": sha256_file(predictions_path),
                    "provider_manifest": relative_display(
                        run_manifest_path, project_root
                    ),
                    "provider_manifest_sha256": sha256_file(run_manifest_path),
                    "artifact_source_audit_status": audit_status,
                    "row_count": int(len(frame)),
                    "distinct_admin_codes": int(
                        frame["FEWSNET_admin_code"].nunique()
                    ),
                    "available_dates": list(EXPECTED_DATES),
                }
            )

    expected_dataset_count = len(MODEL_SOURCES) * len(SCOPE_HORIZONS) * len(
        EXPECTED_DATES
    )
    if len(datasets) != expected_dataset_count:
        raise DashboardRefreshError(
            f"Built {len(datasets)} dashboard datasets; expected {expected_dataset_count}"
        )

    availability = {
        "models": [model for _, model in MODEL_SOURCES],
        "scopesByModel": {
            model: list(SCOPE_HORIZONS) for _, model in MODEL_SOURCES
        },
        "datesByModelScope": {
            f"{model}|{scope}": list(EXPECTED_DATES)
            for _, model in MODEL_SOURCES
            for scope in SCOPE_HORIZONS
        },
    }
    return datasets, included_sources, availability


def replace_or_confirm(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new and new in text:
        return text
    if not new:
        return text
    raise DashboardRefreshError(f"Could not apply HTML transformation: {label}")


def remove_alpha_interaction(html_text: str) -> str:
    replacements = (
        (
            ".controls { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr));",
            ".controls { display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr));",
            "three-column controls",
        ),
        (
            "select, input[type=range] { width: 100%; }",
            "select { width: 100%; }",
            "select-only width rule",
        ),
        (
            "Exploratory diagnostics using existing GeoRF/GeoRF-single month-ind outputs and the global FEWSNET shapefile.",
            "Paper-aligned interactive companion using the current frozen no-temporal-leak GeoRF/GeoRF-Single Stage 3 outputs.",
            "header provenance",
        ),
        (
            '<div class="control"><label for="scopeSelect">forecasting horizon</label><select id="scopeSelect"></select></div>',
            '<div class="control"><label for="scopeSelect">Forecasting horizon</label><select id="scopeSelect"></select></div>',
            "scope label",
        ),
        (
            '  <div class="control"><label for="leftAlpha">Left prediction alpha: <span id="leftAlphaValue">0.00</span></label><input id="leftAlpha" type="range" min="0" max="1" step="0.05" value="0"></div>\n',
            "",
            "left alpha control",
        ),
        (
            '  <div class="control"><label for="rightAlpha">Right prediction alpha: <span id="rightAlphaValue">1.00</span></label><input id="rightAlpha" type="range" min="0" max="1" step="0.05" value="1"></div>\n',
            "",
            "right alpha control",
        ),
        (
            "Base layer: actual labels (y_true). Prediction overlay defaults to alpha 0.",
            "Observed binary crisis labels (y_true) for the selected target month.",
            "actual subtitle",
        ),
        (
            "Base layer: actual labels (y_true). Prediction overlay defaults to alpha 1.",
            "Partitioned binary predictions (y_pred_partitioned) for the same polygons and target month.",
            "predicted subtitle",
        ),
        (
            "const leftAlpha = document.getElementById('leftAlpha');\nconst rightAlpha = document.getElementById('rightAlpha');\nconst leftAlphaValue = document.getElementById('leftAlphaValue');\nconst rightAlphaValue = document.getElementById('rightAlphaValue');\n",
            "",
            "alpha DOM bindings",
        ),
        (
            "function fillSelect(select, values) {\n  select.textContent = '';\n  values.forEach(value => select.appendChild(option(value, value)));\n}",
            "function fillSelect(select, values, labels = {}) {\n  select.textContent = '';\n  values.forEach(value => select.appendChild(option(value, labels[value] || value)));\n}",
            "labeled selectors",
        ),
        (
            "  fillSelect(scopeSelect, DASHBOARD_DATA.availability.scopesByModel[model] || []);",
            "  fillSelect(scopeSelect, DASHBOARD_DATA.availability.scopesByModel[model] || [], DASHBOARD_DATA.horizonLabels);",
            "horizon labels",
        ),
        (
            "function buildLayer(records, field, opacity) {\n  const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');\n  group.setAttribute('class', field === 'actual' ? 'base' : 'overlay');\n  group.setAttribute('opacity', opacity);",
            "function buildLayer(records, field) {\n  const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');\n  group.setAttribute('class', field === 'actual' ? 'base' : 'overlay');",
            "single-layer builder",
        ),
        (
            "function renderMap(svg, records, predOpacity) {\n  svg.textContent = '';\n  svg.appendChild(buildLayer(records, 'actual', 1));\n  svg.appendChild(buildLayer(records, 'predicted', predOpacity));\n}",
            "function renderMap(svg, records, field) {\n  svg.textContent = '';\n  svg.appendChild(buildLayer(records, field));\n}",
            "fixed panel semantics",
        ),
        (
            "  leftAlphaValue.textContent = Number(leftAlpha.value).toFixed(2);\n  rightAlphaValue.textContent = Number(rightAlpha.value).toFixed(2);\n",
            "",
            "alpha value rendering",
        ),
        (
            "  renderMap(document.getElementById('actualMap'), records, Number(leftAlpha.value));\n  renderMap(document.getElementById('predictedMap'), records, Number(rightAlpha.value));",
            "  renderMap(document.getElementById('actualMap'), records, 'actual');\n  renderMap(document.getElementById('predictedMap'), records, 'predicted');",
            "fixed map fields",
        ),
        (
            "leftAlpha.addEventListener('input', renderAll);\nrightAlpha.addEventListener('input', renderAll);\n",
            "",
            "alpha listeners",
        ),
        (
            "document.getElementById('provenance').textContent = `Generated from ${DASHBOARD_DATA.manifest.source_directory} | Shapefile: ${DASHBOARD_DATA.manifest.shapefile_path} | Threshold: ${DASHBOARD_DATA.manifest.threshold_contract}`;",
            "document.getElementById('provenance').textContent = `Frozen providers: ${DASHBOARD_DATA.manifest.source_directory} | Geometry: ${DASHBOARD_DATA.manifest.geometry.source_name} | Labels: ${DASHBOARD_DATA.manifest.label_contract.actual_field} vs ${DASHBOARD_DATA.manifest.label_contract.predicted_field} | Threshold: ${DASHBOARD_DATA.manifest.threshold_contract}`;",
            "public provenance footer",
        ),
    )
    for old, new, label in replacements:
        html_text = replace_or_confirm(html_text, old, new, label)
    return html_text


def build_manifest(
    project_root: Path,
    input_dir: Path,
    included_sources: list[dict[str, Any]],
    availability: dict[str, Any],
    features: list[dict[str, str]],
    old_manifest: dict[str, Any],
    datasets: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    geometry_codes = {str(feature["code"]) for feature in features}
    preferred_key = "GeoRF|fs2|2024-06-01"
    preferred_records = datasets[preferred_key]
    source_name = str(old_manifest.get("shapefile_path", "FEWS_Admin_LZ_v3.shp"))
    source_name = source_name.replace("\\", "/").rsplit("/", 1)[-1]
    return {
        "schema_version": 2,
        "artifact_role": "paper-aligned interactive companion",
        "workflow_mode": "current frozen no-temporal-leak Stage 3 release",
        "production_status": "interactive companion; not a replacement for static paper figures",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_directory": relative_display(input_dir, project_root),
        "final_artifact_contract": {
            "artifact_source_audit": AUDIT_RELATIVE.as_posix(),
            "current_results_reproduction": REPRODUCTION_RELATIVE.as_posix(),
            "provider_status_required": "clean",
        },
        "included_sources": included_sources,
        "join_key": "FEWSNET_admin_code",
        "available_models": availability["models"],
        "available_scopes": list(SCOPE_HORIZONS),
        "scope_contract": {
            scope: {"forecast_horizon_months": horizon}
            for scope, horizon in SCOPE_HORIZONS.items()
        },
        "available_dates": list(EXPECTED_DATES),
        "label_contract": {
            "actual_field": "y_true",
            "predicted_field": "y_pred_partitioned",
            "non_crisis_value": 0,
            "crisis_value": 1,
        },
        "threshold_contract": "none; existing binary labels only",
        "geometry": {
            "source_name": source_name,
            "method": "reused embedded SVG paths from the archived baseline dashboard",
            "feature_count": len(features),
            "distinct_feature_codes": len(geometry_codes),
            "features_sha256": sha256_json(features),
            "baseline_generated_at_utc": old_manifest.get("generated_at_utc"),
        },
        "smoke_test": {
            "model": "GeoRF",
            "scope": "fs2",
            "date": "2024-06-01",
            "join_rows": len(set(preferred_records) & geometry_codes),
            "passed": bool(set(preferred_records) & geometry_codes),
        },
        "validation": {
            "expected_rows_per_provider": EXPECTED_ROWS,
            "expected_provider_count": len(MODEL_SOURCES) * len(SCOPE_HORIZONS),
            "expected_dataset_count": len(datasets),
            "key_population_identical_across_providers": True,
            "actual_labels_identical_across_providers": True,
            "duplicate_admin_month_keys": 0,
        },
        "omissions_or_warnings": [
            "Geometry was not regenerated; verified baseline SVG paths were retained while prediction data were refreshed."
        ],
    }


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def validate_committed_dashboard(
    html_text: str,
    embedded_payload: dict[str, Any],
    expected_payload: dict[str, Any],
) -> None:
    forbidden = (
        "leftAlpha",
        "rightAlpha",
        'type="range"',
        "prediction alpha",
        "predOpacity",
        "group.setAttribute('opacity'",
    )
    present = [token for token in forbidden if token in html_text]
    if present:
        raise DashboardRefreshError(f"Alpha interaction remnants remain: {present}")
    if embedded_payload.get("datasets") != expected_payload["datasets"]:
        raise DashboardRefreshError("Embedded datasets do not match the frozen providers")
    if embedded_payload.get("availability") != expected_payload["availability"]:
        raise DashboardRefreshError("Embedded availability does not match providers")
    if embedded_payload.get("horizonLabels") != expected_payload["horizonLabels"]:
        raise DashboardRefreshError("Embedded horizon labels are incorrect")
    manifest = embedded_payload.get("manifest", {})
    expected_hashes = {
        (row["model"], row["scope"]): row["sha256"]
        for row in expected_payload["manifest"]["included_sources"]
    }
    committed_hashes = {
        (row["model"], row["scope"]): row["sha256"]
        for row in manifest.get("included_sources", [])
    }
    if committed_hashes != expected_hashes:
        raise DashboardRefreshError("Committed provider hashes do not match current frozen files")
    if manifest.get("smoke_test", {}).get("passed") is not True:
        raise DashboardRefreshError("Committed geometry join smoke test did not pass")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = (
        args.project_root.resolve()
        if args.project_root
        else find_project_root(script_dir)
    )
    input_dir = (
        args.input_dir.resolve()
        if args.input_dir
        else project_root / SOURCE_ROOT_RELATIVE
    )
    html_path = script_dir / HTML_NAME
    manifest_path = script_dir / MANIFEST_NAME
    if not html_path.is_file():
        raise DashboardRefreshError(f"Dashboard HTML not found: {html_path}")

    original_html = html_path.read_text(encoding="utf-8")
    original_payload = read_embedded_payload(original_html)
    features = original_payload["features"]
    old_manifest = original_payload.get("manifest", {})
    datasets, included_sources, availability = build_datasets(
        project_root,
        input_dir,
    )
    manifest = build_manifest(
        project_root,
        input_dir,
        included_sources,
        availability,
        features,
        old_manifest,
        datasets,
    )
    payload = {
        "features": features,
        "datasets": datasets,
        "availability": availability,
        "horizonLabels": {
            scope: f"{months}-month horizon ({scope})"
            for scope, months in SCOPE_HORIZONS.items()
        },
        "colors": original_payload["colors"],
        "manifest": manifest,
    }

    if args.check:
        validate_committed_dashboard(original_html, original_payload, payload)
        print(
            "Dashboard verified: 6 providers, 72 model-scope-month datasets, "
            "identical keys/actual labels, current hashes, no alpha interaction."
        )
        return 0

    updated_html = remove_alpha_interaction(original_html)
    updated_html = DATA_LINE_RE.sub(
        f"const DASHBOARD_DATA = {compact_json(payload)};",
        updated_html,
        count=1,
    )
    validate_committed_dashboard(updated_html, read_embedded_payload(updated_html), payload)

    html_tmp = html_path.with_suffix(".html.tmp")
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    html_tmp.write_text(updated_html, encoding="utf-8")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    html_tmp.replace(html_path)
    manifest_tmp.replace(manifest_path)

    print(f"Updated: {html_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Providers: {len(included_sources)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Geometry features: {len(features)}")
    print(f"Join smoke test: {manifest['smoke_test']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DashboardRefreshError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
