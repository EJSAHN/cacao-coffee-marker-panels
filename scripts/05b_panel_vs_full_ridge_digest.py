
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def safe_divide(a: float, b: float) -> float:
    try:
        if not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare low-density marker panels with full-marker ridge benchmarks.")
    parser.add_argument("--project-root", default=".", help="Project root containing analysis/outputs.")
    parser.add_argument("--deployment-r-threshold", type=float, default=0.40, help="Minimum cluster-out Pearson r for deployment shortlist.")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    pred_dir = root / "analysis/outputs" / "prediction_benchmarks"
    panel_dir = root / "analysis/outputs" / "marker_panel_simulation"
    digest_dir = panel_dir / "digest"
    digest_dir.mkdir(parents=True, exist_ok=True)

    pred = read_tsv(pred_dir / "04_prediction_summary.tsv")
    panel = read_tsv(panel_dir / "05_marker_panel_summary.tsv")

    full = pred[(pred["model"] == "ridge")].copy()
    full = full.rename(columns={
        "pooled_pearson_r": "full_ridge_pearson_r",
        "pooled_r2": "full_ridge_r2",
        "pooled_rmse": "full_ridge_rmse",
    })
    full_cols = ["dataset", "trait", "validation_scheme", "full_ridge_pearson_r", "full_ridge_r2", "full_ridge_rmse"]
    full = full[[c for c in full_cols if c in full.columns]].drop_duplicates(["dataset", "trait", "validation_scheme"])

    all_panels = panel.merge(full, on=["dataset", "trait", "validation_scheme"], how="left")
    all_panels["delta_r_vs_full_ridge"] = all_panels["pearson_r_mean"] - all_panels["full_ridge_pearson_r"]
    all_panels["fraction_of_full_ridge_r"] = [safe_divide(a, b) for a, b in zip(all_panels["pearson_r_mean"], all_panels["full_ridge_pearson_r"])]

    sort_cols = ["dataset", "trait", "validation_scheme", "pearson_r_mean", "r2_mean"]
    best = all_panels.sort_values(sort_cols, ascending=[True, True, True, False, False]).groupby(
        ["dataset", "trait", "validation_scheme"], as_index=False
    ).head(1)

    def minimum_panel(threshold: float) -> pd.DataFrame:
        rows = []
        for keys, sub in all_panels.groupby(["dataset", "trait", "validation_scheme"]):
            sub = sub.copy()
            full_r = pd.to_numeric(sub["full_ridge_pearson_r"], errors="coerce").dropna()
            if full_r.empty or full_r.iloc[0] <= 0:
                continue
            target = threshold * full_r.iloc[0]
            candidates = sub[pd.to_numeric(sub["pearson_r_mean"], errors="coerce") >= target]
            if candidates.empty:
                row = sub.sort_values("pearson_r_mean", ascending=False).iloc[0].to_dict()
                row["threshold_met"] = False
            else:
                row = candidates.sort_values(["panel_size_requested", "pearson_r_mean"], ascending=[True, False]).iloc[0].to_dict()
                row["threshold_met"] = True
            row["threshold"] = threshold
            row["target_r"] = target
            rows.append(row)
        return pd.DataFrame(rows)

    min80 = minimum_panel(0.80)
    min90 = minimum_panel(0.90)

    deployment = best[
        (best["validation_scheme"] == "cluster_out") &
        (pd.to_numeric(best["pearson_r_mean"], errors="coerce") >= args.deployment_r_threshold) &
        (pd.to_numeric(best["r2_mean"], errors="coerce") > 0)
    ].copy()
    deployment = deployment.sort_values("pearson_r_mean", ascending=False)

    paths = {
        "best": write_tsv(best, digest_dir / "05b_best_panel_vs_full_ridge.tsv"),
        "all": write_tsv(all_panels, digest_dir / "05b_all_panels_vs_full_ridge.tsv"),
        "min80": write_tsv(min80, digest_dir / "05b_minimum_panel_for_80pct_full_ridge.tsv"),
        "min90": write_tsv(min90, digest_dir / "05b_minimum_panel_for_90pct_full_ridge.tsv"),
        "deployment": write_tsv(deployment, digest_dir / "05b_deployment_shortlist_cluster_out.tsv"),
    }

    config_path = panel_dir / "05_marker_panel_run_config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}

    lines = [
        "# Panel-vs-full-ridge digest",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "This step compares the best low-density ridge marker-panel results against the full-marker ridge benchmark from the prediction benchmark step.",
        "",
        "## Configuration",
        "",
        f"- Datasets: {', '.join(config.get('datasets', sorted(panel['dataset'].dropna().unique())))}",
        f"- Validation schemes: {', '.join(config.get('schemes', sorted(panel['validation_scheme'].dropna().unique())))}",
        f"- Marker-selection strategies: {', '.join(config.get('strategies', sorted(panel['panel_strategy'].dropna().unique())))}",
        f"- Marker panel sizes: {config.get('panel_sizes', sorted(panel['panel_size_requested'].dropna().unique().tolist()))}",
        "",
        "## Outputs",
        "",
    ]
    for path in paths.values():
        lines.append(f"- {path.name}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Trait-predictive ANOVA-F panels are selected within each training fold and should be treated as candidate panels rather than finalized assays.",
        "- Cluster-out results are prioritized for deployment-oriented interpretation.",
    ])
    report_path = digest_dir / "05b_panel_vs_full_ridge_digest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Panel-vs-full-ridge digest written: {report_path}")


if __name__ == "__main__":
    main()
