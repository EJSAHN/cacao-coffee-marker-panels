from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prediction benchmark tables.")
    parser.add_argument("--project-root", default=".", help="Project folder containing analysis outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    pred_dir = root / "analysis" / "outputs" / "prediction_benchmarks"
    out_dir = pred_dir / "digest"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = pred_dir / "04_prediction_summary.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing prediction summary: {summary_path}")

    df = pd.read_csv(summary_path, sep="\t")
    metric_cols = ["pooled_pearson_r", "pooled_spearman_r", "pooled_rmse", "pooled_mae", "pooled_r2"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["dataset", "trait", "trait_label", "validation_scheme"]
    best = (
        df.sort_values(["pooled_pearson_r", "pooled_r2"], ascending=[False, False])
          .groupby(group_cols, as_index=False)
          .first()
    )
    best.to_csv(out_dir / "04b_best_model_per_trait_scheme.tsv", sep="\t", index=False)

    for scheme in sorted(best["validation_scheme"].dropna().unique()):
        top = best[best["validation_scheme"] == scheme].sort_values("pooled_pearson_r", ascending=False)
        top.to_csv(out_dir / f"04b_top_best_models_{scheme}.tsv", sep="\t", index=False)

    baseline = best[best["validation_scheme"] == "random_five_fold"].set_index(["dataset", "trait", "trait_label"])
    stress = best[best["validation_scheme"] == "cluster_out"].set_index(["dataset", "trait", "trait_label"])
    common = baseline[["model", "pooled_pearson_r", "pooled_r2"]].join(
        stress[["model", "pooled_pearson_r", "pooled_r2"]],
        how="inner",
        lsuffix="_random",
        rsuffix="_cluster",
    ).reset_index()
    if not common.empty:
        common["delta_r_random_minus_cluster"] = common["pooled_pearson_r_random"] - common["pooled_pearson_r_cluster"]
        common["delta_r2_random_minus_cluster"] = common["pooled_r2_random"] - common["pooled_r2_cluster"]
        common = common.sort_values("delta_r_random_minus_cluster", ascending=False)
    common.to_csv(out_dir / "04b_random_vs_cluster_drop.tsv", sep="\t", index=False)

    if not common.empty:
        dataset_drop = (
            common.groupby("dataset", as_index=False)
            .agg(
                n_traits=("trait", "count"),
                mean_random_r=("pooled_pearson_r_random", "mean"),
                mean_cluster_r=("pooled_pearson_r_cluster", "mean"),
                mean_delta_r=("delta_r_random_minus_cluster", "mean"),
                median_delta_r=("delta_r_random_minus_cluster", "median"),
            )
            .sort_values("mean_delta_r", ascending=False)
        )
    else:
        dataset_drop = pd.DataFrame()
    dataset_drop.to_csv(out_dir / "04b_dataset_level_random_vs_cluster_drop.tsv", sep="\t", index=False)

    wins = (
        best.groupby(["validation_scheme", "dataset", "model"], as_index=False)
        .size()
        .rename(columns={"size": "n_trait_wins"})
        .sort_values(["validation_scheme", "dataset", "n_trait_wins"], ascending=[True, True, False])
    )
    wins.to_csv(out_dir / "04b_model_win_counts.tsv", sep="\t", index=False)

    robust = best[
        (best["validation_scheme"] == "cluster_out")
        & (best["pooled_pearson_r"] >= 0.30)
        & (best["pooled_r2"] > 0)
    ].copy()
    robust = robust.sort_values("pooled_pearson_r", ascending=False)
    robust.to_csv(out_dir / "04b_cluster_out_shortlist.tsv", sep="\t", index=False)

    report = [
        "# Prediction benchmark digest",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Input table: `{summary_path.relative_to(root)}`",
        f"Total model-by-trait rows: {len(df)}",
        f"Best model rows: {len(best)}",
        "",
        "## Cluster-out shortlist",
        "",
    ]
    if robust.empty:
        report.append("No cluster-out entries met the selected shortlist thresholds.")
    else:
        for _, row in robust.head(15).iterrows():
            report.append(
                f"- {row['dataset']} / {row.get('trait_label', row['trait'])}: "
                f"{row['model']}, Pearson r={row['pooled_pearson_r']:.3f}, R2={row['pooled_r2']:.3f}."
            )
    (out_dir / "04b_prediction_digest_report.md").write_text("\n".join(report), encoding="utf-8")

    print("Prediction digest complete")
    print(out_dir / "04b_prediction_digest_report.md")


if __name__ == "__main__":
    main()
