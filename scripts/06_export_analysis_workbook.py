
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from tropical_perennial_breeding.labels import DATASET_LABELS, MODEL_LABELS, PANEL_STRATEGY_LABELS, TRAIT_LABELS, VALIDATION_LABELS


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", compression="infer", low_memory=False)


def polish_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out.insert(out.columns.get_loc("dataset") + 1, "Dataset label", out["dataset"].map(DATASET_LABELS).fillna(out["dataset"]))
    if "trait" in out.columns:
        loc = out.columns.get_loc("trait") + 1
        out.insert(loc, "Trait label", out["trait"].map(TRAIT_LABELS).fillna(out["trait"]))
    if "validation_scheme" in out.columns:
        out["Validation scheme"] = out["validation_scheme"].map(VALIDATION_LABELS).fillna(out["validation_scheme"])
    if "model" in out.columns:
        out["Model label"] = out["model"].map(MODEL_LABELS).fillna(out["model"])
    if "panel_strategy" in out.columns:
        out["Panel strategy label"] = out["panel_strategy"].map(PANEL_STRATEGY_LABELS).fillna(out["panel_strategy"])
    return out


def write_sheet(writer: pd.ExcelWriter, name: str, df: pd.DataFrame) -> None:
    if df.empty:
        df = pd.DataFrame({"Note": ["No records were available for this sheet."]})
    df.to_excel(writer, sheet_name=name[:31], index=False)


def style_workbook(path: Path) -> None:
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="E9EEF7")
    thin = Side(style="thin", color="D9D9D9")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.border = Border(bottom=thin)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for col in ws.columns:
            max_len = 8
            col_letter = get_column_letter(col[0].column)
            for cell in col[:200]:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, min(len(val), 80))
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            ws.column_dimensions[col_letter].width = min(max_len + 2, 45)
    wb.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a public-facing analysis workbook from tabular pipeline outputs.")
    parser.add_argument("--project-root", default=".", help="Project root containing analysis/intermediate and analysis/outputs.")
    parser.add_argument("--output", default=None, help="Output Excel workbook path.")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out_path = Path(args.output).resolve() if args.output else root / "analysis/outputs" / "analysis_workbook.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "README": None,
        "Cleaned_dataset_summary": root / "analysis/intermediate" / "qc" / "cleaned_dataset_summary.tsv",
        "Identity_structure_QC": root / "analysis/intermediate" / "qc" / "03_qc_identity_structure_summary.tsv",
        "Prediction_summary": root / "analysis/outputs" / "prediction_benchmarks" / "04_prediction_summary.tsv",
        "Prediction_fold_metrics": root / "analysis/outputs" / "prediction_benchmarks" / "04_prediction_fold_metrics.tsv",
        "Prediction_trait_inventory": root / "analysis/outputs" / "prediction_benchmarks" / "04_prediction_trait_inventory.tsv",
        "Marker_panel_summary": root / "analysis/outputs" / "marker_panel_simulation" / "05_marker_panel_summary.tsv",
        "Marker_panel_best": root / "analysis/outputs" / "marker_panel_simulation" / "05_marker_panel_best_per_trait_scheme.tsv",
        "Marker_panel_fold_metrics": root / "analysis/outputs" / "marker_panel_simulation" / "05_marker_panel_fold_metrics.tsv",
        "Panel_vs_full_best": root / "analysis/outputs" / "marker_panel_simulation" / "digest" / "05b_best_panel_vs_full_ridge.tsv",
        "Minimum_panel_80pct": root / "analysis/outputs" / "marker_panel_simulation" / "digest" / "05b_minimum_panel_for_80pct_full_ridge.tsv",
        "Minimum_panel_90pct": root / "analysis/outputs" / "marker_panel_simulation" / "digest" / "05b_minimum_panel_for_90pct_full_ridge.tsv",
        "Deployment_shortlist": root / "analysis/outputs" / "marker_panel_simulation" / "digest" / "05b_deployment_shortlist_cluster_out.tsv",
    }
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        readme = pd.DataFrame({
            "Field": ["Purpose", "Primary metric", "Validation schemes", "Marker-panel note"],
            "Description": [
                "Analysis workbook containing the tabular outputs of the cacao and coffee deployment-aware prediction pipeline.",
                "Pearson correlation between observed and predicted phenotypes; RMSE and R-squared are reported where available.",
                "Random five-fold validation is a conventional baseline; cluster-out validation withholds genetic clusters as a population-structure-aware stress test.",
                "ANOVA-F panels are selected within each training fold and should be treated as candidate panels pending locked-panel validation.",
            ],
        })
        write_sheet(writer, "README", readme)
        for sheet, path in paths.items():
            if sheet == "README":
                continue
            df = polish_values(read_tsv(path))
            write_sheet(writer, sheet, df)
    style_workbook(out_path)
    print(f"Analysis workbook written: {out_path}")


if __name__ == "__main__":
    main()
