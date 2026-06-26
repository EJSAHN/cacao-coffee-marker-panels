from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260621
PROJECT_ROOT = Path(".")

FOCUSED_TRAITS: Dict[str, List[str]] = {
    "icgt_cacao": ["FR_BEN", "FL_OV_NUM", "FR_BW", "FR_BEW", "PodIndex"],
    "agrosavia_matina": [
        "AUDPC flower cushion broom infected by WBD",
        "AUDPC deformed branches infected by WBD",
        "AUDPC pods infected by FPRD",
        "Total pods",
        "Healthy pods",
    ],
    "agrosavia_criollo": [
        "AUDPC flower cushion broom infected by WBD",
        "AUDPC deformed branches infected by WBD",
        "AUDPC pods infected by FPRD",
        "Total pods",
        "Healthy pods",
    ],
    "coffee_int": ["production", "rust", "green"],
    "coffee_prec": ["production", "rust", "green"],
}

ALL_CURATED_TRAITS: Dict[str, List[str]] = {
    "icgt_cacao": [
        "FL_SEP_LEN", "FL_LIG_WID", "FL_OV_NUM", "FL_STY_LEN",
        "FR_POL", "FR_POW", "PODL_W", "FR_WABW", "FR_BEN", "FR_BW", "FR_BL", "FR_BEW", "BENL_W", "PodIndex",
    ],
    "agrosavia_matina": FOCUSED_TRAITS["agrosavia_matina"],
    "agrosavia_criollo": FOCUSED_TRAITS["agrosavia_criollo"],
    "coffee_int": ["production", "rust", "green"],
    "coffee_prec": ["production", "rust", "green"],
}

DATASETS = ["icgt_cacao", "agrosavia_matina", "agrosavia_criollo", "coffee_int", "coffee_prec"]
DEFAULT_PANEL_SIZES = [20, 45, 96, 219, 500, 1000]
DEFAULT_STRATEGIES = ["random", "high_maf", "anova_f"]
DEFAULT_SCHEMES = ["random_five_fold", "cluster_out"]

@dataclass
class Bundle:
    dataset: str
    X: pd.DataFrame
    pheno: pd.DataFrame
    splits: pd.DataFrame


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "features": root / "analysis/intermediate" / "features",
        "qc": root / "analysis/intermediate" / "qc",
        "out": root / "analysis/outputs" / "marker_panel_simulation",
        "logs": root / "analysis/logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_name(x: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ["_", "-", "."] else "_" for ch in str(x))


def load_paths(root: Path, dirs: Dict[str, Path]) -> Dict[str, dict]:
    p = dirs["qc"] / "downstream_data_paths.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing downstream_data_paths.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def read_genotype(path: Path) -> pd.DataFrame:
    X = pd.read_csv(path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X.index.name = "sample_id"
    X.columns = [str(c) for c in X.columns]
    return X.apply(pd.to_numeric, errors="coerce")


def load_bundle(dataset: str, root: Path, dirs: Dict[str, Path], paths: Dict[str, dict]) -> Bundle:
    geno = dirs["features"] / f"{dataset}_genotype_dosage_qc_filtered.tsv.gz"
    if not geno.exists():
        geno = Path(paths[dataset]["genotype_matched"])
    pheno_path = Path(paths[dataset]["phenotype_matched"])
    split_path = dirs["features"] / f"{dataset}_validation_splits.tsv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    X = read_genotype(geno)
    if dataset.startswith("coffee") and np.nanmin(X.to_numpy(dtype="float32", copy=False)) < 0:
        X = X + 1.0

    pheno = pd.read_csv(pheno_path, sep="\t")
    if "sample_id" not in pheno.columns:
        pheno = pheno.rename(columns={pheno.columns[0]: "sample_id"})
    pheno["sample_id"] = pheno["sample_id"].astype(str)
    splits = pd.read_csv(split_path, sep="\t")
    splits["sample_id"] = splits["sample_id"].astype(str)

    common = [i for i in X.index.astype(str) if i in set(pheno.sample_id) and i in set(splits.sample_id)]
    X = X.loc[common]
    pheno = pheno.set_index("sample_id").loc[common].reset_index()
    splits = splits.set_index("sample_id").loc[common].reset_index()
    return Bundle(dataset, X, pheno, splits)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    if np.nanstd(y_true[m]) == 0 or np.nanstd(y_pred[m]) == 0:
        return np.nan
    return float(np.corrcoef(y_true[m], y_pred[m])[0, 1])


def marker_train_stats(X_train: pd.DataFrame) -> pd.DataFrame:
    A = X_train.to_numpy(dtype="float32", copy=False)
    missing = np.isnan(A).mean(axis=0)
    mean_dosage = np.nanmean(A, axis=0)
    p = np.clip(mean_dosage / 2.0, 0, 1)
    maf = np.minimum(p, 1 - p)
    var = np.nanvar(A, axis=0)
    return pd.DataFrame({"marker": X_train.columns, "missing_rate": missing, "maf": maf, "variance": var})


def select_markers(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    strategy: str,
    rng: np.random.Generator,
) -> List[str]:
    stats = marker_train_stats(X_train)
    eligible = stats[(stats["variance"] > 1e-10) & np.isfinite(stats["maf"]) & (stats["maf"] > 0)].copy()
    if len(eligible) == 0:
        return list(X_train.columns[: min(k, X_train.shape[1])])
    k = min(k, len(eligible))

    if strategy == "random":
        return list(rng.choice(eligible["marker"].to_numpy(), size=k, replace=False))

    if strategy == "high_maf":
        return list(eligible.sort_values(["maf", "missing_rate"], ascending=[False, True])["marker"].head(k))

    if strategy == "anova_f":
        cols = list(eligible["marker"])
        Xt = X_train[cols]
        imp = SimpleImputer(strategy="mean")
        A = imp.fit_transform(Xt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fvals, _ = f_regression(A, y_train)
        fvals = np.nan_to_num(fvals, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        order = np.argsort(fvals)[::-1][:k]
        return [cols[i] for i in order]

    raise ValueError(f"Unknown strategy: {strategy}")


def fit_predict_ridge(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 10))),
    ])
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pearson_r": safe_corr(y_true, y_pred),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }


def run_one_trait(
    bundle: Bundle,
    trait: str,
    scheme: str,
    panel_sizes: Sequence[int],
    strategies: Sequence[str],
    n_random_repeats: int,
    include_full: bool,
) -> Tuple[List[dict], List[dict]]:
    ph = bundle.pheno.copy()
    y_all = pd.to_numeric(ph[trait], errors="coerce")
    valid = y_all.notna().to_numpy()
    if valid.sum() < 30 or y_all[valid].nunique() < 5:
        return [], [{"dataset": bundle.dataset, "trait": trait, "scheme": scheme, "status": "skipped_low_trait_variation", "n_valid": int(valid.sum())}]

    X = bundle.X.loc[valid].copy()
    y = y_all[valid].to_numpy(dtype=float)
    splits = bundle.splits.loc[valid].reset_index(drop=True)
    sample_ids = ph.loc[valid, "sample_id"].astype(str).to_numpy()

    if scheme not in splits.columns:
        return [], [{"dataset": bundle.dataset, "trait": trait, "scheme": scheme, "status": "missing_split_column", "n_valid": int(valid.sum())}]

    folds = sorted([f for f in pd.unique(splits[scheme]) if str(f) not in ["", "nan", "NA"]])
    if len(folds) < 2:
        return [], [{"dataset": bundle.dataset, "trait": trait, "scheme": scheme, "status": "too_few_folds", "n_valid": int(valid.sum()), "n_folds": len(folds)}]

    rows = []
    fold_rows = []
    rng_base = np.random.default_rng(SEED + abs(hash((bundle.dataset, trait, scheme))) % 100000)

    panel_specs = []
    for size in panel_sizes:
        for strat in strategies:
            reps = n_random_repeats if strat == "random" else 1
            for rep in range(reps):
                panel_specs.append((int(size), strat, rep))
    if include_full:
        panel_specs.append((int(X.shape[1]), "full", 0))

    for panel_size, strategy, rep in panel_specs:
        pred_all = np.full(len(y), np.nan, dtype=float)
        fold_metrics = []
        for fold in folds:
            test_mask = (splits[scheme].to_numpy() == fold)
            train_mask = ~test_mask
            if train_mask.sum() < 20 or test_mask.sum() < 3:
                continue
            X_train_full = X.iloc[train_mask]
            X_test_full = X.iloc[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            if np.nanstd(y_train) == 0:
                continue

            if strategy == "full":
                markers = list(X_train_full.columns)
            else:
                rng = np.random.default_rng(rng_base.integers(0, 2**32 - 1) + rep)
                markers = select_markers(X_train_full, y_train, panel_size, strategy, rng)
            y_pred = fit_predict_ridge(X_train_full[markers], y_train, X_test_full[markers])
            pred_all[test_mask] = y_pred
            fm = evaluate_predictions(y_test, y_pred)
            fold_metrics.append(fm)
            fold_rows.append({
                "dataset": bundle.dataset,
                "trait": trait,
                "validation_scheme": scheme,
                "panel_strategy": strategy,
                "panel_size_requested": panel_size,
                "panel_size_actual": len(markers),
                "repeat": rep,
                "fold": fold,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **{f"fold_{k}": v for k, v in fm.items()},
            })
        ok = np.isfinite(pred_all)
        if ok.sum() < 10:
            continue
        pooled = evaluate_predictions(y[ok], pred_all[ok])
        rows.append({
            "dataset": bundle.dataset,
            "trait": trait,
            "validation_scheme": scheme,
            "panel_strategy": strategy,
            "panel_size_requested": panel_size,
            "panel_size_actual_median": np.nanmedian([r["panel_size_actual"] for r in fold_rows if r["dataset"] == bundle.dataset and r["trait"] == trait and r["validation_scheme"] == scheme and r["panel_strategy"] == strategy and r["panel_size_requested"] == panel_size and r["repeat"] == rep]),
            "repeat": rep,
            "n_predictions": int(ok.sum()),
            "pooled_pearson_r": pooled["pearson_r"],
            "pooled_rmse": pooled["rmse"],
            "pooled_mae": pooled["mae"],
            "pooled_r2": pooled["r2"],
            "n_folds_ok": len(fold_metrics),
        })
    return rows, fold_rows


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "trait", "validation_scheme", "panel_strategy", "panel_size_requested"]
    agg = df.groupby(keys, as_index=False).agg(
        n_repeats=("repeat", "nunique"),
        n_predictions=("n_predictions", "max"),
        pearson_r_mean=("pooled_pearson_r", "mean"),
        pearson_r_sd=("pooled_pearson_r", "std"),
        pearson_r_max=("pooled_pearson_r", "max"),
        r2_mean=("pooled_r2", "mean"),
        r2_sd=("pooled_r2", "std"),
        rmse_mean=("pooled_rmse", "mean"),
        mae_mean=("pooled_mae", "mean"),
        n_folds_ok_median=("n_folds_ok", "median"),
    )
    return agg.sort_values(["dataset", "trait", "validation_scheme", "panel_strategy", "panel_size_requested"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    ap.add_argument("--panel-sizes", nargs="+", type=int, default=DEFAULT_PANEL_SIZES)
    ap.add_argument("--n-random-repeats", type=int, default=5)
    ap.add_argument("--trait-set", choices=["focused", "all"], default="focused")
    ap.add_argument("--include-full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root)
    dirs = ensure_dirs(root)
    paths = load_paths(root, dirs)

    if args.quick:
        args.datasets = ["icgt_cacao", "agrosavia_matina", "coffee_prec"]
        args.schemes = ["random_five_fold", "cluster_out"]
        args.panel_sizes = [20, 96, 219]
        args.strategies = ["random", "high_maf", "anova_f"]
        args.n_random_repeats = 2

    config = vars(args).copy()
    config["started"] = datetime.now().isoformat(timespec="seconds")
    (dirs["out"] / "05_marker_panel_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("===== 05 MARKER PANEL SIMULATION =====")
    print(f"Project root: {root}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Panel sizes: {args.panel_sizes}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Random repeats: {args.n_random_repeats}")

    all_rows: List[dict] = []
    all_fold_rows: List[dict] = []
    status_rows: List[dict] = []
    trait_map = FOCUSED_TRAITS if args.trait_set == "focused" else ALL_CURATED_TRAITS

    for dataset in args.datasets:
        print(f"\n[{dataset}] Loading data...")
        bundle = load_bundle(dataset, root, dirs, paths)
        traits = [t for t in trait_map.get(dataset, []) if t in bundle.pheno.columns]
        print(f"[{dataset}] Matrix: {bundle.X.shape[0]} samples x {bundle.X.shape[1]} markers; traits: {len(traits)}")
        for trait in traits:
            print(f"  - trait: {trait}")
            for scheme in args.schemes:
                rows, fold_rows = run_one_trait(bundle, trait, scheme, args.panel_sizes, args.strategies, args.n_random_repeats, args.include_full)
                all_rows.extend(rows)
                all_fold_rows.extend(fold_rows)
                if not rows:
                    status_rows.append({"dataset": dataset, "trait": trait, "scheme": scheme, "status": "no_rows"})
                else:
                    status_rows.append({"dataset": dataset, "trait": trait, "scheme": scheme, "status": "ok", "n_rows": len(rows)})

    raw = pd.DataFrame(all_rows)
    folds = pd.DataFrame(all_fold_rows)
    status = pd.DataFrame(status_rows)
    if raw.empty:
        raise RuntimeError("No panel simulation results were produced.")
    agg = aggregate_repeats(raw)

    raw_path = dirs["out"] / "05_marker_panel_raw_results.tsv"
    fold_path = dirs["out"] / "05_marker_panel_fold_metrics.tsv"
    summary_path = dirs["out"] / "05_marker_panel_summary.tsv"
    status_path = dirs["out"] / "05_marker_panel_status.tsv"
    raw.to_csv(raw_path, sep="\t", index=False)
    folds.to_csv(fold_path, sep="\t", index=False)
    agg.to_csv(summary_path, sep="\t", index=False)
    status.to_csv(status_path, sep="\t", index=False)

    best = agg.sort_values(["pearson_r_mean", "r2_mean"], ascending=[False, False]).groupby(["dataset", "trait", "validation_scheme"], as_index=False).first()
    best_path = dirs["out"] / "05_marker_panel_best_per_trait_scheme.tsv"
    best.to_csv(best_path, sep="\t", index=False)

    report = []
    report.append("# 05 marker-panel simulation report")
    report.append("")
    report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append("This step evaluates whether low-density marker panels can approximate genomic prediction performance under random and cluster-out validation. Marker selection is nested within each training fold for trait-predictive ANOVA panels, avoiding test-fold leakage.")
    report.append("")
    report.append("## Outputs")
    for p in [raw_path, fold_path, summary_path, best_path, status_path]:
        report.append(f"- `{p}`")
    report.append("")
    report.append("## Best results by trait/scheme")
    report.append("")
    top = best.sort_values("pearson_r_mean", ascending=False).head(25)
    for _, r in top.iterrows():
        report.append(f"- {r.dataset} / {r.trait} / {r.validation_scheme}: {r.panel_strategy} {int(r.panel_size_requested)} SNPs, r={r.pearson_r_mean:.3f}, R2={r.r2_mean:.3f}")
    report.append("")
    report.append("## Rigor notes")
    report.append("")
    report.append("- Random, high-MAF and ANOVA-F panels are selected using training-fold data only.")
    report.append("- Ridge regression is used as the primary marker-density response model to avoid excessive local runtime and to keep comparisons stable.")
    report.append("- Random panels are repeated; their mean and standard deviation are recorded.")
    report.append("- `cluster_out` remains the deployment stress test; `random_five_fold` is a baseline.")
    report_path = dirs["out"] / "05_marker_panel_simulation_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n===== 05 MARKER PANEL SIMULATION COMPLETE =====")
    print(best.sort_values("pearson_r_mean", ascending=False).head(20).to_string(index=False))
    print(f"\nReport: {report_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
