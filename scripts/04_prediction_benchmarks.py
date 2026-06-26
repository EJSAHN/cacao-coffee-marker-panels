from __future__ import annotations

# Keep Windows/MKL thread behavior predictable on small local machines.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260621
PROJECT_ROOT = Path(".")

DATASETS = [
    "icgt_cacao",
    "agrosavia_matina",
    "agrosavia_criollo",
    "coffee_int",
    "coffee_prec",
]

DEFAULT_MODELS = ["mean", "ridge", "elastic_net", "random_forest"]
ALL_MODELS = ["mean", "ridge", "elastic_net", "random_forest", "gradient_boosting"]
DEFAULT_SCHEMES = ["random_five_fold", "cluster_out"]

TRAIT_WHITELIST: Dict[str, List[str]] = {
    "icgt_cacao": [
        "FL_SEP_LEN",
        "FL_LIG_WID",
        "FL_OV_NUM",
        "FL_STY_LEN",
        "FR_POL",
        "FR_POW",
        "PODL_W",
        "FR_WABW",
        "FR_BEN",
        "FR_BW",
        "FR_BL",
        "FR_BEW",
        "BENL_W",
        "PodIndex",
    ],
    "agrosavia_matina": [
        "Total pods",
        "Healthy pods",
        "AUDPC pods infected by FPRD",
        "AUDPC flower cushion broom infected by WBD",
        "AUDPC deformed branches infected by WBD",
    ],
    "agrosavia_criollo": [
        "Total pods",
        "Healthy pods",
        "AUDPC pods infected by FPRD",
        "AUDPC flower cushion broom infected by WBD",
        "AUDPC deformed branches infected by WBD",
    ],
    "coffee_int": ["production", "rust", "green"],
    "coffee_prec": ["production", "rust", "green"],
}

TRAIT_LABELS: Dict[str, str] = {
    "PodIndex": "pod index",
    "Total pods": "total pods",
    "Healthy pods": "healthy pods",
    "AUDPC pods infected by FPRD": "frosty pod rot AUDPC",
    "AUDPC flower cushion broom infected by WBD": "witches broom flower-cushion AUDPC",
    "AUDPC deformed branches infected by WBD": "witches broom branch AUDPC",
    "production": "coffee bean production",
    "rust": "leaf rust incidence",
    "green": "green bean yield",
}

EXCLUDE_NUMERIC_COLUMNS = {
    "sample_id",
    "sample_key",
    "Acc Group",
    "n_env_records",
}


@dataclass
class DatasetBundle:
    dataset: str
    X: pd.DataFrame
    phenotype: pd.DataFrame
    splits: pd.DataFrame
    genotype_path: Path
    phenotype_path: Path
    split_path: Path


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "features": root / "analysis/intermediate" / "features",
        "qc": root / "analysis/intermediate" / "qc",
        "pred": root / "analysis/outputs" / "prediction_benchmarks",
        "logs": root / "analysis/logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_name(text: object) -> str:
    s = str(text)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def read_table_indexed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "sample_id"
    # Ensure unique columns for sklearn.
    cols = list(map(str, df.columns))
    if len(cols) != len(set(cols)):
        seen: Dict[str, int] = {}
        new_cols = []
        for c in cols:
            seen[c] = seen.get(c, 0) + 1
            if seen[c] == 1:
                new_cols.append(c)
            else:
                new_cols.append(f"{c}__dup{seen[c]}")
        df.columns = new_cols
    return df.apply(pd.to_numeric, errors="coerce")


def load_downstream_paths(root: Path, dirs: Dict[str, Path]) -> Dict[str, dict]:
    path = dirs["qc"] / "downstream_data_paths.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing downstream path file from 02 step: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset: str, root: Path, dirs: Dict[str, Path], downstream: Dict[str, dict]) -> DatasetBundle:
    geno_path = dirs["features"] / f"{dataset}_genotype_dosage_qc_filtered.tsv.gz"
    split_path = dirs["features"] / f"{dataset}_validation_splits.tsv"

    if not geno_path.exists():
        # fallback to 02 matched genotypes, but 03 filtered matrix is preferred
        geno_path = Path(downstream[dataset]["genotype_matched"])
    if not split_path.exists():
        raise FileNotFoundError(f"Missing validation split file from 03 step: {split_path}")

    pheno_path = Path(downstream[dataset]["phenotype_matched"])
    if not pheno_path.exists():
        raise FileNotFoundError(f"Missing phenotype file: {pheno_path}")

    X = read_table_indexed(geno_path)
    if dataset.startswith("coffee"):
        # Coffee raw coding is usually -1/0/1. Convert to 0/1/2 dosage for comparability.
        finite = X.to_numpy(dtype="float32", copy=False)
        if np.isfinite(finite).any() and np.nanmin(finite) < 0:
            X = X + 1.0

    phenotype = pd.read_csv(pheno_path, sep="\t")
    if "sample_id" not in phenotype.columns:
        phenotype = phenotype.rename(columns={phenotype.columns[0]: "sample_id"})
    phenotype["sample_id"] = phenotype["sample_id"].astype(str)

    splits = pd.read_csv(split_path, sep="\t")
    splits["sample_id"] = splits["sample_id"].astype(str)

    common = [sid for sid in X.index.astype(str) if sid in set(phenotype["sample_id"]) and sid in set(splits["sample_id"])]
    if len(common) < 20:
        raise RuntimeError(f"Too few common samples for {dataset}: {len(common)}")

    X = X.loc[common].copy()
    phenotype = phenotype.set_index("sample_id").loc[common].reset_index()
    splits = splits.set_index("sample_id").loc[common].reset_index()

    return DatasetBundle(
        dataset=dataset,
        X=X,
        phenotype=phenotype,
        splits=splits,
        genotype_path=geno_path,
        phenotype_path=pheno_path,
        split_path=split_path,
    )


def select_traits(phenotype: pd.DataFrame, dataset: str, mode: str = "curated") -> pd.DataFrame:
    rows = []
    whitelist = TRAIT_WHITELIST.get(dataset, [])

    if mode == "curated":
        candidate_cols = [c for c in whitelist if c in phenotype.columns]
    else:
        candidate_cols = []
        for c in phenotype.columns:
            if c in EXCLUDE_NUMERIC_COLUMNS:
                continue
            if str(c).startswith("FR_POSH_") or str(c).startswith("FR_PAF_") or str(c).startswith("FR_BES_"):
                # one-hot morphology traits are classification-like; keep out of regression benchmark
                continue
            vals = pd.to_numeric(phenotype[c], errors="coerce")
            if vals.notna().sum() >= 30 and vals.nunique(dropna=True) >= 5:
                candidate_cols.append(c)

    for trait in candidate_cols:
        y = pd.to_numeric(phenotype[trait], errors="coerce")
        rows.append({
            "dataset": dataset,
            "trait": trait,
            "trait_label": TRAIT_LABELS.get(trait, trait),
            "n_nonmissing": int(y.notna().sum()),
            "n_unique": int(y.nunique(dropna=True)),
            "mean": float(y.mean()) if y.notna().any() else np.nan,
            "sd": float(y.std(ddof=1)) if y.notna().sum() > 1 else np.nan,
            "min": float(y.min()) if y.notna().any() else np.nan,
            "max": float(y.max()) if y.notna().any() else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[(out["n_nonmissing"] >= 30) & (out["n_unique"] >= 3)].copy()
    return out.reset_index(drop=True)


def build_fold_list(
    splits: pd.DataFrame,
    scheme: str,
    min_train: int = 20,
    min_test: int = 5,
) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]], pd.DataFrame]:
    if scheme not in splits.columns:
        return [], pd.DataFrame()

    s = splits[scheme]
    valid = s.notna() & (s.astype(str) != "") & (s.astype(str) != "-1")
    values = sorted(s[valid].astype(str).unique(), key=lambda x: (len(x), x))

    fold_rows = []
    folds = []
    n = len(splits)
    all_idx = np.arange(n)

    for val in values:
        test_mask = (s.astype(str).values == val)
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        status = "OK"
        if len(test_idx) < min_test:
            status = "SKIP_SMALL_TEST"
        if len(train_idx) < min_train:
            status = "SKIP_SMALL_TRAIN"

        fold_rows.append({
            "scheme": scheme,
            "fold": val,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "status": status,
        })
        if status == "OK":
            folds.append((val, train_idx, test_idx))

    return folds, pd.DataFrame(fold_rows)


def make_estimator(model_name: str, n_features: int):
    if model_name == "mean":
        return None

    if model_name == "ridge":
        alphas = np.logspace(-3, 5, 17)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", RidgeCV(alphas=alphas, cv=3, scoring="neg_mean_squared_error")),
        ])

    if model_name == "elastic_net":
        k = int(min(n_features, 5000))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("select", SelectKBest(score_func=f_regression, k=k)),
            ("enet", ElasticNetCV(
                l1_ratio=[0.05, 0.25, 0.50, 0.75, 0.95],
                n_alphas=50,
                cv=3,
                max_iter=5000,
                random_state=SEED,
                n_jobs=1,
                selection="cyclic",
            )),
        ])

    if model_name == "random_forest":
        k = int(min(n_features, 3000))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("select", SelectKBest(score_func=f_regression, k=k)),
            ("rf", RandomForestRegressor(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=1,
            )),
        ])

    if model_name == "gradient_boosting":
        k = int(min(n_features, 1000))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("select", SelectKBest(score_func=f_regression, k=k)),
            ("gbr", GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.85,
                random_state=SEED,
            )),
        ])

    raise ValueError(f"Unknown model: {model_name}")


def pearson_r(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman_r(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = pd.Series(y_true, dtype="float64")
    p = pd.Series(y_pred, dtype="float64")
    ok = y.notna() & p.notna()
    if ok.sum() < 3:
        return np.nan
    return float(y[ok].rank().corr(p[ok].rank()))


def metric_row(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) == 0:
        return {"n": 0, "pearson_r": np.nan, "spearman_r": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    rmse = float(math.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else np.nan
    return {
        "n": int(len(y)),
        "pearson_r": pearson_r(y, p),
        "spearman_r": spearman_r(y, p),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def fit_predict_one_fold(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, str]:
    y_train = y[train_idx]
    if model_name == "mean":
        return np.full(len(test_idx), np.nanmean(y_train), dtype=float), "OK"

    est = make_estimator(model_name, n_features=X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X[train_idx], y_train)
        pred = est.predict(X[test_idx])
    return np.asarray(pred, dtype=float), "OK"


def run_prediction_for_trait(
    bundle: DatasetBundle,
    trait: str,
    models: List[str],
    schemes: List[str],
    min_train: int,
    min_test: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_all = pd.to_numeric(bundle.phenotype[trait], errors="coerce").to_numpy(dtype=float)
    valid_trait = np.isfinite(y_all)

    X_df = bundle.X.loc[valid_trait].copy()
    y = y_all[valid_trait]
    pheno_sub = bundle.phenotype.loc[valid_trait].reset_index(drop=True)
    splits_sub = bundle.splits.loc[valid_trait].reset_index(drop=True)
    samples = X_df.index.astype(str).tolist()

    X = X_df.to_numpy(dtype="float32", copy=False)

    fold_metric_rows = []
    prediction_rows = []
    split_inventory_rows = []

    for scheme in schemes:
        folds, inv = build_fold_list(splits_sub, scheme, min_train=min_train, min_test=min_test)
        if not inv.empty:
            inv.insert(0, "dataset", bundle.dataset)
            inv.insert(1, "trait", trait)
            split_inventory_rows.extend(inv.to_dict(orient="records"))

        for model_name in models:
            for fold, train_idx, test_idx in folds:
                try:
                    pred, status = fit_predict_one_fold(model_name, X, y, train_idx, test_idx)
                    metrics = metric_row(y[test_idx], pred)
                    fold_metric_rows.append({
                        "dataset": bundle.dataset,
                        "trait": trait,
                        "validation_scheme": scheme,
                        "fold": fold,
                        "model": model_name,
                        "status": status,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        **metrics,
                    })
                    for sid, yt, yp in zip(np.asarray(samples)[test_idx], y[test_idx], pred):
                        prediction_rows.append({
                            "dataset": bundle.dataset,
                            "trait": trait,
                            "validation_scheme": scheme,
                            "fold": fold,
                            "model": model_name,
                            "sample_id": sid,
                            "y_true": float(yt),
                            "y_pred": float(yp),
                        })
                except Exception as exc:
                    fold_metric_rows.append({
                        "dataset": bundle.dataset,
                        "trait": trait,
                        "validation_scheme": scheme,
                        "fold": fold,
                        "model": model_name,
                        "status": "ERROR",
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "n": 0,
                        "pearson_r": np.nan,
                        "spearman_r": np.nan,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "r2": np.nan,
                        "error": repr(exc),
                    })

    return (
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(split_inventory_rows),
    )


def summarize_predictions(pred: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame(rows)

    keys = ["dataset", "trait", "validation_scheme", "model"]
    for key, g in pred.groupby(keys, dropna=False):
        metrics = metric_row(g["y_true"].values, g["y_pred"].values)
        fm = fold_metrics
        for col, val in zip(keys, key):
            fm = fm[fm[col] == val]
        ok = fm[fm["status"] == "OK"].copy() if "status" in fm.columns else fm.copy()
        rows.append({
            "dataset": key[0],
            "trait": key[1],
            "trait_label": TRAIT_LABELS.get(key[1], key[1]),
            "validation_scheme": key[2],
            "model": key[3],
            "n_predictions": int(metrics["n"]),
            "pooled_pearson_r": metrics["pearson_r"],
            "pooled_spearman_r": metrics["spearman_r"],
            "pooled_rmse": metrics["rmse"],
            "pooled_mae": metrics["mae"],
            "pooled_r2": metrics["r2"],
            "n_folds_ok": int(ok.shape[0]),
            "fold_pearson_r_mean": float(ok["pearson_r"].mean()) if "pearson_r" in ok else np.nan,
            "fold_pearson_r_sd": float(ok["pearson_r"].std(ddof=1)) if ok.shape[0] > 1 and "pearson_r" in ok else np.nan,
            "fold_rmse_mean": float(ok["rmse"].mean()) if "rmse" in ok else np.nan,
            "fold_rmse_sd": float(ok["rmse"].std(ddof=1)) if ok.shape[0] > 1 and "rmse" in ok else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "trait", "validation_scheme", "model"]).reset_index(drop=True)
    return out


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def write_report(
    root: Path,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    trait_inventory: pd.DataFrame,
    summary: pd.DataFrame,
    split_inventory: pd.DataFrame,
    elapsed: str,
) -> Path:
    report = dirs["pred"] / "04_prediction_benchmarks_report.md"
    lines: List[str] = []
    lines.append("# 04 Prediction benchmark report")
    lines.append("")
    lines.append(f"Generated: {now()}")
    lines.append(f"Project root: `{root}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This step benchmarks genotype-based prediction under both random cross-validation and "
        "cluster-out validation. The cluster-out split is treated as the main stress test because "
        "it better reflects deployment into genetically distinct material and reduces relatedness leakage."
    )
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Models: {', '.join(args.models)}")
    lines.append(f"- Validation schemes: {', '.join(args.schemes)}")
    lines.append(f"- Trait mode: {args.trait_mode}")
    lines.append(f"- Minimum training samples per fold: {args.min_train}")
    lines.append(f"- Minimum test samples per fold: {args.min_test}")
    lines.append(f"- Elapsed: {elapsed}")
    lines.append("")
    lines.append("## Traits analyzed")
    lines.append("")
    if trait_inventory.empty:
        lines.append("No traits were selected.")
    else:
        lines.append(trait_inventory[["dataset", "trait", "n_nonmissing", "n_unique", "mean", "sd"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Best pooled Pearson r by dataset, trait and validation scheme")
    lines.append("")
    if summary.empty:
        lines.append("No prediction results were generated.")
    else:
        best = summary.sort_values("pooled_pearson_r", ascending=False, na_position="last")
        best = best.groupby(["dataset", "trait", "validation_scheme"], as_index=False).head(1)
        cols = ["dataset", "trait", "validation_scheme", "model", "n_predictions", "pooled_pearson_r", "pooled_rmse", "pooled_r2"]
        lines.append(best[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Split inventory")
    lines.append("")
    if split_inventory.empty:
        lines.append("No split inventory was generated.")
    else:
        si = split_inventory.groupby(["dataset", "trait", "scheme", "status"], as_index=False).agg(
            n_folds=("fold", "count"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
        )
        lines.append(si.to_markdown(index=False))
    lines.append("")
    lines.append("## Rigor notes")
    lines.append("")
    lines.append("- All model fitting, feature selection, imputation and scaling occur inside each training fold.")
    lines.append("- Random 5-fold cross-validation is retained as a familiar baseline, not as the main deployment estimate.")
    lines.append("- Cluster-out validation provides the population-structure-aware stress test for prediction across genetic groups.")
    lines.append("- Mean prediction is included as a non-genomic baseline; a model must beat this baseline before biological interpretation.")
    lines.append("- For high-dimensional coffee and cacao matrices, supervised feature selection is nested within each fold for tree and elastic-net models.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- Fold metrics: `{dirs['pred'] / '04_prediction_fold_metrics.tsv'}`")
    lines.append(f"- Pooled predictions: `{dirs['pred'] / '04_prediction_pooled_predictions.tsv.gz'}`")
    lines.append(f"- Summary: `{dirs['pred'] / '04_prediction_summary.tsv'}`")
    lines.append(f"- Trait inventory: `{dirs['pred'] / '04_prediction_trait_inventory.tsv'}`")
    lines.append(f"- Split inventory: `{dirs['pred'] / '04_prediction_split_inventory.tsv'}`")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append(
        "Use 05_marker_panel_simulation.py to test whether low-density panels "
        "(20/45/96/219/500/1000 SNPs) can recover identity structure and trait-prediction performance."
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction benchmarks for the cacao and coffee genomic breeding pipeline.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root directory.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="Datasets to process.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS, help="Models to benchmark.")
    parser.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES, choices=DEFAULT_SCHEMES, help="Validation schemes.")
    parser.add_argument("--trait-mode", default="curated", choices=["curated", "broad"], help="Curated trait list or broad numeric scan.")
    parser.add_argument("--min-train", type=int, default=20, help="Minimum train samples per evaluated fold.")
    parser.add_argument("--min-test", type=int, default=5, help="Minimum test samples per evaluated fold.")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test: mean and ridge only.")
    return parser.parse_args()


def main() -> None:
    started = datetime.now()
    args = parse_args()
    if args.quick:
        args.models = ["mean", "ridge"]

    root = Path(args.project_root)
    dirs = ensure_dirs(root)
    downstream = load_downstream_paths(root, dirs)

    run_config = {
        "created": now(),
        "project_root": str(root),
        "datasets": args.datasets,
        "models": args.models,
        "schemes": args.schemes,
        "trait_mode": args.trait_mode,
        "min_train": args.min_train,
        "min_test": args.min_test,
        "quick": bool(args.quick),
        "seed": SEED,
        "thread_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
    }
    (dirs["pred"] / "04_prediction_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("===== 04 PREDICTION BENCHMARKS =====")
    print(f"Project root: {root}")
    print(f"Started: {started.isoformat(timespec='seconds')}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Schemes: {', '.join(args.schemes)}")
    print("")

    all_trait_inv = []
    all_fold_metrics = []
    all_predictions = []
    all_split_inv = []
    dataset_status_rows = []

    try:
        for dataset in args.datasets:
            print(f"[{dataset}] Loading data...")
            bundle = load_dataset(dataset, root, dirs, downstream)
            trait_inv = select_traits(bundle.phenotype, dataset, mode=args.trait_mode)
            if trait_inv.empty:
                print(f"[{dataset}] No eligible traits selected; skipping.")
                dataset_status_rows.append({"dataset": dataset, "status": "NO_TRAITS", "n_samples": bundle.X.shape[0], "n_markers": bundle.X.shape[1], "n_traits": 0})
                continue
            all_trait_inv.append(trait_inv)
            dataset_status_rows.append({"dataset": dataset, "status": "OK", "n_samples": bundle.X.shape[0], "n_markers": bundle.X.shape[1], "n_traits": trait_inv.shape[0]})
            print(f"[{dataset}] Matrix: {bundle.X.shape[0]} samples x {bundle.X.shape[1]} markers; traits: {trait_inv.shape[0]}")

            for trait in trait_inv["trait"].tolist():
                print(f"  - trait: {trait}")
                fm, pred, si = run_prediction_for_trait(
                    bundle=bundle,
                    trait=trait,
                    models=args.models,
                    schemes=args.schemes,
                    min_train=args.min_train,
                    min_test=args.min_test,
                )
                if not fm.empty:
                    all_fold_metrics.append(fm)
                if not pred.empty:
                    all_predictions.append(pred)
                if not si.empty:
                    all_split_inv.append(si)

        trait_inventory = pd.concat(all_trait_inv, ignore_index=True) if all_trait_inv else pd.DataFrame()
        fold_metrics = pd.concat(all_fold_metrics, ignore_index=True) if all_fold_metrics else pd.DataFrame()
        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        split_inventory = pd.concat(all_split_inv, ignore_index=True) if all_split_inv else pd.DataFrame()
        dataset_status = pd.DataFrame(dataset_status_rows)
        summary = summarize_predictions(predictions, fold_metrics)

        write_tsv(dataset_status, dirs["pred"] / "04_prediction_dataset_status.tsv")
        write_tsv(trait_inventory, dirs["pred"] / "04_prediction_trait_inventory.tsv")
        write_tsv(fold_metrics, dirs["pred"] / "04_prediction_fold_metrics.tsv")
        write_tsv(summary, dirs["pred"] / "04_prediction_summary.tsv")
        write_tsv(split_inventory, dirs["pred"] / "04_prediction_split_inventory.tsv")
        predictions.to_csv(dirs["pred"] / "04_prediction_pooled_predictions.tsv.gz", sep="\t", index=False, compression="gzip")

        elapsed = str(datetime.now() - started)
        report = write_report(root, dirs, args, trait_inventory, summary, split_inventory, elapsed)

        print("")
        print("===== 04 PREDICTION BENCHMARKS COMPLETE =====")
        if not summary.empty:
            display_cols = ["dataset", "trait", "validation_scheme", "model", "n_predictions", "pooled_pearson_r", "pooled_rmse", "pooled_r2"]
            best = summary.sort_values("pooled_pearson_r", ascending=False, na_position="last")
            best = best.groupby(["dataset", "trait", "validation_scheme"], as_index=False).head(1)
            print(best[display_cols].to_string(index=False))
        print("")
        print(f"Report: {report}")
        print(f"Summary: {dirs['pred'] / '04_prediction_summary.tsv'}")
        print(f"Elapsed: {elapsed}")

    except Exception:
        log_path = dirs["logs"] / f"04_prediction_benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        print("")
        print("ERROR during prediction benchmarks. See log:")
        print(log_path)
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
