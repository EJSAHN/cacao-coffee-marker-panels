from __future__ import annotations

import argparse
import gzip
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


DEFAULT_PROJECT_ROOT = Path(".")
SEED = 20260621


@dataclass
class MatrixQCResult:
    dataset: str
    genotype_path: Path
    phenotype_path: Path
    code_transform: str
    n_samples_raw: int
    n_markers_raw: int
    n_samples_aligned: int
    n_markers_aligned: int
    n_samples_pca: int
    n_markers_pca: int
    sample_missing_max: float
    marker_missing_max: float
    maf_min: float
    n_high_similarity_pairs_0985: int
    max_pair_similarity: float
    best_k: int
    best_k_silhouette: float
    pc1_var: float
    pc2_var: float


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(text: object) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: Path, sep: str = "\t", index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).lower().endswith(".gz"):
        df.to_csv(path, sep=sep, index=index, compression="gzip")
    else:
        df.to_csv(path, sep=sep, index=index)
    return path


def read_tsv(path: Path, index_col: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=index_col, compression="infer", low_memory=False)


def load_downstream_config(project_root: Path) -> Dict:
    config_path = project_root / "analysis/intermediate" / "qc" / "downstream_data_paths.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing downstream config: {config_path}\n"
            "Run 02_clean_and_harmonize.py first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def convert_to_dosage_if_needed(geno: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Convert matched genotype matrix to approximate 0/1/2 dosage when needed.

    Cacao matrices are already 0/1/2. The Coffea canephora matrices in the public
    dataset commonly use -1/0/1 coding; for PCA and IBS-like similarity we add 1.
    """
    X = geno.apply(pd.to_numeric, errors="coerce").astype("float32")
    finite = X.to_numpy(dtype="float32", copy=False)
    finite_vals = finite[np.isfinite(finite)]
    if finite_vals.size == 0:
        return X, "all_missing_numeric"

    mn = float(np.nanmin(finite_vals))
    mx = float(np.nanmax(finite_vals))

    # Conservative detection of -1/0/1 marker coding.
    if mn >= -1.0001 and mx <= 1.0001 and mn < -0.01:
        return (X + 1.0).astype("float32"), "minus101_plus1_to_012"

    return X, "already_numeric_as_loaded"


def marker_qc(X: pd.DataFrame, dataset: str) -> pd.DataFrame:
    A = X.to_numpy(dtype="float32", copy=False)
    miss = np.isnan(A).mean(axis=0)
    mean_dosage = np.nanmean(A, axis=0)
    alt_freq = mean_dosage / 2.0
    maf = np.minimum(alt_freq, 1.0 - alt_freq)
    maf[(alt_freq < 0) | (alt_freq > 1)] = np.nan

    return pd.DataFrame({
        "dataset": dataset,
        "marker": X.columns.astype(str),
        "missing_rate": miss,
        "mean_dosage": mean_dosage,
        "alt_freq": alt_freq,
        "maf": maf,
    })


def sample_qc(X: pd.DataFrame, dataset: str) -> pd.DataFrame:
    A = X.to_numpy(dtype="float32", copy=False)
    return pd.DataFrame({
        "dataset": dataset,
        "sample_id": X.index.astype(str),
        "n_markers": X.shape[1],
        "missing_rate": np.isnan(A).mean(axis=1),
    })


def choose_filter_thresholds(
    X: pd.DataFrame,
    mqc: pd.DataFrame,
    sqc: pd.DataFrame,
    min_samples: int = 20,
    min_markers: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Choose the strictest QC threshold that leaves enough data for PCA.

    This is intentionally adaptive because public perennial crop datasets differ
    strongly in marker density and missingness. All selected thresholds are saved.
    """
    if X.shape[0] < min_samples:
        min_samples = max(3, X.shape[0])
    if X.shape[1] < min_markers:
        min_markers = max(2, X.shape[1])

    candidates = [
        {"sample_missing_max": 0.20, "marker_missing_max": 0.20, "maf_min": 0.010},
        {"sample_missing_max": 0.30, "marker_missing_max": 0.30, "maf_min": 0.010},
        {"sample_missing_max": 0.40, "marker_missing_max": 0.40, "maf_min": 0.010},
        {"sample_missing_max": 0.60, "marker_missing_max": 0.60, "maf_min": 0.010},
        {"sample_missing_max": 0.80, "marker_missing_max": 0.80, "maf_min": 0.005},
        {"sample_missing_max": 1.00, "marker_missing_max": 1.00, "maf_min": 0.000},
    ]

    best = None
    best_X = None

    for thr in candidates:
        keep_samples = sqc.loc[sqc["missing_rate"] <= thr["sample_missing_max"], "sample_id"].astype(str)
        keep_markers = mqc.loc[
            (mqc["missing_rate"] <= thr["marker_missing_max"]) &
            (mqc["maf"].fillna(-1) >= thr["maf_min"]),
            "marker"
        ].astype(str)

        sub = X.loc[X.index.astype(str).isin(set(keep_samples)), X.columns.astype(str).isin(set(keep_markers))]

        # Remove markers still all-missing after sample filter.
        if sub.shape[1] > 0:
            sub = sub.loc[:, sub.notna().any(axis=0)]

        if sub.shape[0] >= min_samples and sub.shape[1] >= min_markers:
            return sub, thr

        # Keep a fallback with maximum retained information.
        score = sub.shape[0] * max(1, sub.shape[1])
        if best is None or score > best:
            best = score
            best_X = sub
            best_thr = thr

    if best_X is None or best_X.empty:
        # Last-resort fallback: use all non-all-missing markers.
        sub = X.loc[:, X.notna().any(axis=0)]
        return sub, {"sample_missing_max": 1.0, "marker_missing_max": 1.0, "maf_min": 0.0}

    return best_X, best_thr


def mean_impute_and_scale(X: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    A = X.to_numpy(dtype="float32", copy=True)
    col_means = np.nanmean(A, axis=0)
    # If a column is somehow all missing, set to zero after imputation.
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype("float32")
    inds = np.where(np.isnan(A))
    if inds[0].size:
        A[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(A).astype("float32")
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z, pd.Series(col_means, index=X.columns, name="imputation_mean_dosage")


def run_pca(X: pd.DataFrame, dataset: str, features_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    Z, impute_means = mean_impute_and_scale(X)
    n_components = int(min(10, max(1, X.shape[0] - 1), X.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=SEED)
    scores_arr = pca.fit_transform(Z)

    score_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.DataFrame(scores_arr, columns=score_cols)
    scores.insert(0, "sample_id", X.index.astype(str))
    scores.insert(1, "dataset", dataset)

    ev = pd.DataFrame({
        "dataset": dataset,
        "PC": score_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
    })

    # Save top absolute loadings per PC rather than a huge full matrix.
    loading_rows = []
    comps = pca.components_
    markers = np.asarray(X.columns.astype(str))
    for pc_i, pc in enumerate(score_cols):
        vals = comps[pc_i]
        order = np.argsort(np.abs(vals))[::-1][: min(300, len(vals))]
        for rank, idx in enumerate(order, start=1):
            loading_rows.append({
                "dataset": dataset,
                "PC": pc,
                "rank_abs_loading": rank,
                "marker": markers[idx],
                "loading": float(vals[idx]),
                "abs_loading": float(abs(vals[idx])),
            })
    loadings_top = pd.DataFrame(loading_rows)

    impute_path = features_dir / f"{dataset}_imputation_means.tsv"
    write_df(impute_means.reset_index().rename(columns={"index": "marker"}), impute_path)

    return scores, ev, loadings_top, Z


def choose_kmeans(scores: pd.DataFrame, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pc_cols = [c for c in scores.columns if c.startswith("PC")]
    if len(pc_cols) == 0 or scores.shape[0] < 4:
        clusters = scores[["dataset", "sample_id"]].copy()
        clusters["kmeans_cluster"] = 1
        clusters["best_k"] = 1
        return clusters, pd.DataFrame([{
            "dataset": dataset, "k": 1, "silhouette": np.nan, "status": "too_few_samples_or_pcs"
        }])

    Xpc = scores[pc_cols[: min(5, len(pc_cols))]].to_numpy(dtype="float32")
    max_k = min(8, scores.shape[0] - 1)
    rows = []
    best_k = 1
    best_sil = -np.inf
    best_labels = np.ones(scores.shape[0], dtype=int)

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=50)
            labels = km.fit_predict(Xpc)
            if len(set(labels)) < 2:
                sil = np.nan
            else:
                sil = float(silhouette_score(Xpc, labels))
            rows.append({"dataset": dataset, "k": k, "silhouette": sil, "status": "OK"})
            if np.isfinite(sil) and sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels + 1
        except Exception as exc:
            rows.append({"dataset": dataset, "k": k, "silhouette": np.nan, "status": repr(exc)})

    clusters = scores[["dataset", "sample_id"] + pc_cols[: min(10, len(pc_cols))]].copy()
    clusters["kmeans_cluster"] = best_labels
    clusters["best_k"] = best_k

    if not np.isfinite(best_sil):
        best_sil = np.nan

    return clusters, pd.DataFrame(rows)


def likely_group_columns(pheno: pd.DataFrame) -> List[str]:
    candidates = []
    for c in pheno.columns:
        cl = str(c).lower()
        if c == "sample_id":
            continue
        if any(token in cl for token in ["group", "population", "env", "location", "region", "country", "source", "family", "block"]):
            # Allow low-cardinality numeric group codes such as Acc Group.
            nunique = pheno[c].nunique(dropna=True)
            if 1 < nunique <= max(50, int(0.75 * len(pheno))):
                candidates.append(c)
    return candidates[:8]


def make_validation_splits(scores: pd.DataFrame, clusters: pd.DataFrame, pheno: pd.DataFrame, dataset: str) -> pd.DataFrame:
    ids = scores["sample_id"].astype(str).tolist()
    n = len(ids)
    out = pd.DataFrame({"dataset": dataset, "sample_id": ids})

    n_splits = min(5, n)
    out["random_five_fold"] = -1
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        folds = np.empty(n, dtype=int)
        for fold, (_, test_idx) in enumerate(kf.split(np.arange(n)), start=1):
            folds[test_idx] = fold
        out["random_five_fold"] = folds

    cl = clusters[["sample_id", "kmeans_cluster"]].copy()
    cl["sample_id"] = cl["sample_id"].astype(str)
    out = out.merge(cl, on="sample_id", how="left")
    out["cluster_out"] = out["kmeans_cluster"]

    # Attach likely phenotype group columns for future leakage-aware validation.
    if "sample_id" in pheno.columns:
        p = pheno.copy()
        p["sample_id"] = p["sample_id"].astype(str)
        group_cols = likely_group_columns(p)
        if group_cols:
            out = out.merge(p[["sample_id"] + group_cols], on="sample_id", how="left")

    return out


def pair_similarity(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    valid = np.isfinite(a) & np.isfinite(b)
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    diff = np.abs(a[valid] - b[valid])
    sim = 1.0 - float(np.mean(diff / 2.0))
    return sim, n


def compute_pairwise_similarity(
    X: pd.DataFrame,
    dataset: str,
    features_dir: Path,
    threshold: float = 0.985,
    min_compared_markers: int = 50,
    top_n: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, float]:
    ids = X.index.astype(str).tolist()
    A = X.to_numpy(dtype="float32", copy=False)
    n = len(ids)
    sim_mat = np.eye(n, dtype="float32")
    pairs = []
    max_sim = np.nan

    if X.shape[1] < min_compared_markers:
        min_compared_markers = max(2, int(0.5 * X.shape[1]))

    for i in range(n):
        ai = A[i]
        for j in range(i + 1, n):
            sim, ncomp = pair_similarity(ai, A[j])
            sim_mat[i, j] = sim if np.isfinite(sim) else np.nan
            sim_mat[j, i] = sim_mat[i, j]
            if np.isfinite(sim):
                if not np.isfinite(max_sim) or sim > max_sim:
                    max_sim = sim
                if ncomp >= min_compared_markers and sim >= threshold:
                    pairs.append({
                        "dataset": dataset,
                        "sample_i": ids[i],
                        "sample_j": ids[j],
                        "ibs_like_similarity": sim,
                        "n_compared_markers": ncomp,
                        "threshold": threshold,
                    })

    pair_df = pd.DataFrame(pairs)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["ibs_like_similarity", "n_compared_markers"], ascending=[False, False])

    # Also save a top-pairs table even if below threshold. Useful for audit.
    top_rows = []
    triu_i, triu_j = np.triu_indices(n, k=1)
    vals = sim_mat[triu_i, triu_j]
    finite = np.isfinite(vals)
    if finite.any():
        order = np.argsort(vals[finite])[::-1][:top_n]
        ii = triu_i[finite][order]
        jj = triu_j[finite][order]
        vv = vals[finite][order]
        for rank, (i, j, v) in enumerate(zip(ii, jj, vv), start=1):
            sim, ncomp = pair_similarity(A[i], A[j])
            top_rows.append({
                "dataset": dataset,
                "rank": rank,
                "sample_i": ids[i],
                "sample_j": ids[j],
                "ibs_like_similarity": float(v),
                "n_compared_markers": int(ncomp),
            })
    top_df = pd.DataFrame(top_rows)

    sim_df = pd.DataFrame(sim_mat, index=ids, columns=ids)
    sim_df.index.name = "sample_id"
    write_df(sim_df, features_dir / f"{dataset}_ibs_like_similarity_matrix.tsv.gz", index=True)

    n_high = int(pair_df.shape[0])
    return pair_df, top_df, n_high, float(max_sim) if np.isfinite(max_sim) else np.nan



def process_dataset(dataset: str, cfg: Dict, dirs: Dict[str, Path], skip_similarity: bool = False) -> MatrixQCResult:
    print(f"\n[{dataset}] Loading matched genotype and phenotype...")
    geno_path = Path(cfg[dataset]["genotype_matched"])
    pheno_path = Path(cfg[dataset]["phenotype_matched"])

    geno_raw = read_tsv(geno_path, index_col=0)
    geno_raw.index = geno_raw.index.astype(str)
    geno_raw.columns = geno_raw.columns.astype(str)

    pheno = read_tsv(pheno_path)
    if "sample_id" not in pheno.columns:
        raise ValueError(f"{dataset} phenotype table does not have sample_id column: {pheno_path}")
    pheno["sample_id"] = pheno["sample_id"].astype(str)

    # Align defensively even though 02 already aligned.
    shared = [sid for sid in geno_raw.index.astype(str).tolist() if sid in set(pheno["sample_id"])]
    if len(shared) == 0:
        raise ValueError(f"{dataset}: zero shared samples after 02 harmonization. Stop and inspect sample IDs.")

    geno_raw = geno_raw.loc[shared].copy()
    pheno = pheno.set_index("sample_id").loc[shared].reset_index()

    geno, code_transform = convert_to_dosage_if_needed(geno_raw)

    mqc = marker_qc(geno, dataset)
    sqc = sample_qc(geno, dataset)
    write_df(mqc, dirs["qc"] / f"{dataset}_matched_marker_qc.tsv")
    write_df(sqc, dirs["qc"] / f"{dataset}_matched_sample_qc.tsv")

    geno_filt, thr = choose_filter_thresholds(geno, mqc, sqc)
    write_df(geno_filt, dirs["features"] / f"{dataset}_genotype_dosage_qc_filtered.tsv.gz", index=True)

    filter_report = pd.DataFrame([{
        "dataset": dataset,
        "code_transform": code_transform,
        "n_samples_before_filter": geno.shape[0],
        "n_markers_before_filter": geno.shape[1],
        "n_samples_after_filter": geno_filt.shape[0],
        "n_markers_after_filter": geno_filt.shape[1],
        **thr,
    }])
    write_df(filter_report, dirs["qc"] / f"{dataset}_filter_decision.tsv")

    if geno_filt.shape[0] < 3 or geno_filt.shape[1] < 2:
        raise ValueError(f"{dataset}: insufficient matrix after QC filtering: {geno_filt.shape}")

    print(f"[{dataset}] PCA matrix: {geno_filt.shape[0]} samples x {geno_filt.shape[1]} markers")
    scores, ev, loadings_top, Z = run_pca(geno_filt, dataset, dirs["features"])
    write_df(scores, dirs["features"] / f"{dataset}_pca_scores.tsv")
    write_df(ev, dirs["features"] / f"{dataset}_pca_explained_variance.tsv")
    write_df(loadings_top, dirs["features"] / f"{dataset}_pca_top_loadings.tsv")

    clusters, k_table = choose_kmeans(scores, dataset)
    write_df(clusters, dirs["features"] / f"{dataset}_pca_kmeans_clusters.tsv")
    write_df(k_table, dirs["qc"] / f"{dataset}_kmeans_silhouette.tsv")

    splits = make_validation_splits(scores, clusters, pheno, dataset)
    write_df(splits, dirs["features"] / f"{dataset}_validation_splits.tsv")

    if skip_similarity:
        pair_df = pd.DataFrame()
        top_df = pd.DataFrame()
        n_high = 0
        max_sim = np.nan
    else:
        print(f"[{dataset}] Computing missing-aware IBS-like similarity...")
        pair_df, top_df, n_high, max_sim = compute_pairwise_similarity(
            geno_filt, dataset, dirs["features"], threshold=0.985, min_compared_markers=50
        )
        write_df(pair_df, dirs["qc"] / f"{dataset}_near_duplicate_pairs_ge_0.985.tsv")
        write_df(top_df, dirs["qc"] / f"{dataset}_top_similarity_pairs.tsv")

    best_k = int(clusters["best_k"].iloc[0]) if "best_k" in clusters.columns and not clusters.empty else 1
    best_sil = np.nan
    if not k_table.empty and "silhouette" in k_table.columns:
        ok = k_table[np.isfinite(k_table["silhouette"])]
        if not ok.empty:
            best_sil = float(ok.loc[ok["silhouette"].idxmax(), "silhouette"])

    pc1 = float(ev.loc[ev["PC"] == "PC1", "explained_variance_ratio"].iloc[0]) if (ev["PC"] == "PC1").any() else np.nan
    pc2 = float(ev.loc[ev["PC"] == "PC2", "explained_variance_ratio"].iloc[0]) if (ev["PC"] == "PC2").any() else np.nan

    return MatrixQCResult(
        dataset=dataset,
        genotype_path=geno_path,
        phenotype_path=pheno_path,
        code_transform=code_transform,
        n_samples_raw=int(geno_raw.shape[0]),
        n_markers_raw=int(geno_raw.shape[1]),
        n_samples_aligned=int(geno.shape[0]),
        n_markers_aligned=int(geno.shape[1]),
        n_samples_pca=int(geno_filt.shape[0]),
        n_markers_pca=int(geno_filt.shape[1]),
        sample_missing_max=float(thr["sample_missing_max"]),
        marker_missing_max=float(thr["marker_missing_max"]),
        maf_min=float(thr["maf_min"]),
        n_high_similarity_pairs_0985=n_high,
        max_pair_similarity=max_sim,
        best_k=best_k,
        best_k_silhouette=best_sil,
        pc1_var=pc1,
        pc2_var=pc2,
    )


def write_report(summary: pd.DataFrame, dirs: Dict[str, Path], datasets: List[str], skip_similarity: bool) -> Path:
    report_path = dirs["qc"] / "03_qc_identity_structure_report.md"
    lines = []
    lines.append("# 03 QC, identity and population-structure report")
    lines.append("")
    lines.append(f"Generated: {now()}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This step audits the cleaned genotype/phenotype matrices before any prediction modeling. "
        "It applies matched-matrix QC, adaptive but recorded marker/sample filters, PCA, K-means clustering, "
        "missing-aware IBS-like similarity, near-duplicate screening, and validation-split scaffolding."
    )
    lines.append("")
    lines.append("## Datasets processed")
    lines.append("")
    for ds in datasets:
        lines.append(f"- {ds}")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Rigor notes")
    lines.append("")
    lines.append("- PCA and clustering are based only on phenotype-matched samples from 02_clean_and_harmonize.py.")
    lines.append("- Coffee -1/0/1 SNP coding is converted to 0/1/2 dosage only for QC/PCA/similarity calculations.")
    lines.append("- QC filters are adaptive across datasets but fully recorded in `*_filter_decision.tsv`.")
    lines.append("- Near-duplicate detection uses missing-aware IBS-like similarity, not imputed genotypes.")
    lines.append("- Random five-fold validation is provided as a conventional baseline; cluster-out validation is used for population-structure-aware stress testing.")
    if skip_similarity:
        lines.append("- Similarity computation was skipped by user option.")
    lines.append("")
    lines.append("## Key output folders")
    lines.append("")
    lines.append(f"- Features: `{dirs['features']}`")
    lines.append(f"- QC tables: `{dirs['qc']}`")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append(
        "Use these QC-filtered matrices, PCA clusters and validation splits for 04_prediction_benchmarks.py. "
        "The first modeling pass should compare mean/linear baselines, ridge/GBLUP-like models, elastic net, "
        "random forest and gradient boosting under random and cluster-out validation."
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="03 QC, identity, PCA and validation-split scaffolding.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Project root folder.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["icgt_cacao", "agrosavia_matina", "agrosavia_criollo", "coffee_int", "coffee_prec"],
        help="Datasets to process. Default: all cleaned datasets.",
    )
    parser.add_argument("--skip-similarity", action="store_true", help="Skip pairwise IBS-like similarity step.")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    qc_dir = project_root / "analysis/intermediate" / "qc"
    features_dir = project_root / "analysis/intermediate" / "features"
    table_dir = project_root / "analysis/outputs" / "tables"
    log_dir = project_root / "analysis/logs"
    ensure_dirs(qc_dir, features_dir, table_dir, log_dir)

    log_path = log_dir / f"03_qc_identity_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("===== 03 QC / IDENTITY / STRUCTURE =====")
    print(f"Project root: {project_root}")
    print(f"Started: {now()}")

    try:
        cfg = load_downstream_config(project_root)
        dirs = {"qc": qc_dir, "features": features_dir, "tables": table_dir, "logs": log_dir}

        results: List[MatrixQCResult] = []
        for dataset in args.datasets:
            if dataset not in cfg:
                print(f"[SKIP] {dataset}: not present in downstream config")
                continue
            results.append(process_dataset(dataset, cfg, dirs, skip_similarity=args.skip_similarity))

        summary = pd.DataFrame([r.__dict__ for r in results])
        # Convert path objects to strings for TSV output.
        for c in ["genotype_path", "phenotype_path"]:
            if c in summary.columns:
                summary[c] = summary[c].astype(str)

        summary_path = qc_dir / "03_qc_identity_structure_summary.tsv"
        write_df(summary, summary_path)
        write_df(summary, table_dir / "03_qc_identity_structure_summary.tsv")

        report_path = write_report(summary, dirs, args.datasets, args.skip_similarity)

        # Save a path manifest for later scripts.
        manifest = {
            "created": now(),
            "project_root": str(project_root),
            "summary": str(summary_path),
            "report": str(report_path),
            "features_dir": str(features_dir),
            "datasets": args.datasets,
        }
        manifest_path = qc_dir / "03_downstream_structure_paths.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("\n===== 03 QC / IDENTITY / STRUCTURE COMPLETE =====")
        print(summary[[
            "dataset", "n_samples_pca", "n_markers_pca", "sample_missing_max", "marker_missing_max",
            "maf_min", "max_pair_similarity", "n_high_similarity_pairs_0985", "best_k", "pc1_var", "pc2_var"
        ]].to_string(index=False))
        print("")
        print(f"Report: {report_path}")
        print(f"Summary: {summary_path}")

    except Exception:
        err = traceback.format_exc()
        log_path.write_text(err, encoding="utf-8")
        print("\nERROR during 03 QC/identity/structure. See log:")
        print(log_path)
        print(err)
        raise


if __name__ == "__main__":
    main()
