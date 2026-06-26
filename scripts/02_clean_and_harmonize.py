from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

def _project_root_from_args() -> Path:
    import argparse, os
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "."))
    args, _ = parser.parse_known_args()
    return Path(args.project_root).resolve()


PROJECT_ROOT = _project_root_from_args()
RAW = PROJECT_ROOT / "data/raw"
QC = PROJECT_ROOT / "analysis/intermediate" / "qc"
CLEAN = PROJECT_ROOT / "analysis/intermediate" / "cleaned"
FEATURES = PROJECT_ROOT / "analysis/intermediate" / "features"
TABLES = PROJECT_ROOT / "analysis/outputs" / "tables"
LOGS = PROJECT_ROOT / "analysis/logs"

for d in [QC, CLEAN, FEATURES, TABLES, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

PATHS = {
    "icgt_pheno": RAW / "cacao_icgt_plos2022" / "S02_phenotype_346_accessions.xlsx",
    "icgt_geno": RAW / "cacao_icgt_plos2022" / "S10_genotype_data_for_gwas.txt",
    "agro_s1": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S1 (1).xls",
    "agro_s2": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S2.xls",
    "agro_s3": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S3.xls",
    "agro_s4": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S4.xlsx",
    "agro_vcf_criollo": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S5_criollo.vcf",
    "agro_vcf_matina": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S6_matina.vcf",
    "cacao_219_panel": RAW / "cacao_identity_panels" / "BPR2021_S2_snp_information_selected_loci.xlsx",
    "coffee_int_snp": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Int_SNPdata.csv",
    "coffee_prec_snp": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Prec_SNPdata.csv",
    "coffee_int_pheno": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Int_PHENOdata.csv",
    "coffee_prec_pheno": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Prec_PHENOdata.csv",
}

META_COLS_ICGT = [
    "rs_", "alleles", "chrom", "pos", "strand", "assembly_", "center",
    "protLSID", "assayLSID", "panel", "QCcode"
]


# -----------------------------
# General helpers
# -----------------------------

def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def write_df(df: pd.DataFrame, path: Path, sep: str = "\t", index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        df.to_csv(path, sep=sep, index=index, compression="gzip")
    else:
        df.to_csv(path, sep=sep, index=index)
    return path


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_sample_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    return s


def key_basic(x: object) -> str:
    s = normalize_sample_id(x).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def candidate_suffix_base(s: str) -> str:
    """Return base name for common technical replicate suffixes such as _1 or .1.

    This function is intentionally conservative. We only use it as a fallback when
    exact matching fails and the base accession is unique.
    """
    s = normalize_sample_id(s)
    return re.sub(r"([_.-])([0-9]+)$", "", s)


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    exclude_set = set(exclude)
    cols = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# -----------------------------
# Genotype QC helpers
# -----------------------------

def marker_stats_from_dosage(df: pd.DataFrame, dataset: str, assume_code: str = "dosage012") -> pd.DataFrame:
    """Compute marker missingness and MAF.

    df must be samples x markers, numeric. For assume_code='dosage012', values are
    treated as 0/1/2. For assume_code='minus101', values are treated as -1/0/1 and
    converted to 0/1/2 by adding 1 for MAF calculation.
    """
    if df.empty:
        return pd.DataFrame(columns=["dataset", "marker", "missing_rate", "mean_dosage", "alt_freq", "maf"])

    X = df.apply(pd.to_numeric, errors="coerce")
    miss = X.isna().mean(axis=0)

    if assume_code == "minus101":
        Xdos = X + 1.0
    else:
        Xdos = X

    mean_dosage = Xdos.mean(axis=0, skipna=True)
    alt_freq = mean_dosage / 2.0
    maf = np.minimum(alt_freq, 1.0 - alt_freq)
    maf = maf.where((alt_freq >= 0) & (alt_freq <= 1))

    return pd.DataFrame({
        "dataset": dataset,
        "marker": X.columns.astype(str),
        "missing_rate": miss.values,
        "mean_dosage": mean_dosage.values,
        "alt_freq": alt_freq.values,
        "maf": maf.values,
    })


def sample_stats_from_matrix(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    X = df.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({
        "dataset": dataset,
        "sample_id": X.index.astype(str),
        "n_markers": X.shape[1],
        "missing_rate": X.isna().mean(axis=1).values,
    })


def exact_duplicate_groups(df: pd.DataFrame, dataset: str, max_groups: int = 5000) -> pd.DataFrame:
    """Find exact duplicate genotype rows after replacing NaN by a sentinel.

    This is a strict duplicate check. Near-duplicates will be handled in the next
    PCA/IBS script; here we only record perfect row equality.
    """
    if df.empty:
        return pd.DataFrame(columns=["dataset", "group_id", "n_samples", "sample_ids"])

    X = df.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(-9999)

    hashes = []
    for _, row in X.iterrows():
        arr = np.asarray(row.values, dtype=np.float32)
        hashes.append(hashlib.md5(arr.tobytes()).hexdigest())

    tmp = pd.DataFrame({"sample_id": X.index.astype(str), "hash": hashes})
    grouped = tmp.groupby("hash")["sample_id"].apply(list).reset_index()
    grouped = grouped[grouped["sample_id"].map(len) > 1].copy()

    rows = []
    for i, r in enumerate(grouped.itertuples(index=False), start=1):
        rows.append({
            "dataset": dataset,
            "group_id": i,
            "n_samples": len(r.sample_id),
            "sample_ids": ";".join(r.sample_id),
            "hash": r.hash,
        })
        if len(rows) >= max_groups:
            break

    return pd.DataFrame(rows)


# -----------------------------
# ICGT cacao
# -----------------------------

def decode_hapmap_gt(gt: object, alleles: object) -> float:
    if pd.isna(gt):
        return np.nan
    if pd.isna(alleles):
        return np.nan

    a = str(alleles).strip().upper()
    parts = re.split(r"[/|]", a)
    if len(parts) < 2:
        return np.nan
    ref, alt = parts[0], parts[1]

    s = str(gt).strip().upper()
    if s in {"", "N", "NN", "NA", "NAN", "--", ".", "./.", "0", "00"}:
        return np.nan

    if "/" in s or "|" in s:
        tokens = re.split(r"[/|]", s)
    elif len(s) == 2:
        tokens = list(s)
    elif len(s) == 1:
        tokens = [s]
    else:
        # Some files encode missing as unusual strings. Be conservative.
        return np.nan

    if any(t in {"", "N", "-", "."} for t in tokens):
        return np.nan

    if any(t not in {ref, alt} for t in tokens):
        return np.nan

    return float(sum(t == alt for t in tokens))


def load_icgt() -> Dict[str, pd.DataFrame]:
    print("\n[ICGT] Loading phenotype...")
    pheno = pd.read_excel(PATHS["icgt_pheno"], sheet_name="Phenotypewithoutmissingvalues")
    pheno = pheno.copy()
    pheno["sample_id"] = pheno["Accession"].map(normalize_sample_id)
    pheno["sample_key"] = pheno["sample_id"].map(key_basic)

    print("[ICGT] Loading HapMap-like genotype table...")
    raw = pd.read_csv(PATHS["icgt_geno"], sep="\t", dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]

    meta_cols = [c for c in META_COLS_ICGT if c in raw.columns]
    sample_cols = [c for c in raw.columns if c not in meta_cols]

    marker_meta = raw[meta_cols].copy()
    marker_meta = marker_meta.rename(columns={"rs_": "marker"})
    if "marker" not in marker_meta.columns:
        marker_meta.insert(0, "marker", [f"icgt_marker_{i+1}" for i in range(raw.shape[0])])
    marker_meta["marker"] = marker_meta["marker"].astype(str)

    # Ensure unique marker IDs.
    duplicated = marker_meta["marker"].duplicated(keep=False)
    if duplicated.any():
        marker_meta.loc[duplicated, "marker"] = [
            f"{m}__dup{i}" for i, m in enumerate(marker_meta.loc[duplicated, "marker"], start=1)
        ]

    print(f"[ICGT] Decoding {len(sample_cols)} samples x {raw.shape[0]} markers...")
    geno_rows = []
    alleles = raw["alleles"] if "alleles" in raw.columns else pd.Series([np.nan] * raw.shape[0])
    for col in sample_cols:
        vals = [decode_hapmap_gt(gt, al) for gt, al in zip(raw[col].values, alleles.values)]
        geno_rows.append(pd.Series(vals, name=normalize_sample_id(col)))

    geno = pd.DataFrame(geno_rows)
    geno.columns = marker_meta["marker"].tolist()
    geno.index.name = "sample_id"

    marker_qc = marker_stats_from_dosage(geno, "icgt_cacao", assume_code="dosage012")
    sample_qc = sample_stats_from_matrix(geno, "icgt_cacao")
    dup = exact_duplicate_groups(geno, "icgt_cacao")

    # Conservative phenotype-genotype matching.
    pheno_ids = pheno["sample_id"].tolist()
    pheno_keys_exact = {x: x for x in pheno_ids}
    pheno_key_basic = {}
    for sid in pheno_ids:
        k = key_basic(sid)
        pheno_key_basic.setdefault(k, []).append(sid)

    rows = []
    for gid in geno.index.astype(str):
        method = "unmatched"
        pid = ""
        ambiguous = False

        if gid in pheno_keys_exact:
            method = "exact"
            pid = gid
        else:
            base = candidate_suffix_base(gid)
            if base in pheno_keys_exact:
                method = "suffix_base_exact"
                pid = base
            else:
                k = key_basic(gid)
                hits = pheno_key_basic.get(k, [])
                if len(hits) == 1:
                    method = "basic_key"
                    pid = hits[0]
                elif len(hits) > 1:
                    method = "ambiguous_basic_key"
                    ambiguous = True

        rows.append({
            "genotype_sample_id": gid,
            "phenotype_sample_id": pid,
            "match_method": method,
            "ambiguous": ambiguous,
        })

    harmon = pd.DataFrame(rows)
    matched = harmon[(harmon["phenotype_sample_id"] != "") & (~harmon["ambiguous"])].copy()

    # Keep only one genotype sample per phenotype accession. Prefer exact matches.
    method_rank = {"exact": 0, "suffix_base_exact": 1, "basic_key": 2}
    matched["rank"] = matched["match_method"].map(method_rank).fillna(99)
    matched = matched.sort_values(["phenotype_sample_id", "rank", "genotype_sample_id"])
    matched_unique = matched.drop_duplicates("phenotype_sample_id", keep="first").copy()
    duplicate_pheno_matches = matched[matched.duplicated("phenotype_sample_id", keep=False)].copy()

    geno_matched = geno.loc[matched_unique["genotype_sample_id"].tolist()].copy()
    geno_matched.index = matched_unique["phenotype_sample_id"].tolist()
    geno_matched.index.name = "sample_id"

    pheno_matched = pheno.set_index("sample_id").loc[geno_matched.index].reset_index()

    return {
        "pheno": pheno,
        "geno": geno,
        "marker_meta": marker_meta,
        "marker_qc": marker_qc,
        "sample_qc": sample_qc,
        "duplicates": dup,
        "harmonization": harmon,
        "matched_unique": matched_unique,
        "duplicate_pheno_matches": duplicate_pheno_matches,
        "geno_matched": geno_matched,
        "pheno_matched": pheno_matched,
    }


# -----------------------------
# Agrosavia cacao
# -----------------------------

def load_agrosavia_tables() -> Dict[str, pd.DataFrame]:
    print("\n[AGROSAVIA] Loading supplemental tables...")

    accessions = pd.read_excel(PATHS["agro_s1"], sheet_name="S1_table", header=1)
    accessions = accessions.dropna(how="all").copy()
    accessions.columns = [str(c).strip() for c in accessions.columns]
    if "Sample" in accessions.columns:
        accessions["sample_id"] = accessions["Sample"].map(normalize_sample_id)

    ph = pd.read_excel(PATHS["agro_s2"], sheet_name="Table S2", header=1)
    ph = ph.dropna(how="all").copy()
    ph.columns = [str(c).strip() for c in ph.columns]
    first = ph.columns[0]
    ph = ph.rename(columns={first: "Accession_or_Harvest"})
    ph["Accession_or_Harvest"] = ph["Accession_or_Harvest"].map(normalize_sample_id)

    # Summary rows are accession IDs; harvest rows look like 2016-II, 2017-I, etc.
    current = None
    summary_rows = []
    long_rows = []
    trait_cols = [c for c in ph.columns if c != "Accession_or_Harvest"]

    for _, r in ph.iterrows():
        label = r["Accession_or_Harvest"]
        if label == "":
            continue
        is_harvest = bool(re.match(r"^\d{4}[-_][A-Za-z0-9]+$", label))

        if not is_harvest:
            current = label
            rec = {"sample_id": current, "harvest": "TOTAL"}
            for c in trait_cols:
                rec[c] = r[c]
            summary_rows.append(rec)
            long_rows.append(rec.copy())
        else:
            rec = {"sample_id": current, "harvest": label}
            for c in trait_cols:
                rec[c] = r[c]
            long_rows.append(rec)

    pheno_summary = pd.DataFrame(summary_rows)
    pheno_long = pd.DataFrame(long_rows)

    for df in [pheno_summary, pheno_long]:
        for c in df.columns:
            if c not in ["sample_id", "harvest"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    seq_stats = pd.read_excel(PATHS["agro_s3"], sheet_name="Recuperado_Hoja1", header=1)
    seq_stats = seq_stats.dropna(how="all").copy()
    seq_stats.columns = [str(c).strip() for c in seq_stats.columns]
    # The first data row after header=1 contains subheaders, so remove rows with missing or 'Sample'.
    first_col = seq_stats.columns[0]
    seq_stats = seq_stats[seq_stats[first_col].map(lambda x: normalize_sample_id(x).upper() not in {"", "SAMPLE"})].copy()
    seq_stats = seq_stats.rename(columns={first_col: "sample_id"})
    seq_stats["sample_id"] = seq_stats["sample_id"].map(normalize_sample_id)

    pos_sel = pd.read_excel(PATHS["agro_s4"], sheet_name="Hoja1", header=1)
    pos_sel = pos_sel.dropna(how="all").copy()
    pos_sel.columns = [str(c).strip() for c in pos_sel.columns]

    return {
        "accessions": accessions,
        "pheno_summary": pheno_summary,
        "pheno_long": pheno_long,
        "seq_stats": seq_stats,
        "positive_selection_genes": pos_sel,
    }


def parse_vcf_to_dosage(vcf_path: Path, dataset_name: str) -> Dict[str, pd.DataFrame]:
    print(f"[AGROSAVIA] Parsing VCF: {vcf_path.name}")

    opener = gzip.open if str(vcf_path).lower().endswith(".gz") else open
    samples: List[str] = []
    markers: List[str] = []
    meta_rows: List[Dict[str, object]] = []
    dosage_rows: List[List[float]] = []

    with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                samples = [normalize_sample_id(x) for x in cols[9:]]
                continue
            if not line.strip():
                continue

            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info, fmt = cols[:9]
            marker = vid if vid not in {"", "."} else f"{chrom}:{pos}:{ref}>{alt}"
            if marker in markers:
                marker = f"{marker}__{len(markers)+1}"
            markers.append(marker)

            fmt_fields = fmt.split(":")
            try:
                gt_idx = fmt_fields.index("GT")
            except ValueError:
                gt_idx = 0

            vals: List[float] = []
            for cell in cols[9:]:
                parts = cell.split(":")
                gt = parts[gt_idx] if gt_idx < len(parts) else parts[0]
                if gt in {".", "./.", ".|.", ""}:
                    vals.append(np.nan)
                    continue
                alle = re.split(r"[/|]", gt)
                if any(a == "." or a == "" for a in alle):
                    vals.append(np.nan)
                    continue
                try:
                    # Count any non-reference allele as ALT dosage. This is robust for biallelic
                    # and conservative for rare multiallelic records.
                    vals.append(float(sum(int(a) > 0 for a in alle)))
                except Exception:
                    vals.append(np.nan)

            dosage_rows.append(vals)
            meta_rows.append({
                "dataset": dataset_name,
                "marker": marker,
                "chrom": chrom,
                "pos": pd.to_numeric(pos, errors="coerce"),
                "id": vid,
                "ref": ref,
                "alt": alt,
                "qual": qual,
                "filter": filt,
            })

    if not samples:
        raise RuntimeError(f"No VCF samples parsed from {vcf_path}")

    geno = pd.DataFrame(np.asarray(dosage_rows, dtype=np.float32).T, index=samples, columns=markers)
    geno.index.name = "sample_id"
    meta = pd.DataFrame(meta_rows)
    marker_qc = marker_stats_from_dosage(geno, dataset_name, assume_code="dosage012")
    sample_qc = sample_stats_from_matrix(geno, dataset_name)
    dup = exact_duplicate_groups(geno, dataset_name)

    return {
        "geno": geno,
        "marker_meta": meta,
        "marker_qc": marker_qc,
        "sample_qc": sample_qc,
        "duplicates": dup,
    }


def harmonize_agrosavia(geno: pd.DataFrame, pheno_summary: pd.DataFrame, dataset_name: str) -> Dict[str, pd.DataFrame]:
    """Match Agrosavia VCF sample IDs to phenotype accession IDs.

    The Agrosavia VCF sample names include sequencing/index suffixes, for example:
        830713_TAGCTT
        CCN_51_GCCAAT
        830698_GGTTATGC-CGCGATTA

    The phenotype table uses only the biological accession ID, for example:
        830713
        CCN_51

    We therefore remove only a final barcode-like suffix. This is deliberately
    conservative so that accession names that naturally contain underscores,
    such as CCN_51 or EET_400, are preserved.
    """

    def strip_final_barcode(sample_id: object) -> str:
        sid = normalize_sample_id(sample_id)
        if "_" not in sid:
            return sid
        base, tail = sid.rsplit("_", 1)
        tail_u = tail.upper()
        # Illumina/index suffixes in these VCFs are A/C/G/T strings, sometimes
        # dual-index strings separated by a hyphen. Accession suffixes such as
        # _51, _400, or _007 are not stripped by this rule.
        if re.fullmatch(r"[ACGTN]+(?:-[ACGTN]+)?", tail_u) and len(tail_u.replace("-", "")) >= 4:
            return base
        return sid

    ph = pheno_summary.copy()
    ph["sample_id"] = ph["sample_id"].map(normalize_sample_id)
    ph["sample_key"] = ph["sample_id"].map(key_basic)

    pheno_exact = set(ph["sample_id"].astype(str))
    pheno_key_map: Dict[str, List[str]] = {}
    for sid in ph["sample_id"].astype(str):
        pheno_key_map.setdefault(key_basic(sid), []).append(sid)

    rows = []
    for gid in geno.index.astype(str):
        gid_norm = normalize_sample_id(gid)
        gid_base = strip_final_barcode(gid_norm)
        pid = ""
        method = "unmatched"
        ambiguous = False

        candidates: List[Tuple[str, str, int]] = []

        if gid_norm in pheno_exact:
            candidates.append((gid_norm, "exact_full_vcf_id", 0))
        if gid_base in pheno_exact:
            candidates.append((gid_base, "barcode_stripped_exact", 1))

        # Basic-key fallback handles punctuation/spacing differences but is only
        # allowed when the key is unique in the phenotype table.
        for key_source, key_method, rank in [
            (gid_base, "barcode_stripped_basic_key", 2),
            (gid_norm, "full_vcf_basic_key", 3),
        ]:
            hits = pheno_key_map.get(key_basic(key_source), [])
            if len(hits) == 1:
                candidates.append((hits[0], key_method, rank))
            elif len(hits) > 1:
                ambiguous = True

        if candidates:
            # Deduplicate candidate rows while retaining best rank.
            best = sorted(candidates, key=lambda x: (x[2], x[0]))[0]
            pid, method, _ = best
            ambiguous = False
        elif ambiguous:
            method = "ambiguous_basic_key"

        rows.append({
            "genotype_sample_id": gid_norm,
            "genotype_accession_id_inferred": gid_base,
            "phenotype_sample_id": pid,
            "match_method": method,
            "ambiguous": ambiguous,
        })

    h = pd.DataFrame(rows)

    matched = h[(h["phenotype_sample_id"] != "") & (~h["ambiguous"])].copy()
    method_rank = {
        "exact_full_vcf_id": 0,
        "barcode_stripped_exact": 1,
        "barcode_stripped_basic_key": 2,
        "full_vcf_basic_key": 3,
    }
    if not matched.empty:
        matched["rank"] = matched["match_method"].map(method_rank).fillna(99)
        matched = matched.sort_values(["phenotype_sample_id", "rank", "genotype_sample_id"])
        matched_unique = matched.drop_duplicates("phenotype_sample_id", keep="first").copy()
    else:
        matched_unique = matched.copy()

    matched_ids = matched_unique["genotype_sample_id"].tolist()
    matched_pheno_ids = matched_unique["phenotype_sample_id"].tolist()

    geno_m = geno.loc[matched_ids].copy()
    geno_m.index = matched_pheno_ids
    geno_m.index.name = "sample_id"

    pheno_indexed = ph.set_index("sample_id", drop=False)
    pheno_m = pheno_indexed.loc[matched_pheno_ids].drop(columns=["sample_key"], errors="ignore").reset_index(drop=True)

    # Hard guard: zero matched Agrosavia samples is almost certainly an ID-parsing
    # bug and should not be allowed to pass silently.
    if ph.shape[0] > 0 and geno.shape[0] > 0 and geno_m.shape[0] == 0:
        example_geno = "; ".join(list(geno.index.astype(str)[:8]))
        example_pheno = "; ".join(list(ph["sample_id"].astype(str)[:8]))
        raise RuntimeError(
            f"{dataset_name}: zero phenotype-genotype matches after barcode stripping. "
            f"Example genotype IDs: {example_geno}. Example phenotype IDs: {example_pheno}."
        )

    return {"harmonization": h, "geno_matched": geno_m, "pheno_matched": pheno_m}


# -----------------------------
# Cacao identity panel
# -----------------------------

def load_cacao_219_panel() -> pd.DataFrame:
    print("\n[PANEL] Loading cacao 219-SNP panel information...")
    panel = pd.read_excel(PATHS["cacao_219_panel"], sheet_name="Sheet1", header=1)
    panel = panel.dropna(how="all").copy()
    panel.columns = [str(c).strip() for c in panel.columns]
    if "Loci" in panel.columns:
        panel = panel[panel["Loci"].notna()].copy()
        panel["marker"] = panel["Loci"].astype(str).str.replace("*", "", regex=False).str.strip()
        panel["selected_locus_in_paper"] = panel["Loci"].astype(str).str.startswith("*")
    return panel


# -----------------------------
# Coffee
# -----------------------------

def load_coffee_population(pop: str, snp_path: Path, pheno_path: Path) -> Dict[str, pd.DataFrame]:
    print(f"\n[COFFEE {pop}] Loading genotype...")
    geno = pd.read_csv(snp_path)
    first = geno.columns[0]
    geno = geno.rename(columns={first: "sample_id"})
    geno["sample_id"] = geno["sample_id"].map(normalize_sample_id)
    geno = geno.set_index("sample_id")
    geno = geno.apply(pd.to_numeric, errors="coerce")
    geno.index.name = "sample_id"

    # Raw coding is generally -1/0/1. Keep it as original code and also compute QC after +1.
    unique_preview = pd.Series(geno.values.ravel()).dropna().sample(
        n=min(10000, int(np.isfinite(geno.values).sum())), random_state=1
    ) if np.isfinite(geno.values).sum() > 0 else pd.Series(dtype=float)
    vals = set(np.unique(unique_preview.astype(float).values)) if len(unique_preview) else set()
    assume_code = "minus101" if vals.issubset({-1.0, 0.0, 1.0}) else "dosage012"

    marker_qc = marker_stats_from_dosage(geno, f"coffee_{pop}", assume_code=assume_code)
    sample_qc = sample_stats_from_matrix(geno, f"coffee_{pop}")
    dup = exact_duplicate_groups(geno, f"coffee_{pop}")

    print(f"[COFFEE {pop}] Loading phenotype...")
    pheno = pd.read_csv(pheno_path)
    pheno = pheno.rename(columns={pheno.columns[0]: "sample_id"})
    pheno["sample_id"] = pheno["sample_id"].map(normalize_sample_id)
    for c in ["production", "rust", "green"]:
        if c in pheno.columns:
            pheno[c] = pd.to_numeric(pheno[c], errors="coerce")

    trait_cols = [c for c in ["production", "rust", "green"] if c in pheno.columns]
    pheno_mean = pheno.groupby("sample_id", as_index=False)[trait_cols].mean()
    pheno_mean["n_env_records"] = pheno.groupby("sample_id").size().reindex(pheno_mean["sample_id"]).values

    g_ids = set(geno.index.astype(str))
    p_ids = set(pheno_mean["sample_id"].astype(str))
    rows = []
    for sid in sorted(g_ids.union(p_ids)):
        rows.append({
            "sample_id": sid,
            "in_genotype": sid in g_ids,
            "in_phenotype": sid in p_ids,
            "matched": sid in g_ids and sid in p_ids,
        })
    harmon = pd.DataFrame(rows)

    matched_ids = [sid for sid in geno.index.astype(str) if sid in p_ids]
    geno_matched = geno.loc[matched_ids].copy()
    pheno_matched = pheno_mean.set_index("sample_id").loc[matched_ids].reset_index()

    return {
        "geno": geno,
        "pheno_long": pheno,
        "pheno_mean": pheno_mean,
        "marker_qc": marker_qc,
        "sample_qc": sample_qc,
        "duplicates": dup,
        "harmonization": harmon,
        "geno_matched": geno_matched,
        "pheno_matched": pheno_matched,
        "assume_code": pd.DataFrame([{"population": pop, "genotype_code_assumption": assume_code}]),
    }



def df_to_markdown_simple(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)
    rows = [[fmt(v) for v in row] for row in df.to_numpy()]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        safe_row = [cell.replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    return "\n".join(lines)

# -----------------------------
# Output management
# -----------------------------

def record_output(records: List[Dict[str, object]], label: str, path: Path, df: Optional[pd.DataFrame] = None) -> None:
    exists = path.exists()
    records.append({
        "label": label,
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "n_rows": df.shape[0] if df is not None else "",
        "n_cols": df.shape[1] if df is not None else "",
        "sha256": sha256sum(path) if exists and path.stat().st_size < 300_000_000 else "skipped_large_or_missing",
    })


def save_all() -> Dict[str, object]:
    output_records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    # ICGT
    icgt = load_icgt()
    for label, df, fname in [
        ("icgt_pheno_all", icgt["pheno"], "icgt_cacao_phenotype_all.tsv"),
        ("icgt_pheno_matched", icgt["pheno_matched"], "icgt_cacao_phenotype_matched.tsv"),
        ("icgt_genotype_dosage_all", icgt["geno"], "icgt_cacao_genotype_dosage_all.tsv.gz"),
        ("icgt_genotype_dosage_matched", icgt["geno_matched"], "icgt_cacao_genotype_dosage_matched.tsv.gz"),
        ("icgt_marker_metadata", icgt["marker_meta"], "icgt_cacao_marker_metadata.tsv"),
        ("icgt_marker_qc", icgt["marker_qc"], "icgt_cacao_marker_qc.tsv"),
        ("icgt_sample_qc", icgt["sample_qc"], "icgt_cacao_sample_qc.tsv"),
        ("icgt_exact_duplicate_groups", icgt["duplicates"], "icgt_cacao_exact_duplicate_groups.tsv"),
        ("icgt_sample_harmonization", icgt["harmonization"], "icgt_cacao_sample_harmonization.tsv"),
        ("icgt_duplicate_pheno_matches", icgt["duplicate_pheno_matches"], "icgt_cacao_duplicate_pheno_matches.tsv"),
    ]:
        p = CLEAN / fname if "genotype" in label or "pheno" in label else QC / fname
        write_df(df, p, sep="\t", index=("genotype" in label))
        record_output(output_records, label, p, df)

    summary_rows.append({
        "dataset": "icgt_cacao",
        "n_pheno_samples": icgt["pheno"].shape[0],
        "n_genotype_samples": icgt["geno"].shape[0],
        "n_markers": icgt["geno"].shape[1],
        "n_matched_samples": icgt["geno_matched"].shape[0],
        "mean_marker_missing_rate": icgt["marker_qc"]["missing_rate"].mean(),
        "mean_marker_maf": icgt["marker_qc"]["maf"].mean(),
        "n_exact_duplicate_groups": icgt["duplicates"].shape[0],
    })

    # Agrosavia
    agro_tables = load_agrosavia_tables()
    for label, df, fname in [
        ("agrosavia_accessions", agro_tables["accessions"], "agrosavia_accessions.tsv"),
        ("agrosavia_pheno_summary", agro_tables["pheno_summary"], "agrosavia_pheno_summary.tsv"),
        ("agrosavia_pheno_long", agro_tables["pheno_long"], "agrosavia_pheno_long.tsv"),
        ("agrosavia_sequence_stats", agro_tables["seq_stats"], "agrosavia_sequence_stats.tsv"),
        ("agrosavia_positive_selection_genes", agro_tables["positive_selection_genes"], "agrosavia_positive_selection_genes.tsv"),
    ]:
        p = CLEAN / fname if "pheno" in label or label == "agrosavia_accessions" else QC / fname
        write_df(df, p, sep="\t", index=False)
        record_output(output_records, label, p, df)

    for ref_name, path_key in [("matina", "agro_vcf_matina"), ("criollo", "agro_vcf_criollo")]:
        v = parse_vcf_to_dosage(PATHS[path_key], f"agrosavia_{ref_name}")
        h = harmonize_agrosavia(v["geno"], agro_tables["pheno_summary"], f"agrosavia_{ref_name}")

        save_list = [
            (f"agrosavia_{ref_name}_genotype_dosage_all", v["geno"], f"agrosavia_{ref_name}_genotype_dosage_all.tsv.gz", CLEAN, True),
            (f"agrosavia_{ref_name}_genotype_dosage_matched", h["geno_matched"], f"agrosavia_{ref_name}_genotype_dosage_matched.tsv.gz", CLEAN, True),
            (f"agrosavia_{ref_name}_phenotype_matched", h["pheno_matched"], f"agrosavia_{ref_name}_phenotype_matched.tsv", CLEAN, False),
            (f"agrosavia_{ref_name}_marker_metadata", v["marker_meta"], f"agrosavia_{ref_name}_marker_metadata.tsv", QC, False),
            (f"agrosavia_{ref_name}_marker_qc", v["marker_qc"], f"agrosavia_{ref_name}_marker_qc.tsv", QC, False),
            (f"agrosavia_{ref_name}_sample_qc", v["sample_qc"], f"agrosavia_{ref_name}_sample_qc.tsv", QC, False),
            (f"agrosavia_{ref_name}_exact_duplicate_groups", v["duplicates"], f"agrosavia_{ref_name}_exact_duplicate_groups.tsv", QC, False),
            (f"agrosavia_{ref_name}_sample_harmonization", h["harmonization"], f"agrosavia_{ref_name}_sample_harmonization.tsv", QC, False),
        ]
        for label, df, fname, folder, index in save_list:
            p = folder / fname
            write_df(df, p, sep="\t", index=index)
            record_output(output_records, label, p, df)

        summary_rows.append({
            "dataset": f"agrosavia_{ref_name}",
            "n_pheno_samples": agro_tables["pheno_summary"].shape[0],
            "n_genotype_samples": v["geno"].shape[0],
            "n_markers": v["geno"].shape[1],
            "n_matched_samples": h["geno_matched"].shape[0],
            "mean_marker_missing_rate": v["marker_qc"]["missing_rate"].mean(),
            "mean_marker_maf": v["marker_qc"]["maf"].mean(),
            "n_exact_duplicate_groups": v["duplicates"].shape[0],
        })

    # Panel
    panel = load_cacao_219_panel()
    p = CLEAN / "cacao_219snp_identity_panel.tsv"
    write_df(panel, p, sep="\t", index=False)
    record_output(output_records, "cacao_219snp_identity_panel", p, panel)

    # Coffee
    coffee_int = load_coffee_population("int", PATHS["coffee_int_snp"], PATHS["coffee_int_pheno"])
    coffee_prec = load_coffee_population("prec", PATHS["coffee_prec_snp"], PATHS["coffee_prec_pheno"])

    for pop, obj in [("int", coffee_int), ("prec", coffee_prec)]:
        save_list = [
            (f"coffee_{pop}_genotype_code_all", obj["geno"], f"coffee_{pop}_genotype_code_all.tsv.gz", CLEAN, True),
            (f"coffee_{pop}_genotype_code_matched", obj["geno_matched"], f"coffee_{pop}_genotype_code_matched.tsv.gz", CLEAN, True),
            (f"coffee_{pop}_pheno_long", obj["pheno_long"], f"coffee_{pop}_pheno_long.tsv", CLEAN, False),
            (f"coffee_{pop}_pheno_mean", obj["pheno_mean"], f"coffee_{pop}_pheno_mean.tsv", CLEAN, False),
            (f"coffee_{pop}_pheno_matched", obj["pheno_matched"], f"coffee_{pop}_pheno_matched.tsv", CLEAN, False),
            (f"coffee_{pop}_marker_qc", obj["marker_qc"], f"coffee_{pop}_marker_qc.tsv", QC, False),
            (f"coffee_{pop}_sample_qc", obj["sample_qc"], f"coffee_{pop}_sample_qc.tsv", QC, False),
            (f"coffee_{pop}_exact_duplicate_groups", obj["duplicates"], f"coffee_{pop}_exact_duplicate_groups.tsv", QC, False),
            (f"coffee_{pop}_sample_harmonization", obj["harmonization"], f"coffee_{pop}_sample_harmonization.tsv", QC, False),
            (f"coffee_{pop}_genotype_code_assumption", obj["assume_code"], f"coffee_{pop}_genotype_code_assumption.tsv", QC, False),
        ]
        for label, df, fname, folder, index in save_list:
            p = folder / fname
            write_df(df, p, sep="\t", index=index)
            record_output(output_records, label, p, df)

        summary_rows.append({
            "dataset": f"coffee_{pop}",
            "n_pheno_samples": obj["pheno_mean"].shape[0],
            "n_genotype_samples": obj["geno"].shape[0],
            "n_markers": obj["geno"].shape[1],
            "n_matched_samples": obj["geno_matched"].shape[0],
            "mean_marker_missing_rate": obj["marker_qc"]["missing_rate"].mean(),
            "mean_marker_maf": obj["marker_qc"]["maf"].mean(),
            "n_exact_duplicate_groups": obj["duplicates"].shape[0],
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = QC / "cleaned_dataset_summary.tsv"
    write_df(summary, summary_path, sep="\t", index=False)
    record_output(output_records, "cleaned_dataset_summary", summary_path, summary)

    output_manifest = pd.DataFrame(output_records)
    manifest_path = QC / "cleaned_output_manifest.tsv"
    write_df(output_manifest, manifest_path, sep="\t", index=False)

    # Save a simple path config for downstream scripts.
    downstream = {
        "project_root": str(PROJECT_ROOT),
        "created": now(),
        "primary_cacao_prediction_dataset": "agrosavia_matina",
        "icgt_cacao": {
            "genotype_matched": str(CLEAN / "icgt_cacao_genotype_dosage_matched.tsv.gz"),
            "phenotype_matched": str(CLEAN / "icgt_cacao_phenotype_matched.tsv"),
        },
        "agrosavia_matina": {
            "genotype_matched": str(CLEAN / "agrosavia_matina_genotype_dosage_matched.tsv.gz"),
            "phenotype_matched": str(CLEAN / "agrosavia_matina_phenotype_matched.tsv"),
            "marker_metadata": str(QC / "agrosavia_matina_marker_metadata.tsv"),
        },
        "agrosavia_criollo": {
            "genotype_matched": str(CLEAN / "agrosavia_criollo_genotype_dosage_matched.tsv.gz"),
            "phenotype_matched": str(CLEAN / "agrosavia_criollo_phenotype_matched.tsv"),
            "marker_metadata": str(QC / "agrosavia_criollo_marker_metadata.tsv"),
        },
        "coffee_int": {
            "genotype_matched": str(CLEAN / "coffee_int_genotype_code_matched.tsv.gz"),
            "phenotype_matched": str(CLEAN / "coffee_int_pheno_matched.tsv"),
        },
        "coffee_prec": {
            "genotype_matched": str(CLEAN / "coffee_prec_genotype_code_matched.tsv.gz"),
            "phenotype_matched": str(CLEAN / "coffee_prec_pheno_matched.tsv"),
        },
    }
    config_path = QC / "downstream_data_paths.json"
    config_path.write_text(json.dumps(downstream, indent=2), encoding="utf-8")

    report_path = QC / "02_clean_and_harmonize_report.md"
    report_lines = []
    report_lines.append("# 02 Clean and harmonize report")
    report_lines.append("")
    report_lines.append(f"Generated: {now()}")
    report_lines.append("")
    report_lines.append("## Cleaned dataset summary")
    report_lines.append("")
    report_lines.append(df_to_markdown_simple(summary))
    report_lines.append("")
    report_lines.append("## Key rigor notes")
    report_lines.append("")
    report_lines.append("- ICGT cacao genotypes were decoded from HapMap-like nucleotide calls into alternate-allele dosage 0/1/2.")
    report_lines.append("- Agrosavia cacao VCF files were decoded into alternate-allele dosage 0/1/2. Matina is treated as the primary reference-aligned dataset for downstream candidate-region work; Criollo is retained for sensitivity checks.")
    report_lines.append("- Coffea canephora genotype matrices were preserved in their original numeric code. Marker MAF was calculated after treating -1/0/1 coding as 0/1/2 where applicable.")
    report_lines.append("- Phenotype-genotype harmonization was explicit and saved as separate sample-harmonization files. Ambiguous ICGT genotype-to-phenotype matches were not silently used.")
    report_lines.append("- Agrosavia VCF sample names contain sequencing/index barcodes; these were stripped only when the final suffix matched a conservative A/C/G/T barcode pattern, preserving accession names such as CCN_51 and EET_400.")
    report_lines.append("- Exact duplicate genotype groups were recorded using strict row equality. Near-duplicate/IBS-based checks are intentionally deferred to the next script.")
    report_lines.append("")
    report_lines.append("## Output manifest")
    report_lines.append("")
    report_lines.append(f"- `{manifest_path}`")
    report_lines.append(f"- `{config_path}`")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "summary": summary,
        "manifest": output_manifest,
        "report_path": report_path,
        "config_path": config_path,
    }


def main() -> None:
    started = datetime.now()
    log_path = LOGS / f"02_clean_and_harmonize_{started.strftime('%Y%m%d_%H%M%S')}.log"
    print("===== 02 CLEAN AND HARMONIZE =====")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Started: {started.isoformat(timespec='seconds')}")

    try:
        result = save_all()
    except Exception:
        err = traceback.format_exc()
        log_path.write_text(err, encoding="utf-8")
        print("\nERROR during clean/harmonize. See log:")
        print(log_path)
        raise

    elapsed = datetime.now() - started
    print("\n===== CLEAN AND HARMONIZE COMPLETE =====")
    print(result["summary"].to_string(index=False))
    print("")
    print(f"Report: {result['report_path']}")
    print(f"Downstream path config: {result['config_path']}")
    print(f"Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
