from __future__ import annotations

import csv
import gzip
import hashlib
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
TABLES = PROJECT_ROOT / "analysis/outputs" / "tables"
LOGS = PROJECT_ROOT / "analysis/logs"

for d in [QC, TABLES, LOGS, QC / "table_heads"]:
    d.mkdir(parents=True, exist_ok=True)


EXPECTED_FILES = {
    "icgt_cacao_phenotype_xlsx": RAW / "cacao_icgt_plos2022" / "S02_phenotype_346_accessions.xlsx",
    "icgt_cacao_gwas_genotype_txt": RAW / "cacao_icgt_plos2022" / "S10_genotype_data_for_gwas.txt",

    "agrosavia_file_s1_xls": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S1 (1).xls",
    "agrosavia_file_s2_xls": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S2.xls",
    "agrosavia_file_s3_xls": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S3.xls",
    "agrosavia_file_s4_xlsx": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S4.xlsx",
    "agrosavia_criollo_vcf": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S5_criollo.vcf",
    "agrosavia_matina_vcf": RAW / "cacao_agrosavia_g3_2020" / "figshare_files" / "File_S6_matina.vcf",

    "cacao_219snp_panel_xlsx": RAW / "cacao_identity_panels" / "BPR2021_S2_snp_information_selected_loci.xlsx",

    "coffee_int_snp_csv": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Int_SNPdata.csv",
    "coffee_prec_snp_csv": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Prec_SNPdata.csv",
    "coffee_int_pheno_csv": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Int_PHENOdata.csv",
    "coffee_prec_pheno_csv": RAW / "coffee_canephora_dryad2018" / "unzipped" / "data_set" / "Pop_Prec_PHENOdata.csv",
}


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_lines_binary(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def read_sample_text(path: Path, nbytes: int = 65536) -> Tuple[str, str]:
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            with path.open("r", encoding=enc, errors="replace") as f:
                return f.read(nbytes), enc
        except Exception:
            continue
    with path.open("r", errors="replace") as f:
        return f.read(nbytes), "unknown"


def sniff_delimiter(path: Path) -> Tuple[str, str]:
    sample, enc = read_sample_text(path)
    candidates = ["\t", ",", ";", " "]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
        return dialect.delimiter, enc
    except Exception:
        pass

    lines = [x for x in sample.splitlines() if x.strip()][:20]
    best_delim = "\t"
    best_score = -1

    for delim in candidates:
        counts = [len(line.split(delim)) for line in lines]
        if counts:
            score = sum(counts) / len(counts)
            if score > best_score:
                best_score = score
                best_delim = delim

    return best_delim, enc


def inspect_delimited(path: Path, label: str) -> Dict:
    delim, enc = sniff_delimiter(path)
    nlines = count_lines_binary(path)

    try:
        df_head = pd.read_csv(path, sep=delim, nrows=10, engine="python", encoding=enc)
        ncols = df_head.shape[1]
        cols = list(map(str, df_head.columns))

        head_file = QC / "table_heads" / f"{safe_name(label)}__head.csv"
        df_head.to_csv(head_file, index=False)

        return {
            "dataset": label,
            "path": str(path),
            "format": "delimited",
            "sheet": "",
            "status": "OK",
            "delimiter": repr(delim),
            "encoding": enc,
            "n_rows_est": max(0, nlines - 1),
            "n_cols": ncols,
            "first_column": cols[0] if cols else "",
            "columns_preview": " | ".join(cols[:30]),
            "head_file": str(head_file),
            "error": "",
        }

    except Exception as exc:
        return {
            "dataset": label,
            "path": str(path),
            "format": "delimited",
            "sheet": "",
            "status": "ERROR",
            "delimiter": repr(delim),
            "encoding": enc,
            "n_rows_est": max(0, nlines - 1),
            "n_cols": "",
            "first_column": "",
            "columns_preview": "",
            "head_file": "",
            "error": repr(exc),
        }


def inspect_excel(path: Path, label: str) -> List[Dict]:
    rows = []

    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:
        rows.append({
            "dataset": label,
            "path": str(path),
            "format": "excel",
            "sheet": "",
            "status": "ERROR",
            "delimiter": "",
            "encoding": "",
            "n_rows_est": "",
            "n_cols": "",
            "first_column": "",
            "columns_preview": "",
            "head_file": "",
            "error": repr(exc) + " | For old .xls files, install xlrd: python -m pip install xlrd",
        })
        return rows

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
            nrows, ncols = df.shape
            cols = list(map(str, df.columns))

            head_file = QC / "table_heads" / f"{safe_name(label)}__{safe_name(sheet)}__head.csv"
            df.head(10).to_csv(head_file, index=False)

            rows.append({
                "dataset": label,
                "path": str(path),
                "format": "excel",
                "sheet": sheet,
                "status": "OK",
                "delimiter": "",
                "encoding": "",
                "n_rows_est": nrows,
                "n_cols": ncols,
                "first_column": cols[0] if cols else "",
                "columns_preview": " | ".join(cols[:30]),
                "head_file": str(head_file),
                "error": "",
            })

        except Exception as exc:
            rows.append({
                "dataset": label,
                "path": str(path),
                "format": "excel",
                "sheet": sheet,
                "status": "ERROR",
                "delimiter": "",
                "encoding": "",
                "n_rows_est": "",
                "n_cols": "",
                "first_column": "",
                "columns_preview": "",
                "head_file": "",
                "error": repr(exc),
            })

    return rows


def open_text_or_gzip(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def inspect_vcf(path: Path, label: str) -> Dict:
    meta_lines = 0
    variant_lines = 0
    samples = []
    chrom_counts: Dict[str, int] = {}
    header_n_cols = 0

    with open_text_or_gzip(path) as f:
        for line in f:
            if line.startswith("##"):
                meta_lines += 1
                continue

            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                header_n_cols = len(cols)
                samples = cols[9:] if len(cols) > 9 else []
                continue

            if not line.strip():
                continue

            variant_lines += 1
            chrom = line.split("\t", 1)[0]
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1

    sample_file = QC / f"{safe_name(label)}__vcf_samples.txt"
    sample_file.write_text("\n".join(samples), encoding="utf-8")

    chrom_file = QC / f"{safe_name(label)}__vcf_chrom_counts.tsv"
    pd.DataFrame(
        [{"chrom": chrom, "n_variants": n} for chrom, n in chrom_counts.items()]
    ).to_csv(chrom_file, sep="\t", index=False)

    return {
        "dataset": label,
        "path": str(path),
        "format": "vcf",
        "status": "OK",
        "meta_lines": meta_lines,
        "n_variants": variant_lines,
        "n_samples": len(samples),
        "header_n_cols": header_n_cols,
        "sample_file": str(sample_file),
        "chrom_count_file": str(chrom_file),
        "chrom_counts_preview": "; ".join([f"{k}:{v}" for k, v in list(chrom_counts.items())[:20]]),
        "error": "",
    }


def build_file_inventory() -> pd.DataFrame:
    rows = []

    for label, path in EXPECTED_FILES.items():
        if path.exists():
            stat = path.stat()
            rows.append({
                "dataset": label,
                "status": "OK",
                "bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 ** 2), 3),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256sum(path),
                "path": str(path),
            })
        else:
            rows.append({
                "dataset": label,
                "status": "MISSING",
                "bytes": 0,
                "size_mb": 0,
                "modified": "",
                "sha256": "",
                "path": str(path),
            })

    return pd.DataFrame(rows)


def main() -> None:
    started = datetime.now()
    print("Starting raw data inventory...")
    print(f"Project root: {PROJECT_ROOT}")

    file_inventory = build_file_inventory()
    file_inventory_path = QC / "raw_file_inventory.tsv"
    file_inventory.to_csv(file_inventory_path, sep="\t", index=False)

    table_rows = []
    vcf_rows = []

    for label, path in EXPECTED_FILES.items():
        if not path.exists():
            print(f"[MISSING] {label}")
            continue

        print(f"[INSPECT] {label}")

        try:
            suffix = path.suffix.lower()

            if suffix in [".xls", ".xlsx"]:
                table_rows.extend(inspect_excel(path, label))

            elif suffix in [".csv", ".txt", ".tsv"]:
                table_rows.append(inspect_delimited(path, label))

            elif str(path).lower().endswith(".vcf") or str(path).lower().endswith(".vcf.gz"):
                vcf_rows.append(inspect_vcf(path, label))

            elif suffix == ".pdf":
                table_rows.append({
                    "dataset": label,
                    "path": str(path),
                    "format": "pdf",
                    "sheet": "",
                    "status": "OK",
                    "delimiter": "",
                    "encoding": "",
                    "n_rows_est": "",
                    "n_cols": "",
                    "first_column": "",
                    "columns_preview": "",
                    "head_file": "",
                    "error": "PDF retained for documentation; not parsed here.",
                })

            else:
                table_rows.append({
                    "dataset": label,
                    "path": str(path),
                    "format": suffix,
                    "sheet": "",
                    "status": "SKIPPED",
                    "delimiter": "",
                    "encoding": "",
                    "n_rows_est": "",
                    "n_cols": "",
                    "first_column": "",
                    "columns_preview": "",
                    "head_file": "",
                    "error": "Unsupported file type for table inventory.",
                })

        except Exception as exc:
            log_file = LOGS / f"01_inventory_error_{safe_name(label)}.log"
            log_file.write_text(traceback.format_exc(), encoding="utf-8")

            table_rows.append({
                "dataset": label,
                "path": str(path),
                "format": path.suffix.lower(),
                "sheet": "",
                "status": "ERROR",
                "delimiter": "",
                "encoding": "",
                "n_rows_est": "",
                "n_cols": "",
                "first_column": "",
                "columns_preview": "",
                "head_file": "",
                "error": repr(exc),
            })

    table_inventory = pd.DataFrame(table_rows)
    vcf_inventory = pd.DataFrame(vcf_rows)

    table_inventory_path = QC / "raw_table_inventory.tsv"
    vcf_inventory_path = QC / "raw_vcf_inventory.tsv"

    table_inventory.to_csv(table_inventory_path, sep="\t", index=False)
    vcf_inventory.to_csv(vcf_inventory_path, sep="\t", index=False)

    n_expected = len(EXPECTED_FILES)
    n_present = int((file_inventory["status"] == "OK").sum())
    n_missing = int((file_inventory["status"] == "MISSING").sum())
    n_table_errors = int((table_inventory["status"] == "ERROR").sum()) if len(table_inventory) else 0

    report_lines = []
    report_lines.append("# Raw data inventory report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Project root: `{PROJECT_ROOT}`")
    report_lines.append("")
    report_lines.append("## File readiness")
    report_lines.append("")
    report_lines.append(f"- Expected files: {n_expected}")
    report_lines.append(f"- Present files: {n_present}")
    report_lines.append(f"- Missing files: {n_missing}")
    report_lines.append("")
    report_lines.append("## Output files")
    report_lines.append("")
    report_lines.append(f"- Raw file inventory: `{file_inventory_path}`")
    report_lines.append(f"- Raw table inventory: `{table_inventory_path}`")
    report_lines.append(f"- Raw VCF inventory: `{vcf_inventory_path}`")
    report_lines.append(f"- Table head previews: `{QC / 'table_heads'}`")
    report_lines.append("")
    report_lines.append("## VCF summary")
    report_lines.append("")

    if len(vcf_inventory):
        for _, row in vcf_inventory.iterrows():
            report_lines.append(
                f"- {row['dataset']}: {row['n_samples']} samples, {row['n_variants']} variants."
            )
    else:
        report_lines.append("- No VCF files inspected.")

    report_lines.append("")
    report_lines.append("## Table parsing errors")
    report_lines.append("")

    if n_table_errors == 0:
        report_lines.append("- None.")
    else:
        err = table_inventory[table_inventory["status"] == "ERROR"]
        for _, row in err.iterrows():
            report_lines.append(f"- {row['dataset']} / sheet `{row.get('sheet', '')}`: {row['error']}")

    report_lines.append("")
    report_lines.append("## Next step")
    report_lines.append("")
    report_lines.append(
        "Build cleaned genotype and phenotype matrices with explicit sample-ID harmonization, "
        "missingness reports, duplicate checks, and leakage-aware validation splits."
    )

    report_path = QC / "raw_inventory_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    elapsed = datetime.now() - started

    print("")
    print("===== RAW DATA INVENTORY COMPLETE =====")
    print(f"Expected files: {n_expected}")
    print(f"Present files:  {n_present}")
    print(f"Missing files:  {n_missing}")
    print(f"Table errors:   {n_table_errors}")
    print(f"Elapsed:        {elapsed}")
    print("")
    print("Outputs:")
    print(f"  {file_inventory_path}")
    print(f"  {table_inventory_path}")
    print(f"  {vcf_inventory_path}")
    print(f"  {report_path}")
    print("")


if __name__ == "__main__":
    main()