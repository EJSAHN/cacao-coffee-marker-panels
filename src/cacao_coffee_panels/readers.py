"""Source-specific input decoders. No imputation, rounding, or inferred coordinates."""
from __future__ import annotations
import collections, csv, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
OLE = bytes.fromhex("D0CF11E0A1B11AE1")
TRAITS_COFFEE = ["production", "rust", "green"]
BASES = set("ACGT")
MISSING_GT = {"", "NN", "N", "NA", "NAN", ".", "./.", ".|.", "--"}


def text_id(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x).strip()

def accession_key(x):
    return re.sub(r"[^A-Z0-9]", "", text_id(x).upper())

def detect_workbook(path):
    """OLE must precede ZIP detection: BIFF workbooks can embed ZIP themes."""
    with Path(path).open("rb") as f:
        sig = f.read(8)
    if sig == OLE:
        return "xls"
    if sig.startswith(b"PK"):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or "xl/workbook.xml" not in z.namelist():
                raise ValueError("ZIP container is not a valid XLSX workbook")
        return "xlsx"
    raise ValueError("Unknown workbook signature (not OLE XLS or OOXML XLSX)")

def read_workbook(path):
    """Return raw cell values by sheet, with no rewriting of source workbooks."""
    kind = detect_workbook(path)
    if kind == "xls":
        import xlrd
        book = xlrd.open_workbook(file_contents=Path(path).read_bytes(), on_demand=True)
        out = {s.name: [s.row_values(i) for i in range(s.nrows)] for s in book.sheets()}
        book.release_resources()
    else:
        import openpyxl
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        out = {s.title: [list(row) for row in s.iter_rows(values_only=True)] for s in book.worksheets}
        book.close()
    return kind, out

def rows_to_frame(rows, header):
    names = [text_id(x) for x in rows[header]]
    while names and not names[-1]:
        names.pop()
    if not names or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate source column names")
    data = [list(r[:len(names)]) + [None] * max(0, len(names)-len(r)) for r in rows[header+1:]
            if any(text_id(v) for v in r)]
    return pd.DataFrame(data, columns=names)

def numeric_column(series, name):
    vals = pd.to_numeric(series.replace({"": np.nan}), errors="coerce")
    present = series.map(lambda x: text_id(x) not in {"", "NA", "NaN", "nan", "-", "."})
    bad = present & vals.isna()
    if bad.any():
        raise ValueError(f"Non-numeric source values in {name}: {series[bad].head().tolist()}")
    if np.isinf(vals.to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite source values in {name}")
    return vals

def matrix_summary(X):
    n, p = X.shape
    seen = np.isfinite(X)
    return dict(n_samples=n, n_markers=p, n_missing=int((~seen).sum()),
                missing_fraction=float((~seen).mean()) if X.size else None,
                all_missing_markers=int((~seen).all(axis=0).sum()),
                all_missing_samples=int((~seen).all(axis=1).sum()),
                min_sample_missing=float((~seen).mean(axis=1).min()) if n and p else None,
                max_sample_missing=float((~seen).mean(axis=1).max()) if n and p else None)

def qc_profile(X, sample_max, marker_max, maf_min):
    """Descriptive cohort QC, not predictive preprocessing.

    Sample missingness uses all eligible observed source loci. Marker missingness
    and MAF then use samples passing that sample threshold. One pass, no fallback.
    Model-stage marker QC must be fitted again in each training fold.
    """
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Empty marker matrix")
    sample_keep = np.isfinite(X).mean(axis=1) >= 1.0-sample_max-1e-12
    S = X[sample_keep]
    if not len(S):
        return sample_keep, np.zeros(X.shape[1], dtype=bool), np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    count = np.isfinite(S).sum(axis=0)
    mu = np.divide(np.nansum(S, axis=0), count, out=np.full(S.shape[1], np.nan), where=count > 0)
    freq = mu / 2.0
    maf = np.minimum(freq, 1.0-freq)
    missing = 1.0-count / len(S)
    marker_keep = (count > 0) & (missing <= marker_max+1e-12) & (maf >= maf_min-1e-12) & (maf <= .5+1e-12)
    return sample_keep, marker_keep, missing, maf

def decode_hapmap(path):
    """Decode observed calls, never impute a missing genotype.

    Explicit allele pairs preserve their source order. For '*' allele metadata,
    the two observed bases define an analysis coding alphabet in sorted order.
    This is not a claimed genomic REF/ALT or strand assignment. Monomorphic
    observed rows are coded zero and will fail positive-MAF filters.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header[:4] != ["rs_", "alleles", "chrom", "pos"] or len(header) < 12:
            raise ValueError("Unexpected HapMap source schema")
        samples = [s.strip() for s in header[11:]]
        if len(samples) != len(set(samples)) or any(not s for s in samples):
            raise ValueError("Invalid source sample identifiers")
        markers, metadata, values, row_log, token_log = [], [], [], [], []
        for source_line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Ragged HapMap row at source line {source_line}")
            if not any(v.strip() for v in row):
                row_log.append(dict(source_line=source_line, marker_id="", action="excluded_empty_record", reason="all fields empty"))
                continue
            marker = row[0].strip()
            if not marker or marker in markers:
                raise ValueError(f"Missing/duplicate nonempty marker ID at line {source_line}: {marker}")
            calls = [t.strip().upper() for t in row[11:]]
            observed = [t for t in calls if t not in MISSING_GT]
            bad = [t for t in observed if len(t) != 2 or not set(t).issubset(BASES)]
            if bad:
                raise ValueError(f"Undecodable calls for {marker}: {sorted(set(bad))}")
            bases = sorted(set("".join(observed)))
            field = row[1].strip().upper()
            if re.fullmatch(r"[ACGT]/[ACGT]", field) and field[0] != field[2]:
                pair = field.split("/")
                if not set(bases).issubset(set(pair)):
                    raise ValueError(f"Allele descriptor/call conflict at {marker}")
                kind = "source_allele_pair"
                counted = pair[1]
            elif field == "*" and len(bases) == 2:
                pair = bases
                counted = pair[1]
                kind = "observed_biallelic_pair"
            elif field == "*" and len(bases) == 1:
                pair = bases
                counted = ""
                kind = "observed_monomorphic"
            else:
                raise ValueError(f"Unresolved allele alphabet for {marker}: {field} / {bases}")
            val = [np.nan if t in MISSING_GT else float(t.count(counted)) if counted else 0.0 for t in calls]
            values.append(val)
            markers.append(marker)
            metadata.append(dict(marker_id=marker, source_line=source_line, source_alleles=field,
                                 analysis_allele_0=pair[0], counted_allele=counted, encoding=kind,
                                 source_chrom=row[2], source_position=row[3],
                                 coordinate_unit="unverified_not_used_as_bp", genomic_ref_alt="not_assigned",
                                 chromosome_status="source_code_11_unresolved" if row[2]=="11" else "source_label_only"))
            row_log.append(dict(source_line=source_line, marker_id=marker, action="decoded_observed_calls", reason=kind))
            for token, n in collections.Counter(calls).items():
                token_log.append(dict(marker_id=marker, encoding=kind, token=token, n_calls=n))
    return samples, markers, np.asarray(values, dtype=np.float64).T, pd.DataFrame(metadata), pd.DataFrame(row_log), pd.DataFrame(token_log)

def match_ids(genotype_ids, phenotype_ids, strip_barcode=False, allow_multiple_genotypes=False):
    """Match IDs without selecting candidates. Multiple matches require an explicit
    caller policy and remain visible in the returned crosswalk.
    """
    if len(phenotype_ids) != len(set(phenotype_ids)):
        raise ValueError("Duplicate phenotype accession IDs")
    exact = set(phenotype_ids)
    keymap = collections.defaultdict(list)
    for sid in phenotype_ids:
        keymap[accession_key(sid)].append(sid)
    rows = []
    for gid in genotype_ids:
        base = gid
        if strip_barcode and "_" in gid:
            stem, tail = gid.rsplit("_", 1)
            if re.fullmatch(r"[ACGTN]+(?:-[ACGTN]+)?", tail.upper()) and len(tail.replace("-", "")) >= 4:
                base = stem
        if gid in exact:
            pid, method = gid, "exact"
        elif base in exact:
            pid, method = base, "barcode_suffix_removed"
        elif not strip_barcode and re.sub(r"([_.-])([0-9]+)$", "", gid) in exact:
            pid, method = re.sub(r"([_.-])([0-9]+)$", "", gid), "terminal_numeric_suffix_exact_base"
        else:
            hits = keymap.get(accession_key(base), [])
            if len(hits)>1:
                raise ValueError(f"Ambiguous ID match: {gid}: {hits}")
            pid, method = (hits[0], "unique_punctuation_normalization") if hits else ("", "unmatched")
        rows.append(dict(source_genotype_id=gid, accession_id=pid, match_method=method))
    mapped = [r["accession_id"] for r in rows if r["accession_id"]]
    if not allow_multiple_genotypes and len(mapped)!=len(set(mapped)):
        raise ValueError("More than one source genotype maps to one phenotype accession; manual review required")
    return pd.DataFrame(rows)

def uniquify_appended_ids(base_ids):
    """Reproduce concatenated numeric collision suffixes for a second block.

    Used only as a checkable source-row-name reconstruction hypothesis. The full
    output sequence must exactly equal the second environment's source row names.
    No arbitrary terminal-digit stripping or position-only merge is allowed.
    """
    used = set(base_ids)
    out = []
    for sid in base_ids:
        number=1
        while sid+str(number) in used:
            number+=1
        new=sid+str(number)
        used.add(new)
        out.append(new)
    return out

def coffee_crosswalk(ph, genotype_ids):
    p = ph.copy()
    p["source_record_id"] = p.iloc[:, 0].astype(str)
    if set(p["Env"].str.lower()) != {"fem", "fes"}:
        raise ValueError("Unexpected coffee environments; not guessing crosswalk")
    first = p[p["Env"].str.lower()=="fem"].copy()
    second = p[p["Env"].str.lower()=="fes"].copy()
    if first["source_record_id"].tolist()!=genotype_ids:
        raise ValueError("FEM source row order differs from genotype order")
    expected = uniquify_appended_ids(genotype_ids)
    if second["source_record_id"].tolist()!=expected:
        raise ValueError("FES names do not match complete collision-suffix reconstruction")
    first["accession_id"]=genotype_ids
    first["id_evidence"]="exact_genotype_id"
    second["accession_id"]=genotype_ids
    second["id_evidence"]="inferred_rbind_rowname_collision_rule_and_order"
    out=pd.concat([first,second],ignore_index=True)
    out["source_environment"]=out["Env"].str.upper()
    if out.duplicated(["accession_id","source_environment"]).any():
        raise ValueError("Non-unique coffee accession-environment mapping")
    return out

def parse_agrosavia_phenotypes(rows):
    df=rows_to_frame(rows,1)
    label_col=df.columns[0]
    traits=list(df.columns[1:])
    current=None
    totals=[];long=[]
    for i,record in df.iterrows():
        label=text_id(record[label_col])
        if not label:
            continue
        is_harvest=bool(re.fullmatch(r"\d{4}[-_][A-Za-z0-9]+",label))
        if is_harvest:
            if current is None:
                raise ValueError("Harvest record before accession summary")
            harvest=label
        else:
            source_accession_label=label
            current=re.sub(r"\s+", "", label)
            harvest="TOTAL"
        d={"sample_id":current,"source_accession_label":source_accession_label,"harvest":harvest,"source_row":int(i)+3}
        for t in traits:
            v=record[t]
            d[t]=float(v) if text_id(v) not in {"", "NA", "-", "."} else np.nan
        long.append(d)
        if not is_harvest:
            totals.append(d.copy())
    totals=pd.DataFrame(totals)
    if totals["sample_id"].duplicated().any():
        raise ValueError("Repeated Agrosavia accession-summary row")
    return totals,pd.DataFrame(long),traits

def parse_vcf(path):
    samples=None;markers=[];metadata=[];vals=[]
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("##"):
                continue
            fields=line.rstrip("\r\n").split("\t")
            if line.startswith("#CHROM"):
                samples=fields[9:]
                continue
            if not line.strip():
                continue
            if samples is None or len(fields)!=9+len(samples):
                raise ValueError("Malformed VCF record")
            chrom,pos,vid,ref,alt,qual,filt,info,fmt=fields[:9]
            if "," in alt:
                raise ValueError("Multiallelic variant needs a separate policy; not pooling ALT alleles")
            if "GT" not in fmt.split(":"):
                raise ValueError("VCF record without GT")
            gt_index=fmt.split(":").index("GT")
            marker=vid if vid not in {"", "."} else f"{chrom}:{pos}:{ref}>{alt}"
            if marker in markers:
                raise ValueError(f"Duplicate VCF marker: {marker}")
            row=[]
            for value in fields[9:]:
                gt=value.split(":")[gt_index]
                alleles=re.split(r"[/|]",gt)
                if any(a=="." for a in alleles):
                    row.append(np.nan)
                elif len(alleles)==2 and set(alleles).issubset({"0","1"}):
                    row.append(float(sum(a=="1" for a in alleles)))
                else:
                    raise ValueError(f"Unsupported GT at {marker}: {gt}")
            markers.append(marker);vals.append(row)
            metadata.append(dict(marker_id=marker,source_chrom=chrom,source_position=int(pos),ref=ref,alt=alt,
                                 coordinate_unit="bp_as_published_vcf",filter=filt))
    if samples is None or len(set(samples))!=len(samples):
        raise ValueError("Invalid VCF sample header")
    return samples,markers,np.asarray(vals,dtype=np.float64).T,pd.DataFrame(metadata)
