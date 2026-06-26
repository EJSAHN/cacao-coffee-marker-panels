
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", low_memory=False, **kwargs)


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz" or str(path).endswith(".tsv.gz"):
        df.to_csv(path, sep="\t", index=index, compression="gzip")
    else:
        df.to_csv(path, sep="\t", index=index)
    return path


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(str(p) for p in paths))
