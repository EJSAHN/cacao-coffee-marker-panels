"""Strict, read-only loading of matched dosage matrices and explicit fold IDs."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Cohort:
    dosage: np.ndarray
    sample_ids: np.ndarray
    marker_ids: np.ndarray
    phenotypes: pd.DataFrame
    samples: pd.DataFrame
    markers: pd.DataFrame

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with np.load(directory / 'genotypes.npz', allow_pickle=False) as z:
            required = {'dosage', 'sample_ids', 'marker_ids'}
            if not required.issubset(z.files):
                raise ValueError('Genotype archive requires dosage, sample_ids and marker_ids')
            X = np.asarray(z['dosage'], dtype=float)
            ids = np.asarray(z['sample_ids'], dtype=str)
            mids = np.asarray(z['marker_ids'], dtype=str)
        if X.ndim != 2 or ids.ndim != 1 or mids.ndim != 1 or X.shape != (len(ids), len(mids)) or len(ids) < 2 or len(mids) < 1:
            raise ValueError('Genotype dimensions disagree')
        for name, values in [('sample', ids), ('marker', mids)]:
            if len(set(values)) != len(values) or any(not v.strip() for v in values):
                raise ValueError('Empty or duplicate ' + name + ' identifiers')
        if np.isinf(X).any() or np.any(X[np.isfinite(X)] < -1e-9) or np.any(X[np.isfinite(X)] > 2 + 1e-9):
            raise ValueError('Dosages must be finite 0-2 values or NaN; no rounding is performed')
        def table(filename, key, expected):
            t = pd.read_csv(directory / filename, sep='\t', dtype=str, keep_default_na=False)
            if key not in t or t[key].duplicated().any() or t[key].tolist() != expected.tolist():
                raise ValueError(filename + ': identifiers/order do not match genotype archive')
            return t
        y = table('phenotypes.tsv', 'sample_id', ids)
        for c in y.columns[1:]:
            y[c] = pd.to_numeric(y[c], errors='raise')
            if not np.isfinite(y[c].to_numpy(float)).all():
                raise ValueError('Phenotypes must be complete; specify eligibility before analysis')
        return cls(X, ids, mids, y, table('samples.tsv', 'sample_id', ids), table('markers.tsv', 'marker_id', mids))

    def split(self, design):
        lookup = {s: i for i, s in enumerate(self.sample_ids)}
        def indices(key):
            values = list(map(str, design[key]))
            if len(values) != len(set(values)) or any(x not in lookup for x in values):
                raise ValueError('Unknown or duplicate ID in ' + key)
            return np.array([lookup[x] for x in values], dtype=int)
        tr, te = indices('train_ids'), indices('test_ids')
        if not len(tr) or not len(te) or set(tr) & set(te):
            raise ValueError('Empty or overlapping training/test IDs')
        traits = design.get('traits', self.phenotypes.columns[1:].tolist())
        if not traits or len(traits) != len(set(traits)) or any(t not in self.phenotypes.columns[1:] for t in traits):
            raise ValueError('Invalid phenotype selection')
        Y = self.phenotypes[traits].to_numpy(float)
        return tr, te, traits, Y
