"""Frozen-split sensitivity to missingness filters and candidate redundancy.

No phenotype is accepted by cohort-definition or representative-selection helpers.
Predictive transformations are refit separately in every inherited inner split.
"""
from __future__ import annotations
import collections
import numpy as np
from .ridge import Preprocessor, linear_kernels, pc_kernels, ridge_grid, choose_lambda

def representative_indices(X, ids, components):
    """One member per connected component: missing fraction, then literal ID."""
    X = np.asarray(X, float)
    ids = np.asarray(ids, str)
    components = np.asarray(components, str)
    if X.shape[0] != len(ids) or len(ids) != len(components) or len(set(ids)) != len(ids):
        raise ValueError('Invalid cohort identifiers')
    miss = np.mean(~np.isfinite(X), axis=1)
    keep = []
    records = []
    for group in sorted(set(components)):
        members = np.flatnonzero(components == group)
        chosen = min(members, key=lambda i: (float(miss[i]), str(ids[i])))
        keep.append(chosen)
        records.extend((dict(sample_id=str(ids[i]), identity_component=group, n_component_members=len(members), raw_missing_fraction=float(miss[i]), representative_id=str(ids[chosen]), retained=bool(i == chosen)) for i in members))
    return (np.array(sorted(keep), int), records)

def build_variants(s, cfg):
    n = len(s['ids'])
    allidx = np.arange(n)
    base = float(cfg['baseline']['marker_missing_max'])
    out = []

    def add(name, indices, q, reason):
        if len(set(indices)) != len(indices):
            raise ValueError('Repeated cohort index')
        out.append(dict(name=name, parent_scenario=s['name'], dataset=s['dataset'], indices=np.asarray(indices, int), marker_missing_max=float(q), reason=reason))
    if s['dataset'] == 'icgt_cacao':
        for q in cfg['icgt_marker_missingness']:
            add(f'icgt_marker_q{round(100 * q):02d}_same_samples', allidx, q, 'Marker filter only; frozen 20%-sample cohort')
        q = cfg['icgt_stricter_sample_missingness']
        idx = np.flatnonzero(np.mean(~np.isfinite(s['X']), axis=1) <= q + 1e-12)
        add('icgt_sample_q10_marker_q20', idx, base, 'Stricter sample filter only; original folds restricted to retained IDs')
        add('icgt_sample_q10_marker_q10', idx, q, 'Stricter sample and marker filters; original folds restricted to retained IDs')
    idx, records = representative_indices(s['X'], s['ids'], s['components'])
    if len(idx) < n:
        add(s['dataset'] + '_component_representatives_q20', idx, base, 'One phenotype-blind representative per frozen high-similarity connected component; not confirmed clone removal')
    return (out, records)

def restrict_inner(ids_train, assignment, cfg):
    ids = np.asarray(ids_train, str)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training IDs')
    loc = {v: i for i, v in enumerate(ids)}
    rows = [r for r in assignment if r['sample_id'] in loc]
    if len(rows) != len(ids) or len({r['sample_id'] for r in rows}) != len(ids):
        raise ValueError('Inherited inner assignments do not cover each training ID exactly once')
    labels = np.empty(len(ids), int)
    for r in rows:
        labels[loc[r['sample_id']]] = int(r['inner_fold'])
    expected = sorted({int(r['inner_fold']) for r in assignment})
    if len(expected) < 2:
        return ([], 'fewer_than_two_inherited_inner_folds')
    splits = []
    idx = np.arange(len(ids))
    for f in expected:
        va = idx[labels == f]
        tr = idx[labels != f]
        if len(tr) < cfg['inner_min_train'] or len(va) < cfg['inner_min_test']:
            return ([], 'restricted_inner_split_below_sample_gate')
        splits.append((tr, va))
    return (splits, 'ok')

def fit_frozen_inner(Xtrain, Ytrain, Xtest, ids_train, assignment, cfg):
    """Training targets only. No outer-test phenotype argument exists."""
    Ytrain = np.asarray(Ytrain, float)
    if Ytrain.ndim != 2 or not np.isfinite(Ytrain).all():
        raise ValueError('Invalid training targets')
    splits, reason = restrict_inner(ids_train, assignment, cfg)
    if not splits:
        return dict(state='NOT_ESTIMABLE', reason=reason)
    lambdas = np.asarray(cfg['lambda_grid'], float)
    nt = Ytrain.shape[1]
    loss = {m: np.zeros((len(lambdas), nt)) for m in ('full_ridge', 'pc_ridge')}
    audits = []
    inner_rows = []
    nval = 0
    try:
        for j, (tr, va) in enumerate(splits, 1):
            pre = Preprocessor.fit(Xtrain[tr], cfg)
            A = pre.transform(Xtrain[tr])
            B = pre.transform(Xtrain[va])
            K, C = linear_kernels(A, B)
            PK, PC, npc = pc_kernels(K, C, cfg['pca_components'])
            nval += len(va)
            for m, kk, cc in [('full_ridge', K, C), ('pc_ridge', PK, PC)]:
                P = ridge_grid(kk, cc, Ytrain[tr], lambdas)
                loss[m] += np.sum((P - Ytrain[va, None, :]) ** 2, axis=0)
            audits.append(pre.audit(np.asarray(ids_train)[tr], f'inner_{j}'))
            inner_rows.extend((dict(sample_id=str(ids_train[i]), inner_fold=j) for i in va))
        pre = Preprocessor.fit(Xtrain, cfg)
        A = pre.transform(Xtrain)
        B = pre.transform(Xtest)
    except ValueError as exc:
        if str(exc) in ('No eligible training markers', 'No variable training markers', 'No PCA dimensions'):
            return dict(state='NOT_ESTIMABLE', reason=str(exc))
        raise
    audits.append(pre.audit(ids_train, 'outer'))
    K, C = linear_kernels(A, B)
    PK, PC, npc = pc_kernels(K, C, cfg['pca_components'])
    results = {}
    tuning = []
    for m, kk, cc in [('full_ridge', K, C), ('pc_ridge', PK, PC)]:
        mse = loss[m] / nval
        best = np.array([choose_lambda(mse[:, q]) for q in range(nt)])
        P = ridge_grid(kk, cc, Ytrain, lambdas)
        results[m] = dict(predictions=P[:, best, np.arange(nt)], lambdas=lambdas[best], inner_mse=mse[best, np.arange(nt)])
        for q in range(nt):
            tuning.extend((dict(model=m, trait_index=q, lambda_value=float(x), inner_mse=float(mse[j, q]), chosen=bool(j == best[q]), inner_validation_n=nval) for j, x in enumerate(lambdas)))
    results['mean'] = dict(predictions=np.tile(Ytrain.mean(0), (len(Xtest), 1)), lambdas=[None] * nt, inner_mse=[None] * nt)
    return dict(state='OK', reason='', results=results, preprocessing=audits, tuning=tuning, inner_assignments=inner_rows, eligible_indices=pre.indices, n_pc=npc)
