"""Sample-count-matched validation geometry with a common inner tuning rule.

Cluster labels and outer-fold sizes come from a frozen, genotype-only design.
Random partitions change membership, never the phenotype or fold-size vector.
This is not a permutation hypothesis test or causal decomposition.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from .ridge import Preprocessor, linear_kernels, ridge_grid, choose_lambda, seed_for, id_hash, statistics

def matched_partitions(ids, cluster_tests, valid_folds, repeats, seed):
    """Yield identical-evaluation-pool partitions and exact matching fold sizes."""
    n = len(ids)
    all_idx = np.arange(n)
    valid = sorted(valid_folds)
    original = np.concatenate([np.asarray(cluster_tests[f], int) for f in sorted(cluster_tests)])
    if sorted(original.tolist()) != all_idx.tolist():
        raise ValueError('Original clusters must partition all sample IDs exactly once')
    if not valid:
        return ([], np.array([], int), all_idx)
    eligible = np.sort(np.concatenate([np.asarray(cluster_tests[f], int) for f in valid]))
    training_only = np.setdiff1d(all_idx, eligible)
    sizes = [len(cluster_tests[f]) for f in valid]
    tasks = []
    for f in valid:
        te = np.sort(cluster_tests[f])
        tr = np.setdiff1d(all_idx, te)
        tasks.append(dict(design='cluster_common_inner', repeat=0, fold=f, train=tr, test=te))
    for rep in range(1, repeats + 1):
        perm = np.random.default_rng(seed_for(seed, 'outer_membership', rep)).permutation(eligible)
        off = 0
        for f, size in zip(valid, sizes):
            te = np.sort(perm[off:off + size])
            off += size
            tasks.append(dict(design='size_matched_random', repeat=rep, fold=f, train=np.setdiff1d(all_idx, te), test=te))
    for task in tasks:
        if len(task['test']) != len(cluster_tests[task['fold']]):
            raise AssertionError('Changed fold size')
        if set(task['train']) & set(task['test']):
            raise AssertionError('Train/test overlap')
        if not set(training_only).issubset(set(task['train'])):
            raise AssertionError('Training-only samples entered evaluation')
    return (tasks, eligible, training_only)

def fit_full_ridge(Xtrain, Ytrain, Xtest, ids_train, cfg, inner_seed):
    """Never receives outer-test phenotypes; transforms are fit per inner train."""
    Xtrain = np.asarray(Xtrain, float)
    Xtest = np.asarray(Xtest, float)
    Ytrain = np.asarray(Ytrain, float)
    if Ytrain.ndim != 2 or len(Xtrain) != len(Ytrain) or len(ids_train) != len(Xtrain):
        raise ValueError('Training arrays disagree')
    if not np.isfinite(Ytrain).all():
        raise ValueError('Nonfinite phenotype')
    ntraits = Ytrain.shape[1]
    lambdas = np.asarray(cfg['lambda_grid'], float)
    if not np.all(np.isfinite(lambdas)) or np.any(lambdas <= 0):
        raise ValueError('Invalid penalty grid')
    if len(Xtrain) < cfg['outer_min_train'] or len(Xtest) < cfg['outer_min_test']:
        return dict(state='NOT_ESTIMABLE', reason='outer_sample_gate')
    cv = list(KFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=int(inner_seed)).split(np.arange(len(Xtrain))))
    if any((len(tr) < cfg['inner_min_train'] or len(va) < cfg['inner_min_test'] for tr, va in cv)):
        return dict(state='NOT_ESTIMABLE', reason='inner_sample_gate')
    sse = np.zeros((len(lambdas), ntraits))
    nval = 0
    audit = []
    assign = []
    for f, (tr, va) in enumerate(cv, 1):
        prep = Preprocessor.fit(Xtrain[tr], cfg)
        A = prep.transform(Xtrain[tr])
        B = prep.transform(Xtrain[va])
        K, C = linear_kernels(A, B)
        grid = ridge_grid(K, C, Ytrain[tr], lambdas)
        sse += np.sum((grid - Ytrain[va, None, :]) ** 2, axis=0)
        nval += len(va)
        audit.append(prep.audit(np.asarray(ids_train)[tr], f'inner_{f}'))
        assign.extend((dict(sample_id=str(ids_train[i]), inner_fold=f) for i in va))
    mse = sse / nval
    best = np.array([choose_lambda(mse[:, j]) for j in range(ntraits)], int)
    prep = Preprocessor.fit(Xtrain, cfg)
    Z = prep.transform(Xtrain)
    V = prep.transform(Xtest)
    K, C = linear_kernels(Z, V)
    grid = ridge_grid(K, C, Ytrain, lambdas)
    pred = grid[:, best, np.arange(ntraits)]
    audit.append(prep.audit(ids_train, 'outer'))
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite predictions; no clipping permitted')
    return dict(state='OK', predictions=pred, mean_predictions=np.tile(Ytrain.mean(axis=0), (len(Xtest), 1)), lambda_values=lambdas[best], inner_mse=mse, chosen_indices=best, n_markers=len(prep.indices), selected_marker_indices=prep.indices, preprocessing=audit, inner_assignments=assign, n_inner_validation=nval)
