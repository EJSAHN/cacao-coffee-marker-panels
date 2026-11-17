"""Inner-CV design diagnostics with unchanged outer full-marker ridge fits.

Selection functions receive training phenotypes only. No outer test target is
passed to loss calculation or penalty selection. The full outer prediction
path is computed once and shared across all inner designs.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from sklearn.model_selection import KFold
from .ridge import Preprocessor, linear_kernels, ridge_grid, choose_lambda, balanced_group_folds, seed_for, id_hash

def validate_splits(pairs, n, groups=None, require_group_separation=False):
    """Require each training-pool sample to be validated exactly once."""
    universe = np.arange(n)
    seen = []
    for train, valid in pairs:
        train = np.asarray(train, int)
        valid = np.asarray(valid, int)
        if not len(train) or not len(valid):
            raise ValueError('Empty inner split')
        if len(set(train)) != len(train) or len(set(valid)) != len(valid):
            raise ValueError('Duplicate index in split')
        if set(train) & set(valid) or sorted(np.r_[train, valid].tolist()) != universe.tolist():
            raise ValueError('Inner split does not partition outer training set')
        if require_group_separation:
            if groups is None or set(groups[train]) & set(groups[valid]):
                raise ValueError('Group leakage in grouped inner split')
        seen.extend(valid.tolist())
    if sorted(seen) != universe.tolist():
        raise ValueError('Inner validation does not cover outer training set exactly once')

def archived_splits(rows, ids, groups, grouped):
    ids = np.asarray(ids, str)
    lookup = {v: i for i, v in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError('Duplicate outer training IDs')
    seen = [r['sample_id'] for r in rows]
    if len(set(seen)) != len(seen) or set(seen) != set(ids):
        raise ValueError('Archived inner assignments disagree with outer training IDs')
    labels = np.empty(len(ids), int)
    for r in rows:
        labels[lookup[r['sample_id']]] = int(r['inner_fold'])
    pairs = [(np.flatnonzero(labels != f), np.flatnonzero(labels == f)) for f in sorted(set(labels))]
    validate_splits(pairs, len(ids), groups, grouped)
    return pairs

def random_matching_sizes(n, sizes, seed):
    if sum(sizes) != n or any((s < 1 for s in sizes)):
        raise ValueError('Bad validation size vector')
    perm = np.random.default_rng(seed).permutation(n)
    pairs = []
    off = 0
    for size in sizes:
        valid = np.sort(perm[off:off + size])
        off += size
        pairs.append((np.setdiff1d(np.arange(n), valid), valid))
    validate_splits(pairs, n)
    return pairs

def design_list(ids, groups, archive_grouped, archive_random, cfg, dataset, fold):
    """Two archived anchors plus five paired repeats of three inner designs."""
    result = [dict(design='archived_grouped', repeat=0, pairs=archived_splits(archive_grouped, ids, groups, True)), dict(design='archived_random', repeat=0, pairs=archived_splits(archive_random, ids, groups, False))]
    for rep in range(1, cfg['inner_repeats'] + 1):
        seed = seed_for(cfg['seed'], 'inner_design_audit', dataset, fold, rep)
        k = min(cfg['baseline']['inner_folds'], len(np.unique(groups)))
        if k < 2:
            raise ValueError('Fewer than two outer-training clusters')
        grouped = balanced_group_folds(groups, k, seed)
        validate_splits(grouped, len(ids), groups, True)
        matched = random_matching_sizes(len(ids), [len(v) for _, v in grouped], seed_for(seed, 'size_matched'))
        balanced = list(KFold(n_splits=cfg['baseline']['inner_folds'], shuffle=True, random_state=seed_for(seed, 'balanced')).split(np.arange(len(ids))))
        for name, pairs in [('grouped_repeated', grouped), ('random_size_matched', matched), ('random_balanced', balanced)]:
            validate_splits(pairs, len(ids))
            result.append(dict(design=name, repeat=rep, pairs=pairs))
    return result

def inner_diagnostic(X, Y, ids, groups, pairs, base):
    """Fit train-only transforms per inner split; return complete loss surfaces."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    ids = np.asarray(ids, str)
    groups = np.asarray(groups, str)
    if Y.ndim != 2 or len(X) != len(Y) or len(ids) != len(X) or (not np.isfinite(Y).all()):
        raise ValueError('Invalid training arrays')
    validate_splits(pairs, len(X))
    if any((len(a) < base['inner_min_train'] or len(b) < base['inner_min_test'] for a, b in pairs)):
        return dict(state='NOT_ESTIMABLE', reason='inner_sample_gate')
    lambdas = np.asarray(base['lambda_grid'], float)
    loss = []
    sizes = []
    details = []
    audits = []
    assignments = []
    for fold, (tr, va) in enumerate(pairs, 1):
        prep = Preprocessor.fit(X[tr], base)
        A = prep.transform(X[tr])
        B = prep.transform(X[va])
        K, C = linear_kernels(A, B)
        predictions = ridge_grid(K, C, Y[tr], lambdas)
        sse = np.sum((predictions - Y[va, None, :]) ** 2, axis=0)
        loss.append(sse)
        sizes.append(len(va))
        audits.append(prep.audit(ids[tr], f'inner_{fold}'))
        shared = np.array([np.any(groups[tr] == g) for g in groups[va]], bool)
        details.append(dict(inner_fold=fold, n_train=len(tr), n_validation=len(va), train_clusters=len(np.unique(groups[tr])), validation_clusters=len(np.unique(groups[va])), overlapping_clusters=len(set(groups[tr]) & set(groups[va])), validation_fraction_with_training_cluster=float(shared.mean()), train_mean=Y[tr].mean(axis=0), validation_mean=Y[va].mean(axis=0), training_sd=Y[tr].std(axis=0), validation_sd=Y[va].std(axis=0), mean_baseline_sse=np.sum((Y[va] - Y[tr].mean(axis=0)) ** 2, axis=0), n_markers=len(prep.indices)))
        assignments.extend((dict(sample_id=str(ids[j]), cluster=str(groups[j]), inner_fold=fold) for j in va))
    loss = np.asarray(loss)
    sizes = np.asarray(sizes)
    return dict(state='OK', sse=loss, validation_sizes=sizes, sample_weighted_mse=loss.sum(axis=0) / sizes.sum(), equal_fold_mse=np.mean(loss / sizes[:, None, None], axis=0), details=details, audits=audits, assignments=assignments)

def select_loss(mse, lambdas, relative_band=0.01):
    mse = np.asarray(mse, float)
    lambdas = np.asarray(lambdas, float)
    if mse.ndim != 1 or mse.shape != lambdas.shape or (not np.isfinite(mse).all()) or np.any(mse < 0):
        raise ValueError('Invalid inner losses')
    i = choose_lambda(mse)
    minimum = float(mse.min())
    tol = max(1e-12, abs(minimum) * 1e-10)
    near = np.flatnonzero(mse <= minimum * (1 + relative_band) + tol)
    return dict(index=i, lambda_value=float(lambdas[i]), minimum_loss=minimum, selected_loss=float(mse[i]), boundary='lower' if i == 0 else 'upper' if i == len(lambdas) - 1 else 'interior', near_minimum_n=len(near), near_minimum_lambda_low=float(lambdas[near[0]]), near_minimum_lambda_high=float(lambdas[near[-1]]), relative_loss_range=float((mse.max() - minimum) / max(minimum, 1e-30)))

def outer_path(Xtrain, Ytrain, Xtest, ids_train, base):
    prep = Preprocessor.fit(Xtrain, base)
    A = prep.transform(Xtrain)
    B = prep.transform(Xtest)
    K, C = linear_kernels(A, B)
    predictions = ridge_grid(K, C, Ytrain, np.asarray(base['lambda_grid']))
    ev = np.maximum(eigh((K + K.T) / 2, eigvals_only=True), 0)
    return dict(predictions=predictions, mean=np.mean(Ytrain, axis=0), eigenvalues=ev, n_markers=len(prep.indices), indices=prep.indices, audit=prep.audit(ids_train, 'outer'))

def df_for_lambda(ev, lam):
    """Effective marker degrees of freedom; excludes fitted intercept."""
    return float(np.sum(ev / (ev + lam)))
