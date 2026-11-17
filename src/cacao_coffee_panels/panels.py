"""Training-only extended density candidates and source-label-stratified sampling.

Source chromosome/scaffold codes are strata, not inferred physical positions.
The original ridge, preprocessing and inner-split implementations are reused.
"""
from collections import defaultdict
import numpy as np
from .ridge import Preprocessor, seed_for, inner_splits, rank_markers, f_scores, linear_kernels, ridge_grid, choose_lambda

def extra_specs(sizes, core_sizes):
    specs = []
    for strategy in ('random', 'high_maf', 'f_regression', 'stratified_random'):
        for size in sizes:
            if strategy != 'stratified_random' and size in core_sizes:
                continue
            specs.append(dict(model=f'{strategy}_{size}' + ('_draw1' if strategy in ('random', 'stratified_random') else ''), strategy=strategy, size=int(size), draw=0))
    return specs

def specs_for_all(sizes):
    return {s['model']: s for s in extra_specs(sizes, [])}

def stratified_order(marker_ids, labels, seed):
    """Nested prefixes: one random marker per source group, then proportional remainder.

    For k >= the number of nonempty groups, every group has at least one marker.
    If k < group count, k groups are sampled by the seeded random group order.
    Remaining markers are interleaved by quantile ranks proportional to the
    remaining marker counts, not physical chromosome length. No phenotype used.
    """
    ids = np.asarray(marker_ids, str)
    labels = np.asarray(labels, str)
    if ids.ndim != 1 or labels.shape != ids.shape or len(ids) == 0 or (len(set(ids)) != len(ids)):
        raise ValueError('Invalid marker/group identifiers')
    if any((not v.strip() for v in labels)):
        raise ValueError('Missing source label must be explicitly coded')
    rng = np.random.default_rng(seed)
    groups = sorted(set(labels))
    order = rng.permutation(len(groups))
    heads = []
    tail = []
    priority = []
    tie = []
    for gi in order:
        idx = np.flatnonzero(labels == groups[gi])
        idx = idx[np.argsort(ids[idx], kind='stable')]
        idx = rng.permutation(idx)
        heads.append(int(idx[0]))
        rest = idx[1:]
        m = len(rest)
        if m:
            tail.extend(rest.tolist())
            priority.extend(((np.arange(m) + 0.5) / m).tolist())
            tie.extend(rng.random(m).tolist())
    if tail:
        perm = np.lexsort((np.asarray(ids)[tail], np.asarray(tie), np.asarray(priority)))
        tail = np.asarray(tail, int)[perm].tolist()
    result = np.asarray(heads + tail, int)
    if len(result) != len(ids) or len(np.unique(result)) != len(ids):
        raise ValueError('Invalid stratified order')
    return result

def fit_extra(Xtrain, Ytrain, Xtest, ids_train, marker_ids, source_groups, groups_train, scheme, cfg, seed):
    """No outer-test phenotypes are accepted. Same folds/tuning as archived core fits."""
    Ytrain = np.asarray(Ytrain, float)
    if Ytrain.ndim != 2 or not np.isfinite(Ytrain).all():
        raise ValueError('Invalid training targets')
    source_groups = np.asarray(source_groups, str)
    if len(source_groups) != Xtrain.shape[1]:
        raise ValueError('Marker-group dimension mismatch')
    splits, reason = inner_splits(ids_train, groups_train, scheme, cfg, seed_for(seed, 'inner'))
    if not splits:
        return dict(state='NOT_ESTIMABLE', reason=reason)
    cache = []
    audit = []
    assign = []
    for f, (tr, va) in enumerate(splits):
        prep = Preprocessor.fit(Xtrain[tr], cfg)
        A = prep.transform(Xtrain[tr])
        B = prep.transform(Xtrain[va])
        cache.append((prep, A, B, Ytrain[tr], Ytrain[va]))
        audit.append(prep.audit(np.asarray(ids_train)[tr], f'inner_{f + 1}'))
        assign.extend((dict(sample_id=str(ids_train[i]), inner_fold=f + 1) for i in va))
    outer = Preprocessor.fit(Xtrain, cfg)
    Z = outer.transform(Xtrain)
    V = outer.transform(Xtest)
    audit.append(outer.audit(ids_train, 'outer'))
    minp = min((len(x[0].indices) for x in cache + [(outer,)]))
    specs = extra_specs(cfg['extended_sizes'], cfg['core_sizes'])
    lambdas = np.asarray(cfg['lambda_grid'])
    nt = Ytrain.shape[1]
    rank_cache = {}

    def ranks(strategy, prep, A, Y, f, q=0):
        key = (strategy, f, q if strategy == 'f_regression' else -1)
        if key not in rank_cache:
            if strategy == 'stratified_random':
                rank_cache[key] = stratified_order(np.asarray(marker_ids)[prep.indices], source_groups[prep.indices], seed_for(seed, 'stratified_panel', 0, f))
            else:
                rank_cache[key] = rank_markers(strategy, prep, A, Y[:, q], marker_ids, seed_for(seed, 'panel', 0, f))
        return rank_cache[key]
    results = {}
    tuning = []
    unavailable = []
    for spec in specs:
        strategy = spec['strategy']
        k = spec['size']
        key = spec['model']
        if k > minp:
            unavailable.append(dict(model=key, reason='requested_size_exceeds_eligible_markers_in_at_least_one_training_split', minimum_eligible_markers=minp, requested_size=k))
            continue
        sse = np.zeros((len(lambdas), nt))
        nval = 0
        for f, (prep, A, B, Ytr, Yva) in enumerate(cache):
            nval += len(Yva)
            if strategy == 'f_regression':
                for q in range(nt):
                    idx = ranks(strategy, prep, A, Ytr, f, q)[:k]
                    K, C = linear_kernels(A[:, idx], B[:, idx])
                    pred = ridge_grid(K, C, Ytr[:, q], lambdas)[:, :, 0]
                    sse[:, q] += ((pred - Yva[:, q, None]) ** 2).sum(0)
            else:
                idx = ranks(strategy, prep, A, Ytr, f)[:k]
                K, C = linear_kernels(A[:, idx], B[:, idx])
                pred = ridge_grid(K, C, Ytr, lambdas)
                sse += ((pred - Yva[:, None, :]) ** 2).sum(0)
        mse = sse / nval
        best = np.array([choose_lambda(mse[:, q]) for q in range(nt)])
        selected = []
        predout = np.empty((len(Xtest), nt))
        if strategy == 'f_regression':
            for q in range(nt):
                idx = ranks(strategy, outer, Z, Ytrain, 'outer', q)[:k]
                K, C = linear_kernels(Z[:, idx], V[:, idx])
                predout[:, q] = ridge_grid(K, C, Ytrain[:, q], lambdas)[:, best[q], 0]
                selected.append(outer.indices[idx])
        else:
            idx = ranks(strategy, outer, Z, Ytrain, 'outer')[:k]
            K, C = linear_kernels(Z[:, idx], V[:, idx])
            grid = ridge_grid(K, C, Ytrain, lambdas)
            predout = grid[:, best, np.arange(nt)]
            selected = [outer.indices[idx].copy() for q in range(nt)]
        for q in range(nt):
            for j, lam in enumerate(lambdas):
                tuning.append(dict(model=key, trait_index=q, lambda_value=float(lam), inner_mse=float(mse[j, q]), chosen=bool(j == best[q]), inner_validation_n=nval))
        results[key] = dict(predictions=predout, lambdas=lambdas[best], inner_mse=mse[best, np.arange(nt)], selected=selected, spec=spec)
    return dict(state='OK', results=results, tuning=tuning, preprocessing=audit, inner_assignments=assign, candidate_failures=unavailable, outer_eligible_indices=outer.indices)

def choose_procedure(records, specs):
    """Choose from available fixed candidate procedures by inner MSE alone."""
    vals = {r['model']: float(r['inner_mse']) for r in records if r['model'] in specs}
    if not vals:
        raise ValueError('No available panel procedure')
    if not all((np.isfinite(v) for v in vals.values())):
        raise ValueError('Nonfinite inner loss')
    low = min(vals.values())
    tied = [k for k, v in vals.items() if v <= low + max(1e-12, abs(low) * 1e-10)]
    return min(tied, key=lambda k: (specs[k]['size'], specs[k]['strategy'], k))
