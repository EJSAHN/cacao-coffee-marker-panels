"""Training-only preprocessing, fold construction, and paired nested ridge evaluation.

The normalized linear kernel is Z Z.T / p. Its penalty lambda is equivalent to
primal Ridge(alpha=lambda*p) on the same standardized markers with an intercept.
All selection and transformations are re-fit separately in every inner split.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

def seed_for(seed: int, *parts) -> int:
    text = json.dumps([int(seed), *map(str, parts)], ensure_ascii=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')

def array_hash(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def id_hash(ids) -> str:
    return hashlib.sha256(json.dumps(list(map(str, ids)), ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()

def statistics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.ndim != 1 or y.shape != pred.shape:
        raise ValueError('Metric shapes differ')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError('Nonfinite predictions/targets; never silently drop them')
    n = len(y)
    if n == 0:
        return dict(n=0, pearson_r=None, rmse=None, mae=None, r2=None, correlation_status='no_predictions')
    a = y - y.mean()
    b = pred - pred.mean()
    sa = float(a @ a)
    sb = float(b @ b)
    ta = np.finfo(float).eps * max(1.0, float(y @ y)) * 10
    tb = np.finfo(float).eps * max(1.0, float(pred @ pred)) * 10
    ok = n >= 2 and sa > ta and (sb > tb)
    reason = 'defined' if ok else 'constant_target' if sa <= ta else 'constant_prediction' if sb <= tb else 'too_few_samples'
    err = y - pred
    sse = float(err @ err)
    return dict(n=n, pearson_r=float(np.clip(a @ b / math.sqrt(sa * sb), -1, 1)) if ok else None, rmse=float(np.sqrt(sse / n)), mae=float(np.mean(abs(err))), r2=float(1 - sse / sa) if n >= 2 and sa > ta else None, correlation_status=reason)

@dataclass
class Preprocessor:
    indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    maf: np.ndarray
    missing: np.ndarray

    @classmethod
    def fit(cls, X, cfg):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError('Invalid training matrix')
        if np.isinf(X).any():
            raise ValueError('Infinite genotype')
        obs = np.isfinite(X)
        n = obs.sum(axis=0)
        s = np.where(obs, X, 0).sum(axis=0)
        mu = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
        freq = mu / 2.0
        maf = np.minimum(freq, 1 - freq)
        missing = 1 - n / len(X)
        eligible = (n > 0) & (missing <= cfg['marker_missing_max'] + 1e-12) & (maf >= cfg['maf_min'] - 1e-12) & (maf <= 0.5 + 1e-12)
        ind = np.flatnonzero(eligible)
        if len(ind) == 0:
            raise ValueError('No eligible training markers')
        Z = np.where(obs[:, ind], X[:, ind], mu[ind]) - mu[ind]
        scale = np.sqrt(np.mean(Z * Z, axis=0))
        variable = scale ** 2 > cfg['variance_min']
        ind = ind[variable]
        if len(ind) == 0:
            raise ValueError('No variable training markers')
        mode = cfg.get('scaling', 'empirical_sd')
        if mode == 'empirical_sd':
            scales = scale[variable]
        elif mode == 'center_only':
            scales = np.ones(len(ind), dtype=float)
        elif mode == 'allele_frequency':
            freq_kept = mu[ind] / 2.0
            scales = np.sqrt(2.0 * freq_kept * (1.0 - freq_kept))
        else:
            raise ValueError('Unknown scaling mode: ' + str(mode))
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Invalid training-only scale')
        return cls(ind, mu[ind], scales, maf[ind], missing[ind])

    def transform(self, X):
        a = np.asarray(X, dtype=float)[:, self.indices]
        return (np.where(np.isfinite(a), a, self.means) - self.means) / self.scales

    def audit(self, train_ids, level):
        return dict(level=level, n_train=len(train_ids), n_markers=len(self.indices), train_id_sha256=id_hash(train_ids), eligible_indices_sha256=array_hash(self.indices), means_sha256=array_hash(self.means), scales_sha256=array_hash(self.scales))

def linear_kernels(Z, V):
    p = Z.shape[1]
    if p < 1:
        raise ValueError('Empty features')
    return (Z @ Z.T / p, V @ Z.T / p)

def ridge_grid(K, C, Y, lambdas):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not np.isfinite(Y).all():
        raise ValueError('Nonfinite training target')
    ev, U = eigh((K + K.T) / 2, check_finite=True)
    if ev.min() < -1e-07 * max(1.0, float(ev.max())):
        raise ValueError('Kernel is not PSD')
    ev = np.maximum(ev, 0)
    mu = Y.mean(axis=0)
    coef = U.T @ (Y - mu)
    CU = C @ U
    out = np.empty((C.shape[0], len(lambdas), Y.shape[1]))
    for j, lam in enumerate(lambdas):
        if lam <= 0:
            raise ValueError('Ridge penalty must be positive')
        out[:, j, :] = mu + CU @ (coef / (ev[:, None] + lam))
    return out

def pc_kernels(K, C, ncomp):
    ev, U = eigh((K + K.T) / 2)
    order = np.argsort(ev)[::-1]
    use = order[ev[order] > max(1.0, float(ev.max())) * 1e-10][:ncomp]
    if len(use) == 0:
        raise ValueError('No PCA dimensions')
    U = U[:, use]
    ev = ev[use]
    scale = len(K) / float(ev.sum())
    return (U * ev @ U.T * scale, C @ U @ U.T * scale, len(use))

def f_scores(Z, y):
    yy = np.asarray(y, dtype=float) - float(np.mean(y))
    ss = float(yy @ yy)
    if ss <= 1e-20:
        return np.zeros(Z.shape[1])
    denom = np.sum(Z * Z, axis=0) * ss
    r2 = np.divide((Z.T @ yy) ** 2, denom, out=np.zeros(Z.shape[1]), where=denom > 0)
    r2 = np.clip(r2, 0, 1)
    return (len(y) - 2) * r2 / np.maximum(1 - r2, np.finfo(float).eps)

def rank_markers(strategy, prep, Z, y, marker_ids, seed):
    ids = np.asarray(marker_ids, dtype=str)[prep.indices]
    if strategy == 'random':
        return np.random.default_rng(seed).permutation(len(ids))
    if strategy == 'high_maf':
        return np.lexsort((ids, prep.missing, -prep.maf))
    if strategy == 'f_regression':
        return np.lexsort((ids, -f_scores(Z, y)))
    raise ValueError('Unknown marker strategy: ' + strategy)

def balanced_group_folds(groups, k, seed):
    groups = np.asarray(groups, dtype=str)
    u, c = np.unique(groups, return_counts=True)
    if len(u) < k:
        raise ValueError('Insufficient groups')
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(u))
    order = np.lexsort((jitter, -c))
    sizes = np.zeros(k, dtype=int)
    mapping = {}
    for j in order:
        cand = np.flatnonzero(sizes == sizes.min())
        fold = int(rng.choice(cand))
        mapping[u[j]] = fold
        sizes[fold] += c[j]
    labels = np.array([mapping[g] for g in groups])
    all_idx = np.arange(len(groups))
    return [(all_idx[labels != f], all_idx[labels == f]) for f in range(k)]

def inner_splits(ids, groups, scheme, cfg, seed):
    n = len(ids)
    if scheme == 'random5':
        pairs = list(KFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=seed).split(np.arange(n)))
    else:
        k = min(cfg['inner_folds'], len(np.unique(groups)))
        if k < 2:
            return ([], 'fewer_than_two_training_groups')
        pairs = balanced_group_folds(groups, k, seed)
    if any((len(a) < cfg['inner_min_train'] or len(b) < cfg['inner_min_test'] for a, b in pairs)):
        return ([], 'inner_split_below_predeclared_sample_gate')
    return (pairs, 'ok')

def candidate_configs(cfg):
    out = [dict(model='full_ridge', strategy='full', size=None, draw=0), dict(model='pc_ridge', strategy='pc', size=None, draw=0)]
    for strat in ['random', 'high_maf', 'f_regression']:
        for k in cfg['panel_sizes']:
            for draw in range(cfg['random_panel_draws'] if strat == 'random' else 1):
                out.append(dict(model=f'{strat}_{k}' + (f'_draw{draw + 1}' if strat == 'random' else ''), strategy=strat, size=k, draw=draw))
    return out

def choose_lambda(losses):
    m = float(np.min(losses))
    return int(np.flatnonzero(losses <= m + max(1e-12, abs(m) * 1e-10))[-1])

def nested_fold(Xtrain, Ytrain, Xtest, ids_train, marker_ids, groups_train, scheme, cfg, seed):
    """Accepts training phenotypes only. Returns no metrics using outer test y."""
    Ytrain = np.asarray(Ytrain, dtype=float)
    if Ytrain.ndim != 2 or not np.isfinite(Ytrain).all():
        raise ValueError('Targets must be a complete numeric matrix')
    configs = candidate_configs(cfg)
    lambdas = np.asarray(cfg['lambda_grid'])
    t = Ytrain.shape[1]
    splits, reason = inner_splits(ids_train, groups_train, scheme, cfg, seed_for(seed, 'inner'))
    if not splits:
        return dict(state='NOT_ESTIMABLE', reason=reason)
    cache = []
    audit = []
    inner_records = []
    for f, (tr, va) in enumerate(splits):
        prep = Preprocessor.fit(Xtrain[tr], cfg)
        A = prep.transform(Xtrain[tr])
        B = prep.transform(Xtrain[va])
        cache.append((prep, A, B, Ytrain[tr], Ytrain[va]))
        audit.append(prep.audit(np.asarray(ids_train)[tr], f'inner_{f + 1}'))
        for i in va:
            inner_records.append(dict(sample_id=str(ids_train[i]), inner_fold=f + 1))
    outer = Preprocessor.fit(Xtrain, cfg)
    Z = outer.transform(Xtrain)
    V = outer.transform(Xtest)
    audit.append(outer.audit(ids_train, 'outer'))
    minp = min([len(p.indices) for p, *_ in cache] + [len(outer.indices)])
    results = {}
    tuning = []
    not_estimable = []
    for spec in configs:
        key = spec['model']
        strategy = spec['strategy']
        k = spec['size']
        if k is not None and k > minp:
            not_estimable.append(dict(model=key, reason='requested_size_exceeds_eligible_markers_in_at_least_one_training_split', minimum_eligible_markers=minp, requested_size=k))
            continue
        sse = np.zeros((len(lambdas), t))
        nval = 0
        for f, (prep, A, B, Ytr, Yva) in enumerate(cache):
            nval += len(Yva)
            if strategy in ('full', 'pc'):
                K, C = linear_kernels(A, B)
                if strategy == 'pc':
                    K, C, _ = pc_kernels(K, C, cfg['pca_components'])
                pred = ridge_grid(K, C, Ytr, lambdas)
                sse += np.sum((pred - Yva[:, None, :]) ** 2, axis=0)
            elif strategy == 'f_regression':
                for q in range(t):
                    order = rank_markers(strategy, prep, A, Ytr[:, q], marker_ids, 0)[:k]
                    K, C = linear_kernels(A[:, order], B[:, order])
                    pred = ridge_grid(K, C, Ytr[:, q], lambdas)[:, :, 0]
                    sse[:, q] += np.sum((pred - Yva[:, q, None]) ** 2, axis=0)
            else:
                order = rank_markers(strategy, prep, A, Ytr[:, 0], marker_ids, seed_for(seed, 'panel', spec['draw'], f))[:k]
                K, C = linear_kernels(A[:, order], B[:, order])
                pred = ridge_grid(K, C, Ytr, lambdas)
                sse += np.sum((pred - Yva[:, None, :]) ** 2, axis=0)
        mse = sse / nval
        best = np.array([choose_lambda(mse[:, q]) for q in range(t)])
        selected = []
        predout = np.empty((len(Xtest), t))
        rankscores = []
        if strategy in ('full', 'pc'):
            K, C = linear_kernels(Z, V)
            if strategy == 'pc':
                K, C, dim = pc_kernels(K, C, cfg['pca_components'])
            grid = ridge_grid(K, C, Ytrain, lambdas)
            predout = grid[:, best, np.arange(t)]
            selected = [outer.indices.copy() for _ in range(t)]
            rankscores = [None] * t
        elif strategy == 'f_regression':
            for q in range(t):
                order = rank_markers(strategy, outer, Z, Ytrain[:, q], marker_ids, 0)[:k]
                K, C = linear_kernels(Z[:, order], V[:, order])
                predout[:, q] = ridge_grid(K, C, Ytrain[:, q], lambdas)[:, best[q], 0]
                selected.append(outer.indices[order])
                rankscores.append(f_scores(Z, Ytrain[:, q])[order])
        else:
            order = rank_markers(strategy, outer, Z, Ytrain[:, 0], marker_ids, seed_for(seed, 'panel', spec['draw'], 'outer'))[:k]
            K, C = linear_kernels(Z[:, order], V[:, order])
            grid = ridge_grid(K, C, Ytrain, lambdas)
            predout = grid[:, best, np.arange(t)]
            selected = [outer.indices[order] for _ in range(t)]
            rankscores = [None] * t
        for q in range(t):
            for j, lam in enumerate(lambdas):
                tuning.append(dict(model=key, trait_index=q, lambda_value=float(lam), inner_mse=float(mse[j, q]), chosen=bool(j == best[q]), inner_validation_n=nval))
        results[key] = dict(predictions=predout, lambdas=lambdas[best], inner_mse=mse[best, np.arange(t)], selected=selected, scores=rankscores, spec=spec)
    panelkeys = [k for k, v in results.items() if v['spec']['strategy'] not in ('full', 'pc') and v['spec']['draw'] == 0]
    if panelkeys:
        winners = []
        for q in range(t):
            vals = {k: float(results[k]['inner_mse'][q]) for k in panelkeys}
            m = min(vals.values())
            tied = [k for k in panelkeys if vals[k] <= m + max(1e-12, abs(m) * 1e-10)]
            winners.append(min(tied, key=lambda k: (results[k]['spec']['size'], results[k]['spec']['strategy'], k)))
        results['selected_panel'] = dict(predictions=np.column_stack([results[k]['predictions'][:, q] for q, k in enumerate(winners)]), lambdas=np.array([results[k]['lambdas'][q] for q, k in enumerate(winners)]), inner_mse=np.array([results[k]['inner_mse'][q] for q, k in enumerate(winners)]), selected=[results[k]['selected'][q] for q, k in enumerate(winners)], scores=[results[k]['scores'][q] for q, k in enumerate(winners)], spec=dict(model='selected_panel', strategy='inner_selected', size=None, draw=0), chosen_models=winners)
    results['mean'] = dict(predictions=np.tile(Ytrain.mean(axis=0), (len(Xtest), 1)), lambdas=[None] * t, inner_mse=[None] * t, selected=[np.array([], dtype=int) for _ in range(t)], scores=[None] * t, spec=dict(model='mean', strategy='mean', size=0, draw=0))
    return dict(state='OK', results=results, tuning=tuning, preprocessing=audit, inner_assignments=inner_records, candidate_failures=not_estimable, outer_eligible_indices=outer.indices)

class UnionFind:

    def __init__(self, n):
        self.p = list(range(n))

    def root(self, i):
        while i != self.p[i]:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def join(self, i, j):
        a, b = (self.root(i), self.root(j))
        if a != b:
            self.p[max(a, b)] = min(a, b)

    def labels(self, ids):
        roots = [self.root(i) for i in range(len(self.p))]
        sets = {}
        for i, g in enumerate(roots):
            sets.setdefault(g, []).append(str(ids[i]))
        names = {g: 'G_' + hashlib.sha256('\n'.join(sorted(ss)).encode()).hexdigest()[:12] for g, ss in sets.items()}
        return np.array([names[g] for g in roots])

def pair_audit(X, ids, cfg):
    n, p = X.shape
    uf = UnionFind(n)
    records = []
    threshold = max(cfg['minimum_pair_overlap'], int(math.ceil(p * cfg['minimum_pair_overlap_fraction'])))
    for i in range(n - 1):
        a = X[i]
        B = X[i + 1:]
        mask = np.isfinite(B) & np.isfinite(a)
        overlap = mask.sum(axis=1)
        delta = np.where(mask, np.abs(B - a), 0).sum(axis=1)
        sim = 1 - np.divide(delta, 2 * overlap, out=np.full(len(B), np.nan), where=overlap > 0)
        for pos, j in enumerate(range(i + 1, n)):
            qualified = bool(overlap[pos] >= threshold)
            high = bool(qualified and sim[pos] >= cfg['similarity_threshold'] - 1e-12)
            if high:
                uf.join(i, j)
            records.append(dict(sample_a=str(ids[i]), sample_b=str(ids[j]), n_overlap=int(overlap[pos]), overlap_fraction=float(overlap[pos] / p), similarity=float(sim[pos]) if np.isfinite(sim[pos]) else None, qualifies_overlap=qualified, high_similarity=high))
    return (records, uf.labels(ids))

def structure(X, ids, cfg, seed):
    prep = Preprocessor.fit(X, cfg)
    Z = prep.transform(X)
    K = Z @ Z.T / Z.shape[1]
    ev, U = eigh((K + K.T) / 2)
    order = np.argsort(ev)[::-1]
    ev = np.maximum(ev[order], 0)
    U = U[:, order]
    q = min(cfg['pca_components'], int(np.sum(ev > max(1, ev[0]) * 1e-10)))
    if q < 1:
        raise ValueError('No structure PCs')
    U = U[:, :q]
    for j in range(q):
        if U[np.argmax(np.abs(U[:, j])), j] < 0:
            U[:, j] *= -1
    pcs = U * np.sqrt(ev[:q])
    var = ev[:q] / ev.sum()
    candidate = []
    solutions = {}
    cluster_pcs = pcs[:, :min(cfg.get('clustering_pc_count', 5), q)]
    for k in cfg['cluster_candidates']:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=cfg['kmeans_n_init'], algorithm='lloyd').fit(cluster_pcs)
        raw = km.labels_
        u, c = np.unique(raw, return_counts=True)
        if len(u) < 2 or len(u) != k:
            candidate.append(dict(k=k, silhouette=None, min_cluster_size=int(c.min()), status='fewer_distinct_clusters'))
            continue
        score = float(silhouette_score(cluster_pcs, raw))
        candidate.append(dict(k=k, silhouette=score, min_cluster_size=int(c.min()), status='evaluated'))
        ordered = sorted(u, key=lambda g: min((str(ids[i]) for i in np.flatnonzero(raw == g))))
        mapping = {g: j + 1 for j, g in enumerate(ordered)}
        solutions[k] = np.array([mapping[g] for g in raw])
    good = [x for x in candidate if x['status'] == 'evaluated']
    if not good:
        raise ValueError('No evaluable cluster solution')
    best = sorted(good, key=lambda x: (-x['silhouette'], x['k']))[0]
    return dict(labels=solutions[best['k']], pcs=pcs, variance=var, candidates=candidate, selected_k=best['k'], marker_indices=prep.indices)

def outer_splits(ids, clusters, components, cfg, seed):
    out = []
    idx = np.arange(len(ids))
    for scheme in cfg['schemes']:
        if scheme == 'cluster_out':
            for f, cl in enumerate(sorted(np.unique(clusters)), 1):
                out.append((scheme, 0, f, idx[clusters != cl], idx[clusters == cl]))
        else:
            for rep in range(cfg['random_cv_repeats']):
                sd = seed_for(seed, scheme, rep)
                if scheme == 'random5':
                    pairs = list(KFold(5, shuffle=True, random_state=sd).split(idx))
                elif scheme == 'identity_grouped5':
                    if len(np.unique(components)) < 5:
                        continue
                    pairs = balanced_group_folds(components, 5, sd)
                else:
                    raise ValueError('Unknown scheme')
                for f, (tr, te) in enumerate(pairs, 1):
                    out.append((scheme, rep + 1, f, tr, te))
    return out
