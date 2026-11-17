"""Nested comparisons on an identical training-only feature screen.

The primary full-marker ridge analysis is unchanged. These additional models
share a capped F-ranking screen, re-fitted in every inner training split. No
outer-test targets are accepted by the model fitting functions.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from .ridge import Preprocessor, linear_kernels, ridge_grid, choose_lambda, rank_markers, id_hash, array_hash, seed_for
from .cohorts import restrict_inner

def model_grid(name, cfg):
    if name == 'elastic_net':
        return [dict(l1_ratio=r, alpha_fraction=f) for r in cfg[name]['l1_ratios'] for f in cfg[name]['alpha_fractions']]
    if name == 'random_forest':
        return [dict(min_samples_leaf=n) for n in cfg[name]['min_samples_leaf']]
    if name == 'gradient_boosting':
        return [dict(max_depth=d) for d in cfg[name]['max_depth']]
    raise ValueError('Unknown model: ' + name)

def fit_estimator(name, A, y, B, params, cfg, seed):
    """Fit one model with training-only target scaling; return original-unit predictions."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    y = np.asarray(y, float)
    if not np.isfinite(A).all() or not np.isfinite(B).all() or (not np.isfinite(y).all()):
        raise ValueError('Nonfinite model inputs')
    if A.shape[1] != B.shape[1] or A.shape[0] != len(y) or y.ndim != 1:
        raise ValueError('Model dimensions differ')
    mu = float(np.mean(y))
    scale = float(np.std(y))
    diag = dict(target_mean=mu, target_scale=scale, seed=int(seed), status='OK', reason='', retries=0)
    if scale <= np.finfo(float).eps * max(1.0, float(np.max(abs(y)))) * 10:
        diag.update(status='CONSTANT_TRAINING_TARGET', reason='Return training mean')
        return (np.full(len(B), mu), diag)
    yz = (y - mu) / scale
    if name == 'elastic_net':
        c = cfg[name]
        am = float(np.max(abs(A.T @ yz))) / (len(y) * params['l1_ratio'])
        alpha = max(1e-12, float(params['alpha_fraction'] * am))
        diag.update(alpha=alpha, alpha_max=am, l1_ratio=params['l1_ratio'], alpha_fraction=params['alpha_fraction'])
        for attempt, lim in enumerate([c['max_iter'], c['retry_max_iter']]):
            est = ElasticNet(alpha=alpha, l1_ratio=params['l1_ratio'], fit_intercept=True, precompute=False, max_iter=lim, tol=c['tol'], selection=c['selection'], random_state=seed)
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter('always', ConvergenceWarning)
                est.fit(A, yz)
            warned = [str(w.message) for w in records if issubclass(w.category, ConvergenceWarning)]
            other = [str(w.message) for w in records if not issubclass(w.category, ConvergenceWarning)]
            diag.update(n_iter=int(est.n_iter_), dual_gap=float(est.dual_gap_), convergence_warnings=' | '.join(warned), other_warnings=' | '.join(other), retries=attempt)
            if not warned:
                break
        if warned:
            diag.update(status='NOT_ESTIMABLE', reason='Elastic net failed convergence after deterministic retry')
            return (None, diag)
    elif name == 'random_forest':
        c = cfg[name]
        est = RandomForestRegressor(n_estimators=c['n_estimators'], max_features=c['max_features'], min_samples_leaf=params['min_samples_leaf'], max_depth=c['max_depth'], bootstrap=c['bootstrap'], n_jobs=1, random_state=seed, criterion='squared_error')
        est.fit(A, yz)
    elif name == 'gradient_boosting':
        c = cfg[name]
        est = GradientBoostingRegressor(n_estimators=c['n_estimators'], learning_rate=c['learning_rate'], max_depth=params['max_depth'], min_samples_leaf=c['min_samples_leaf'], max_features=c['max_features'], subsample=c['subsample'], loss=c['loss'], n_iter_no_change=None, random_state=seed)
        est.fit(A, yz)
    else:
        raise ValueError('Unknown estimator')
    yp = mu + scale * est.predict(B)
    if not np.isfinite(yp).all():
        raise ValueError('Nonfinite model prediction')
    return (yp, diag)

def make_split_cache(X, Y, V, ids, assignment, cfg):
    """One QC transform per split, shared by all traits and model families."""
    base = {**cfg['baseline'], 'scaling': 'center_only'}
    splits, reason = restrict_inner(ids, assignment, base)
    if not splits:
        return (None, reason)
    cache = []
    aud = []
    for j, (tr, va) in enumerate(splits, 1):
        prep = Preprocessor.fit(X[tr], base)
        cache.append(dict(level=f'inner_{j}', prep=prep, A=prep.transform(X[tr]), B=prep.transform(X[va]), Y=Y[tr], Yval=Y[va], n_val=len(va)))
        aud.append(prep.audit(np.asarray(ids)[tr], f'inner_{j}'))
    prep = Preprocessor.fit(X, base)
    outer = dict(level='outer', prep=prep, A=prep.transform(X), B=prep.transform(V), Y=Y)
    aud.append(prep.audit(ids, 'outer'))
    return (dict(inner=cache, outer=outer, audits=aud), 'ok')

def fit_trait(cache, q, marker_ids, cfg, task_seed):
    """Evaluate one trait; all four families use the same screened markers per split."""
    limit = int(cfg['shared_screen']['max_markers'])
    lambdas = np.asarray(cfg['baseline']['lambda_grid'], float)
    family = ['elastic_net', 'random_forest', 'gradient_boosting']
    grids = {m: model_grid(m, cfg) for m in family}
    losses = {'screened_ridge': np.zeros(len(lambdas))} | {m: np.zeros(len(grids[m])) for m in family}
    valid = {m: np.ones(len(grids[m]), bool) for m in family}
    details = []
    marker_sets = {}
    screen_audits = []
    nval = 0

    def screen(block):
        pre = block['prep']
        order = rank_markers('f_regression', pre, block['A'], block['Y'][:, q], marker_ids, task_seed)
        order = order[:min(limit, len(order))]
        glob = pre.indices[order]
        marker_sets[block['level']] = glob
        screen_audits.append(dict(level=block['level'], n_eligible=len(pre.indices), n_used=len(glob), marker_indices_sha256=array_hash(glob), method='training_f_regression'))
        return (block['A'][:, order], block['B'][:, order])
    for j, block in enumerate(cache['inner'], 1):
        A, B = screen(block)
        y = block['Y'][:, q]
        truth = block['Yval'][:, q]
        nval += len(truth)
        K, C = linear_kernels(A, B)
        P = ridge_grid(K, C, y, lambdas)[:, :, 0]
        losses['screened_ridge'] += np.sum((P - truth[:, None]) ** 2, axis=0)
        for m in family:
            for z, params in enumerate(grids[m]):
                yp, diag = fit_estimator(m, A, y, B, params, cfg, seed_for(task_seed, q, m, j))
                details.append({'model': m, 'level': block['level'], 'candidate': z, **params, **diag})
                if yp is None:
                    valid[m][z] = False
                else:
                    losses[m][z] += float(np.sum((yp - truth) ** 2))
    A, B = screen(cache['outer'])
    y = cache['outer']['Y'][:, q]
    results = {}
    tuning = []
    chosen = []
    states = []
    mse = losses['screened_ridge'] / nval
    best = choose_lambda(mse)
    K, C = linear_kernels(A, B)
    pred = ridge_grid(K, C, y, lambdas)[:, best, 0]
    results['screened_ridge'] = dict(predictions=pred, inner_mse=float(mse[best]), config=dict(lambda_value=float(lambdas[best])), n_markers=A.shape[1])
    for z, lam in enumerate(lambdas):
        tuning.append(dict(model='screened_ridge', candidate=z, lambda_value=float(lam), inner_mse=float(mse[z]), valid=True, chosen=z == best, inner_validation_n=nval))
    states.append(dict(model='screened_ridge', status='OK', reason=''))
    for m in family:
        mse = losses[m] / nval
        candidate_mse = np.where(valid[m], mse, np.inf)
        if not np.isfinite(candidate_mse).any():
            best = None
            states.append(dict(model=m, status='NOT_ESTIMABLE', reason='No candidate converged in every inner split'))
        else:
            best = int(np.argmin(candidate_mse))
            params = grids[m][best]
            yp, diag = fit_estimator(m, A, y, B, params, cfg, seed_for(task_seed, q, m, 'outer'))
            details.append({'model': m, 'level': 'outer', 'candidate': best, **params, **diag})
            if yp is not None:
                results[m] = dict(predictions=yp, inner_mse=float(mse[best]), config=params, n_markers=A.shape[1])
                states.append(dict(model=m, status='OK', reason=''))
            else:
                states.append(dict(model=m, status='NOT_ESTIMABLE', reason=diag['reason']))
        for z, params in enumerate(grids[m]):
            tuning.append(dict(model=m, candidate=z, **params, inner_mse=float(mse[z]) if valid[m][z] else None, valid=bool(valid[m][z]), chosen=z == best, inner_validation_n=nval))
    return dict(results=results, tuning=tuning, fit_diagnostics=details, screen_audits=screen_audits, marker_sets=marker_sets, model_status=states)
