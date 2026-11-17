"""Replay a recorded training-only screen without sorting floating-point ties again.

No test phenotype is accepted. The screen archive is an explicit reproduction
input, not a panel selected from the outer test. Fitting and tuning are otherwise
identical to models.fit_trait. Original fitting implementation remains unchanged.
"""
from pathlib import Path
import numpy as np
from .models import model_grid,fit_estimator
from .ridge import linear_kernels,ridge_grid,choose_lambda,seed_for,array_hash,id_hash
from .io import read_json,sha

def load_recorded_screen(index, scenario, scheme, repeat, fold, trait, train_ids, test_ids, input_hashes):
    path=Path(index).resolve();data=read_json(path)
    rows=[r for r in data['screens'] if (r['scenario'],r['scheme'],r['repeat'],r['fold'],r['trait'])==(scenario,scheme,repeat,fold,trait)]
    if len(rows)!=1:raise ValueError('Missing or duplicate recorded screen entry')
    row=rows[0]
    if row['train_ids_sha256']!=id_hash(train_ids) or row['test_ids_sha256']!=id_hash(test_ids):
        raise ValueError('Recorded screen belongs to different train/test IDs')
    for name in ('genotypes.npz','phenotypes.tsv','markers.tsv'):
        if data['cohort_hashes'][scenario][name]!=input_hashes[name]:raise ValueError('Recorded screen input hash differs')
    src=(path.parent/row['path']).resolve()
    if path.parent not in src.parents:raise ValueError('Screen path escapes input folder')
    if sha(src)!=row['sha256']:raise ValueError('Recorded screen archive hash differs')
    with np.load(src,allow_pickle=False) as z:result={k:z[k].copy() for k in z.files}
    return result

def fit_with_recorded_screen(cache, q, marker_ids, cfg, task_seed, screen_indices):
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
        global_order = np.asarray(screen_indices[block['level']], dtype=int)
        if len(global_order) != min(limit, len(pre.indices)) or len(set(global_order)) != len(global_order):
            raise ValueError('Recorded screen size or uniqueness differs')
        lookup = {int(value): index for index, value in enumerate(pre.indices)}
        if any(int(value) not in lookup for value in global_order):
            raise ValueError('Recorded screen includes a marker ineligible in this training split')
        order = np.array([lookup[int(value)] for value in global_order], dtype=int)
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
