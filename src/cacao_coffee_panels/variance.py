"""One-kernel, Gaussian REML using orthonormal error contrasts.

K has mean diagonal one. The covariance is tau * (h*K + (1-h)*I).
This estimates a kernel-conditioned SNP variance proportion for the supplied
phenotype target, not field-scale heritability or predictive accuracy.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.linalg import eigh, qr
from scipy.optimize import minimize_scalar, brentq

def stable_seed(*items):
    b = json.dumps(items, ensure_ascii=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:4], 'little')

def marker_data(X, cfg):
    """Cohort-level, phenotype-blind QC. No source genotype rounding."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 5 or np.isinf(X).any():
        raise ValueError('Invalid genotype matrix')
    observed = np.isfinite(X)
    values = X[observed]
    if values.size == 0 or np.any(values < -1e-09) or np.any(values > 2 + 1e-09):
        raise ValueError('Expected allele dosages on the 0-2 scale')
    n = observed.sum(0)
    total = np.where(observed, X, 0).sum(0)
    means = np.divide(total, n, out=np.zeros(X.shape[1]), where=n > 0)
    maf = np.minimum(means / 2, 1 - means / 2)
    missing = 1 - n / len(X)
    ind = np.flatnonzero((n > 0) & (missing <= cfg['marker_missing_max'] + 1e-12) & (maf >= cfg['maf_min'] - 1e-12))
    Z = np.where(observed[:, ind], X[:, ind], means[ind]) - means[ind]
    var = np.mean(Z * Z, axis=0)
    keep = var > cfg['variance_min']
    ind = ind[keep]
    Z = Z[:, keep]
    if not len(ind):
        raise ValueError('No eligible variable markers')
    return dict(Z=Z, indices=ind, means=means[ind], maf=maf[ind], missing=missing[ind], variance=var[keep])

def scaled_matrix(md, mode):
    if mode == 'center_only':
        return md['Z'].copy()
    if mode == 'allele_frequency':
        p = md['means'] / 2
        return md['Z'] / np.sqrt(2 * p * (1 - p))
    raise ValueError('Unknown scaling: ' + mode)

def normalized_kernel(Z):
    if Z.ndim != 2 or Z.shape[1] < 1 or (not np.isfinite(Z).all()):
        raise ValueError('Invalid kernel inputs')
    raw = Z @ Z.T / Z.shape[1]
    scale = float(np.trace(raw) / len(raw))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Zero kernel scale')
    K = (raw + raw.T) / (2 * scale)
    return (K, scale)

def full_pcs(K, count):
    values, U = eigh((K + K.T) / 2, check_finite=True)
    order = np.argsort(values)[::-1]
    order = order[values[order] > max(1.0, float(values.max())) * 1e-10][:count]
    U = U[:, order]
    for j in range(U.shape[1]):
        if U[np.argmax(np.abs(U[:, j])), j] < 0:
            U[:, j] *= -1
    return (U, values[order])

def kernel_diagnostics(K, full):
    values = eigh((K + K.T) / 2, eigvals_only=True)
    if values.min() < -max(1.0, values.max()) * 1e-08:
        raise ValueError('Kernel not positive semidefinite')
    ev = np.maximum(values, 0)
    ii = np.triu_indices(len(K), 1)
    a = K[ii]
    b = full[ii]
    ac = a - a.mean()
    bc = b - b.mean()
    den = float(ac @ ac) * float(bc @ bc)
    return dict(mean_diagonal=float(np.diag(K).mean()), min_eigenvalue=float(values.min()), numerical_rank=int(np.sum(ev > max(1.0, ev.max()) * 1e-10)), effective_rank=float(ev.sum() ** 2 / (ev @ ev)), offdiagonal_r_vs_full=float(ac @ bc / np.sqrt(den)) if den > 1e-24 else None, frobenius_relative_difference=float(np.linalg.norm(K - full) / np.linalg.norm(full)))

def neutral_panels(marker_ids, md, cfg, scenario):
    """Selection depends only on genotype metadata and deterministic seeds."""
    ids = np.asarray(marker_ids, dtype=str)[md['indices']]
    m = len(ids)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate eligible marker IDs')
    out = [dict(panel='full', strategy='full', requested_size=m, draw=0, selected=np.arange(m))]
    unavailable = []
    base = np.argsort(ids, kind='stable')
    high = np.lexsort((ids, md['missing'], -md['maf']))
    random = [np.random.default_rng(stable_seed(cfg['seed'], scenario, 'neutral_panel', d)).permutation(base) for d in range(1, cfg['random_panel_draws'] + 1)]
    for k in cfg['panel_sizes']:
        if k > m:
            unavailable.append(dict(requested_size=k, available_markers=m, reason='requested_size_exceeds_eligible_markers'))
            continue
        out.append(dict(panel=f'high_maf_{k}', strategy='high_maf', requested_size=k, draw=0, selected=high[:k]))
        for d, order in enumerate(random, 1):
            out.append(dict(panel=f'random_{k}_draw{d}', strategy='random', requested_size=k, draw=d, selected=order[:k]))
    return (out, unavailable)

@dataclass
class ContrastModel:
    eigenvalues: np.ndarray
    rotation: np.ndarray
    n: int
    fixed_rank: int
    residual_df: int

    @classmethod
    def build(cls, K, C):
        K = np.asarray(K, float)
        C = np.asarray(C, float)
        n = len(K)
        if K.shape != (n, n) or C.ndim != 2 or len(C) != n:
            raise ValueError('Model dimensions differ')
        if not np.isfinite(K).all() or not np.isfinite(C).all():
            raise ValueError('Nonfinite model input')
        if not np.allclose(K, K.T, rtol=1e-09, atol=1e-10):
            raise ValueError('Kernel not symmetric')
        if not np.allclose(C[:, 0], 1.0):
            raise ValueError('Fixed design must begin with an intercept')
        Q, R, piv = qr(C, mode='full', pivoting=True)
        d = np.abs(np.diag(R))
        tol = np.finfo(float).eps * max(C.shape) * max(float(d.max()), 1.0)
        rank = int(np.sum(d > tol))
        if rank != C.shape[1]:
            raise ValueError('Rank-deficient fixed-effect design')
        if n - rank < 3:
            raise ValueError('Insufficient residual degrees of freedom')
        Q2 = Q[:, rank:]
        Kq = Q2.T @ K @ Q2
        ev, U = eigh((Kq + Kq.T) / 2, check_finite=True)
        if ev.min() < -1e-08 * max(1.0, float(ev.max())):
            raise ValueError('Projected kernel not PSD')
        return cls(np.maximum(ev, 0), U.T @ Q2.T, n, rank, n - rank)

    def objective(self, h, z):
        """Profile -2 restricted log likelihood, with additive constants removed."""
        if h < 0 or h >= 1:
            return np.inf
        v = 1 - h + h * self.eigenvalues
        if np.any(v <= 0):
            return np.inf
        s = float(np.sum(z * z / v))
        if not np.isfinite(s) or s <= 0:
            return np.inf
        return float(np.log(v).sum() + self.residual_df * np.log(s / self.residual_df))

    def fit(self, y, cfg):
        y = np.asarray(y, float)
        if y.shape != (self.n,) or not np.isfinite(y).all():
            raise ValueError('Nonfinite or wrong-sized phenotype')
        z = self.rotation @ (y - y.mean())
        scale = float(z @ z / self.residual_df)
        if scale <= np.finfo(float).eps * max(float(np.var(y)), 1e-20) * 10:
            return (dict(status='NOT_ESTIMABLE', reason='constant_target_after_fixed_effects', h2=None), [])
        z = z / np.sqrt(scale)
        grid = np.unique(np.r_[np.linspace(0, cfg['h_upper'], cfg['profile_grid_points']), 1 - np.logspace(-8, -2, 37), np.logspace(-8, -2, 19)])
        grid = grid[(grid >= 0) & (grid <= cfg['h_upper'])]
        vals = np.array([self.objective(h, z) for h in grid])
        m = self.residual_df
        common = dict(n=self.n, fixed_rank=self.fixed_rank, residual_df=m, projected_eigen_min=float(self.eigenvalues.min()), projected_eigen_max=float(self.eigenvalues.max()), h_search_upper=cfg['h_upper'], lr_support_cutoff=cfg['lr_support_cutoff'], interval_type='approximate_profile_LR_support_not_guaranteed_coverage')
        if np.ptp(vals) < 1e-08:
            result = dict(common, status='NOT_IDENTIFIABLE', reason='profile_likelihood_flat', h2=None, support_lower=0.0, support_upper=cfg['h_upper'], support_components=1, support_intervals_json=json.dumps([[0.0, cfg['h_upper']]]), touches_lower=True, touches_upper=True, variance_g=None, variance_e=None, total_variance=None, boundary='not_identifiable', lr_vs_h0=None)
            return (result, [dict(h=float(h), two_loglik_drop=0.0) for h in grid])
        candidates = [(float(vals[0]), float(grid[0])), (float(vals[-1]), float(grid[-1]))]
        for i in range(1, len(grid) - 1):
            if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
                opt = minimize_scalar(lambda h: self.objective(h, z), bounds=(grid[i - 1], grid[i + 1]), method='bounded', options={'xatol': 1e-11})
                if not opt.success:
                    raise ValueError('REML scalar optimization failed')
                candidates.append((float(opt.fun), float(opt.x)))
        fmin, hhat = min(candidates, key=lambda v: (v[0], v[1]))
        grid = np.unique(np.r_[grid, hhat])
        vals = np.array([self.objective(h, z) for h in grid])
        delta = vals - fmin
        cutoff = cfg['lr_support_cutoff']
        roots = []
        for a, b, da, db in zip(grid[:-1], grid[1:], delta[:-1] - cutoff, delta[1:] - cutoff):
            if da == 0:
                roots.append(float(a))
            if da * db < 0:
                roots.append(float(brentq(lambda h: self.objective(h, z) - fmin - cutoff, a, b, xtol=1e-12)))
        points = sorted(set([0.0, cfg['h_upper']] + roots))
        intervals = []
        for a, b in zip(points[:-1], points[1:]):
            if self.objective((a + b) / 2, z) - fmin <= cutoff + 1e-07:
                intervals.append([a, b])
        if not intervals:
            raise ValueError('Empty likelihood support set')
        tau = float(np.sum(z * z / (1 - hhat + hhat * self.eigenvalues)) / m) * scale
        boundary = 'lower' if hhat <= 1e-07 else 'upper_search_limit' if hhat >= cfg['h_upper'] - 1e-07 else 'interior'
        result = dict(common, status='OK', reason='', h2=hhat, total_variance=tau, variance_g=hhat * tau, variance_e=(1 - hhat) * tau, boundary=boundary, support_lower=intervals[0][0], support_upper=intervals[-1][1], support_components=len(intervals), support_intervals_json=json.dumps(intervals), touches_lower=intervals[0][0] == 0.0, touches_upper=intervals[-1][1] == cfg['h_upper'], lr_vs_h0=max(0.0, float(self.objective(0, z) - fmin)))
        profile = [dict(h=float(h), two_loglik_drop=max(0.0, float(d))) for h, d in zip(grid, delta)]
        return (result, profile)
