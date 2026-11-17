import unittest,sys,json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cacao_coffee_panels.variance import *
CFG=json.loads((Path(__file__).resolve().parents[1]/'config/variance.json').read_text())


def dense_objective(K,C,y,h):
    n=len(y);q=C.shape[1];V=(1-h)*np.eye(n)+h*K
    cf=cho_factor(V,lower=True);viC=cho_solve(cf,C);viy=cho_solve(cf,y)
    A=C.T@viC;beta=np.linalg.solve(A,C.T@viy);r=y-C@beta
    rss=r@cho_solve(cf,r)
    return (2*np.log(np.diag(cf[0])).sum()+np.linalg.slogdet(A)[1]
       -np.linalg.slogdet(C.T@C)[1]+(n-q)*np.log(rss/(n-q)))

class TestVariance(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(357);self.X=self.rng.binomial(2,.35,size=(64,120)).astype(float)
        self.X[0,0]=np.nan;self.md=marker_data(self.X,CFG)
        self.K,_=normalized_kernel(self.md['Z']);self.C=np.ones((64,1));self.y=self.rng.normal(size=64)
    def test_mean_diagonal(self):assert_allclose(np.trace(self.K)/64,1,atol=1e-12)
    def test_psd(self):self.assertGreater(np.linalg.eigvalsh(self.K).min(),-1e-9)
    def test_input_not_mutated(self):
        x=self.X.copy();marker_data(self.X,CFG);assert_array_equal(x,self.X)
    def test_fractional_preserved(self):
        X=self.X.copy();X[0,3]=.375;m=marker_data(X,CFG);j=np.flatnonzero(m['indices']==3)[0]
        self.assertAlmostEqual(m['Z'][0,j]+m['means'][j],.375)
    def test_allele_complement_kernel(self):
        m=marker_data(2-self.X,CFG);K,_=normalized_kernel(m['Z']);assert_allclose(K,self.K,atol=1e-12)
    def test_missing_imputed_to_center(self):
        j=np.flatnonzero(self.md['indices']==0)[0];self.assertEqual(self.md['Z'][0,j],0)
    def test_monomorphic_excluded(self):
        X=self.X.copy();X[:,7]=1.;self.assertNotIn(7,marker_data(X,CFG)['indices'])
    def test_all_missing_excluded(self):
        X=self.X.copy();X[:,7]=np.nan;self.assertNotIn(7,marker_data(X,CFG)['indices'])
    def test_range_rejected(self):
        X=self.X.copy();X[0,0]=3
        with self.assertRaises(ValueError):marker_data(X,CFG)
    def test_spectral_equals_dense_intercept(self):
        model=ContrastModel.build(self.K,self.C);z=model.rotation@self.y
        for h in [0.,.05,.45,.95,.999]:assert_allclose(model.objective(h,z),dense_objective(self.K,self.C,self.y,h),atol=1e-9)
    def test_spectral_equals_dense_pc(self):
        pcs,_=full_pcs(self.K,5);C=np.column_stack((self.C,pcs));model=ContrastModel.build(self.K,C);z=model.rotation@self.y
        for h in [.0,.2,.8,.999]:assert_allclose(model.objective(h,z),dense_objective(self.K,C,self.y,h),atol=1e-9)
    def test_kernel_identity_not_identifiable(self):
        fit,_=ContrastModel.build(np.eye(64),self.C).fit(self.y,CFG)
        self.assertEqual(fit['status'],'NOT_IDENTIFIABLE');self.assertIsNone(fit['h2'])
    def test_constant_not_estimable(self):
        fit,_=ContrastModel.build(self.K,self.C).fit(np.ones(64),CFG)
        self.assertEqual(fit['status'],'NOT_ESTIMABLE')
    def test_rank_deficient_design_rejected(self):
        with self.assertRaises(ValueError):ContrastModel.build(self.K,np.ones((64,2)))
    def test_nonnormal_inputs_not_rejected(self):
        fit,_=ContrastModel.build(self.K,self.C).fit(abs(self.y),CFG);self.assertEqual(fit['status'],'OK')
    def test_phenotype_scale_shift_invariance(self):
        m=ContrastModel.build(self.K,self.C);a,_=m.fit(self.y,CFG);b,_=m.fit(self.y*5+10,CFG)
        assert_allclose(a['h2'],b['h2'],atol=1e-6);assert_allclose(a['total_variance']*25,b['total_variance'],rtol=1e-5)
    def test_permutation_invariance(self):
        ii=self.rng.permutation(64);a,_=ContrastModel.build(self.K,self.C).fit(self.y,CFG)
        b,_=ContrastModel.build(self.K[np.ix_(ii,ii)],self.C[ii]).fit(self.y[ii],CFG)
        assert_allclose(a['h2'],b['h2'],atol=1e-6)
    def test_profile_contains_estimate(self):
        fit,profile=ContrastModel.build(self.K,self.C).fit(self.y,CFG)
        self.assertLessEqual(fit['support_lower'],fit['h2']);self.assertGreaterEqual(fit['support_upper'],fit['h2'])
        self.assertLess(min(r['two_loglik_drop'] for r in profile),1e-7)
    def test_components_sum(self):
        f,_=ContrastModel.build(self.K,self.C).fit(self.y,CFG)
        assert_allclose(f['variance_g']+f['variance_e'],f['total_variance'])
    def test_dense_optimum(self):
        # Exhaustive coarse scan plus local optimization is independent of contrast rotation.
        m=ContrastModel.build(self.K,self.C);f,_=m.fit(self.y,CFG)
        hs=np.linspace(0,1-1e-8,151);values=[dense_objective(self.K,self.C,self.y,h) for h in hs]
        i=int(np.argmin(values));cand=[(values[0],hs[0]),(values[-1],hs[-1])]
        if 0<i<len(hs)-1:
            x=minimize_scalar(lambda h:dense_objective(self.K,self.C,self.y,h),bounds=(hs[i-1],hs[i+1]),method='bounded',options={'xatol':1e-10})
            cand.append((x.fun,x.x))
        assert_allclose(f['h2'],min(cand)[1],atol=2e-6)
    def test_panels_available_sizes_only(self):
        ids=np.array([f'm{i:04d}' for i in range(120)]);sp,no=neutral_panels(ids,self.md,CFG,'toy')
        self.assertTrue(all(len(s['selected'])==s['requested_size'] for s in sp));self.assertEqual({r['requested_size'] for r in no},{219,500,1000,2000,5000})
    def test_no_f_selected_variance(self):
        sp,_=neutral_panels(np.arange(120).astype(str),self.md,CFG,'toy')
        self.assertEqual(set(r['strategy'] for r in sp),{'full','random','high_maf'})
    def test_random_reproducible(self):
        ids=np.arange(120).astype(str);a,_=neutral_panels(ids,self.md,CFG,'toy');b,_=neutral_panels(ids,self.md,CFG,'toy')
        for p,q in zip(a,b):assert_array_equal(p['selected'],q['selected'])
    def test_random_nested(self):
        cfg={**CFG,'panel_sizes':[20,45,96]};sp,_=neutral_panels(np.arange(120).astype(str),self.md,cfg,'toy');d={s['panel']:s for s in sp}
        for draw in range(1,4):self.assertTrue(set(d[f'random_20_draw{draw}']['selected'])<set(d[f'random_45_draw{draw}']['selected']))
    def test_high_maf_sorted(self):
        sp,_=neutral_panels(np.arange(120).astype(str),self.md,CFG,'toy');s=[r for r in sp if r['strategy']=='high_maf'][0]
        vals=self.md['maf'][s['selected']];self.assertTrue(np.all(np.diff(vals)<=1e-12))
    def test_pc_orthonormal(self):
        p,_=full_pcs(self.K,5);assert_allclose(p.T@p,np.eye(5),atol=1e-10);assert_allclose(p.mean(0),0,atol=1e-10)
    def test_frequency_mode(self):
        Z=scaled_matrix(self.md,'allele_frequency');p=self.md['means']/2
        assert_allclose(Z,self.md['Z']/np.sqrt(2*p*(1-p)))
    def test_kernel_self_concordance(self):
        d=kernel_diagnostics(self.K,self.K);self.assertAlmostEqual(d['offdiagonal_r_vs_full'],1);self.assertEqual(d['frobenius_relative_difference'],0)
    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):scaled_matrix(self.md,'unknown')
    def test_nonfinite_y_rejected(self):
        y=self.y.copy();y[0]=np.nan
        with self.assertRaises(ValueError):ContrastModel.build(self.K,self.C).fit(y,CFG)

if __name__=='__main__':unittest.main()
