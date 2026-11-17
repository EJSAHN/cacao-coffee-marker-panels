import inspect,json,sys,unittest,tempfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cacao_coffee_panels.ridge import Preprocessor,statistics,choose_lambda,balanced_group_folds
from cacao_coffee_panels.tuning import (validate_splits,archived_splits,random_matching_sizes,
    design_list,inner_diagnostic,select_loss,outer_path,df_for_lambda)
from cacao_coffee_panels.io import tsv,read_tsv,manifest,verify,child,sha

CFG=json.loads((ROOT/'config/tuning.json').read_text())
class TestTuning(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(22);self.X=rng.binomial(2,.35,(36,18)).astype(float);self.X[1,0]=np.nan
        self.X[2,1]=.82;self.Y=rng.normal(size=(36,2));self.ids=np.array([f'S{i:02d}' for i in range(36)])
        self.groups=np.array(['A']*12+['B']*9+['C']*6+['D']*5+['E']*4);self.base=CFG['baseline']
        self.pairs=list(KFold(3,shuffle=True,random_state=42).split(self.X))
    def test_01_partition(self):validate_splits(self.pairs,36)
    def test_02_duplicate_rejected(self):
        with self.assertRaises(ValueError):validate_splits([(np.arange(18),np.arange(18))],36)
    def test_03_group_separation(self):
        p=balanced_group_folds(self.groups,3,1);validate_splits(p,36,self.groups,True)
    def test_04_random_not_grouped(self):
        with self.assertRaises(ValueError):validate_splits(self.pairs,36,self.groups,True)
    def test_05_size_match(self):
        p=random_matching_sizes(36,[16,12,8],3);self.assertEqual([len(v) for _,v in p],[16,12,8]);validate_splits(p,36)
    def test_06_bad_sizes(self):
        with self.assertRaises(ValueError):random_matching_sizes(36,[12,12,11],3)
    def test_07_deterministic(self):
        a=random_matching_sizes(36,[16,12,8],3);b=random_matching_sizes(36,[16,12,8],3)
        for x,y in zip(a,b):np.testing.assert_array_equal(x[1],y[1])
    def test_08_different_seeds(self):
        a=random_matching_sizes(36,[16,12,8],3);b=random_matching_sizes(36,[16,12,8],4);self.assertFalse(np.array_equal(a[0][1],b[0][1]))
    def test_09_archived_order(self):
        rows=[dict(sample_id=self.ids[j],inner_fold=f+1) for f,(_,v) in enumerate(self.pairs) for j in v]
        p=archived_splits(rows[::-1],self.ids,self.groups,False)
        for a,b in zip(p,self.pairs):np.testing.assert_array_equal(a[1],b[1])
    def test_10_archived_duplicates(self):
        rows=[dict(sample_id=self.ids[j],inner_fold=f+1) for f,(_,v) in enumerate(self.pairs) for j in v]
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):archived_splits(rows,self.ids,self.groups,False)
    def test_11_design_pairing(self):
        pg=balanced_group_folds(self.groups,3,20)
        def rows(p):return [dict(sample_id=self.ids[j],inner_fold=f+1) for f,(_,v) in enumerate(p) for j in v]
        d=design_list(self.ids,self.groups,rows(pg),rows(self.pairs),CFG,'test',1)
        self.assertEqual(len(d),17)
        for rep in range(1,6):
            g=next(x for x in d if x['design']=='grouped_repeated' and x['repeat']==rep)
            r=next(x for x in d if x['design']=='random_size_matched' and x['repeat']==rep)
            self.assertEqual([len(v) for _,v in g['pairs']],[len(v) for _,v in r['pairs']])
    def test_12_source_unchanged(self):
        x=self.X.copy();y=self.Y.copy();inner_diagnostic(self.X,self.Y,self.ids,self.groups,self.pairs,self.base)
        np.testing.assert_array_equal(self.X,x);np.testing.assert_array_equal(self.Y,y)
    def test_13_inner_matches_independent_ridge(self):
        fit=inner_diagnostic(self.X,self.Y,self.ids,self.groups,self.pairs,self.base)
        total=np.zeros((17,2))
        for tr,va in self.pairs:
            raw=self.X[tr];obs=np.isfinite(raw);n=obs.sum(0);mu=np.divide(np.where(obs,raw,0).sum(0),n,out=np.zeros(raw.shape[1]),where=n>0)
            maf=np.minimum(mu/2,1-mu/2);miss=1-n/len(tr);A=np.where(obs,raw,mu)-mu
            keep=(n>0)&(miss<=.2+1e-12)&(maf>=.01-1e-12)&(maf<=.5+1e-12)&(np.mean(A*A,0)>1e-12)
            A=A[:,keep];B=np.where(np.isfinite(self.X[va][:,keep]),self.X[va][:,keep],mu[keep])-mu[keep]
            for j,lam in enumerate(self.base['lambda_grid']):
                pred=Ridge(alpha=lam*sum(keep),solver='svd').fit(A,self.Y[tr]).predict(B)
                total[j]+=np.sum((self.Y[va]-pred)**2,0)
        np.testing.assert_allclose(fit['sample_weighted_mse'],total/36,rtol=1e-8,atol=1e-10)
    def test_14_loss_weighting(self):
        p=random_matching_sizes(36,[18,10,8],10);f=inner_diagnostic(self.X,self.Y,self.ids,self.groups,p,self.base)
        np.testing.assert_allclose(f['equal_fold_mse'],(f['sse']/f['validation_sizes'][:,None,None]).mean(0))
        self.assertGreater(np.max(abs(f['sample_weighted_mse']-f['equal_fold_mse'])),1e-6)
    def test_15_sample_gate(self):
        p=random_matching_sizes(36,[30,3,3],4);f=inner_diagnostic(self.X,self.Y,self.ids,self.groups,p,self.base);self.assertEqual(f['state'],'NOT_ESTIMABLE')
    def test_16_no_outer_y_argument(self):
        self.assertEqual(list(inspect.signature(inner_diagnostic).parameters),['X','Y','ids','groups','pairs','base'])
        self.assertNotIn('Ytest',inspect.signature(outer_path).parameters)
    def test_17_tie_stronger_penalty(self):self.assertEqual(choose_lambda(np.ones(17)),16)
    def test_18_loss_band(self):
        s=select_loss(np.array([2.,1.,1.005,3.]),np.array([.1,1,10,100]),.01)
        self.assertEqual(s['lambda_value'],1);self.assertEqual(s['near_minimum_n'],2)
    def test_19_nonfinite_loss(self):
        with self.assertRaises(ValueError):select_loss(np.array([1.,np.nan]),np.array([1.,2.]))
    def test_20_constant_correlation(self):self.assertIsNone(statistics(np.arange(4.),np.ones(4))['pearson_r'])
    def test_21_test_transform_does_not_refit(self):
        p=Preprocessor.fit(self.X[:24],self.base);mu=p.means.copy();p.transform(self.X[24:]*100);np.testing.assert_array_equal(p.means,mu)
    def test_22_outer_independent_ridge(self):
        o=outer_path(self.X[:24],self.Y[:24],self.X[24:],self.ids[:24],self.base);p=Preprocessor.fit(self.X[:24],self.base)
        for k in [0,8,16]:
            pred=Ridge(alpha=self.base['lambda_grid'][k]*o['n_markers'],solver='svd').fit(p.transform(self.X[:24]),self.Y[:24]).predict(p.transform(self.X[24:]))
            np.testing.assert_allclose(pred,o['predictions'][:,k,:],rtol=1e-8,atol=1e-10)
    def test_23_df_monotone(self):
        ev=np.array([0,1,5,10]);self.assertGreater(df_for_lambda(ev,.1),df_for_lambda(ev,10));self.assertLessEqual(df_for_lambda(ev,.1),3)
    def test_numeric_module_integrity(self):
        hashes=json.loads((ROOT/'tests/core_identity.json').read_text())
        for name,expected in hashes.items():self.assertEqual(sha(ROOT/'src/cacao_coffee_panels'/name),expected)
    def test_25_file_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);tsv(d/'a.tsv',[{'x':'A','y':.123}]);manifest(d);self.assertEqual(verify(d),1)
            (d/'a.tsv').write_text('changed')
            with self.assertRaises(ValueError):verify(d)
    def test_26_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):child(d,'../elsewhere')
    def test_27_gzip_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.tsv.gz';tsv(p,[dict(sample_id='x',value=.123)]);self.assertEqual(read_tsv(p)[0]['sample_id'],'x')
if __name__=='__main__':unittest.main(verbosity=2)
