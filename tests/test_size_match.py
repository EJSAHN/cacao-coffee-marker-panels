import sys,unittest,copy
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cacao_coffee_panels.size_match import matched_partitions,fit_full_ridge
from cacao_coffee_panels.ridge import Preprocessor,choose_lambda,statistics

CFG=dict(marker_missing_max=.2,maf_min=.01,variance_min=1e-12,outer_min_train=20,outer_min_test=5,inner_min_train=10,inner_min_test=3,inner_folds=3,lambda_grid=[.001,.1,10,1000],scaling='center_only')
class Geometry(unittest.TestCase):
    def setUp(self):
        self.ids=np.array([f'S{i}' for i in range(53)]);self.cl={1:np.arange(0,12),2:np.arange(12,30),3:np.arange(30,50),4:np.arange(50,53)}
        self.tasks,self.eval,self.only=matched_partitions(self.ids,self.cl,[1,2,3],4,10)
    def test_exact_sizes(self):
        for t in self.tasks:self.assertEqual(len(t['test']),len(self.cl[t['fold']]))
    def test_exact_training_sizes(self):
        for t in self.tasks:self.assertEqual(len(t['train']),53-len(self.cl[t['fold']]))
    def test_no_overlap(self):
        for t in self.tasks:self.assertFalse(set(t['train'])&set(t['test']))
    def test_coverage(self):
        for rep in range(1,5):self.assertEqual(sorted(np.concatenate([t['test'] for t in self.tasks if t['repeat']==rep]).tolist()),list(range(50)))
    def test_training_only(self):
        for t in self.tasks:self.assertTrue(set([50,51,52]).issubset(t['train']))
    def test_reference(self):
        for t in self.tasks:
            if t['design']=='cluster_common_inner':np.testing.assert_array_equal(t['test'],self.cl[t['fold']])
    def test_determinism(self):
        again,_,_=matched_partitions(self.ids,self.cl,[1,2,3],4,10)
        for a,b in zip(self.tasks,again):np.testing.assert_array_equal(a['test'],b['test'])
    def test_seed_changes_membership(self):
        other,_,_=matched_partitions(self.ids,self.cl,[1,2,3],4,11)
        self.assertTrue(any(not np.array_equal(a['test'],b['test']) for a,b in zip(self.tasks,other)))
    def test_invalid_partition(self):
        with self.assertRaises(ValueError):matched_partitions(self.ids,{1:np.arange(52)},[1],2,1)
    def test_no_valid_cluster(self):
        tasks,ev,only=matched_partitions(self.ids,self.cl,[],3,10)
        self.assertFalse(tasks);self.assertEqual(len(ev),0);self.assertEqual(len(only),53)

class Numerics(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123);self.X=rng.binomial(2,.4,size=(65,45)).astype(float);self.X[3:6,2]=np.nan
        self.Y=rng.normal(size=(65,2));self.ids=np.array([f'ID{i}' for i in range(50)]);self.out=fit_full_ridge(self.X[:50],self.Y[:50],self.X[50:],self.ids,CFG,9)
    def test_independent_outer_ridge(self):
        train=self.X[:50];mu=np.nanmean(train,axis=0);mask=(np.mean(~np.isfinite(train),axis=0)<=.2)&(np.minimum(mu/2,1-mu/2)>=.01)
        A=np.where(np.isfinite(train[:,mask]),train[:,mask],mu[mask])-mu[mask];V=np.where(np.isfinite(self.X[50:,mask]),self.X[50:,mask],mu[mask])-mu[mask]
        good=np.mean(A*A,axis=0)>1e-12;A=A[:,good];V=V[:,good]
        for j in range(2):
            fit=Ridge(alpha=self.out['lambda_values'][j]*A.shape[1],solver='cholesky').fit(A,self.Y[:50,j]);np.testing.assert_allclose(fit.predict(V),self.out['predictions'][:,j],atol=1e-10)
    def test_inner_mse_reproduction(self):
        sse=np.zeros((4,2));nval=0
        for tr,va in KFold(3,shuffle=True,random_state=9).split(np.arange(50)):
            xx=self.X[tr];mu=np.nanmean(xx,axis=0);mask=(np.mean(~np.isfinite(xx),axis=0)<=.2)&(np.minimum(mu/2,1-mu/2)>=.01)
            A=np.where(np.isfinite(xx[:,mask]),xx[:,mask],mu[mask])-mu[mask];B=np.where(np.isfinite(self.X[va][:,mask]),self.X[va][:,mask],mu[mask])-mu[mask];good=np.mean(A*A,axis=0)>1e-12;A=A[:,good];B=B[:,good]
            for k,lam in enumerate(CFG['lambda_grid']):
                pred=Ridge(alpha=lam*A.shape[1],solver='cholesky').fit(A,self.Y[tr]).predict(B);sse[k]+=np.sum((pred-self.Y[va])**2,axis=0)
            nval+=len(va)
        np.testing.assert_allclose(sse/nval,self.out['inner_mse'],atol=1e-10)
    def test_outer_test_X_not_in_tuning(self):
        b=fit_full_ridge(self.X[:50],self.Y[:50],self.X[50:]*10,self.ids,CFG,9)
        np.testing.assert_array_equal(b['inner_mse'],self.out['inner_mse']);np.testing.assert_array_equal(b['lambda_values'],self.out['lambda_values'])
    def test_train_read_only(self):
        before=self.X.copy();fit_full_ridge(self.X[:50],self.Y[:50],self.X[50:],self.ids,CFG,9);np.testing.assert_array_equal(before,self.X)
    def test_no_outer_test_y_parameter(self):
        import inspect
        self.assertEqual(list(inspect.signature(fit_full_ridge).parameters),['Xtrain','Ytrain','Xtest','ids_train','cfg','inner_seed'])
    def test_finite_prediction(self):self.assertTrue(np.isfinite(self.out['predictions']).all())
    def test_mean_uses_train_only(self):np.testing.assert_allclose(self.out['mean_predictions'],np.tile(self.Y[:50].mean(axis=0),(15,1)))
    def test_small_outer_skipped(self):self.assertEqual(fit_full_ridge(self.X[:50],self.Y[:50],self.X[50:53],self.ids,CFG,9)['state'],'NOT_ESTIMABLE')
    def test_nonfinite_y_fails(self):
        y=self.Y[:50].copy();y[0,0]=np.nan
        with self.assertRaises(ValueError):fit_full_ridge(self.X[:50],y,self.X[50:],self.ids,CFG,9)
    def test_constant_prediction_correlation(self):self.assertIsNone(statistics([1,2,3],[2,2,2])['pearson_r'])
    def test_negative_r2_not_hidden(self):self.assertLess(statistics([1,2,3],[20,20,20])['r2'],0)
    def test_tie_prefers_shrinkage(self):self.assertEqual(choose_lambda(np.array([1.,1.,1.])),2)
if __name__=='__main__':unittest.main()
