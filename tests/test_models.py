import copy,inspect,json,sys,tempfile,unittest,warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cacao_coffee_panels.models import fit_estimator,make_split_cache,fit_trait,model_grid
from cacao_coffee_panels.ridge import Preprocessor,rank_markers,linear_kernels,ridge_grid
from cacao_coffee_panels.io import tsv,verify,manifest,child
class TestComparison(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/'config/models.json').read_text());self.cfg['shared_screen']['max_markers']=12
  self.cfg['random_forest']['n_estimators']=4;self.cfg['gradient_boosting']['n_estimators']=6
  rng=np.random.default_rng(811);self.X=rng.integers(0,3,(48,20)).astype(float);self.X[0,0]=np.nan
  self.V=rng.integers(0,3,(8,20)).astype(float);self.y=2*np.nan_to_num(self.X[:,2])+rng.normal(size=48)
  self.Y=np.c_[self.y,self.y*.3+2];self.ids=np.array([f'S{i:02d}' for i in range(48)]);self.mid=np.array([f'M{i:03d}' for i in range(20)])
  self.assign=[dict(sample_id=s,inner_fold=i%3+1) for i,s in enumerate(self.ids)]
 def cache(self,X=None,V=None):return make_split_cache(self.X if X is None else X,self.Y,self.V if V is None else V,self.ids,self.assign,self.cfg)[0]
 def test_no_test_target_argument(self):
  self.assertNotIn('y_test',inspect.signature(fit_trait).parameters);self.assertNotIn('y_test',inspect.signature(make_split_cache).parameters)
 def test_common_cap(self):
  r=fit_trait(self.cache(),0,self.mid,self.cfg,3)
  self.assertTrue(all(v['n_markers']==12 for v in r['results'].values()))
  self.assertTrue(all(len(v)==12 for v in r['marker_sets'].values()))
 def test_cap_respects_available_features(self):
  self.cfg['shared_screen']['max_markers']=1000;r=fit_trait(self.cache(),0,self.mid,self.cfg,3)
  self.assertTrue(all(v['n_markers']==20 for v in r['results'].values()))
 def test_independent_ridge(self):
  A=np.nan_to_num(self.X);B=self.V;A=A-A.mean(0);B=B-np.nan_to_num(self.X).mean(0)
  K,C=linear_kernels(A,B);a=ridge_grid(K,C,self.y,[.3])[:,0,0]
  b=Ridge(alpha=.3*A.shape[1],solver='svd').fit(A,self.y).predict(B);np.testing.assert_allclose(a,b,atol=1e-10)
 def test_enet_independent(self):
  A=np.nan_to_num(self.X);p={'l1_ratio':.5,'alpha_fraction':.1};a,d=fit_estimator('elastic_net',A,self.y,self.V,p,self.cfg,3)
  mu=self.y.mean();sd=self.y.std();z=(self.y-mu)/sd;alpha=.1*np.max(abs(A.T@z))/(len(z)*.5)
  b=mu+sd*ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=20000,tol=1e-5,selection='cyclic').fit(A,z).predict(self.V)
  np.testing.assert_allclose(a,b,atol=1e-8)
 def test_forest_independent(self):
  A=np.nan_to_num(self.X);p={'min_samples_leaf':2};a,_=fit_estimator('random_forest',A,self.y,self.V,p,self.cfg,3)
  b=self.y.mean()+self.y.std()*RandomForestRegressor(n_estimators=4,max_features='sqrt',min_samples_leaf=2,n_jobs=1,random_state=3).fit(A,(self.y-self.y.mean())/self.y.std()).predict(self.V)
  np.testing.assert_array_equal(a,b)
 def test_boosting_independent(self):
  A=np.nan_to_num(self.X);p={'max_depth':2};a,_=fit_estimator('gradient_boosting',A,self.y,self.V,p,self.cfg,3)
  b=self.y.mean()+self.y.std()*GradientBoostingRegressor(n_estimators=6,learning_rate=.05,max_depth=2,min_samples_leaf=3,max_features='sqrt',subsample=.85,n_iter_no_change=None,random_state=3,loss='squared_error').fit(A,(self.y-self.y.mean())/self.y.std()).predict(self.V)
  np.testing.assert_array_equal(a,b)
 def test_fscore_matches_library(self):
  c=self.cache();b=c['inner'][0];ind=rank_markers('f_regression',b['prep'],b['A'],b['Y'][:,0],self.mid,3)
  scores,_=f_regression(b['A'],b['Y'][:,0]);want=np.lexsort((self.mid[b['prep'].indices],-scores));np.testing.assert_array_equal(ind,want)
 def test_inner_validation_not_used_to_fit_preprocessor(self):
  c=self.cache();X=self.X.copy();va=np.array([i for i in range(48) if i%3==0]);X[va,0]=2
  d=self.cache(X=X);np.testing.assert_array_equal(c['inner'][0]['prep'].means,d['inner'][0]['prep'].means)
 def test_outer_predictors_do_not_change_training_selection(self):
  a=fit_trait(self.cache(),0,self.mid,self.cfg,9);b=fit_trait(self.cache(V=np.zeros_like(self.V)),0,self.mid,self.cfg,9)
  self.assertEqual(a['tuning'],b['tuning'])
  for k in a['marker_sets']:np.testing.assert_array_equal(a['marker_sets'][k],b['marker_sets'][k])
 def test_inputs_not_modified(self):
  x=self.X.copy();y=self.Y.copy();v=self.V.copy();fit_trait(self.cache(),0,self.mid,self.cfg,6)
  np.testing.assert_array_equal(x,self.X);np.testing.assert_array_equal(y,self.Y);np.testing.assert_array_equal(v,self.V)
 def test_constant_targets(self):
  for m,p in [('elastic_net',{'l1_ratio':.5,'alpha_fraction':.1}),('random_forest',{'min_samples_leaf':2}),('gradient_boosting',{'max_depth':2})]:
   a,d=fit_estimator(m,np.nan_to_num(self.X),np.ones(48)*7,self.V,p,self.cfg,1);np.testing.assert_array_equal(a,np.ones(8)*7);self.assertEqual(d['status'],'CONSTANT_TRAINING_TARGET')
 def test_invalid_nonfinite(self):
  with self.assertRaises(ValueError):fit_estimator('elastic_net',self.X,self.y,self.V,{'l1_ratio':.5,'alpha_fraction':.1},self.cfg,1)
 def test_bad_assignment_raises(self):
  with self.assertRaises(ValueError):make_split_cache(self.X,self.Y,self.V,self.ids,self.assign[:-1],self.cfg)
 def test_small_inner_split_not_estimable(self):
  a=[dict(sample_id=s,inner_fold=1 if i<2 else 2) for i,s in enumerate(self.ids)];cache,reason=make_split_cache(self.X,self.Y,self.V,self.ids,a,self.cfg);self.assertIsNone(cache)
 def test_unknown_model_raises(self):
  with self.assertRaises(ValueError):model_grid('fake',self.cfg)
 def test_feature_ties_deterministic(self):
  pre=Preprocessor.fit(np.tile([0,1,2,1],(20,1)).T,self.cfg['baseline']);X=pre.transform(np.tile([0,1,2,1],(20,1)).T)
  mid=self.mid[::-1];order=rank_markers('f_regression',pre,X,np.array([0,1,2,1]),mid,1);self.assertEqual(list(mid[order]),sorted(mid))
 def test_resume_manifest_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'a.txt').write_text('a');manifest(root);self.assertEqual(verify(root),1);(root/'a.txt').write_text('b')
   with self.assertRaises(ValueError):verify(root)
 def test_source_escape_rejected(self):
  with self.assertRaises(ValueError):child(ROOT,'../outside.txt')
 def test_enet_nonconvergence_is_not_silenced(self):
  class Fake:
   def __init__(self,**kw):self.n_iter_=kw['max_iter'];self.dual_gap_=1.
   def fit(self,A,y):warnings.warn('fake convergence problem',ConvergenceWarning);return self
  with patch('cacao_coffee_panels.models.ElasticNet',Fake):
   p,d=fit_estimator('elastic_net',np.nan_to_num(self.X),self.y,self.V,{'l1_ratio':.5,'alpha_fraction':.1},self.cfg,1)
  self.assertIsNone(p);self.assertEqual(d['status'],'NOT_ESTIMABLE');self.assertEqual(d['retries'],1)
 def test_enet_shift_invariance(self):
  A=np.nan_to_num(self.X);p={'l1_ratio':.5,'alpha_fraction':.1};a,_=fit_estimator('elastic_net',A,self.y,self.V,p,self.cfg,1);b,_=fit_estimator('elastic_net',A,self.y+100,self.V,p,self.cfg,1);np.testing.assert_allclose(a+100,b,atol=1e-10)
 def test_inner_val_targets_not_outer_targets(self):
  c=self.cache();self.assertNotIn('Yval',c['outer']);self.assertTrue(all('Yval' in b for b in c['inner']))
if __name__=='__main__':unittest.main(verbosity=2)
