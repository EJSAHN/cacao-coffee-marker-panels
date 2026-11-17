from pathlib import Path
import sys,unittest,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cacao_coffee_panels.panels import extra_specs,stratified_order,fit_extra,choose_procedure,specs_for_all
from cacao_coffee_panels.ridge import nested_fold,Preprocessor,linear_kernels,ridge_grid,seed_for,rank_markers,inner_splits
from sklearn.linear_model import Ridge

CFG=dict(marker_missing_max=.2,maf_min=.01,variance_min=1e-12,inner_folds=3,inner_min_train=10,inner_min_test=3,
         lambda_grid=[.001,.01,.1,1.,10.],pca_components=5,random_panel_draws=1,panel_sizes=[5,9],
         core_sizes=[],extended_sizes=[5,9],scaling='center_only')
class Tests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(22);self.X=rng.binomial(2,.35,size=(70,25)).astype(float);self.Y=np.c_[self.X[:,0]*2+rng.normal(size=70),self.X[:,2]+rng.normal(size=70)]
  self.ids=np.array(['s%03d'%i for i in range(70)]);self.mid=np.array(['m%03d'%i for i in range(25)]);self.groups=np.array(['A']*10+['B']*10+['unknown']*5)
  self.X[3,4]=np.nan
 def fit(self,scheme='random5',cfg=CFG):
  return fit_extra(self.X[:55],self.Y[:55],self.X[55:],self.ids[:55],self.mid,self.groups,np.array(['g'+str(i%7) for i in range(55)]),scheme,cfg,777)
 def test_specs(self):
  ss=extra_specs([20,45,96,219,500,1000,2000,5000],[45,219,500]);self.assertEqual(len(ss),23);self.assertEqual(len(set(s['model'] for s in ss)),23)
 def test_order_unique(self):
  r=stratified_order(self.mid,self.groups,7);self.assertEqual(sorted(r),list(range(25)))
 def test_group_heads(self):
  r=stratified_order(self.mid,self.groups,7);self.assertEqual(set(self.groups[r[:3]]),set(self.groups))
 def test_nested_prefix(self):
  a=stratified_order(self.mid,self.groups,7);b=stratified_order(self.mid,self.groups,7);np.testing.assert_array_equal(a,b);self.assertTrue(set(a[:5])<=set(b[:10]))
 def test_row_order_invariance(self):
  ix=np.arange(25)[::-1];a=self.mid[stratified_order(self.mid,self.groups,7)];b=self.mid[ix][stratified_order(self.mid[ix],self.groups[ix],7)];np.testing.assert_array_equal(a,b)
 def test_seed_changes(self):
  self.assertFalse(np.array_equal(stratified_order(self.mid,self.groups,1),stratified_order(self.mid,self.groups,2)))
 def test_missing_group_rejected(self):
  with self.assertRaises(ValueError):stratified_order(['a','b'],['x',''],1)
 def test_duplicate_marker_rejected(self):
  with self.assertRaises(ValueError):stratified_order(['a','a'],['x','y'],1)
 def test_one_group(self):
  r=stratified_order(self.mid,['x']*25,3);self.assertEqual(len(r),25)
 def test_singletons(self):
  r=stratified_order(self.mid,self.mid,3);self.assertEqual(len(set(r)),25)
 def test_constant_and_all_missing_removed(self):
  X=self.X.copy();X[:,0]=np.nan;X[:,1]=1;pr=Preprocessor.fit(X[:55],CFG);self.assertNotIn(0,pr.indices);self.assertNotIn(1,pr.indices)
 def test_original_numerics_preserved(self):
  new=self.fit();old=nested_fold(self.X[:55],self.Y[:55],self.X[55:],self.ids[:55],self.mid,np.array(['g'+str(i%7) for i in range(55)]),'random5',CFG,777)
  for key,val in new['results'].items():
   if key.startswith('stratified'):continue
   np.testing.assert_allclose(val['predictions'],old['results'][key]['predictions'],atol=1e-12)
   np.testing.assert_array_equal(val['lambdas'],old['results'][key]['lambdas'])
 def test_unavailable_not_capped(self):
  r=self.fit(cfg={**CFG,'extended_sizes':[5,500]});self.assertEqual(len(r['candidate_failures']),4);self.assertFalse(any('500' in k for k in r['results']))
 def test_no_outer_target_parameter(self):
  import inspect
  self.assertNotIn('Ytest',inspect.signature(fit_extra).parameters)
 def test_outer_x_not_used_in_training(self):
  a=self.fit();saved=self.X.copy();self.X[55:]=2;b=self.fit();self.X=saved
  self.assertEqual(a['preprocessing'],b['preprocessing']);self.assertEqual(a['tuning'],b['tuning'])
  for k in a['results']:
   for x,y in zip(a['results'][k]['selected'],b['results'][k]['selected']):np.testing.assert_array_equal(x,y)
 def test_neutral_selection_no_phenotype(self):
  a=self.fit();self.Y=self.Y[::-1].copy();b=self.fit()
  for k in a['results']:
   if k.startswith('f_regression'):continue
   for x,y in zip(a['results'][k]['selected'],b['results'][k]['selected']):np.testing.assert_array_equal(x,y)
 def test_primal_equivalence(self):
  a=self.fit();pr=Preprocessor.fit(self.X[:55],CFG);Z=pr.transform(self.X[:55]);V=pr.transform(self.X[55:]);loc={x:i for i,x in enumerate(pr.indices)}
  for k,val in a['results'].items():
   for q in range(2):
    idx=[loc[x] for x in val['selected'][q]];m=Ridge(alpha=float(val['lambdas'][q])*len(idx),solver='cholesky').fit(Z[:,idx],self.Y[:55,q]);np.testing.assert_allclose(m.predict(V[:,idx]),val['predictions'][:,q],atol=1e-10)
 def test_grouped_splits(self):
  a=self.fit('identity_grouped5');self.assertEqual(a['state'],'OK');self.assertEqual(len(a['inner_assignments']),55)
 def test_inner_group_gate(self):
  a=fit_extra(self.X[:55],self.Y[:55],self.X[55:],self.ids[:55],self.mid,self.groups,['x']*55,'cluster_out',CFG,7);self.assertEqual(a['state'],'NOT_ESTIMABLE')
 def test_selector_uses_inner_only(self):
  specs=specs_for_all([5,9]);r=[dict(model='random_5_draw1',inner_mse=4,y_test=0),dict(model='high_maf_9',inner_mse=2,y_test=100)];self.assertEqual(choose_procedure(r,specs),'high_maf_9')
 def test_selector_tie_smaller(self):
  specs=specs_for_all([5,9]);r=[dict(model='random_5_draw1',inner_mse=2),dict(model='high_maf_9',inner_mse=2)];self.assertEqual(choose_procedure(r,specs),'random_5_draw1')
 def test_all_missing_group_is_not_silently_excluded(self):
  self.groups[:]='UNSPECIFIED';r=self.fit();self.assertIn('stratified_random_5_draw1',r['results'])

if __name__=='__main__':unittest.main()
