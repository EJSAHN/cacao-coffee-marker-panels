import unittest,sys,math,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prediction_reporting import pearson,ranks,fold_stats,aggregate_folds,fisher_summary,fingerprint
from scipy.stats import pearsonr,rankdata,spearmanr

class TestMetrics(unittest.TestCase):
 def fold(self,y,p,b=1):return fold_stats(y,p,np.full(len(y),b),[str(i) for i in range(len(y))])
 def test_exact_constant_prediction(self):
  r,s=pearson([1,2,3],[2,2,2]);self.assertTrue(np.isnan(r));self.assertEqual(s,'constant_predicted')
 def test_constant_outcome(self):self.assertEqual(pearson([1,1],[2,3])[1],'constant_observed')
 def test_small_nonzero_sd_not_dropped(self):self.assertAlmostEqual(pearson(np.arange(10),2+np.arange(10)*1e-6)[0],1)
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):pearson([1,np.nan],[1,2])
 def test_r_matches_scipy(self):
  rng=np.random.default_rng(1)
  for n in [5,17,100]:
   x=rng.normal(size=n);y=.4*x+rng.normal(size=n);self.assertAlmostEqual(pearson(x,y)[0],pearsonr(x,y).statistic)
 def test_ranks_ties(self):np.testing.assert_array_equal(ranks([2,1,2,3,1]),rankdata([2,1,2,3,1]))
 def test_spearman(self):
  r=self.fold([1,2,3,4,5],[3,2,1,2,7]);self.assertAlmostEqual(r['spearman_r'],spearmanr([1,2,3,4,5],[3,2,1,2,7]).statistic)
 def test_fisher_manual(self):
  a=self.fold([1,2,3,4,5],[1,4,2,5,3]);b=self.fold([1,2,3,4,5,6],[2,3,5,1,4,8]);g=fisher_summary([a,b]);expected=np.tanh((2*np.arctanh(a['pearson_r'])+3*np.arctanh(b['pearson_r']))/5);self.assertAlmostEqual(g['within_fisher_r'],expected)
 def test_n_le_three(self):self.assertEqual(fisher_summary([self.fold([1,2,3],[1,3,2])])['fisher_folds'],0)
 def test_perfect_clip_reported(self):
  x=self.fold([1,2,3,4,5],[1,2,3,4,5]);self.assertTrue(x['fisher_clipped']);self.assertTrue(np.isfinite(x['fisher_z']))
 def test_invalid_not_zero(self):self.assertTrue(np.isnan(fisher_summary([self.fold([1,2,3,4],[2,2,2,2])])['within_fisher_r']))
 def test_baseline_skill_zero(self):self.assertEqual(self.fold([1,2,3,4],[2,2,2,2],2)['mse_skill'],0)
 def test_zero_baseline_denominator(self):self.assertTrue(np.isnan(self.fold([1,1,1,1],[2,3,2,3],1)['mse_skill']))
 def test_bad_training_mean(self):
  with self.assertRaises(ValueError):fold_stats([1,2],[1,2],[1,2],['a','b'])
 def test_duplicate_ids(self):
  with self.assertRaises(ValueError):fingerprint(['a','a'])
 def test_fingerprint_order(self):self.assertEqual(fingerprint(['a','b'],[1,2]),fingerprint(['b','a'],[2,1]))
 def test_pooled_decomposition(self):
  rng=np.random.default_rng(7);ys=[];ps=[];bs=[];rows=[]
  for n,mu in [(5,-4),(13,2),(17,8)]:
   y=rng.normal(mu,1,n);p=.5*y+rng.normal(size=n);b=np.full(n,2.3);ys.extend(y);ps.extend(p);bs.extend(b);rows.append(self.fold(y,p,2.3))
  a=aggregate_folds(rows);self.assertAlmostEqual(a['pooled_r'],pearsonr(ys,ps).statistic);self.assertAlmostEqual(a['pooled_r'],a['within_covariance_component']+a['between_covariance_component']);self.assertAlmostEqual(a['mse_skill'],1-np.sum((np.array(ys)-ps)**2)/np.sum((np.array(ys)-bs)**2))
 def test_pooled_baseline_negative(self):
  ys=[np.arange(5.),np.arange(5.)+10];rows=[self.fold(y,np.full(5,ys[1-i].mean()),ys[1-i].mean()) for i,y in enumerate(ys)]
  a=aggregate_folds(rows);self.assertLess(a['pooled_r'],0);self.assertTrue(np.isnan(a['within_fisher_r']));self.assertEqual(a['mse_skill'],0)
 def test_affine_does_not_rescue_calibration(self):
  y=np.arange(1.,9.);a=self.fold(y,y,3);b=self.fold(y,100+1e-6*y,3);self.assertAlmostEqual(a['pearson_r'],b['pearson_r']);self.assertLess(b['mse_skill'],0)
 def test_no_fitting_imports(self):
  text=(ROOT/'scripts/prediction_reporting.py').read_text();self.assertNotIn('sklearn',text);self.assertNotIn('.fit(',text)
 def test_sizes_coverage(self):
  a=aggregate_folds([self.fold([1,2,3,4,5],[1,2,3,4,5]),self.fold([1,2,3,4,5],[0,0,0,0,0])]);self.assertEqual(a['fisher_n'],5);self.assertEqual(a['fisher_sample_coverage'],.5)
 def test_finite_extreme_small(self):
  a=self.fold(np.arange(1,9.),10+1e-10*np.arange(1,9.));self.assertTrue(a['small_prediction_spread']);self.assertTrue(a['fisher_eligible'])
 def test_no_ci_or_pvalue_in_summary(self):
  a=aggregate_folds([self.fold([1,2,3,4,5],[2,1,3,4,5])]);self.assertNotIn('pvalue',a);self.assertNotIn('ci',a)
 def test_signed_component(self):
  a=aggregate_folds([self.fold([1,2,3,4,5],[11,12,13,14,15]),self.fold([11,12,13,14,15],[1,2,3,4,5])]);self.assertGreater(a['within_cross_product'],0);self.assertLess(a['between_cross_product'],0);self.assertLess(a['pooled_r'],0)


import tempfile, csv, json
from prediction_reporting import summarize_file

class TestReportingGuard(unittest.TestCase):
    def test_roundoff_only_is_unresolved_but_errors_remain(self):
        p = np.array([57.747, np.nextafter(57.747, np.inf)] * 5)
        y = np.arange(10, dtype=float)
        r = fold_stats(y, p, np.full(10, 3.0), list(map(str, range(10))))
        self.assertEqual(r['pearson_status'], 'numerically_unresolved')
        self.assertFalse(r['fisher_eligible'])
        self.assertTrue(math.isnan(r['pearson_r']))
        self.assertTrue(math.isnan(r['spearman_r']))
        self.assertAlmostEqual(r['sse'], float(np.sum((y-p)**2)))
        self.assertEqual(aggregate_folds([r])['n_predictions'], 10)

    def test_small_resolved_spread_is_not_removed(self):
        y = np.arange(10, dtype=float)
        r = fold_stats(y, 38.0 + y*0.0001, np.full(10, 38.0), list(map(str, range(10))))
        self.assertTrue(r['fisher_eligible'])
        self.assertEqual(r['pearson_status'], 'ok')

    def test_serialized_false_rejected(self):
        with self.assertRaises(TypeError):
            fisher_summary([{'fisher_eligible': 'False'}])

    def test_nonfinite_baseline_rejected(self):
        with self.assertRaises(ValueError):
            fold_stats([1., 2.], [2., 3.], [math.inf, math.inf], ['a', 'b'])

class TestFileInterface(unittest.TestCase):
    columns = ['dataset', 'scheme', 'repeat', 'trait', 'model', 'fold',
               'sample_id', 'observed', 'predicted', 'train_mean']
    groups = columns[:5]
    def sample(self):
        return [dict(zip(self.columns, ['cohort', 'cluster_out', '0', 'trait', 'ridge',
                     str(i//5), str(i), i, i*0.7+2, 4.0])) for i in range(10)]
    def write(self, path, records):
        with path.open('w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, self.columns, delimiter='\t')
            wr.writeheader(); wr.writerows(records)
    def test_round_trip_and_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src=root/'p.tsv'; self.write(src, self.sample())
            before=src.read_bytes(); out=root/'out'
            result=summarize_file(src, out, self.groups)
            self.assertEqual(result['prediction_rows'], 10)
            self.assertEqual(result['groups'], 1)
            self.assertFalse(result['model_fitting'])
            self.assertEqual(before, src.read_bytes())
            self.assertFalse((out/'INCOMPLETE.txt').exists())
            self.assertTrue((out/'report.json').exists())
    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'p.tsv';self.write(src,self.sample())
            with self.assertRaises(FileExistsError):summarize_file(src,root,self.groups)
    def test_duplicate_id_across_folds_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'p.tsv';a=self.sample();a[-1]['sample_id']='0';self.write(src,a)
            with self.assertRaises(ValueError):summarize_file(src,root/'out',self.groups)
            self.assertTrue((root/'out/INCOMPLETE.txt').exists())
            self.assertFalse((root/'out/report.json').exists())
    def test_noncontiguous_group_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'p.tsv';a=self.sample();a[5]['model']='other';self.write(src,a)
            with self.assertRaises(ValueError):summarize_file(src,root/'out',self.groups)
    def test_empty_input_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'p.tsv';self.write(src,[])
            with self.assertRaises(ValueError):summarize_file(src,root/'out',self.groups)
    def test_missing_column_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'p.tsv';src.write_text('other\n1\n')
            with self.assertRaises(ValueError):summarize_file(src,root/'out',self.groups)

if __name__ == '__main__':
    unittest.main()
