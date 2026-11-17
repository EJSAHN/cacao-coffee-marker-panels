import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cacao_coffee_panels.data import Cohort
from cacao_coffee_panels.cli import main,output_directory

class InputTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name)/'data';self.p.mkdir()
  self.ids=np.array(['s%02d'%i for i in range(30)]);self.mids=np.array(['m%02d'%i for i in range(12)])
  self.X=np.random.default_rng(7).binomial(2,.35,(30,12)).astype(float);self.X[0,0]=np.nan;self.X[1,1]=.123
  np.savez_compressed(self.p/'genotypes.npz',dosage=self.X,sample_ids=self.ids,marker_ids=self.mids)
  pd.DataFrame({'sample_id':self.ids,'trait':np.arange(30)/3}).to_csv(self.p/'phenotypes.tsv',sep='\t',index=False)
  pd.DataFrame({'sample_id':self.ids,'cluster':['1']*30}).to_csv(self.p/'samples.tsv',sep='\t',index=False)
  pd.DataFrame({'marker_id':self.mids,'source_chrom':['unknown']*12}).to_csv(self.p/'markers.tsv',sep='\t',index=False)
 def tearDown(self):self.temp.cleanup()
 def test_load_fractional_and_nan(self):
  c=Cohort.load(self.p);self.assertTrue(np.isnan(c.dosage[0,0]));self.assertEqual(c.dosage[1,1],.123)
 def test_order_not_silently_changed(self):
  p=self.p/'phenotypes.tsv';t=pd.read_csv(p,sep='\t');t.iloc[::-1].to_csv(p,sep='\t',index=False)
  with self.assertRaises(ValueError):Cohort.load(self.p)
 def test_duplicate_ids_rejected(self):
  ids=self.ids.copy();ids[1]=ids[0];np.savez(self.p/'genotypes.npz',dosage=self.X,sample_ids=ids,marker_ids=self.mids)
  with self.assertRaises(ValueError):Cohort.load(self.p)
 def test_infinite_rejected(self):
  self.X[0,0]=np.inf;np.savez(self.p/'genotypes.npz',dosage=self.X,sample_ids=self.ids,marker_ids=self.mids)
  with self.assertRaises(ValueError):Cohort.load(self.p)
 def test_out_of_range_rejected(self):
  self.X[0,0]=3.;np.savez(self.p/'genotypes.npz',dosage=self.X,sample_ids=self.ids,marker_ids=self.mids)
  with self.assertRaises(ValueError):Cohort.load(self.p)
 def test_split_overlap_rejected(self):
  with self.assertRaises(ValueError):Cohort.load(self.p).split(dict(train_ids=self.ids[:25],test_ids=self.ids[24:]))
 def test_unknown_id_rejected(self):
  with self.assertRaises(ValueError):Cohort.load(self.p).split(dict(train_ids=self.ids[:25],test_ids=['absent']))
 def test_existing_output_refused(self):
  with self.assertRaises(ValueError):output_directory(self.p)
 def test_verify_cli(self):self.assertEqual(main(['verify-input','--data',str(self.p)]),0)
 def test_predict_cli(self):
  root=Path(self.temp.name);cfg=dict(marker_missing_max=.2,maf_min=.01,variance_min=1e-12,scaling='center_only',outer_min_train=20,outer_min_test=5,inner_min_train=10,inner_min_test=3,inner_folds=3,lambda_grid=[.01,1.,100.],pca_components=3)
  design=dict(train_ids=self.ids[:24].tolist(),test_ids=self.ids[24:].tolist(),inner_assignments=[dict(sample_id=s,inner_fold=i%3+1) for i,s in enumerate(self.ids[:24])])
  (root/'c.json').write_text(json.dumps(cfg));(root/'d.json').write_text(json.dumps(design))
  self.assertEqual(main(['predict-fold','--data',str(self.p),'--design',str(root/'d.json'),'--config',str(root/'c.json'),'--out',str(root/'out')]),0)
  out=pd.read_csv(root/'out/predictions.tsv',sep='\t');self.assertEqual(len(out),18)
 def test_output_in_source_rejected(self):
  root=Path(self.temp.name);(root/'c.json').write_text('{}');(root/'d.json').write_text('{}')
  self.assertEqual(main(['predict-fold','--data',str(self.p),'--design',str(root/'d.json'),'--config',str(root/'c.json'),'--out',str(self.p/'newoutput')]),2)
  self.assertFalse((self.p/'newoutput').exists())
 def test_inner_test_leakage_rejected(self):
  from cacao_coffee_panels.cli import predict
  cfg=dict(marker_missing_max=.2,maf_min=.01,variance_min=1e-12,outer_min_train=20,outer_min_test=5)
  design=dict(train_ids=self.ids[:24].tolist(),test_ids=self.ids[24:].tolist(),inner_assignments=[dict(sample_id=s,inner_fold=i%3+1) for i,s in enumerate(self.ids)])
  with self.assertRaises(ValueError):predict(Cohort.load(self.p),design,cfg,'ridge')
if __name__=='__main__':unittest.main()
