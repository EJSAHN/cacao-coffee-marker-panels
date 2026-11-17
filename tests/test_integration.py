import sys,json,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cacao_coffee_panels.workflow import checkpoint_start,checkpoint_finish,cohort_record,prediction_rows
from cacao_coffee_panels.io import tsv,js,sha
from cacao_coffee_panels.exports import collect_results
from cacao_coffee_panels.study_reporting import build_summary
from cacao_coffee_panels.readers import decode_hapmap,match_ids,detect_workbook,parse_agrosavia_phenotypes
from cacao_coffee_panels.model_replay import fit_with_recorded_screen
from cacao_coffee_panels.models import make_split_cache,fit_trait
from cacao_coffee_panels.ridge import seed_for

class IntegrationTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_checkpoint_same(self):
  p=self.d/'cp';self.assertTrue(checkpoint_start(p,{'a':1}));(p/'x').write_text('abc');checkpoint_finish(p);self.assertFalse(checkpoint_start(p,{'a':1},True))
 def test_checkpoint_changed(self):
  p=self.d/'cp';checkpoint_start(p,{'a':1});checkpoint_finish(p)
  with self.assertRaises(ValueError):checkpoint_start(p,{'a':2},True)
 def test_checkpoint_tampered(self):
  p=self.d/'cp';checkpoint_start(p,{'a':1});(p/'x').write_text('a');checkpoint_finish(p);(p/'x').write_text('b')
  with self.assertRaises(ValueError):checkpoint_start(p,{'a':1},True)
 def test_checkpoint_partial(self):
  p=self.d/'cp';checkpoint_start(p,{})
  with self.assertRaises(ValueError):checkpoint_start(p,{},True)
 def test_checkpoint_no_resume(self):
  p=self.d/'cp';checkpoint_start(p,{})
  with self.assertRaises(FileExistsError):checkpoint_start(p,{})
 def records(self):
  return [dict(analysis='ridge',dataset='toy',scenario='q20',scaling='center_only',scheme='cluster_out',repeat=0,trait='trait',model=m,fold=f,sample_id=str(f*10+i),observed=float(i+f),predicted=float(i+f+.1) if m=='full_ridge' else 2.,train_mean=2.) for m in ['full_ridge','mean'] for f in [1,2] for i in range(6)]
 def test_reporting(self):
  p=self.d/'p.tsv';tsv(p,self.records());before=sha(p);s,f,c=build_summary([p],self.d/'out');self.assertEqual(len(s),2);self.assertEqual(before,sha(p));self.assertTrue(np.isnan(next(x for x in s if x['model']=='mean')['within_fisher_r']))
 def test_duplicate_predictions(self):
  p=self.d/'p.tsv';rr=self.records();rr.append(rr[0]);tsv(p,rr)
  with self.assertRaises(ValueError):build_summary([p],self.d/'out')
 def test_export_hash_rejected(self):
  p=self.d/'p.tsv';tsv(p,[{'x':1}]);c=self.d/'c.json';js(c,dict(tables=[dict(label='x',path=p.name,sha256='bad')]))
  with self.assertRaises(ValueError):collect_results(c,self.d/'out',False)
 def test_export_values_preserved(self):
  p=self.d/'p.tsv';tsv(p,[{'x':'001','y':1.23456789}]);c=self.d/'c.json';js(c,dict(tables=[dict(label='x',path=p.name,sha256=sha(p))]));collect_results(c,self.d/'out',False);self.assertEqual(p.read_bytes(),(self.d/'out/tables/p.tsv').read_bytes())
 def test_export_escape(self):
  p=self.d/'p.tsv';tsv(p,[{'x':1}]);c=self.d/'c.json';js(c,dict(tables=[dict(label='x',path=p.name,filename='../oops')]))
  with self.assertRaises(ValueError):collect_results(c,self.d/'out',False)
 def test_metadata_asterisk(self):
  p=self.d/'source.txt';p.write_text('rs_\talleles\tchrom\tpos\tstrand\tassembly#\tcenter\tprotLSID\tassayLSID\tpanelLSID\tQCcode\ta\tb\nM1\t*\t1\t1.1\t+\tNA\tNA\tNA\tNA\tNA\tNA\tAA\tAG\n'+ '\t'*12+'\n')
  ids,mids,x,meta,action,tokens=decode_hapmap(p);self.assertEqual(x.shape,(2,1));self.assertTrue(np.array_equal(x[:,0],[0.,1.]))
 def test_match_ambiguous(self):
  with self.assertRaises(ValueError):match_ids(['A-1','A1'],['A 1'])
 def test_ole_before_zip(self):
  p=self.d/'x.xls';p.write_bytes(bytes.fromhex('D0CF11E0A1B11AE1')+b'PK\x03\x04');self.assertEqual(detect_workbook(p),'xls')
 def test_invalid_workbook(self):
  p=self.d/'bad.xls';p.write_text('<html>not a workbook</html>')
  with self.assertRaises(ValueError):detect_workbook(p)
 def test_model_screen_replay(self):
  cfg=json.loads((ROOT/'config/models.json').read_text());rng=np.random.default_rng(8);X=rng.binomial(2,.4,(42,14)).astype(float);Y=rng.normal(size=(42,1));ids=np.array([str(i) for i in range(36)]);assignment=[dict(sample_id=str(i),inner_fold=i%3+1) for i in range(36)]
  cfg['random_forest']['n_estimators']=4;cfg['gradient_boosting']['n_estimators']=4;cache,reason=make_split_cache(X[:36],Y[:36],X[36:],ids,assignment,cfg);a=fit_trait(cache,0,np.array(['m'+str(i) for i in range(14)]),cfg,9);b=fit_with_recorded_screen(cache,0,np.array(['m'+str(i) for i in range(14)]),cfg,9,a['marker_sets'])
  for name in a['results']:np.testing.assert_allclose(a['results'][name]['predictions'],b['results'][name]['predictions'],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

class CommonSupportTests(unittest.TestCase):
 def test_same_IDs_different_partitions(self):
  from cacao_coffee_panels.comparisons import common_support
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);base=[dict(analysis='ridge',scaling='center_only',scheme='cluster_out',repeat=0,trait='T',model='full_ridge',sample_id=str(i),fold=i//5,observed=float(i),predicted=float(i+.2),train_mean=4.,dataset='left',scenario='primary') for i in range(10)]
   right=[dict(x,dataset='right',fold=i%2) for i,x in enumerate(base)];tsv(d/'l.tsv',base);tsv(d/'r.tsv',right);res=common_support(d/'l.tsv',d/'r.tsv',d/'out');self.assertEqual(res[0]['n_common'],10);self.assertFalse(res[0]['same_partition'])
 def test_phenotype_mismatch(self):
  from cacao_coffee_panels.comparisons import common_support
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);base=[dict(analysis='ridge',scaling='center_only',scheme='random5',repeat=0,trait='T',model='full_ridge',sample_id=str(i),fold=0,observed=float(i),predicted=float(i),train_mean=2.,dataset='A',scenario='s') for i in range(5)];right=[dict(x) for x in base];right[0]['observed']=88;tsv(d/'l.tsv',base);tsv(d/'r.tsv',right)
   with self.assertRaises(ValueError):common_support(d/'l.tsv',d/'r.tsv',d/'out')

class WorkbookContractTests(unittest.TestCase):
 def test_export_cell_types_without_spreadsheet_engine(self):
  # Exercise serializer/layout contracts with a stub, not an installed Excel writer.
  from unittest.mock import patch
  import types,collections
  sheets=[]
  class Cell:
   def __init__(self,ws,value):self.value=value;self.data_type='n'
  class Sheet:
   def __init__(self,name):self.title=name;self.rows=[];self.column_dimensions=collections.defaultdict(types.SimpleNamespace);self.row_dimensions=collections.defaultdict(types.SimpleNamespace)
   def append(self,row):self.rows.append(row)
  class Book:
   def __init__(self,**kw):self.properties=types.SimpleNamespace()
   def create_sheet(self,n):s=Sheet(n);sheets.append(s);return s
   def save(self,p):Path(p).write_bytes(b'stub workbook for serializer test only')
  simple=lambda **kw:types.SimpleNamespace(**kw)
  fake={'openpyxl':types.SimpleNamespace(Workbook=Book),'openpyxl.styles':types.SimpleNamespace(Font=simple,PatternFill=lambda *a,**kw:simple(**kw),Alignment=simple),'openpyxl.cell':types.SimpleNamespace(WriteOnlyCell=Cell),'openpyxl.utils':types.SimpleNamespace(get_column_letter=lambda i:chr(64+i))}
  with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,fake):
   from cacao_coffee_panels.workbook import export_tables
   d=Path(d);tsv(d/'x.tsv',[dict(sample_id='001',rmse=1.234567890123,comment='=1+1'),dict(sample_id='002',rmse=None,comment='plain')]);before=sha(d/'x.tsv');records=export_tables([('Results',d/'x.tsv')],d/'x.xlsx');self.assertEqual(records[0]['rows'],2);self.assertEqual(before,sha(d/'x.tsv'))
   table=sheets[2];self.assertEqual(table.rows[1][0].value,'001');self.assertEqual(table.rows[1][0].data_type,'s');self.assertEqual(table.rows[1][2].data_type,'s');self.assertEqual(table.rows[1][1].value,1.234567890123);self.assertIsNone(table.rows[2][1].value)
