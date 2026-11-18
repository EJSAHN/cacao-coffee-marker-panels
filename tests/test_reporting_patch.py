import tempfile, unittest
from pathlib import Path
import numpy as np
from cacao_coffee_panels.reporting import fold_stats
from cacao_coffee_panels.workbook import export_tables

class ReportingPrecisionAndExportTests(unittest.TestCase):
    def test_zero_mean_roundoff_is_undefined(self):
        y=np.array([-2.,-1.,0.,1.,2.]);p=y*1e-12
        x=fold_stats(y,p,np.ones(5),list('abcde'))
        self.assertEqual(x['pearson_status'],'numerically_unresolved')
        self.assertFalse(x['fisher_eligible'])
        self.assertAlmostEqual(x['sse'],float(np.sum((y-p)**2)))
    def test_resolved_strong_shrinkage_retains_ranking(self):
        y=np.array([-2.,-1.,0.,1.,2.]);p=y*1e-5
        x=fold_stats(y,p,np.ones(5),list('abcde'))
        self.assertTrue(x['fisher_eligible']);self.assertAlmostEqual(x['pearson_r'],1.)
    def test_actual_excel_engine_retains_numeric_columns_and_ids(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=folder/'input.tsv';out=folder/'output.xlsx'
            source.write_text('sample_id\tresidual_df\tinner_validation_n\tmarker_indices_sha256\ttrait\n0012\t99\t87\t00017\t=1+2\n',encoding='utf-8')
            export_tables([('test',source)],out)
            wb=load_workbook(out,data_only=False);ws=wb['test']
            self.assertEqual(ws['A2'].value,'0012');self.assertEqual(ws['A2'].data_type,'s')
            self.assertEqual(ws['B2'].value,99);self.assertEqual(ws['B2'].data_type,'n')
            self.assertEqual(ws['C2'].value,87);self.assertEqual(ws['C2'].data_type,'n')
            self.assertEqual(ws['D2'].value,'00017');self.assertEqual(ws['E2'].data_type,'s')
            wb.close()
