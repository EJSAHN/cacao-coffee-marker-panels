"""Compare stored procedures on common test IDs; never refit or recalibrate."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
import pandas as pd
from .io import tsv,js,sha
from .reporting import fold_stats,aggregate_folds,fisher_summary

def common_support(left,right,out,keys=None):
    """Each input is a long prediction table with a single dataset/scenario.

    KEY distinguishes procedures, not the dataset/scenario being compared.
    Distinct fold partitions retain separate correlations, not a paired-r claim.
    """
    keys=keys or ['analysis','scaling','scheme','repeat','trait','model']
    sources=[Path(left),Path(right)];before=[sha(p) for p in sources]
    data=[pd.read_csv(p,sep='\t',dtype={'sample_id':str},keep_default_na=False) for p in sources]
    for d in data:
        for col in ('dataset','scenario'):
            if col in d and d[col].nunique()!=1:raise ValueError('Supply one dataset and scenario per side')
        if not set(keys+['sample_id','fold','observed','predicted','train_mean'])<=set(d):raise ValueError('Missing comparison columns')
        if d.duplicated(keys+['sample_id']).any():raise ValueError('Duplicate test ID within a procedure')
    groups=[{k:g for k,g in d.groupby(keys,sort=False,dropna=False)} for d in data]
    rows=[];missing=[]
    for key in sorted(set(groups[0])|set(groups[1]),key=str):
        meta=dict(zip(keys,key))
        if key not in groups[0] or key not in groups[1]:missing.append(dict(**meta,reason='procedure_missing_on_one_side'));continue
        a,b=groups[0][key],groups[1][key];common=sorted(set(a.sample_id)&set(b.sample_id))
        if not common:missing.append(dict(**meta,reason='no_common_test_IDs'));continue
        aa=a.set_index('sample_id').loc[common];bb=b.set_index('sample_id').loc[common]
        if not np.allclose(aa.observed,bb.observed,rtol=0,atol=1e-12):raise ValueError('Common accession has different phenotype')
        summaries=[];folds=[];signatures=[]
        for d in [aa,bb]:
            rr=[];membership=[]
            for f,g in d.groupby('fold',sort=True):
                z=fold_stats(g.observed.to_numpy(float),g.predicted.to_numpy(float),g.train_mean.to_numpy(float),g.index.to_numpy(str));z['fold']=f;rr.append(z);membership.append(tuple(sorted(g.index)))
            signatures.append(tuple(sorted(membership)));folds.append(rr);summaries.append(aggregate_folds(rr))
        same=signatures[0]==signatures[1]
        row=dict(**meta,n_common=len(common),left_n=len(a),right_n=len(b),same_partition=same,left_dataset=str(a.dataset.iloc[0]),right_dataset=str(b.dataset.iloc[0]),comparison='Common evaluation IDs; training sets and marker representations may differ')
        for side,s in zip(['left','right'],summaries):
            row.update({side+'_'+k:s[k] for k in ['within_fisher_r','fisher_n','fisher_folds','fisher_sample_coverage','rmse','mae','mse_skill','pooled_r','r2']})
        row['delta_rmse_left_minus_right']=summaries[0]['rmse']-summaries[1]['rmse'];rows.append(row)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);tsv(out/'common_test_comparison.tsv',rows);tsv(out/'unpaired_procedures.tsv',missing)
    if [sha(p) for p in sources]!=before:raise ValueError('Input changed during comparison')
    js(out/'sources.json',[dict(file=p.name,sha256=h) for p,h in zip(sources,before)]);return rows
