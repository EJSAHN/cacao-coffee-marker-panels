"""Evaluate stored predictions and paired comparisons without any fitting."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
import pandas as pd
from .io import tsv,sha,js
from .reporting import fold_stats,aggregate_folds,fisher_summary,fingerprint

KEY=['analysis','dataset','scenario','scaling','scheme','repeat','trait','model']

def build_summary(files,out):
    """Consume completed per-fold files. Duplicates and support changes fail closed."""
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);store=defaultdict(list);used=[];statuses=[]
    for p in files:
        p=Path(p)
        if (p.parent/'signature.json').exists() and not (p.parent/'complete.json').exists():raise ValueError('Incomplete prediction checkpoint')
        before=sha(p);d=pd.read_csv(p,sep='\t',dtype={'sample_id':str},keep_default_na=False)
        required=set(KEY+['fold','sample_id','observed','predicted','train_mean'])
        if not required<=set(d):raise ValueError('Prediction schema missing: '+str(required-set(d)))
        for c in ('observed','predicted','train_mean'):d[c]=pd.to_numeric(d[c],errors='raise')
        if not np.isfinite(d[['observed','predicted','train_mean']].to_numpy(float)).all():raise ValueError('Nonfinite prediction input')
        for key,g in d.groupby(KEY+['fold'],sort=False,dropna=False):
            meta=dict(zip(KEY+['fold'],key));stats=fold_stats(g.observed.to_numpy(),g.predicted.to_numpy(),g.train_mean.to_numpy(),g.sample_id.to_numpy())
            stats.update(meta);stats['_ids']=g.sample_id.tolist();stats['_y']=g.observed.tolist();store[tuple(meta[k] for k in KEY)].append(stats)
        if sha(p)!=before:raise ValueError('Prediction source changed')
        used.append(dict(source_index=len(used)+1,path=p.parent.name+'/'+p.name,sha256=before,rows=len(d)))
    summary=[];folds=[];paired=[]
    for key,rr in store.items():
        meta=dict(zip(KEY,key));rr=sorted(rr,key=lambda r:r['fold'])
        if len({r['fold'] for r in rr})!=len(rr):raise ValueError('Duplicate fold per procedure')
        ids=[v for r in rr for v in r['_ids']];y=[v for r in rr for v in r['_y']]
        if len(set(ids))!=len(ids):raise ValueError('Repeated test individual in one repeat')
        a=aggregate_folds(rr);a.update(meta);a['eval_ids_sha256']=fingerprint(ids);a['observations_sha256']=fingerprint(ids,y)
        a['partition_signature']=hashlib.sha256(json.dumps([(str(r['fold']),r['eval_ids_sha256']) for r in rr]).encode()).hexdigest();summary.append(a)
        folds +=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rr]
        if meta['model'] in ('full_ridge','mean'):continue
        rk=list(key);rk[KEY.index('model')]='full_ridge';rk=tuple(rk)
        if rk not in store:continue
        ref=store[rk];fm={r['fold']:r for r in rr};rm={r['fold']:r for r in ref}
        if set(fm)!=set(rm) or any(fm[f]['observations_sha256']!=rm[f]['observations_sha256'] for f in fm):
            paired.append(dict(**meta,status='DIFFERENT_EVALUATION_SUPPORT',reference_model='full_ridge'));continue
        common=[f for f in fm if fm[f]['fisher_eligible'] and rm[f]['fisher_eligible']]
        ar=fisher_summary([fm[f] for f in common]);br=fisher_summary([rm[f] for f in common]);b=aggregate_folds(ref)
        paired.append(dict(**meta,status='PAIRED',reference_model='full_ridge',n_predictions=a['n_predictions'],common_fisher_folds=len(common),common_fisher_n=ar['fisher_n'],model_common_fisher_r=ar['within_fisher_r'],reference_common_fisher_r=br['within_fisher_r'],delta_within_fisher_r=ar['within_fisher_r']-br['within_fisher_r'],delta_rmse=a['rmse']-b['rmse'],delta_mse_skill=a['mse_skill']-b['mse_skill']))
    repeats=[];df=pd.DataFrame(summary);cols=[k for k in KEY if k!='repeat']
    if not df.empty:
        for key,g in df.groupby(cols,sort=False,dropna=False):
            if g['repeat'].duplicated().any():raise ValueError('Duplicate repetition')
            row=dict(zip(cols,key));row.update(n_repeats=len(g),n_unique_partitions=g.partition_signature.nunique(),min_n=int(g.n_predictions.min()),max_n=int(g.n_predictions.max()),min_fisher_coverage=float(g.fisher_sample_coverage.min()),dispersion='Reused-sample partition variation, not an independent-sample confidence interval')
            for c in ('within_fisher_r','pooled_r','mse_skill','rmse','mae','r2'):
                x=g[c].dropna();row.update({c+'_mean':x.mean(),c+'_sd':x.std(ddof=1) if len(x)>1 else np.nan,c+'_min':x.min(),c+'_max':x.max(),c+'_n':len(x)})
            repeats.append(row)
    tsv(out/'prediction_by_repeat.tsv',summary);tsv(out/'prediction_repeat_summary.tsv',repeats);tsv(out/'fold_metrics.tsv.gz',folds);tsv(out/'paired_comparisons.tsv',paired);tsv(out/'source_files.tsv',used)
    js(out/'status.json',dict(state='COMPLETE',prediction_rows=sum(r['rows'] for r in used),summary_rows=len(summary),fold_rows=len(folds),paired_rows=len(paired),fits=0))
    return summary,folds,paired


def marker_stability(root,out):
    """Describe overlap across outer-training sets; no fixed assay selection."""
    groups=defaultdict(list);freq=[];rows=[]
    for p in sorted(Path(root).rglob('selected_markers.json')):
        sig=json.loads((p.parent/'signature.json').read_text());
        for r in json.loads(p.read_text()):
            if r['model'] in ('full_ridge','pc_ridge','mean'):continue
            key=(sig['dataset'],sig['scenario'],sig['scaling'],sig['scheme'],r['analysis'],r['trait'],r['model']);groups[key].append(set(r['marker_ids']))
    for key,sets in groups.items():
        val=[len(a&b)/len(a|b) if a|b else 1. for i,a in enumerate(sets) for b in sets[i+1:]]
        common=set.intersection(*sets);union=set.union(*sets)
        m=dict(zip(['dataset','scenario','scaling','scheme','analysis','trait','model'],key))
        rows.append(dict(**m,n_training_sets=len(sets),n_unique_sets=len({tuple(sorted(x)) for x in sets}),mean_jaccard=float(np.mean(val)) if val else None,n_intersection=len(common),n_union=len(union),min_panel_size=min(map(len,sets)),max_panel_size=max(map(len,sets))))
        freq +=[dict(**m,marker_id=k,selection_count=sum(k in x for x in sets),n_sets=len(sets)) for k in sorted(union)]
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tsv(out/'marker_stability.tsv',rows);tsv(out/'marker_selection_frequency.tsv.gz',freq);return rows
