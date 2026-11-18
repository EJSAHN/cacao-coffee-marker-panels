"""Metrics on frozen out-of-fold predictions. No estimators and no fitting.

Within-fold Fisher summaries are descriptive, not inverse-variance meta-analysis
inference: folds reuse training samples and accessions may be genetically related.
"""
from __future__ import annotations
import hashlib, json, math
import numpy as np

VERSION='0.2.1'
FISHER_CLIP=1e-12
SMALL_SPREAD_RATIO=1e-3
NUMERIC_SPREAD_RATIO=1e-10

def fingerprint(ids, values=None):
    ids=np.asarray(ids,dtype=str)
    order=np.argsort(ids,kind='stable')
    if len(set(ids))!=len(ids): raise ValueError('Duplicate sample ID within evaluation unit')
    rows=[]
    for i in order:
        rows.append([str(ids[i])]+([] if values is None else [float(values[i]).hex()]))
    return hashlib.sha256(json.dumps(rows,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def pearson(x,y):
    """Return Pearson r and status. Constant predictions are undefined, never zero."""
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if x.ndim!=1 or y.shape!=x.shape: raise ValueError('Mismatched paired arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite predictions/targets')
    if len(x)<2:return float('nan'),'too_few'
    if np.max(x)==np.min(x):return float('nan'),'constant_observed'
    if np.max(y)==np.min(y):return float('nan'),'constant_predicted'
    a=x-x.mean();b=y-y.mean(); na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    if na==0 or nb==0:return float('nan'),'unresolved_variance'
    r=float(np.dot(a/na,b/nb))
    return min(1.,max(-1.,r)),'ok'

def ranks(x):
    """Average ranks for ties; stable, no scipy dependency."""
    x=np.asarray(x,float); idx=np.argsort(x,kind='stable');v=x[idx]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];ends=np.r_[starts[1:],len(v)]
    out=np.empty(len(v),float)
    for a,b in zip(starts,ends):out[idx[a:b]]=(a+b-1)/2+1
    return out

def fold_stats(y,p,b,ids):
    y=np.asarray(y,float);p=np.asarray(p,float);b=np.asarray(b,float)
    if y.ndim!=1 or y.shape!=p.shape or y.shape!=b.shape or len(y)!=len(ids) or not len(y):raise ValueError('Invalid evaluation arrays')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(b).all():raise ValueError('Nonfinite values: not silently dropped')
    # The baseline must be the stored mean of the corresponding training sample.
    if not np.allclose(b,b[0],rtol=0,atol=1e-12*max(1,abs(float(b[0])))):raise ValueError('Training mean varies within a fold/trait')
    fingerprint(ids)  # Reject duplicate IDs even when the caller does not use hashes.
    n=len(y);ym=float(y.mean());pm=float(p.mean());bm=float(b[0]);a=y-ym;c=p-pm
    sy=float(a@a);sp=float(c@c);cross=float(a@c)
    se=float(np.sum((y-p)**2)); sb=float(np.sum((y-b)**2))
    r,status=pearson(y,p);rho,_=pearson(ranks(y),ranks(p))
    sd_y=math.sqrt(sy/(n-1)) if n>1 else float('nan');sd_p=math.sqrt(sp/(n-1)) if n>1 else float('nan')
    ratio=sd_p/sd_y if sd_y>0 else float('nan')
    near_p=((math.sqrt(sp)<1e-13*abs(pm)) or (np.isfinite(ratio) and ratio<NUMERIC_SPREAD_RATIO)) if sp>0 else False
    near_y=(math.sqrt(sy)<1e-13*abs(ym)) if sy>0 else False
    if (near_p or near_y) and status == 'ok':
        r = float('nan')
        rho = float('nan')
        status = 'numerically_unresolved'
    fisher_ok=n>3 and status=='ok'
    clipped=fisher_ok and abs(r)>1-FISHER_CLIP
    return dict(n=n,obs_mean=ym,pred_mean=pm,train_mean=bm,ss_y=sy,ss_pred=sp,cross_product=cross,
        sse=se,baseline_sse=sb,sae=float(np.sum(abs(y-p))),pearson_r=r,pearson_status=status,
        spearman_r=rho,obs_sd=sd_y,pred_sd=sd_p,pred_obs_sd_ratio=ratio,
        small_prediction_spread=bool(np.isfinite(ratio) and ratio<SMALL_SPREAD_RATIO),
        numeric_near_constant=bool(near_p or near_y),fisher_eligible=fisher_ok,
        fisher_exclusion=('' if fisher_ok else ('numeric_near_constant' if status == 'numerically_unresolved' else ('n_le_3' if n<=3 else status))),
        fisher_weight=(n-3 if fisher_ok else 0),
        fisher_z=(float(np.arctanh(np.clip(r,-1+FISHER_CLIP,1-FISHER_CLIP))) if fisher_ok else float('nan')),
        fisher_clipped=bool(clipped),rmse=math.sqrt(se/n),mae=float(np.mean(abs(y-p))),
        mse_skill=(1-se/sb if sb>0 else float('nan')),r2=(1-se/sy if sy>0 else float('nan')),
        eval_ids_sha256=fingerprint(ids),observations_sha256=fingerprint(ids,y))

def fisher_summary(rows):
    if any(not isinstance(r['fisher_eligible'], (bool, np.bool_)) for r in rows):
        raise TypeError('fisher_eligible must be boolean, not a serialized string')
    eligible=[r for r in rows if r['fisher_eligible']]
    if not eligible:return dict(within_fisher_r=float('nan'),fisher_folds=0,fisher_n=0,fisher_weight=0,
        fold_r_nweighted=float('nan'),fold_r_equal=float('nan'),fold_r_min=float('nan'),fold_r_max=float('nan'),fisher_clipped_folds=0)
    w=np.array([r['fisher_weight'] for r in eligible],float);rr=np.array([r['pearson_r'] for r in eligible],float)
    ns=np.array([r['n'] for r in eligible],float);zz=np.array([r['fisher_z'] for r in eligible],float)
    return dict(within_fisher_r=float(np.tanh(np.sum(w*zz)/sum(w))),fisher_folds=len(eligible),fisher_n=int(sum(ns)),fisher_weight=int(sum(w)),
        fold_r_nweighted=float(np.average(rr,weights=ns)),fold_r_equal=float(np.mean(rr)),
        fold_r_min=float(min(rr)),fold_r_max=float(max(rr)),fisher_clipped_folds=sum(bool(r['fisher_clipped']) for r in eligible))

def aggregate_folds(rows):
    """Pooled/within/between summaries, per single repeat and fixed procedure."""
    if not rows:raise ValueError('No evaluable folds')
    n=sum(int(r['n']) for r in rows);ym=sum(r['n']*r['obs_mean'] for r in rows)/n
    pm=sum(r['n']*r['pred_mean'] for r in rows)/n;bm=sum(r['n']*r['train_mean'] for r in rows)/n
    wy=sum(r['ss_y'] for r in rows);wp=sum(r['ss_pred'] for r in rows);wc=sum(r['cross_product'] for r in rows)
    by=sum(r['n']*(r['obs_mean']-ym)**2 for r in rows);bp=sum(r['n']*(r['pred_mean']-pm)**2 for r in rows)
    bc=sum(r['n']*(r['obs_mean']-ym)*(r['pred_mean']-pm) for r in rows)
    bb=sum(r['n']*(r['train_mean']-bm)**2 for r in rows)
    yb=sum(r['n']*(r['obs_mean']-ym)*(r['train_mean']-bm) for r in rows)
    sy=wy+by;sp=wp+bp;den=math.sqrt(sy*sp)
    sse=sum(r['sse'] for r in rows);base=sum(r['baseline_sse'] for r in rows)
    out=fisher_summary(rows)
    out.update(n_predictions=n,n_folds=len(rows),n_constant_pred_folds=sum(r['pearson_status']=='constant_predicted' for r in rows),
        n_undefined_folds=sum(r['pearson_status']!='ok' for r in rows),
        n_near_constant_folds=sum(bool(r['numeric_near_constant']) for r in rows),
        n_small_spread_folds=sum(bool(r['small_prediction_spread']) for r in rows),
        fisher_sample_coverage=out['fisher_n']/n,fisher_fold_coverage=out['fisher_folds']/len(rows),min_fold_n=min(r['n'] for r in rows),max_fold_n=max(r['n'] for r in rows),
        pooled_r=(wc+bc)/den if den else float('nan'),
        within_covariance_component=wc/den if den else float('nan'),
        between_covariance_component=bc/den if den else float('nan'),
        within_cross_product=wc,between_cross_product=bc,
        within_centered_r=wc/math.sqrt(wy*wp) if wy>0 and wp>0 else float('nan'),
        pooled_baseline_r=yb/math.sqrt(sy*bb) if sy>0 and bb>0 else float('nan'),
        baseline_within_r_status='undefined_constant_training_mean',
        rmse=math.sqrt(sse/n),mae=sum(r['sae'] for r in rows)/n,
        mse_skill=1-sse/base if base>0 else float('nan'),
        baseline_rmse=math.sqrt(base/n),r2=1-sse/sy if sy>0 else float('nan'),
        sse=sse,baseline_sse=base,ss_y_total=sy,
        sd_ratio_min=min((r['pred_obs_sd_ratio'] for r in rows if np.isfinite(r['pred_obs_sd_ratio'])),default=float('nan')),
        r_aggregation='tanh(weighted_mean(atanh(r_fold), n_fold-3)); descriptive; no IID CI')
    out['fisher_coverage_status']='complete' if out['fisher_folds']==len(rows) else ('none' if out['fisher_folds']==0 else 'partial')
    return out


def summarize_file(input_path, output_dir, group_columns):
    """Summarize a long TSV of stored predictions; never fit or select models.

    Rows of each evaluation group must be contiguous. Repeat belongs in the
    group key, not in the fold key. Test sample IDs must be unique per group.
    Each fold has one stored training mean. Invalid data cause failure, not
    silent omission. Outputs must not already exist.
    """
    import csv
    import gzip
    import itertools
    from pathlib import Path

    source = Path(input_path).resolve()
    dest = Path(output_dir).resolve()
    groups = list(group_columns)
    fixed = ['fold', 'sample_id', 'observed', 'predicted', 'train_mean']
    if not groups or len(set(groups)) != len(groups) or set(groups) & set(fixed):
        raise ValueError('Group columns must be nonempty, unique, and distinct from value columns')
    if not source.is_file():
        raise FileNotFoundError(source)
    if dest.exists():
        raise FileExistsError('Output directory already exists; use a new directory')
    def file_hash(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    before = file_hash(source)
    dest.mkdir(parents=True)
    marker = dest / 'INCOMPLETE.txt'
    marker.write_text('No completed summary has been certified.\n', encoding='utf-8')
    op = gzip.open if source.suffix == '.gz' else open
    n_groups = 0
    n_rows = 0
    seen = set()
    def cleaned(row):
        return {k: ('' if isinstance(v, (float, np.floating)) and not math.isfinite(v) else v)
                for k, v in row.items()}
    with op(source, 'rt', encoding='utf-8-sig', newline='') as stream, \
         (dest / 'fold_metrics.tsv').open('w', encoding='utf-8', newline='') as fo, \
         (dest / 'summary.tsv').open('w', encoding='utf-8', newline='') as so:
        reader = csv.DictReader(stream, delimiter='\t')
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ValueError('Duplicate column name')
        missing = set(groups + fixed) - set(headers)
        if missing:
            raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
        fw = sw = None
        for key, records in itertools.groupby(reader, lambda r: tuple(r[c] for c in groups)):
            if key in seen:
                raise ValueError('Noncontiguous evaluation group; sort input by group columns first')
            if any(v is None or v == '' for v in key):
                raise ValueError('Empty evaluation group identifier')
            seen.add(key)
            meta = dict(zip(groups, key))
            folds = {}
            all_ids = set()
            for record in records:
                fid = record['fold']
                sid = record['sample_id']
                if not fid or not sid or sid in all_ids:
                    raise ValueError('Empty fold/sample ID or repeated sample within an evaluation group')
                all_ids.add(sid)
                folds.setdefault(fid, []).append(record)
                n_rows += 1
            stats = []
            for fid in sorted(folds):
                r = folds[fid]
                st = fold_stats([float(x['observed']) for x in r],
                                [float(x['predicted']) for x in r],
                                [float(x['train_mean']) for x in r],
                                [x['sample_id'] for x in r])
                stats.append(st)
                row = {**meta, 'fold': fid, **st}
                if fw is None:
                    if set(meta) & set(st):
                        raise ValueError('Group column name collides with a metric name')
                    fw = csv.DictWriter(fo, fieldnames=list(row), delimiter='\t', lineterminator='\n')
                    fw.writeheader()
                fw.writerow(cleaned(row))
            summary = aggregate_folds(stats)
            if set(meta) & set(summary):
                raise ValueError('Group column name collides with a summary name')
            row = {**meta, **summary}
            if sw is None:
                sw = csv.DictWriter(so, fieldnames=list(row), delimiter='\t', lineterminator='\n')
                sw.writeheader()
            sw.writerow(cleaned(row))
            n_groups += 1
    if not n_groups:
        raise ValueError('No prediction rows')
    after = file_hash(source)
    if after != before:
        raise RuntimeError('Input changed during reporting')
    result = dict(status='COMPLETE', reporting_version=VERSION,
                  input_sha256=before, groups=n_groups, prediction_rows=n_rows,
                  group_columns=groups, model_fitting=False,
                  output_sha256={n: file_hash(dest / n)
                                 for n in ['fold_metrics.tsv', 'summary.tsv']})
    (dest / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    marker.unlink()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Summarize stored predictions without fitting models')
    parser.add_argument('--input', required=True, help='Long-format TSV or TSV.gz; see docs/PREDICTION_REPORTING.md')
    parser.add_argument('--output', required=True, help='New output directory; existing directories are refused')
    parser.add_argument('--group-columns', nargs='+', default=['dataset', 'scheme', 'repeat', 'trait', 'model'])
    args = parser.parse_args()
    result = summarize_file(args.input, args.output, args.group_columns)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
