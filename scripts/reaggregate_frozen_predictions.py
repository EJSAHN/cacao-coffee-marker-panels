#!/usr/bin/env python3
"""Reaggregate frozen full-ridge predictions without model fitting.

Requirements: Python 3.10+, numpy, pandas.
Example:
  python reaggregate_frozen_predictions.py --input-dir . --output-dir recomputed

The input directory must contain the two supplied prediction tables:
  R04_full_ridge_OOF_with_groups.tsv.gz
  R10_full_ridge_OOF_with_groups.tsv.gz

No source archive, model code, genotype matrix, or external connection is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

def group_stat(f):
    y=f.y_true.to_numpy(float);p=f.y_pred.to_numpy(float);b=f.train_mean.to_numpy(float)
    n=len(f);ym=y.mean();pm=p.mean();a=y-ym;c=p-pm;sy=a@a;sp=c@c;cross=a@c
    status='ok'
    if n<2:status='too_few'
    elif np.ptp(y)==0:status='constant_observed'
    elif np.ptp(p)==0:status='constant_predicted'
    elif sy==0 or sp==0:status='unresolved_variance'
    elif math.sqrt(sy)<1e-13*abs(ym) or math.sqrt(sp)<1e-13*abs(pm):status='numerically_unresolved'
    r=float(np.clip(cross/math.sqrt(sy*sp),-1,1)) if status=='ok' else np.nan
    eligible=n>3 and status=='ok'
    return {'n':n,'r':r,'status':status,'fisher_eligible':eligible,'fisher_weight':n-3 if eligible else 0,'z':np.arctanh(np.clip(r,-1+1e-12,1-1e-12)) if eligible else np.nan,'y_mean':ym,'pred_mean':pm,'ss_y':sy,'ss_pred':sp,'cross_product':cross,'sse':np.sum((y-p)**2),'baseline_sse':np.sum((y-b)**2),'pred_obs_sd_ratio':math.sqrt(sp/sy) if sy>0 else np.nan}

def aggregate(f,grouping):
    rows=[]
    col='fold' if grouping=='outer_fold' else 'fixed_cluster'
    for key,g in f.groupby(col,sort=True):rows.append({'group':str(key),**group_stat(g)})
    eligible=[x for x in rows if x['fisher_eligible']]
    w=sum(x['fisher_weight'] for x in eligible);n=sum(x['n'] for x in rows)
    y=f.y_true.to_numpy(float);p=f.y_pred.to_numpy(float);b=f.train_mean.to_numpy(float)
    ym=y.mean();pm=p.mean();sy=np.sum((y-ym)**2);sp=np.sum((p-pm)**2);den=math.sqrt(sy*sp)
    within=sum(x['cross_product'] for x in rows);between=sum(x['n']*(x['y_mean']-ym)*(x['pred_mean']-pm) for x in rows)
    bs=np.sum((y-b)**2);sse=np.sum((y-p)**2)
    return rows,{'within_fisher_r':math.tanh(sum(x['fisher_weight']*x['z'] for x in eligible)/w) if w else np.nan,'fisher_weight':w,'eligible_groups':len(eligible),'n_groups':len(rows),'n_predictions':n,'fisher_n':sum(x['n'] for x in eligible),'sample_coverage':sum(x['n'] for x in eligible)/n,'group_coverage':len(eligible)/len(rows),'min_group_n':min(x['n'] for x in rows),'max_group_n':max(x['n'] for x in rows),'mse_skill':1-sse/bs if bs>0 else np.nan,'rmse':math.sqrt(sse/n),'pooled_r':float(np.sum((y-ym)*(p-pm))/den) if den else np.nan,'within_covariance_component':within/den if den else np.nan,'between_covariance_component':between/den if den else np.nan,'sse':sse,'baseline_sse':bs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check-against', type=Path, default=None,
                        help='Optional supplied grouped_metrics_by_repeat.tsv to verify')
    args = parser.parse_args()
    inputs = [args.input_dir / (stage + '_full_ridge_OOF_with_groups.tsv.gz')
              for stage in ['R04', 'R10']]
    required = {'source_stage','dataset','scheme','repeat','fold','trait','model',
                'sample_id','y_true','y_pred','train_mean','fixed_cluster'}
    frames = []
    manifest = []
    for path in inputs:
        frame = pd.read_csv(path, sep='\t', compression='gzip',
                            dtype={'sample_id':str, 'fixed_cluster':str},
                            float_precision='round_trip')
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
        if frame[list(required)].isna().any().any():
            raise ValueError(f'{path.name}: missing required values')
        if set(frame.model) != {'full_ridge'}:
            raise ValueError(f'{path.name}: expected only full_ridge predictions')
        if not np.isfinite(frame[['y_true','y_pred','train_mean']].to_numpy(float)).all():
            raise ValueError(f'{path.name}: nonfinite numerical input')
        frames.append(frame)
        manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'prediction_rows':len(frame)})
    predictions = pd.concat(frames, ignore_index=True)
    unit = ['source_stage','dataset','scheme','repeat','trait','model']
    if predictions.duplicated(unit + ['sample_id']).any():
        raise ValueError('A sample has more than one OOF prediction within a unit')
    if predictions.groupby(['dataset','sample_id']).fixed_cluster.nunique().gt(1).any():
        raise ValueError('Fixed cluster labels are inconsistent across predictions')
    # Common reference scope is defined from the primary eligible cluster-out
    # test IDs for the two Agrosavia representations, not from prediction values.
    main = predictions[predictions.source_stage.eq('R04') & predictions.scheme.eq('cluster_out')]
    common = (set(main.loc[main.dataset.eq('agrosavia_matina'),'sample_id']) &
              set(main.loc[main.dataset.eq('agrosavia_criollo'),'sample_id']))
    if not common:
        raise ValueError('Cannot derive common Agrosavia test IDs from the supplied tables')
    groups = []; repeats = []
    for keys,frame in predictions.groupby(unit,sort=False):
        meta = dict(zip(unit, keys))
        scopes = {'all_retained_tests':frame}
        if meta['dataset'].startswith('agrosavia_'):
            scopes['common_reference_test_accessions'] = frame[frame.sample_id.isin(common)]
        for scope, subset in scopes.items():
            if subset.empty:
                raise ValueError(f'Empty evaluation scope: {meta}, {scope}')
            for grouping in ['outer_fold','fixed_cluster']:
                rows, summary = aggregate(subset, grouping)
                prefix = {**meta,'evaluation_scope':scope,'grouping':grouping}
                repeats.append({**prefix, **summary})
                groups.extend({**prefix, **row} for row in rows)
    repeat_table = pd.DataFrame(repeats)
    summary_rows = []
    summary_keys = ['source_stage','dataset','scheme','trait','model','evaluation_scope','grouping']
    metrics = ['within_fisher_r','mse_skill','rmse','pooled_r',
               'within_covariance_component','between_covariance_component']
    for key,frame in repeat_table.groupby(summary_keys,sort=False):
        row = dict(zip(summary_keys,key))
        row.update(n_repeats=len(frame),n_predictions_min=int(frame.n_predictions.min()),
                   n_predictions_max=int(frame.n_predictions.max()),
                   sample_coverage_min=frame.sample_coverage.min(),
                   group_coverage_min=frame.group_coverage.min())
        for metric in metrics:
            row.update({metric+'_mean':frame[metric].mean(),
                        metric+'_sd':frame[metric].std(ddof=1),
                        metric+'_min':frame[metric].min(),metric+'_max':frame[metric].max()})
        summary_rows.append(row)
    check = None
    if args.check_against is not None:
        expected = pd.read_csv(args.check_against,sep='\t',float_precision='round_trip')
        match_keys = unit + ['evaluation_scope','grouping']
        a = repeat_table.set_index(match_keys).sort_index()
        b = expected.set_index(match_keys).sort_index()
        if not a.index.equals(b.index) or set(a.columns) != set(b.columns):
            raise AssertionError('Expected output keys or columns differ')
        max_differences = {}
        for col in a.columns:
            av = a[col].to_numpy(float); bv = b[col].to_numpy(float)
            if not np.allclose(av,bv,rtol=1e-12,atol=1e-12,equal_nan=True):
                raise AssertionError(f'Output differs: {col}')
            valid = np.isfinite(av) & np.isfinite(bv)
            max_differences[col] = float(np.max(np.abs(av[valid]-bv[valid]))) if valid.any() else None
        check = {'status':'pass','rows':len(a),'max_absolute_differences':max_differences}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,table in [('grouped_metrics_by_group',pd.DataFrame(groups)),
                       ('grouped_metrics_by_repeat',repeat_table),
                       ('grouped_metrics_summary',pd.DataFrame(summary_rows))]:
        table.to_csv(args.output_dir/(name+'.tsv'),sep='\t',index=False,float_format='%.17g')
    report = {'model_fitting':False,'inputs':manifest,'prediction_rows':len(predictions),
              'common_agrosavia_test_accessions':len(common),'repeat_rows':len(repeat_table),
              'group_rows':len(groups),'verification':check,
              'notes':['Correlations use Fisher-z weights n-3 for eligible groups with n>3.',
                       'Numerically unresolved correlations are excluded from r, not from MSE skill.',
                       'MSE skill uses every stored test prediction and its stored training mean.',
                       'Random within-cluster predictions come from multiple outer models.',
                       'Between/within covariance components are descriptive, not causal components.',
                       'Repeat SD is partition dispersion, not an independent-sample standard error.']}
    (args.output_dir/'reaggregation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Completed: {len(predictions):,} stored predictions; no fitting. Outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
