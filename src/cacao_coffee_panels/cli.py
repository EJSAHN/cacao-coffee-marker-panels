"""Explicit matched-input commands; no download, implicit installation or source mutation."""
from pathlib import Path
import argparse
import sys
import json
import numpy as np
from .data import Cohort
from .io import js, tsv, read_json, manifest
from .cohorts import fit_frozen_inner
from .models import make_split_cache, fit_trait
from .ridge import statistics
from .variance import marker_data, scaled_matrix, normalized_kernel, full_pcs, ContrastModel


def output_directory(path):
    p = Path(path)
    if p.exists():
        raise ValueError('Output path already exists; choose a new directory')
    p.mkdir(parents=True)
    return p


def predict(cohort, design, cfg, analysis):
    tr, te, traits, Y = cohort.split(design)
    base = cfg['baseline'] if analysis == 'models' else cfg
    if len(tr) < base['outer_min_train'] or len(te) < base['outer_min_test']:
        raise ValueError('Outer split fails the configured sample gate')
    assignment = design['inner_assignments']
    if {str(r['sample_id']) for r in assignment} != set(cohort.sample_ids[tr]):
        raise ValueError('Inner assignments must cover training IDs, without test IDs')
    X = cohort.dosage
    frozen = fit_frozen_inner(X[tr], Y[tr], X[te], cohort.sample_ids[tr], assignment, base)
    if frozen['state'] != 'OK':
        raise ValueError('Not estimable: ' + frozen['reason'])
    predictions, metrics, tuning = [], [], list(frozen['tuning'])
    for model, rec in frozen['results'].items():
        for q, trait in enumerate(traits):
            yp = rec['predictions'][:, q]
            predictions.extend(dict(sample_id=str(cohort.sample_ids[i]), trait=trait, model=model, y_true=float(Y[i,q]), y_pred=float(p),train_mean=float(Y[tr,q].mean())) for i,p in zip(te,yp))
            metrics.append(dict(trait=trait, model=model, **statistics(Y[te,q],yp)))
    extra = {}
    if analysis == 'models':
        cache, reason = make_split_cache(X[tr], Y[tr], X[te], cohort.sample_ids[tr], assignment, cfg)
        if cache is None:
            raise ValueError('Not estimable: ' + reason)
        for q, trait in enumerate(traits):
            out = fit_trait(cache,q,cohort.marker_ids,cfg,int(design['seed']))
            extra[trait] = out
            tuning.extend(dict(trait=trait, **r) for r in out['tuning'])
            for model, rec in out['results'].items():
                yp = rec['predictions']
                predictions.extend(dict(sample_id=str(cohort.sample_ids[i]), trait=trait, model=model, y_true=float(Y[i,q]), y_pred=float(p),train_mean=float(Y[tr,q].mean())) for i,p in zip(te,yp))
                metrics.append(dict(trait=trait, model=model, **statistics(Y[te,q],yp)))
    return predictions, metrics, tuning, frozen, extra


def main(argv=None):
    from .commands import COMMANDS, main as study_main
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] in COMMANDS:
        return study_main(args_in)
    parser = argparse.ArgumentParser(description='Matched-input genomic prediction and variance components',epilog='Study commands: '+', '.join(sorted(COMMANDS))+'; use run.py COMMAND --help')
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify-input', help='Validate matched input dimensions, values and exact ID order')
    verify.add_argument('--data', required=True)
    fit = sub.add_parser('predict-fold', help='Evaluate a supplied outer split and stored inner assignments')
    for k in ('data','design','config','out'):
        fit.add_argument('--'+k, required=True)
    fit.add_argument('--analysis', choices=['ridge','models'], default='ridge')
    var = sub.add_parser('variance', help='Full-marker kernel-conditioned REML, not field heritability')
    for k in ('data','config','out'):
        var.add_argument('--'+k, required=True)
    var.add_argument('--scaling',choices=['center_only','allele_frequency'],default='center_only')
    args = parser.parse_args(argv)
    try:
        cohort = Cohort.load(args.data)
        if args.command == 'verify-input':
            print(json.dumps(dict(status='VALID',n_samples=len(cohort.sample_ids),n_markers=len(cohort.marker_ids),traits=list(cohort.phenotypes.columns[1:]),missing_fraction=float(np.mean(~np.isfinite(cohort.dosage)))),indent=2))
            return 0
        cfg = read_json(args.config)
        source_root = Path(args.data).resolve()
        target = Path(args.out).resolve()
        if target == source_root or source_root in target.parents:
            raise ValueError('Output must be outside the input cohort directory')
        out = output_directory(target)
        js(out/'configuration.json',cfg)
        if args.command == 'predict-fold':
            design = read_json(args.design)
            p,m,t,f,e = predict(cohort,design,cfg,args.analysis)
            js(out/'design.json',design)
            tsv(out/'predictions.tsv',p)
            tsv(out/'metrics.tsv',m)
            tsv(out/'tuning.tsv',t)
            js(out/'diagnostics.json',dict(preprocessing=f['preprocessing'],additional_models=e))
        else:
            md = marker_data(cohort.dosage,cfg)
            K,scale = normalized_kernel(scaled_matrix(md,args.scaling))
            Kcenter,_ = normalized_kernel(md['Z'])
            U,_ = full_pcs(Kcenter,cfg['pc_count'])
            results=[];profiles=[]
            for name,C in [('intercept',np.ones((len(K),1))),('intercept_pc5',np.c_[np.ones(len(K)),U])]:
                model=ContrastModel.build(K,C)
                for trait in cohort.phenotypes.columns[1:]:
                    row,profile=model.fit(cohort.phenotypes[trait].to_numpy(float),cfg)
                    results.append(dict(trait=trait,fixed_design=name,scaling=args.scaling,n_markers=len(md['indices']),**row))
                    profiles.extend(dict(trait=trait,fixed_design=name,**r) for r in profile)
            tsv(out/'variance_components.tsv',results);tsv(out/'profiles.tsv',profiles)
        js(out/'status.json',dict(state='COMPLETE',scope=args.command,version='2.0.0rc1'))
        manifest(out)
        print('COMPLETE: '+str(out))
        return 0
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
