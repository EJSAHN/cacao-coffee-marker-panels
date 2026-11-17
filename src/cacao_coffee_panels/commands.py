"""Portable command routing for preserved analyses and result export."""
from pathlib import Path
import argparse,json,sys
from .io import read_json,js
from .prepare import prepare_raw,make_design
from .workflow import run_prediction,run_variance,run_sensitivity,run_size_match,run_tuning
from .study_reporting import build_summary,marker_stability
from .exports import collect_results,import_matched,import_screens
from .workbook import export_tables
from .comparisons import common_support

COMMANDS={'prepare','design','import-matched','predict-study','variance-study','qc-sensitivity','size-matched','tuning-diagnostic','summarize-study','marker-stability','collect-results','export-workbook','compare-predictions','import-screens'}

def main(argv=None):
    parser=argparse.ArgumentParser(description='Genomic prediction, marker panels, and transparent result reporting')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--sources',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('design');p.add_argument('--matrices',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--datasets',nargs='+')
    p=sub.add_parser('import-matched');p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    for command in ('predict-study','variance-study','qc-sensitivity','size-matched','tuning-diagnostic'):
        p=sub.add_parser(command);p.add_argument('--input',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
        if command!='variance-study':p.add_argument('--resume',action='store_true');p.add_argument('--max-tasks',type=int)
        if command in ('predict-study','variance-study'):p.add_argument('--datasets',nargs='+')
        if command in ('qc-sensitivity','size-matched','tuning-diagnostic'):p.add_argument('--primary',required=True)
        if command=='tuning-diagnostic':p.add_argument('--size-results',required=True)
        if command=='predict-study':
            p.add_argument('--scaling',choices=['center_only','allele_frequency'],nargs='+');p.add_argument('--skip-models',action='store_true');p.add_argument('--skip-expanded',action='store_true');p.add_argument('--recorded-screens')
    p=sub.add_parser('summarize-study');p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('marker-stability');p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('collect-results');p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);p.add_argument('--no-workbook',action='store_true')
    p=sub.add_parser('export-workbook');p.add_argument('--catalog',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('import-screens');p.add_argument('--input',required=True);p.add_argument('--matched',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('compare-predictions');p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    if hasattr(args,'max_tasks') and args.max_tasks is not None and args.max_tasks<1:raise ValueError('max-tasks must be positive')
    # No output may be nested in an input source; preserve author-supplied data.
    target=Path(args.out).resolve()
    for k in ('input','matrices','primary','size_results'):
        if hasattr(args,k):
            parent=Path(getattr(args,k)).resolve()
            if target==parent or parent in target.parents:raise ValueError('Output must be outside all input directories')
    cfg=read_json(args.config) if hasattr(args,'config') else None
    if args.command=='prepare':result=prepare_raw(args.sources,args.out)
    elif args.command=='design':result=make_design(args.matrices,args.config,args.out,args.datasets)
    elif args.command=='import-screens':result=import_screens(args.input,args.matched,args.out)
    elif args.command=='import-matched':result=import_matched(args.input,args.out)
    elif args.command=='predict-study':result=run_prediction(args.input,cfg,args.out,args.resume,args.max_tasks,args.datasets,args.scaling,not args.skip_models,not args.skip_expanded,args.recorded_screens)
    elif args.command=='variance-study':result=run_variance(args.input,cfg,args.out,args.datasets)
    elif args.command=='qc-sensitivity':result=run_sensitivity(args.input,args.primary,cfg,args.out,args.resume,args.max_tasks)
    elif args.command=='size-matched':result=run_size_match(args.input,args.primary,cfg,args.out,args.resume,args.max_tasks)
    elif args.command=='tuning-diagnostic':result=run_tuning(args.input,args.primary,args.size_results,cfg,args.out,args.resume,args.max_tasks)
    elif args.command=='summarize-study':
        files=sorted(Path(args.input).rglob('predictions.tsv.gz'))
        if not files:raise ValueError('No stored predictions')
        result=build_summary(files,args.out)
    elif args.command=='marker-stability':result=marker_stability(args.input,args.out)
    elif args.command=='compare-predictions':result=common_support(args.left,args.right,args.out)
    elif args.command=='collect-results':result=collect_results(args.catalog,args.out,not args.no_workbook)
    else:
        catalog=Path(args.catalog).resolve();data=read_json(catalog);result=export_tables([(r['label'],catalog.parent/r['path']) for r in data['tables']],args.out,data.get('title','Cacao and coffee analysis results'))
    state='PARTIAL_LIMITED_TEST' if getattr(args,'max_tasks',None) else 'COMPLETE'
    if target.is_dir():
        js(target/'execution.json',dict(state=state,command=args.command,arguments=vars(args),new_fitting=args.command in ('predict-study','variance-study','qc-sensitivity','size-matched','tuning-diagnostic'),completed_tasks=result if isinstance(result,int) else None))
    print(state+': '+str(target));return 0
