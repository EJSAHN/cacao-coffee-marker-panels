#!/usr/bin/env python
"""Ordered execution from recorded matched inputs. No installation or downloads."""
from __future__ import annotations
import argparse,os,subprocess,sys,json,hashlib,shutil
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cacao_coffee_panels.io import sha,js,read_json,manifest,verify

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--recorded-screens');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--steps',nargs='+',choices=['prediction','variance','qc','size','tuning','report','stability','export'])
    a=p.parse_args(argv);src=Path(a.input).resolve();cfg=Path(a.config).resolve();out=Path(a.out).resolve()
    if not src.is_dir() or not cfg.is_file():raise FileNotFoundError('Matched input directory or configuration missing')
    if out==src or src in out.parents:raise ValueError('Output must be outside matched inputs')
    def command(name,*opts):return [sys.executable,'-B',str(ROOT/'run.py'),name,*map(str,opts)]
    pred=command('predict-study','--input',src,'--config',cfg,'--out',out/'prediction')
    if a.recorded_screens:pred+=['--recorded-screens',str(Path(a.recorded_screens).resolve())]
    if a.resume:pred+=['--resume']
    steps=[('prediction',pred),('variance',command('variance-study','--input',src,'--config',cfg,'--out',out/'variance')),('qc',command('qc-sensitivity','--input',src,'--config',cfg,'--primary',out/'prediction','--out',out/'qc')),('size',command('size-matched','--input',src,'--config',cfg,'--primary',out/'prediction','--out',out/'size')),('tuning',command('tuning-diagnostic','--input',src,'--config',cfg,'--primary',out/'prediction','--size-results',out/'size','--out',out/'tuning')),('report',command('summarize-study','--input',out,'--out',out/'reporting')),('stability',command('marker-stability','--input',out/'prediction','--out',out/'stability'))]
    # Summaries are a derived view of stored outputs and may reside beside them.
    # run.py's input/output guard is intentionally stricter, so use a sibling staging path.
    stage=out.parent/(out.name+'_reporting_stage')
    steps[5]=('report',command('summarize-study','--input',out,'--out',stage))
    selected=set(a.steps or ['prediction','variance','qc','size','tuning','report','stability','export'])
    steps=[(name,c) for name,c in steps if name in selected]
    if a.dry_run:
        print(json.dumps(dict(steps=steps,export='export' in selected),indent=2));return 0
    code=hashlib.sha256(''.join(sha(f) for f in sorted((ROOT/'src').rglob('*.py'))).encode()).hexdigest()
    contract=dict(config_sha256=sha(cfg),code_sha256=code,inputs={f.relative_to(src).as_posix():sha(f) for f in sorted(src.rglob('*')) if f.is_file()},screen_index_sha256=sha(a.recorded_screens) if a.recorded_screens else None)
    if out.exists():
        if not a.resume or not (out/'contract.json').is_file() or read_json(out/'contract.json')!=contract:raise ValueError('Existing output or changed run contract')
    else:out.mkdir(parents=True);js(out/'contract.json',contract)
    try:
        for name,cmd in steps:
            mark=out/'phase_checks'/name
            if mark.exists():
                info=read_json(mark/'output.json');verify(Path(info['directory']),'phase_manifest.tsv');print('REUSE '+name,flush=True);continue
            dest=Path(cmd[cmd.index('--out')+1]);print('RUN '+name,flush=True)
            if a.resume and name in ('qc','size','tuning') and dest.exists():cmd+=['--resume']
            result=subprocess.run(cmd,check=False)
            if result.returncode:raise RuntimeError(name+' failed; previous outputs retained')
            if not dest.is_dir():raise RuntimeError('Missing phase output '+name)
            if name=='report':
                final=out/'reporting'
                if final.exists():raise FileExistsError(final)
                shutil.move(str(dest),str(final));dest=final
            manifest(dest,'phase_manifest.tsv');mark.mkdir(parents=True);js(mark/'output.json',dict(directory=str(dest),manifest_sha256=sha(dest/'phase_manifest.tsv')))
        if 'export' in selected:
            tablelist=[]
            for label,file in [('Prediction_summary',out/'reporting'/'prediction_repeat_summary.tsv'),('Predictions_by_repeat',out/'reporting'/'prediction_by_repeat.tsv'),('Paired_model_comparisons',out/'reporting'/'paired_comparisons.tsv'),('Variance_components',out/'variance/variance_components.tsv'),('Marker_stability',out/'stability/marker_stability.tsv')]:
                if not file.is_file():raise FileNotFoundError('Required export source '+str(file))
                tablelist.append(dict(label=label,path=str(file),filename=file.name,sha256=sha(file),workbook=True))
            catalog=out/'result_catalog.json';js(catalog,dict(title='Cacao and coffee analysis results',tables=tablelist))
            export=out/'export'
            if not export.exists():subprocess.run(command('collect-results','--catalog',catalog,'--out',export),check=True)
        js(out/'status.json',dict(state='SELECTED_PHASES_COMPLETE',completed_phases=[n for n,_ in steps],export='export' in selected));print('COMPLETE: '+str(out));return 0
    except BaseException as exc:
        js(out/'status.json',dict(state='INCOMPLETE',error=str(exc)));raise

if __name__=='__main__':raise SystemExit(main())
