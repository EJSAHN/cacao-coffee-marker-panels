"""Checkpointed study execution using the preserved numerical kernels.

Outer test targets are used only after training-only fitting and selection.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json, hashlib, time, platform, sys
import numpy as np
from .data import Cohort
from .io import read_json, read_tsv, tsv, js, sha, clean
from .ridge import nested_fold, seed_for, id_hash
from .panels import fit_extra, specs_for_all, choose_procedure
from .models import make_split_cache, fit_trait
from .model_replay import load_recorded_screen,fit_with_recorded_screen
from .cohorts import build_variants, fit_frozen_inner
from .variance import marker_data, scaled_matrix, normalized_kernel, full_pcs, ContrastModel, neutral_panels, kernel_diagnostics
from .size_match import matched_partitions, fit_full_ridge
from .tuning import design_list, inner_diagnostic, select_loss, outer_path, df_for_lambda
from . import reporting


def cohort_record(directory):
    directory=Path(directory).resolve();c=Cohort.load(directory)
    definition=read_json(directory/'definition.json') if (directory/'definition.json').is_file() else {}
    dataset=definition.get('dataset',directory.name.split('__')[0])
    name=definition.get('scenario',dataset+'__primary_q20')
    rows=read_tsv(directory/'outer_test_assignments.tsv');lookup={v:i for i,v in enumerate(c.sample_ids)};groups=defaultdict(list)
    for r in rows:
        if r['sample_id'] not in lookup:raise ValueError('Unknown sample in outer fold')
        groups[(r['scheme'],int(r['repeat']),int(r['fold']))].append(lookup[r['sample_id']])
    tasks=[];assigned=defaultdict(list)
    for (sch,rep,f),values in sorted(groups.items()):
        if len(values)!=len(set(values)):raise ValueError('Duplicate test sample')
        te=np.array(sorted(values),int);tr=np.setdiff1d(np.arange(len(c.sample_ids)),te)
        tasks.append((sch,rep,f,tr,te));assigned[sch,rep].extend(te)
    for k,v in assigned.items():
        if sorted(v)!=list(range(len(c.sample_ids))):raise ValueError('Incomplete outer partition '+str(k))
    return dict(name=name,dataset=dataset,reference_scenario=definition.get('reference_scenario',name),role=definition.get('role','primary'),
        ids=c.sample_ids,markers=c.marker_ids,X=c.dosage,Y=c.phenotypes.iloc[:,1:].to_numpy(float),traits=c.phenotypes.columns[1:].tolist(),
        clusters=c.samples['cluster'].to_numpy(str),components=c.samples['identity_component'].to_numpy(str),
        source_groups=c.markers['source_chrom'].fillna('').replace('','unresolved').to_numpy(str),tasks=tasks,directory=directory, input_hashes={p.name:sha(p) for p in sorted(directory.glob('*')) if p.is_file()})


def checkpoint_start(folder, signature, resume=False):
    folder=Path(folder)
    code_hash=hashlib.sha256(''.join(sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()
    sig=clean({**signature,'software_sha256':code_hash})
    if folder.exists():
        if not resume:raise FileExistsError('Output already exists: '+str(folder))
        if not (folder/'complete.json').is_file():raise ValueError('Incomplete checkpoint; choose a new output or remove ONLY this incomplete checkpoint: '+str(folder))
        if read_json(folder/'signature.json')!=sig:raise ValueError('Checkpoint inputs/configuration changed')
        for r in read_tsv(folder/'files.tsv'):
            p=folder/r['path']
            if not p.is_file() or sha(p)!=r['sha256']:raise ValueError('Checkpoint content changed')
        return False
    folder.mkdir(parents=True);js(folder/'signature.json',sig);return True


def checkpoint_finish(folder, state='COMPLETE'):
    folder=Path(folder)
    tsv(folder/'files.tsv',[dict(path=p.relative_to(folder).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('files.tsv','complete.json')])
    js(folder/'complete.json',dict(state=state))


def prediction_rows(s, task, results, cfg, analysis, scaling, chosen=None):
    sch,rep,f,tr,te=task;rows=[]
    for model,val in results.items():
        pred=np.asarray(val['predictions']);pred=pred[:,None] if pred.ndim==1 else pred
        if pred.shape!=(len(te),len(s['traits'])):raise ValueError('Prediction dimensions disagree')
        for q,trait in enumerate(s['traits']):
            winner=val.get('chosen_models',[model]*len(s['traits']))[q]
            for j,i in enumerate(te):rows.append(dict(analysis=analysis,dataset=s['dataset'],scenario=s['name'],scheme=sch,repeat=rep,fold=f,
                scaling=scaling,trait=trait,model=model,chosen_model=winner,sample_id=str(s['ids'][i]),observed=float(s['Y'][i,q]),predicted=float(pred[j,q]),train_mean=float(s['Y'][tr,q].mean())))
    return rows


def write_primary_fold(s,task,cfg,dest,scaling='center_only',expanded=True,models=True,resume=False,screen_index=None):
    sch,rep,f,tr,te=task;b={**cfg['baseline'],'scaling':scaling}
    sig=dict(dataset=s['dataset'],scenario=s['name'],scheme=sch,repeat=rep,fold=f,scaling=scaling,expanded=expanded,models=models,
             config=cfg,recorded_screen_sha256=sha(screen_index) if screen_index else None,train=id_hash(s['ids'][tr]),test=id_hash(s['ids'][te]),inputs=s['input_hashes'])
    if not checkpoint_start(dest,sig,resume):return
    if len(tr)<b['outer_min_train'] or len(te)<b['outer_min_test']:
        js(dest/'status.json',dict(status='NOT_ESTIMABLE',reason='outer_sample_gate',n_train=len(tr),n_test=len(te)));checkpoint_finish(dest,'NOT_ESTIMABLE');return
    groups=s['clusters'][tr] if sch=='cluster_out' else s['components'][tr]
    seed=seed_for(b['seed'],s['reference_scenario'],sch,rep,f)
    fit=nested_fold(s['X'][tr],s['Y'][tr],s['X'][te],s['ids'][tr],s['markers'],groups,sch,b,seed)
    if fit['state']!='OK':js(dest/'status.json',fit);checkpoint_finish(dest,'NOT_ESTIMABLE');return
    core=fit['results'];rows=prediction_rows(s,task,core,cfg,'ridge',scaling)
    selections=[];tuning=[{'analysis':'ridge',**r} for r in fit['tuning']];unavailable=list(fit['candidate_failures'])
    for model,val in core.items():
        for q,t in enumerate(s['traits']):
            selections.append(dict(analysis='ridge',model=model,trait=t,marker_ids=s['markers'][val['selected'][q]].tolist(),chosen_model=val.get('chosen_models',[model]*len(s['traits']))[q],lambda_value=val['lambdas'][q],inner_mse=val['inner_mse'][q]))
    if expanded and scaling=='center_only':
        more=fit_extra(s['X'][tr],s['Y'][tr],s['X'][te],s['ids'][tr],s['markers'],s['source_groups'],groups,sch,{**b,**cfg['panel']},seed)
        if more['state']!='OK':raise ValueError('Expanded evaluator disagrees on estimability')
        if fit['inner_assignments']!=more['inner_assignments']:raise ValueError('Inner folds differ across panel ranges')
        combined={**core,**more['results']};specs=specs_for_all(cfg['panel']['extended_sizes']);winners=[]
        for q in range(len(s['traits'])):
            records=[dict(model=m,inner_mse=v['inner_mse'][q]) for m,v in combined.items() if v['inner_mse'][q] is not None]
            winners.append(choose_procedure(records,specs))
        combined['selected_panel_expanded']=dict(predictions=np.column_stack([combined[k]['predictions'][:,q] for q,k in enumerate(winners)]),chosen_models=winners)
        rows+=prediction_rows(s,task,combined,cfg,'density',scaling)
        tuning +=[{'analysis':'density',**r} for r in more['tuning']];unavailable+=more['candidate_failures']
        for model,val in combined.items():
            for q,t in enumerate(s['traits']):
                source=combined[winners[q]] if model=='selected_panel_expanded' else val
                selections.append(dict(analysis='density',model=model,trait=t,marker_ids=s['markers'][source['selected'][q]].tolist(),chosen_model=winners[q] if model=='selected_panel_expanded' else model,lambda_value=source['lambdas'][q],inner_mse=source['inner_mse'][q]))
    if models and scaling=='center_only':
        mc=cfg['models'];cache,reason=make_split_cache(s['X'][tr],s['Y'][tr],s['X'][te],s['ids'][tr],fit['inner_assignments'],mc)
        if cache is None:raise ValueError('Shared model screen unavailable: '+reason)
        diagnostics=[];model_status=[];screen_audits=[];model_markers=[]
        for q,t in enumerate(s['traits']):
            task_seed=seed_for(mc['seed'],s['name'],sch,rep,f)
            if screen_index:
                recorded=load_recorded_screen(screen_index,s['name'],sch,rep,f,t,s['ids'][tr],s['ids'][te],s['input_hashes'])
                out=fit_with_recorded_screen(cache,q,s['markers'],mc,task_seed,recorded)
            else:
                out=fit_trait(cache,q,s['markers'],mc,task_seed)
            options={m:dict(predictions=core[m]['predictions'][:,q],inner_mse=core[m]['inner_mse'][q]) for m in mc['parent_models']};options.update(out['results'])
            cand=[m for m in mc['model_selection_candidates'] if m in options and options[m]['inner_mse'] is not None and np.isfinite(options[m]['inner_mse'])]
            if not cand:raise ValueError('No estimable family-selection candidates')
            win=min(cand,key=lambda m:options[m]['inner_mse']);options['inner_selected_family']={**options[win],'chosen_models':[win]}
            one={**s,'traits':[t],'Y':s['Y'][:,q:q+1]};rows+=prediction_rows(one,task,options,cfg,'models',scaling)
            diagnostics +=[dict(trait=t,**r) for r in out['fit_diagnostics']];model_status +=[dict(trait=t,**r) for r in out['model_status']]
            tuning +=[dict(analysis='models',trait=t,**r) for r in out['tuning']]
            screen_audits +=[dict(trait=t,**r) for r in out['screen_audits']]
            model_markers.append(dict(trait=t,sets=out['marker_sets']))
        tsv(dest/'fit_diagnostics.tsv',diagnostics);tsv(dest/'model_status.tsv',model_status);tsv(dest/'model_screen_audit.tsv',screen_audits);js(dest/'model_markers.json',model_markers)
    tsv(dest/'predictions.tsv.gz',rows);tsv(dest/'inner_assignments.tsv',fit['inner_assignments']);tsv(dest/'preprocessing.tsv',fit['preprocessing']);tsv(dest/'tuning.tsv.gz',tuning);tsv(dest/'unavailable_candidates.tsv',unavailable)
    js(dest/'selected_markers.json',selections);js(dest/'status.json',dict(status='OK',n_train=len(tr),n_test=len(te)))
    checkpoint_finish(dest)


def run_prediction(input_root,cfg,out,resume=False,max_tasks=None,cohort_filter=None,scaling_modes=None,models=True,expanded=True,screen_index=None):
    input_root=Path(input_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    folders=[p for p in sorted(input_root.iterdir()) if p.is_dir() and (p/'genotypes.npz').is_file()]
    if not folders:raise ValueError('No matched input cohorts')
    count=0
    for p in folders:
        s=cohort_record(p)
        if cohort_filter and s['dataset'] not in cohort_filter:continue
        for mode in scaling_modes or cfg['scaling_modes']:
            for task in s['tasks']:
                sch,rep,f,tr,te=task;dest=out/mode/s['name']/f'{sch}_r{rep:02d}_f{f:02d}'
                print(f'{mode} {s["name"]} {sch} {rep} {f}',flush=True)
                write_primary_fold(s,task,cfg,dest,mode,expanded and s['role']=='primary',models and s['role']=='primary',resume,screen_index)
                count+=1
                if max_tasks and count>=max_tasks:return count
    return count


def run_variance(input_root,cfg,out,cohort_filter=None):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);rows=[];profiles=[];missing=[];diagnostics=[]
    vc=cfg['variance']
    for p in sorted(Path(input_root).iterdir()):
        if not p.is_dir() or not (p/'genotypes.npz').is_file():continue
        s=cohort_record(p)
        if cohort_filter and s['dataset'] not in cohort_filter:continue
        md=marker_data(s['X'],vc);panels,absent=neutral_panels(s['markers'],md,vc,s['name'])
        missing +=[dict(scenario=s['name'],**r) for r in absent]
        Kcenter,_=normalized_kernel(md['Z']);U,_=full_pcs(Kcenter,vc['pc_count'])
        for mode in vc['kernel_scalings']:
            Z=scaled_matrix(md,mode);Kfull,_=normalized_kernel(Z)
            for panel in panels:
                K,scale=normalized_kernel(Z[:,panel['selected']]);info=dict(dataset=s['dataset'],scenario=s['name'],scaling=mode,**{k:v for k,v in panel.items() if k!='selected'})
                diagnostics.append({**info,**kernel_diagnostics(K,Kfull)})
                for fixed,C in [('intercept',np.ones((len(K),1))),('intercept_pc5',np.c_[np.ones(len(K)),U])]:
                    model=ContrastModel.build(K,C)
                    for q,t in enumerate(s['traits']):
                        row,prof=model.fit(s['Y'][:,q],vc);rows.append({**info,'trait':t,'fixed_design':fixed,**row});profiles +=[{**info,'trait':t,'fixed_design':fixed,**r} for r in prof]
    tsv(out/'variance_components.tsv',rows);tsv(out/'profiles.tsv.gz',profiles);tsv(out/'unavailable_panels.tsv',missing);tsv(out/'kernel_diagnostics.tsv',diagnostics)
    return rows


def run_sensitivity(input_root,primary_root,cfg,out,resume=False,max_tasks=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);count=0
    for p in sorted(Path(input_root).iterdir()):
        if not p.is_dir() or not (p/'genotypes.npz').is_file():continue
        s=cohort_record(p)
        if s['role']!='primary' or s['dataset'] not in ('icgt_cacao','agrosavia_matina'):continue
        variants,record=build_variants(s,{**cfg['sensitivity'],'baseline':cfg['baseline']});tsv(out/(s['name']+'_representatives.tsv'),record)
        for v in variants:
            idx=v['indices'];subset={s['ids'][i] for i in idx};lookup={s['ids'][i]:j for j,i in enumerate(idx)}
            a={**s,'name':v['name'],'ids':s['ids'][idx],'X':s['X'][idx],'Y':s['Y'][idx],'clusters':s['clusters'][idx],'components':s['components'][idx]}
            for sch,rep,f,tr0,te0 in s['tasks']:
                tr=np.array([lookup[s['ids'][i]] for i in tr0 if s['ids'][i] in subset],int);te=np.array([lookup[s['ids'][i]] for i in te0 if s['ids'][i] in subset],int)
                base={**cfg['baseline'],'scaling':'center_only','marker_missing_max':v['marker_missing_max']};dest=out/v['name']/f'{sch}_r{rep:02d}_f{f:02d}'
                parent=Path(primary_root)/'center_only'/s['name']/dest.name
                parent_file=parent/'inner_assignments.tsv'
                sig=dict(config=base,scenario=v['name'],inputs=s['input_hashes'],parent_inner_sha256=sha(parent_file) if parent_file.is_file() else None,train=id_hash(a['ids'][tr]),test=id_hash(a['ids'][te]))
                if not checkpoint_start(dest,sig,resume):continue
                parent=Path(primary_root)/'center_only'/s['name']/dest.name
                if len(tr)>=base['outer_min_train'] and len(te)>=base['outer_min_test'] and not (parent/'inner_assignments.tsv').is_file():
                    if not (parent/'status.json').is_file() or read_json(parent/'status.json').get('status')=='OK':raise FileNotFoundError('Required inherited inner assignments missing: '+str(parent))
                if len(tr)<base['outer_min_train'] or len(te)<base['outer_min_test'] or not (parent/'inner_assignments.tsv').is_file():
                    js(dest/'status.json',dict(status='NOT_ESTIMABLE',reason='sample_gate'));checkpoint_finish(dest,'NOT_ESTIMABLE');continue
                assignment=read_tsv(parent/'inner_assignments.tsv');fit=fit_frozen_inner(a['X'][tr],a['Y'][tr],a['X'][te],a['ids'][tr],assignment,base)
                if fit['state']=='OK':
                    tsv(dest/'predictions.tsv.gz',prediction_rows(a,(sch,rep,f,tr,te),fit['results'],cfg,'sensitivity','center_only'))
                    tsv(dest/'tuning.tsv',fit['tuning']);tsv(dest/'inner_assignments.tsv',fit['inner_assignments'])
                js(dest/'status.json',dict(status=fit['state'],reason=fit.get('reason','')));checkpoint_finish(dest,fit['state']);count+=1
                if max_tasks and count>=max_tasks:return count
    return count


def run_size_match(input_root,primary_root,cfg,out,resume=False,max_tasks=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);sc=cfg['size_match'];b={**sc['baseline'],'scaling':'center_only'};count=0
    for p in sorted(Path(input_root).iterdir()):
        if not p.is_dir() or not (p/'genotypes.npz').is_file():continue
        s=cohort_record(p)
        if s['role']!='primary':continue
        tests={f:te for sch,rep,f,tr,te in s['tasks'] if sch=='cluster_out'}
        valid=[]
        for f,te in tests.items():
            pp=Path(primary_root)/'center_only'/s['name']/f'cluster_out_r00_f{f:02d}'/'status.json'
            if not pp.is_file():raise FileNotFoundError('Required primary fold status missing: '+str(pp))
            if read_json(pp)['status']=='OK':valid.append(f)
        tasks,eligible,trainonly=matched_partitions(s['ids'],tests,valid,sc['random_repeats'],seed_for(sc['seed'],s['dataset']))
        js(out/(s['dataset']+'_evaluation_pool.json'),dict(eligible=s['ids'][eligible],training_only=s['ids'][trainonly]))
        for t in tasks:
            d,rep,f,tr,te=t['design'],t['repeat'],t['fold'],t['train'],t['test'];dest=out/s['dataset']/f'{d}_r{rep:02d}_f{f:02d}'
            # Same inner seed for the matched sizes as in the original diagnostic.
            inner_seed=seed_for(sc['seed'],s['dataset'],'common_inner',f)
            sig=dict(config=sc,inputs=s['input_hashes'],evaluation_pool=sorted(valid),design=d,repeat=rep,fold=f,train=id_hash(s['ids'][tr]),test=id_hash(s['ids'][te]))
            if not checkpoint_start(dest,sig,resume):continue
            fit=fit_full_ridge(s['X'][tr],s['Y'][tr],s['X'][te],s['ids'][tr],b,inner_seed)
            if fit['state']=='OK':
                results={'full_ridge':dict(predictions=fit['predictions']),'mean':dict(predictions=fit['mean_predictions'])}
                tsv(dest/'predictions.tsv.gz',prediction_rows(s,(d,rep,f,tr,te),results,cfg,'size_match','center_only'))
                tsv(dest/'inner_assignments.tsv',fit['inner_assignments']);tsv(dest/'penalties.tsv',[dict(trait=t,lambda_value=fit['lambda_values'][q]) for q,t in enumerate(s['traits'])])
            js(dest/'status.json',dict(status=fit['state'],reason=fit.get('reason','')));checkpoint_finish(dest,fit['state']);count+=1
            if max_tasks and count>=max_tasks:return count
    return count


def run_tuning(input_root,primary_root,size_root,cfg,out,resume=False,max_tasks=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tc=cfg['tuning'];base={**tc['baseline'],'scaling':'center_only'};count=0
    for p in sorted(Path(input_root).iterdir()):
        if not p.is_dir() or not (p/'genotypes.npz').is_file():continue
        s=cohort_record(p)
        if s['role']!='primary':continue
        for sch,rep,f,tr,te in s['tasks']:
            if sch!='cluster_out':continue
            pg=Path(primary_root)/'center_only'/s['name']/f'cluster_out_r00_f{f:02d}';pr=Path(size_root)/s['dataset']/f'cluster_common_inner_r00_f{f:02d}'
            if not (pg/'status.json').is_file():raise FileNotFoundError('Required primary status missing: '+str(pg))
            if read_json(pg/'status.json')['status']!='OK':continue
            if not (pg/'inner_assignments.tsv').is_file() or not (pr/'inner_assignments.tsv').is_file():raise FileNotFoundError('Required tuning assignments missing')
            dest=out/s['dataset']/f'f{f:02d}';sig=dict(config=tc,inputs=s['input_hashes'],group_inner_sha256=sha(pg/'inner_assignments.tsv'),random_inner_sha256=sha(pr/'inner_assignments.tsv'),train=id_hash(s['ids'][tr]),test=id_hash(s['ids'][te]))
            if not checkpoint_start(dest,sig,resume):continue
            path=outer_path(s['X'][tr],s['Y'][tr],s['X'][te],s['ids'][tr],base);lam=np.asarray(base['lambda_grid'])
            designs=design_list(s['ids'][tr],s['clusters'][tr],read_tsv(pg/'inner_assignments.tsv'),read_tsv(pr/'inner_assignments.tsv'),tc,s['dataset'],f)
            rows=[];loss=[];selection=[];assign=[]
            for d in designs:
                fit=inner_diagnostic(s['X'][tr],s['Y'][tr],s['ids'][tr],s['clusters'][tr],d['pairs'],base)
                if fit['state']!='OK':continue
                assign +=[dict(design=d['design'],repeat=d['repeat'],**r) for r in fit['assignments']]
                for objective in tc['loss_aggregation']:
                    preds=[];sel=[]
                    for q,t in enumerate(s['traits']):
                        chosen=select_loss(fit[objective][:,q],lam,tc['near_minimum_relative_loss_band']);j=chosen.pop('index');preds.append(path['predictions'][:,j,q]);sel.append(chosen)
                        selection.append(dict(dataset=s['dataset'],trait=t,design=d['design'],repeat=d['repeat'],fold=f,objective=objective,marker_effective_df=df_for_lambda(path['eigenvalues'],chosen['lambda_value']),**chosen))
                        loss +=[dict(trait=t,design=d['design'],repeat=d['repeat'],objective=objective,lambda_value=x,inner_loss=float(fit[objective][i,q])) for i,x in enumerate(lam)]
                    task=(d['design']+'__'+objective,d['repeat'],f,tr,te)
                    rows+=prediction_rows(s,task,{'full_ridge':dict(predictions=np.column_stack(preds))},cfg,'tuning','center_only')
            tsv(dest/'predictions.tsv.gz',rows);tsv(dest/'penalty_selection.tsv',selection);tsv(dest/'inner_losses.tsv.gz',loss);tsv(dest/'inner_assignments.tsv',assign);checkpoint_finish(dest);count+=1
            if max_tasks and count>=max_tasks:return count
    return count
