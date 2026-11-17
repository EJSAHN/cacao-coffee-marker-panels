"""Build matched matrices and genotype-only validation designs from explicit sources."""
from __future__ import annotations
from pathlib import Path
import collections, itertools, json
import numpy as np
import pandas as pd
from . import readers
from .io import js, tsv, sha, read_json
from .ridge import structure, pair_audit, outer_splits, seed_for, id_hash


def save_matrix(folder, X, ids, mids, phenotype, markers, sample_meta=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    ids=np.asarray(ids,str);mids=np.asarray(mids,str);X=np.asarray(X,float)
    if X.shape!=(len(ids),len(mids)) or len(set(ids))!=len(ids) or len(set(mids))!=len(mids):
        raise ValueError('Invalid or duplicate genotype axes')
    np.savez_compressed(folder/'genotypes.npz',dosage=X,sample_ids=ids,marker_ids=mids)
    phenotype.to_csv(folder/'phenotypes.tsv',sep='\t',index=False,float_format='%.17g')
    markers.to_csv(folder/'markers.tsv',sep='\t',index=False,float_format='%.17g')
    (sample_meta if sample_meta is not None else pd.DataFrame({'sample_id':ids})).to_csv(folder/'samples.tsv',sep='\t',index=False)


def prepare_raw(source_manifest, out):
    source_manifest=Path(source_manifest).resolve();spec=read_json(source_manifest)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True); used={};catalog=[]
    def path(k):
        p=Path(spec['files'][k]);p=p if p.is_absolute() else source_manifest.parent/p;p=p.resolve()
        if not p.is_file():raise FileNotFoundError(p)
        used[k]={'name':p.name,'sha256':sha(p)}
        return p
    def save(ds,variant,X,ids,mids,ph,meta,**extra):
        save_matrix(out/ds/variant,X,ids,mids,ph,meta)
        catalog.append(dict(dataset=ds,variant=variant,path=ds+'/'+variant,n_samples=len(ids),n_markers=len(mids),**extra))
    # Literal identifiers and observed calls are retained; no source call is imputed.
    samples,mids,X,meta,actions,tokens=readers.decode_hapmap(path('icgt_genotype'))
    _,book=readers.read_workbook(path('icgt_phenotype'))
    ph=readers.rows_to_frame(book['Phenotypewithoutmissingvalues'],0)
    ph.insert(0,'sample_id',ph['Accession'].map(readers.text_id))
    cross=readers.match_ids(samples,ph['sample_id'].tolist(),allow_multiple_genotypes=True)
    cand=cross[cross['accession_id']!=''].copy()
    rank={'exact':0,'terminal_numeric_suffix_exact_base':1,'unique_punctuation_normalization':2}
    cand['match_rank']=cand['match_method'].map(rank)
    cand=cand.sort_values(['accession_id','match_rank','source_genotype_id'])
    matched=cand.drop_duplicates('accession_id',keep='first');ids=matched['accession_id'].tolist()
    lookup={s:i for i,s in enumerate(samples)};M=X[[lookup[s] for s in matched['source_genotype_id']]]
    ph=ph.set_index('sample_id').loc[ids].reset_index()
    save('icgt_cacao','observed_calls',M,ids,mids,ph,meta)
    keep=meta['encoding'].eq('source_allele_pair').to_numpy()
    save('icgt_cacao','declared_alleles_only',M[:,keep],ids,np.asarray(mids)[keep],ph,meta.loc[keep].reset_index(drop=True))
    audit=out/'audit';audit.mkdir()
    for name,frame in [('icgt_sample_crosswalk',cross),('icgt_multiple_candidates',cand[cand.duplicated('accession_id',keep=False)]),('icgt_source_actions',actions),('icgt_tokens',tokens)]:
        frame.to_csv(audit/(name+'.tsv'),sep='\t',index=False)
    # Source-ID alternatives are exported, not selected using phenotypes.
    alternatives=[]
    b=spec['baseline']
    for acc,grp in cand.groupby('accession_id',sort=True):
        for a,c in itertools.combinations(grp['source_genotype_id'],2):
            A=X[lookup[a]];B=X[lookup[c]];ok=np.isfinite(A)&np.isfinite(B)
            n=int(ok.sum());sim=1-float(np.mean(abs(A[ok]-B[ok])))/2 if n else None
            flagged=n<max(b['minimum_pair_overlap'],int(np.ceil(len(A)*b['minimum_pair_overlap_fraction']))) or sim is None or sim<b['similarity_threshold']-1e-12
            alternatives.append(dict(accession_id=acc,source_id_a=a,source_id_b=c,n_overlap=n,dosage_similarity=sim,sensitivity_exclusion=flagged))
    tsv(audit/'alternative_record_comparison.tsv',alternatives)
    # Published accession summaries, not recomputed harvest averages.
    _,book=readers.read_workbook(path('agrosavia_phenotype'))
    totals,long,traits=readers.parse_agrosavia_phenotypes(book['Table S2'])
    long.to_csv(audit/'agrosavia_harvests.tsv',sep='\t',index=False,float_format='%.17g')
    for rep in ('matina','criollo'):
        samples,mids,X,meta=readers.parse_vcf(path('agrosavia_'+rep+'_vcf'))
        cross=readers.match_ids(samples,totals['sample_id'].tolist(),strip_barcode=True)
        m=cross[cross['accession_id']!=''].sort_values('accession_id');ids=m['accession_id'].tolist();lookup={s:i for i,s in enumerate(samples)}
        ph=totals.set_index('sample_id').loc[ids].reset_index()
        save('agrosavia_'+rep,'source_summary',X[[lookup[s] for s in m['source_genotype_id']]],ids,mids,ph,meta)
        cross.to_csv(audit/('agrosavia_'+rep+'_crosswalk.tsv'),sep='\t',index=False)
    for pop in ('int','prec'):
        ds='coffee_'+pop;raw=pd.read_csv(path(ds+'_genotype'),index_col=0);ids=raw.index.astype(str).tolist();mids=raw.columns.astype(str).tolist();X=raw.to_numpy(float)
        if not np.isfinite(X).all() or X.min()<-1-1e-10 or X.max()>1+1e-10:raise ValueError('Coffee source is not finite [-1,1] dosage')
        cross=readers.coffee_crosswalk(pd.read_csv(path(ds+'_phenotype')),ids)
        for t in readers.TRAITS_COFFEE:cross[t]=readers.numeric_column(cross[t],t)
        cross.to_csv(audit/(ds+'_source_crosswalk.tsv'),sep='\t',index=False,float_format='%.17g')
        ph=cross[cross.source_environment=='FEM'].set_index('accession_id').loc[ids,readers.TRAITS_COFFEE].reset_index().rename(columns={'accession_id':'sample_id'})
        meta=[]
        for marker in mids:
            parts=marker.split('.',1);valid=len(parts)==2 and all(v.isdigit() for v in parts)
            meta.append(dict(marker_id=marker,source_chrom=parts[0] if valid else '',source_position=parts[1] if valid else '',coordinate_unit='bp_from_source_README' if valid else 'unresolved',encoding='source_numeric_plus_one_no_rounding'))
        save(ds,'continuous_source_FEM',X+1.,ids,mids,ph,pd.DataFrame(meta))
    js(out/'matrix_index.json',catalog);js(out/'sources.json',used)
    for k,row in used.items():
        if sha(path(k))!=row['sha256']:raise ValueError('Source changed while reading')
    js(out/'status.json',{'state':'COMPLETE','operation':'source_reading'})
    return catalog


def make_design(matrix_root, config_path, out, datasets=None):
    matrix_root=Path(matrix_root);cfg=read_json(config_path);b=cfg['baseline'];out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);entries=read_json(matrix_root/'matrix_index.json');rows=[];base_ids={}
    for entry in entries:
        ds=entry['dataset']
        if entry['variant']=='declared_alleles_only' or (datasets and ds not in datasets):continue
        d=matrix_root/entry['path']
        with np.load(d/'genotypes.npz',allow_pickle=False) as z:X=z['dosage'];ids=z['sample_ids'].astype(str);mids=z['marker_ids'].astype(str)
        order=np.argsort(ids,kind='stable');X=X[order];ids=ids[order]
        meta=pd.read_csv(d/'markers.tsv',sep='\t',dtype=str,keep_default_na=False)
        ph=pd.read_csv(d/'phenotypes.tsv',sep='\t',dtype={'sample_id':str}).set_index('sample_id').loc[ids]
        traits=cfg['traits'][ds];Y=ph[traits].to_numpy(float)
        if not np.isfinite(Y).all():raise ValueError('Missing target; no phenotype imputation')
        keep=np.mean(~np.isfinite(X),axis=1)<=b['sample_missing_max']+1e-12
        X=X[keep];ids=ids[keep];Y=Y[keep];ph=ph.loc[ids]
        st=structure(X,ids,{**b,'scaling':cfg.get('structure_scaling','empirical_sd')},seed_for(b['seed'],ds,'structure'));pairs,components=pair_audit(X[:,st['marker_indices']],ids,b)
        name=ds+'__primary_q20';cohort='agrosavia' if ds.startswith('agrosavia') else ds
        split=outer_splits(ids,st['labels'],components,b,seed_for(b['seed'],cohort,'outer'))
        samples=pd.DataFrame({'sample_id':ids,'cluster':st['labels'],'identity_component':components})
        if 'Acc Group' in ph:samples['source_group_code']=ph['Acc Group'].to_numpy()
        target=pd.DataFrame(Y,columns=traits);target.insert(0,'sample_id',ids)
        dest=out/name;save_matrix(dest,X,ids,mids,target,meta,samples)
        assignments=[dict(scheme=s,repeat=r,fold=f,sample_id=ids[i]) for s,r,f,tr,te in split for i in te]
        tsv(dest/'outer_test_assignments.tsv',assignments);tsv(dest/'identity_pairs.tsv.gz',pairs)
        tsv(dest/'cluster_candidates.tsv',st['candidates']);js(dest/'definition.json',dict(dataset=ds,scenario=name,reference_scenario=name,role='primary'))
        rows.append(dict(dataset=ds,scenario=name,path=name,reference_scenario=name,role='primary'))
        base_ids[ds]=(ids,split,st['labels'],components,traits)
    # The sensitivity copies inherit primary memberships; they never recluster.
    if 'icgt_cacao' in base_ids:
        ds='icgt_cacao';ids,split,clusters,components,traits=base_ids[ds];primary=out/(ds+'__primary_q20')
        with np.load(primary/'genotypes.npz',allow_pickle=False) as z:fullX=z['dosage'];fullmid=z['marker_ids'].astype(str)
        ph=pd.read_csv(primary/'phenotypes.tsv',sep='\t',dtype={'sample_id':str});all_samples=pd.read_csv(primary/'samples.tsv',sep='\t',dtype=str)
        flags=set()
        a=matrix_root/'audit/alternative_record_comparison.tsv'
        if a.is_file() and a.stat().st_size>2:
            aa=pd.read_csv(a,sep='\t');flags=set(aa.loc[aa.sensitivity_exclusion.astype(str).str.lower()=='true','accession_id'])
        for variant,name,role in [('declared_alleles_only',ds+'__declared_same_samples','allele_metadata_sensitivity'),('observed_calls',ds+'__exclude_discordant_names','sample_identity_sensitivity')]:
            e=next(e for e in entries if e['dataset']==ds and e['variant']==variant);d=matrix_root/e['path']
            with np.load(d/'genotypes.npz',allow_pickle=False) as z:
                loc={v:i for i,v in enumerate(z['sample_ids'].astype(str))};mids=z['marker_ids'].astype(str);X=z['dosage'][[loc[v] for v in ids]]
            keep=np.array([v not in flags for v in ids]) if role=='sample_identity_sensitivity' else np.ones(len(ids),bool)
            meta=pd.read_csv(d/'markers.tsv',sep='\t',dtype=str,keep_default_na=False);dest=out/name
            save_matrix(dest,X[keep],ids[keep],mids,ph.loc[keep].reset_index(drop=True),meta,all_samples.loc[keep].reset_index(drop=True))
            subset=set(ids[keep]);tsv(dest/'outer_test_assignments.tsv',[dict(scheme=s,repeat=r,fold=f,sample_id=ids[i]) for s,r,f,tr,te in split for i in te if ids[i] in subset])
            js(dest/'definition.json',dict(dataset=ds,scenario=name,reference_scenario=ds+'__primary_q20',role=role))
            rows.append(dict(dataset=ds,scenario=name,path=name,reference_scenario=ds+'__primary_q20',role=role))
    tsv(out/'cohorts.tsv',rows);js(out/'configuration.json',cfg);return rows
