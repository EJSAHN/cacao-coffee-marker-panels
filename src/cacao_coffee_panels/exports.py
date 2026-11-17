"""Read-only export of approved result tables and matched inputs."""
from pathlib import Path
import shutil,json,csv
import numpy as np
from .io import read_json,read_tsv,tsv,js,sha,verify,manifest
from .data import Cohort
from .workbook import export_tables


def collect_results(config_path,out,workbook=True):
    """Explicit table catalog; no guessing from a latest timestamp or run name."""
    config_path=Path(config_path).resolve();cfg=read_json(config_path);out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);copied=[];sheets=[];seen=set()
    for row in cfg['tables']:
        label=row['label'];src=Path(row['path']);src=src if src.is_absolute() else config_path.parent/src;src=src.resolve()
        if not src.is_file():raise FileNotFoundError(src)
        expected=row.get('sha256');before=sha(src)
        if expected and expected!=before:raise ValueError('Source hash mismatch '+src.name)
        name=row.get('filename',src.name)
        if Path(name).name!=name or name in seen:raise ValueError('Unsafe or duplicate export filename')
        seen.add(name);dest=out/'tables'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest)
        if sha(dest)!=before or sha(src)!=before:raise ValueError('Copy mismatch')
        copied.append(dict(label=label,file='tables/'+name,sha256=before,bytes=dest.stat().st_size,workbook=bool(row.get('workbook',True))))
        if row.get('workbook',True):sheets.append((label,dest))
    tsv(out/'result_sources.tsv',copied)
    if workbook:export_tables(sheets,out/'analysis_summary.xlsx',cfg.get('title','Cacao and coffee analysis results'))
    js(out/'status.json',dict(state='COMPLETE',operation='export_existing_results',fits=0,tables=len(copied),workbook=workbook));manifest(out);return copied


def import_matched(source,out):
    """Copy explicit matched inputs with their recorded partitions, without refitting."""
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    folders=[p for p in sorted(source.iterdir()) if p.is_dir() and (p/'genotypes.npz').is_file()]
    if not folders:raise ValueError('No cohort directories with genotypes.npz')
    out.mkdir(parents=True);records=[]
    for p in folders:
        c=Cohort.load(p);target=out/p.name;target.mkdir();used=[]
        for name in ('genotypes.npz','phenotypes.tsv','samples.tsv','markers.tsv','outer_test_assignments.tsv','definition.json'):
            src=p/name
            if not src.exists():
                if name=='definition.json':continue
                raise FileNotFoundError(src)
            before=sha(src);shutil.copyfile(src,target/name)
            if sha(target/name)!=before or sha(src)!=before:raise ValueError('Source copy mismatch')
            used.append(dict(file=name,sha256=before))
        if (p/'inner_folds').is_dir():shutil.copytree(p/'inner_folds',target/'inner_folds')
        if not (target/'definition.json').exists():
            ds=p.name.split('__')[0];js(target/'definition.json',dict(dataset=ds,scenario=ds+'__primary_q20',reference_scenario=ds+'__primary_q20',role='primary'))
        after=Cohort.load(target)
        if not np.array_equal(c.dosage,after.dosage,equal_nan=True):raise ValueError('Genotype values changed')
        records.append(dict(cohort=p.name,n_samples=len(c.sample_ids),n_markers=len(c.marker_ids),source_files=used))
    js(out/'input_sources.json',records);manifest(out);return records


def import_screens(source,matched,out):
    """Copy recorded training-only model screens and bind them to matched input bytes.

    The source must include one status.json beside each screened_marker_indices.npz.
    This imports prior feature-selection evidence, never derives ranks from test targets.
    """
    source=Path(source).resolve();matched=Path(matched).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    folders={}
    for p in sorted(matched.iterdir()):
        if not p.is_dir() or not (p/'genotypes.npz').is_file():continue
        definition=read_json(p/'definition.json') if (p/'definition.json').is_file() else {}
        key=definition.get('scenario',p.name.split('__')[0]+'__primary_q20');Cohort.load(p)
        folders[key]=p
    inputs={k:{name:sha(p/name) for name in ('genotypes.npz','phenotypes.tsv','markers.tsv')} for k,p in folders.items()}
    records=[];seen=set();out.mkdir(parents=True)
    for f in sorted(source.rglob('screened_marker_indices.npz')):
        status=read_json(f.parent/'status.json')
        if status.get('status')!='OK':raise ValueError('Screen exists without completed model evaluation')
        keys=['scenario','scheme','repeat','fold','trait','train_ids_sha256','test_ids_sha256'];row={k:status[k] for k in keys}
        key=tuple(row[k] for k in keys[:5])
        if key in seen:raise ValueError('Duplicate saved screen');
        seen.add(key)
        if row['scenario'] not in folders:raise ValueError('Screen cohort absent from matched inputs')
        with np.load(f,allow_pickle=False) as z:
            if 'outer' not in z or not any(k.startswith('inner_') for k in z.files):raise ValueError('Incomplete model screen archive')
            for k in z.files:
                a=z[k]
                if a.ndim!=1 or a.dtype.kind not in 'iu' or len(set(a))!=len(a):raise ValueError('Invalid screen indices')
        dest=out/'screens'/('screen_%06d.npz'%(len(records)+1));dest.parent.mkdir(exist_ok=True)
        before=sha(f);shutil.copyfile(f,dest)
        if sha(f)!=before or sha(dest)!=before:raise ValueError('Recorded screen changed while copying')
        records.append({**row,'path':dest.relative_to(out).as_posix(),'sha256':before})
    if not records:raise ValueError('No completed model screen files')
    js(out/'index.json',dict(cohort_hashes=inputs,screens=records,origin='Recorded training-only feature order; no new screening or model selection'))
    manifest(out);return records
