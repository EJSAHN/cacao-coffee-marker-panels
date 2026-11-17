"""Portable Excel export for the existing analysis environment.

TSV files remain the precision-preserving source. Large sheets are split, never
silently truncated. The workbook contains data values, not selected headlines.
"""
from pathlib import Path
import csv,gzip,math,re,hashlib
from .io import sha,js


def export_tables(tables,out,title='Cacao and coffee analysis results'):
    """tables is a sequence of (sheet label, TSV path). No numeric recomputation."""
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.utils import get_column_letter
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);wb=Workbook(write_only=True)
    wb.properties.title=title;wb.properties.creator='Ezekiel Ahn';wb.properties.description='Numeric results with source-file identities; missing values are blank.'
    readme=wb.create_sheet('Read_me');readme.column_dimensions['A'].width=28;readme.column_dimensions['B'].width=105
    for a,b in [('Title',title),('Scope','Exported result tables. This workbook does not fit, select or recalibrate models.'),('Precision','Machine-readable TSV files retain full precision; Excel displays numeric values to six decimal places.'),('Missing values','Blank cells denote unavailable values; undefined correlations are not zero.'),('Repeats','Partition SD is descriptive and is not an independent-sample confidence interval.'),('Fisher summary','Within-fold Pearson correlations, Fisher z with n-3 descriptive weights. Coverage must accompany comparisons.'),('MSE skill','1 minus prediction SSE divided by training-mean baseline SSE on the same test records.')]:
        readme.append([a,b])
    catalog=wb.create_sheet('Sheet_index');catalog.append(['Sheet','Source_file','Rows','SHA256']);catalog.freeze_panes='A2'
    for c,w in [('A',31),('B',65),('C',12),('D',68)]:catalog.column_dimensions[c].width=w
    used={'Read_me','Sheet_index'};manifest=[]
    def sheet_name(label,part):
        base=re.sub(r'[\\/*?:\[\]]','_',str(label));suffix='' if part==1 else '_'+str(part);base=base[:31-len(suffix)]+suffix
        if base in used:raise ValueError('Duplicate workbook sheet label '+base)
        used.add(base);return base
    for label,source in tables:
        source=Path(source);before=sha(source);op=gzip.open if source.suffix=='.gz' else open
        with op(source,'rt',encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(stream,delimiter='\t');headers=next(reader,[])
            if len(headers)>16384:raise ValueError('Too many Excel columns; use the TSV')
            count=0;part=1;part_rows=0;ws=None
            def start_sheet():
                nonlocal ws
                ws=wb.create_sheet(sheet_name(label,part));ws.freeze_panes='A2'
                for i,h in enumerate(headers,1):ws.column_dimensions[get_column_letter(i)].width=min(38,max(14,len(h)+2))
                cells=[]
                for h in headers:
                    c=WriteOnlyCell(ws,value=h);c.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D');c.alignment=Alignment(wrap_text=True,vertical='top');cells.append(c)
                ws.append(cells);ws.row_dimensions[1].height=30
            start_sheet()
            for row in reader:
                if len(row)!=len(headers):raise ValueError('Ragged table '+source.name)
                if part_rows>=1048575:
                    catalog.append([ws.title,source.name,part_rows,before]);part+=1;part_rows=0;start_sheet()
                cells=[]
                for h,v in zip(headers,row):
                    val=None if v in ('','nan','NaN','None') else v
                    # IDs, code labels and hashes must remain text, including leading zeros.
                    is_id=any(k in h.lower() for k in ('id','hash','code','chrom','position','path','name','trait','model','scheme','scenario','dataset','status','reason','design','objective','source','boundary','target'))
                    if val is not None and not is_id and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',v):
                        f=float(v);val=int(v) if re.fullmatch(r'[+-]?\d+',v) and abs(f)<1e15 else f
                    c=WriteOnlyCell(ws,value=val)
                    if isinstance(val,str):c.data_type='s'
                    else:c.number_format='0' if isinstance(val,int) else '0.000000'
                    cells.append(c)
                ws.append(cells);count+=1;part_rows+=1
            catalog.append([ws.title,source.name,part_rows,before]);manifest.append(dict(label=label,source=source.name,rows=count,sha256=before,parts=part))
        if sha(source)!=before:raise ValueError('Source changed during export')
    wb.save(out);js(out.with_suffix('.sources.json'),manifest);return manifest
