"""Read-only source validation and atomic result serialization."""
from __future__ import annotations
import csv, gzip, hashlib, json, os
from pathlib import Path
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and (not np.isfinite(x)):
        return None
    return x

def js(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_tsv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))

def tsv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys((k for row in rows for k in row))) or ['status']
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(v) for k, v in row.items()})

def child(root, rel):
    root = Path(root).resolve()
    result = (root / rel).resolve()
    if root not in result.parents:
        raise ValueError('Path escapes source directory: ' + rel)
    return result

def verify(root, manifest='payload_manifest.tsv'):
    rows = read_tsv(Path(root) / manifest)
    if not rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Invalid manifest')
    for row in rows:
        path = child(root, row['path'])
        if not path.is_file() or path.stat().st_size != int(row['bytes']) or sha(path) != row['sha256']:
            raise ValueError('Source integrity failure: ' + row['path'])
    return len(rows)

def manifest(root, filename='payload_manifest.tsv'):
    root = Path(root)
    rows = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name != filename and (not p.name.endswith('.tmp')) and ('__pycache__' not in p.parts) and (p.suffix != '.pyc')]
    tsv(root / filename, rows)
    return rows
