"""Numeric NPZ audit and streaming CSR adapter for author-style Drebin data.

No pickle, fabricated APK hashes, inferred within-range dates or anonymous features.
"""
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from numpy.lib import format as npformat
from scipy import sparse

SCHEMA = 'drebin-author-selected-v1'
PATTERN = re.compile(r'([0-9]{4}-(?:0[1-9]|1[0-2]))(?:to([0-9]{4}-(?:0[1-9]|1[0-2])))?_selected\.npz')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def header(stream):
    version=npformat.read_magic(stream)
    if version == (1,0): shape,order,dtype=npformat.read_array_header_1_0(stream)
    elif version == (2,0): shape,order,dtype=npformat.read_array_header_2_0(stream)
    else: raise ValueError('Unsupported NPY version: '+str(version))
    return shape,order,dtype


def describe_npz(path):
    entries={}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith('.npy') or info.filename[:-4] in entries:
                raise ValueError('Unexpected or duplicate NPZ member')
            with archive.open(info) as stream: shape,order,dtype=header(stream)
            entries[info.filename[:-4]]=dict(shape=list(shape),dtype=str(dtype),object=dtype.hasobject,
                fortran_order=order,uncompressed_bytes=info.file_size,compressed_bytes=info.compress_size)
    return entries


def numeric_blocks(path, key='X_train', chunk_bytes=8*1024*1024):
    """Read C-order numeric NPY in bounded row chunks, including compressed ZIP."""
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        shape,order,dtype=header(stream)
        if dtype.hasobject:
            raise ValueError('Object/pickle array refused; verify source before any separate trusted conversion')
        if len(shape)!=2 or order or dtype.kind not in 'biuf':
            raise ValueError('Expected C-order numeric 2D feature array')
        if shape[1]<1: raise ValueError('Empty feature dimension')
        row_bytes=shape[1]*dtype.itemsize
        if row_bytes>chunk_bytes: raise ValueError('One row exceeds configured chunk budget')
        rows=max(1,chunk_bytes//row_bytes)
        for start in range(0,shape[0],rows):
            count=min(rows,shape[0]-start); expected=count*row_bytes
            raw=stream.read(expected)
            if len(raw)!=expected: raise ValueError('Truncated numeric array')
            yield start,np.frombuffer(raw,dtype=dtype).reshape(count,shape[1])
        if stream.read(1): raise ValueError('Unexpected trailing feature data')


def family_mapping(root):
    mapping=json.loads((Path(root)/'family_idx.json').read_text(encoding='utf-8'))
    if (not isinstance(mapping,dict) or mapping.get('benign')!=0
            or any(type(v) is not int or v<0 for v in mapping.values())
            or len(set(mapping.values()))!=len(mapping)):
        raise ValueError('Need unique nonnegative integer family_idx with benign=0')
    return mapping


def read_labels(path, mapping):
    with np.load(path,allow_pickle=False) as archive:
        y=archive['y_train']
        families=archive['y_mal_family']
    if (y.ndim!=1 or y.dtype.kind not in 'biuf' or not np.isfinite(y).all()
            or not np.equal(y,np.floor(y)).all() or not set(y).issubset(set(mapping.values()))):
        raise ValueError('Invalid or unmapped numeric family labels')
    positive=y!=0
    if families.ndim!=1 or families.dtype.kind not in 'US' or len(families)!=int(positive.sum()):
        raise ValueError('Malware-only family array has incompatible shape/dtype')
    family_values=np.array([mapping.get(str(f),-1) for f in families])
    if not np.array_equal(family_values,y[positive]):
        raise ValueError('Family names do not align with nonzero y_train in row order')
    return y.astype(np.int64),families.astype(str)


def vocabulary(path):
    mapping=json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(mapping,dict) or not mapping or any(not isinstance(k,str) or not k for k in mapping)
            or any(type(v) is not int for v in mapping.values())
            or set(mapping.values())!=set(range(len(mapping)))):
        raise ValueError('Vocabulary must map real names to unique contiguous column indices')
    names=[None]*len(mapping)
    for name,index in mapping.items(): names[index]=name
    return names


def month_sequence(start,end):
    first=int(start[:4])*12+int(start[5:])-1; last=int(end[:4])*12+int(end[5:])-1
    return [f'{i//12:04}-{i%12+1:02}' for i in range(first,last+1)]


def audit_directory(root, scan_features=True):
    print(f'Audit directory: {Path(root).resolve()} (pickle disabled)',flush=True)
    root=Path(root); mapping=family_mapping(root)
    report=dict(root=str(root.resolve()),feature_schema=SCHEMA,allow_pickle=False,
        family_idx_sha256=sha256(root/'family_idx.json'),family_mapping=mapping,
        label_mapping='Verified locally: family_idx benign=0; every positive y_train matches y_mal_family; binary label=(family_id!=0)',
        apk_hashes_available=False,sample_dates_available=False,
        date_provenance='Monthly filenames declare observation periods; range files do not expose individual row months',
        source_authenticity='Local file consistency only; original download provenance and external authenticity not independently verified',
        files=[],vocabularies=[],issues=[],warnings=[],feature_scan=scan_features)
    vocab_by_dim={}
    for path in sorted(root.glob('*.json')):
        if path.name=='family_idx.json': continue
        print(f'Reading vocabulary {path.name} ({path.stat().st_size/1048576:.1f} MiB)...',flush=True)
        try:
            if '_full_training_features.json' in path.name:
                full=json.loads(path.read_text(encoding='utf-8'))
                empty=[index for name,index in full.items() if not name]
                report['vocabularies'].append(dict(file=path.name,sha256=sha256(path),features=len(full),
                    role='Full reference vocabulary, excluded from selected NPZ adapter',empty_name_indices=empty,
                    contiguous_unique_indices=set(full.values())==set(range(len(full))) and len(set(full.values()))==len(full)))
                if empty: report['warnings'].append(path.name+': empty feature name; excluded full vocabulary, not silently renamed')
                print(f'Vocabulary checked: {len(full)} entries',flush=True)
                del full
                continue
            names=vocabulary(path)
            signature=hashlib.sha256(json.dumps(names,ensure_ascii=False).encode()).hexdigest()
            entry=dict(file=path.name,sha256=sha256(path),features=len(names),ordered_names_sha256=signature)
            report['vocabularies'].append(entry)
            print(f'Vocabulary checked: {len(names)} entries',flush=True)
            if '_selected_training_features.json' in path.name:
                vocab_by_dim.setdefault(len(names),set()).add(signature)
        except (ValueError,TypeError) as error: report['issues'].append(f'{path.name}: {error}')
    for path in sorted(root.glob('*.npz')):
        print(f'Checking {root.name}/{path.name}...',flush=True)
        match=PATTERN.fullmatch(path.name)
        entry=dict(file=path.name,bytes=path.stat().st_size,sha256=sha256(path),issues=[])
        try:
            if not match: raise ValueError('Unrecognized file naming; not selected automatically')
            entry.update(period_start=match[1],period_end=match[2] or match[1],
                         kind='range' if match[2] else 'monthly')
            entries=describe_npz(path); entry['arrays']=entries
            if any(e['object'] for e in entries.values()):
                raise ValueError('Object arrays blocked; pickle remains disabled')
            if set(entries)!={'X_train','y_train','y_mal_family'}:
                raise ValueError('Unexpected keys; inspect schema explicitly')
            y,families=read_labels(path,mapping)
            shape=entries['X_train']['shape']
            if len(shape)!=2 or shape[0]!=len(y): raise ValueError('Feature/label row mismatch')
            if entries['X_train']['fortran_order']: raise ValueError('Fortran feature order unsupported')
            entry.update(samples=len(y),features=shape[1],benign=int(sum(y==0)),malware=int(sum(y!=0)),
                storage='dense numeric NPY inside NPZ',family_alignment_verified=True,
                family_counts={str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))})
            if shape[1] not in vocab_by_dim or len(vocab_by_dim[shape[1]])!=1:
                raise ValueError('Missing/ambiguous matching selected vocabulary')
            entry['ordered_names_sha256']=next(iter(vocab_by_dim[shape[1]]))
            if scan_features:
                nnz=0; zero_rows=0; finite=True; minimum=None; maximum=None
                processed_bytes=0; next_progress=512*1024*1024
                for offset,block in numeric_blocks(path):
                    finite=finite and bool(np.isfinite(block).all())
                    nnz+=int(np.count_nonzero(block)); zero_rows+=int(sum(~np.any(block!=0,axis=1)))
                    if block.size:
                        lo=float(block.min()); hi=float(block.max())
                        minimum=lo if minimum is None else min(minimum,lo)
                        maximum=hi if maximum is None else max(maximum,hi)
                    processed_bytes+=block.nbytes
                    if processed_bytes>=next_progress:
                        print(f'  {path.name}: {offset+len(block)}/{shape[0]} rows scanned ({processed_bytes/1073741824:.2f} GiB)',flush=True)
                        next_progress=processed_bytes+512*1024*1024
                entry.update(nnz=nnz,zero_rows=zero_rows,finite=finite,value_min=minimum,value_max=maximum,
                             density=nnz/(shape[0]*shape[1]) if shape[0] else None)
                if not finite or not len(y): raise ValueError('Nonfinite or empty features')
        except (ValueError,KeyError,TypeError,OSError) as error:
            entry['issues'].append(str(error)); report['issues'].append(f'{path.name}: {error}')
        report['files'].append(entry)
        print(f'Audit {root.name}/{path.name}: '+('FAIL' if entry['issues'] else 'OK'),flush=True)
    monthly=[e for e in report['files'] if e.get('kind')=='monthly']
    if monthly:
        expected=month_sequence(min(e['period_start'] for e in monthly),max(e['period_end'] for e in monthly))
        report['missing_monthly_files_between_observed_bounds']=sorted(set(expected)-{e['period_start'] for e in monthly})
    else: report['missing_monthly_files_between_observed_bounds']=[]
    report['range_only_periods']=[dict(start=e['period_start'],end=e['period_end'],
        limitation='Per-month counts and row dates unavailable; do not fabricate') for e in report['files'] if e.get('kind')=='range']
    report['npz_count']=len(report['files'])
    report['dimensions']=sorted({e['features'] for e in report['files'] if 'features' in e})
    coverage={}
    for entry in report['files']:
        if 'period_start' not in entry: continue
        for month in month_sequence(entry['period_start'],entry['period_end']):
            coverage.setdefault(month,[]).append(entry['file'])
    report['overlapping_file_periods']={m:files for m,files in coverage.items() if len(files)>1}
    report['missing_periods_between_observed_bounds']=(sorted(set(month_sequence(min(coverage),max(coverage)))-set(coverage)) if coverage else [])
    report['content_pass']=bool(scan_features and report['files'] and not report['issues'])
    report['structural_pass']=bool(report['files']) and not report['issues'] and not report['missing_periods_between_observed_bounds']
    report['research_ready']=False
    report['limitations']=['Filename dates and author family labels are declarations, not independently verified ground truth',
        'Original APK hashes absent; source-row IDs are provenance handles only',
        'Selected vocabulary consistency does not prove the upstream feature selector never used future rows',
        'Annual initial range has no row-level month; temporal validation within that year is unavailable']
    return report


def load_spec(path):
    """Load explicit, nonoverlapping periods only. Dense source is chunked into CSR."""
    from .data import Dataset
    path=Path(path); spec=json.loads(path.read_text(encoding='utf-8'))
    if spec.get('feature_schema')!=SCHEMA: raise ValueError('Not a Drebin specification')
    root=Path(spec['source_root']); mapping=family_mapping(root)
    inverse={v:k for k,v in mapping.items()}
    if spec.get('family_idx_sha256')!=sha256(root/'family_idx.json') or spec.get('vocabulary_sha256')!=sha256(root/spec['vocabulary_file']):
        raise ValueError('Vocabulary/family schema hash mismatch')
    names=vocabulary(root/spec['vocabulary_file'])
    matrices=[]; ids=[]; periods=[]; labels=[]; records=[]; covered=set(); used=set(); total_nnz=0
    for filename in spec['files']:
        match=PATTERN.fullmatch(filename)
        if not match or Path(filename).name!=filename or filename in used:
            raise ValueError('Invalid/duplicate source file')
        used.add(filename); start=match[1]; end=match[2] or start
        months=set(month_sequence(start,end))
        if not months or covered.intersection(months): raise ValueError('Overlapping monthly/range input files')
        covered.update(months)
        source=root/filename; digest=sha256(source)
        if spec['source_sha256'].get(filename)!=digest: raise ValueError('Source archive hash mismatch')
        entries=describe_npz(source)
        if set(entries)!={'X_train','y_train','y_mal_family'} or any(e['object'] for e in entries.values()):
            raise ValueError('Unexpected keys or object arrays; pickle remains disabled')
        if entries['X_train']['shape'][1]!=len(names): raise ValueError('Vocabulary dimension mismatch')
        y,families=read_labels(source,mapping)
        row_count=0
        for offset,block in numeric_blocks(source,chunk_bytes=spec.get('chunk_bytes',8*1024*1024)):
            if not np.isfinite(block).all(): raise ValueError('Nonfinite features')
            total_nnz+=int(np.count_nonzero(block))
            if total_nnz*12+(len(ids)+row_count+len(block)+1)*8>spec.get('max_csr_bytes',256*1024*1024):
                raise MemoryError('CSR adapter exceeds declared storage budget; use smaller explicit file selection')
            matrices.append(sparse.csr_matrix(block,dtype=np.float64)); row_count+=len(block)
        if row_count!=len(y): raise ValueError('Features/labels mismatch')
        period=start+'to'+end if match[2] else start
        for i,value in enumerate(y):
            sample_id=f'{root.name}:{filename}:row:{i}'
            ids.append(sample_id); periods.append(period); labels.append(int(value!=0))
            records.append(dict(sample_id=sample_id,source_file=filename,source_row=i,
                family_id=int(value),family_name=inverse[int(value)],
                period_start=start,period_end=end,apk_sha256=None))
    if not matrices: raise ValueError('No data selected')
    matrix=sparse.vstack(matrices,format='csr'); matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    digest=hashlib.sha256(path.read_bytes())
    digest.update(sha256(root/spec['vocabulary_file']).encode()); digest.update(sha256(root/'family_idx.json').encode())
    return Dataset(np.array(ids),np.array(periods),matrix,np.array(labels),names,digest.hexdigest(),
        feature_schema=SCHEMA,provenance=dict(adapter='streaming-numeric-npz',spec=spec,records=records,
            id_kind='source-file-row, not APK hash',timestamp_kind='declared file period; no invented per-row month',
            upstream_feature_selection_verified=False))
