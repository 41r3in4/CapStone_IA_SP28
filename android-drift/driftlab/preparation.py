"""Offline batch extraction and provenance-preserving dataset preparation."""
import csv
import hashlib
import json
from pathlib import Path
import re

from .apk import SCHEMA, inspect_apk, vectorize


def batch_extract(input_dir, output):
    source = Path(input_dir)
    if not source.is_dir():
        raise ValueError('Input must be an APK directory')
    paths = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() == '.apk')
    if not paths:
        raise ValueError('No APK files found')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    counts = dict(ok=0, duplicate=0, error=0)
    seen = set()
    with (out / 'extractions.jsonl').open('x', encoding='utf-8') as stream:
        for path in paths:
            record = {'file': str(path.relative_to(source))}
            try:
                report = inspect_apk(path)
                sample_id = report['sha256']
                status = 'duplicate' if sample_id in seen else 'ok'
                seen.add(sample_id)
                record.update(status=status, sample_id=sample_id, feature_schema=SCHEMA,
                              feature_tokens=report['feature_tokens'])
            except (ValueError, OSError, RuntimeError) as error:
                record.update(status='error', error=str(error))
            counts[record['status']] += 1
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = dict(feature_schema=SCHEMA, input=str(source.resolve()), counts=counts,
                   total=len(paths), apk_execution=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def validate_metadata(path):
    """Require externally supplied labels and observation months; never infer either."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {'sample_id', 'month', 'label', 'source'}.issubset(fields):
            raise ValueError('Metadata requires unique columns: sample_id, month, label, source')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty metadata')
    seen = set()
    for line, row in enumerate(rows, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'Metadata row {line}: malformed CSV row')
        sample_id = row['sample_id']
        if not re.fullmatch('[0-9a-f]{64}', sample_id):
            raise ValueError(f'Metadata row {line}: sample_id must be lowercase SHA256')
        if sample_id in seen:
            raise ValueError(f'Metadata row {line}: duplicate sample_id')
        seen.add(sample_id)
        if not re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', row['month']) or row['month'].startswith('0000'):
            raise ValueError(f'Metadata row {line}: invalid month; expected YYYY-MM')
        if row['label'] not in ('0', '1'):
            raise ValueError(f'Metadata row {line}: label must be 0 or 1')
        if not row['source'].strip():
            raise ValueError(f'Metadata row {line}: source is required')
    return rows


def build_dataset(extractions, metadata, output, train_through, max_elements=5_000_000, fit_ids=None):
    if not re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', train_through) or train_through.startswith('0000'):
        raise ValueError('Invalid train-through month')
    rows = validate_metadata(metadata)
    records = {}
    with Path(extractions).open(encoding='utf-8') as stream:
        for line, text in enumerate(stream, 1):
            record = json.loads(text)
            if record.get('status') not in ('ok', 'duplicate', 'error'):
                raise ValueError(f'Extraction row {line}: invalid status')
            if record['status'] == 'error':
                continue
            sample_id = record.get('sample_id', '')
            tokens = record.get('feature_tokens')
            if (not re.fullmatch('[0-9a-f]{64}', sample_id) or record.get('feature_schema') != SCHEMA
                    or not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens)):
                raise ValueError(f'Extraction row {line}: invalid feature record')
            if tokens:
                vectorize([], tokens)
            canonical = sorted(set(tokens))
            if sample_id in records and records[sample_id] != canonical:
                raise ValueError('Conflicting extraction for ' + sample_id)
            records[sample_id] = canonical
    missing = sorted({r['sample_id'] for r in rows} - records.keys())
    if missing:
        raise ValueError('Metadata samples missing successful extraction: ' + ', '.join(missing))
    rows.sort(key=lambda r: (r['month'], r['sample_id']))
    train = [r for r in rows if r['month'] <= train_through]
    if fit_ids is not None:
        fit_ids = set(fit_ids)
        eligible = {r['sample_id'] for r in train}
        if not fit_ids or not fit_ids.issubset(eligible):
            raise ValueError('fit_ids must be nonempty metadata IDs at or before train-through')
        train = [r for r in train if r['sample_id'] in fit_ids]
    if not train:
        raise ValueError('No training samples at or before train-through')
    vocabulary = sorted({t for r in train for t in records[r['sample_id']]})
    if not vocabulary:
        raise ValueError('Training vocabulary is empty')
    if max_elements < 1 or len(rows) * len(vocabulary) > max_elements:
        raise ValueError('Dataset exceeds dense CSV element limit')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    features = ['f_' + t for t in vocabulary]
    with (out / 'dataset.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'month', 'label'] + features)
        for row in rows:
            present = set(records[row['sample_id']])
            writer.writerow([row['sample_id'], row['month'], row['label']] + [int(t in present) for t in vocabulary])
    provenance = dict(feature_schema=SCHEMA, train_through=train_through, feature_names=features,
                      vocabulary_fit_ids=[r['sample_id'] for r in train],
                      samples=len(rows), training_samples=len(train),
                      metadata_sha256=hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
                      extractions_sha256=hashlib.sha256(Path(extractions).read_bytes()).hexdigest(),
                      dataset_sha256=hashlib.sha256((out / 'dataset.csv').read_bytes()).hexdigest(),
                      unused_extractions=len(records.keys() - {r['sample_id'] for r in rows}),
                      future_only_tokens=sorted({t for r in rows for t in records[r['sample_id']]} - set(vocabulary)),
                      metadata=rows)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    return provenance
