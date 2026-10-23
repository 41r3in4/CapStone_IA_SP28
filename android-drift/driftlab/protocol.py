"""Auditable temporal partitions and train-only feature preprocessing."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def temporal_split(data, train_through, validation_through, calibration_through, monitor_through):
    boundaries = [train_through, validation_through, calibration_through, monitor_through]
    months = sorted(set(data.months))
    if any(b not in months for b in boundaries) or boundaries != sorted(set(boundaries)):
        raise ValueError('Temporal boundaries must be distinct increasing observed months')
    serial = [int(m[:4]) * 12 + int(m[5:]) for m in months]
    if any(b - a != 1 for a, b in zip(serial, serial[1:])):
        raise ValueError('Missing calendar month')
    groups = {}
    lower = ''
    for name, upper in zip(('fit', 'validation', 'calibration', 'regression_monitor'), boundaries):
        idx = np.flatnonzero((data.months > lower) & (data.months <= upper))
        if set(data.y[idx]) != {0, 1}:
            raise ValueError(name + ' must contain both classes')
        groups[name] = idx
        lower = upper
    groups['future_test'] = np.flatnonzero(data.months > lower)
    if not len(groups['future_test']):
        raise ValueError('Need future test after monitor period')
    return groups


def split_manifest(data, groups):
    seen = set()
    rows = []
    for name, indices in groups.items():
        for i in indices:
            i = int(i)
            if i < 0 or i >= len(data.ids) or i in seen:
                raise ValueError('Overlapping or invalid split index')
            seen.add(i)
            rows.append(dict(sample_id=str(data.ids[i]), month=str(data.months[i]), split=name))
    if seen != set(range(len(data.ids))):
        raise ValueError('Split manifest must cover every sample exactly once')
    return rows


def audit(data, groups):
    manifest = split_manifest(data, groups)
    from scipy import sparse
    matrix = data.X.tocsr(copy=True) if sparse.issparse(data.X) else sparse.csr_matrix(data.X)
    matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    hashes = {}
    split_by_id = {r['sample_id']: r['split'] for r in manifest}
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        key = hashlib.sha256(row.indices.astype('<i8').tobytes() + row.data.astype('<f8').tobytes()).hexdigest()
        hashes.setdefault(key, []).append(i)
    duplicates = []
    for indices in hashes.values():
        if len(indices) > 1:
            duplicates.append(dict(sample_ids=data.ids[indices].tolist(),
                conflicting_labels=len(set(data.y[indices])) > 1,
                cross_split=len({split_by_id[str(data.ids[i])] for i in indices}) > 1))
    monthly = []
    for month in sorted(set(data.months)):
        y = data.y[data.months == month]
        monthly.append(dict(month=str(month), samples=len(y), benign=int(sum(y == 0)),
                            malware=int(sum(y == 1)), malware_rate=float(y.mean())))
    return dict(data_sha256=data.sha256, samples=len(data.ids), features=matrix.shape[1],
                nnz=matrix.nnz, monthly=monthly, duplicate_vectors=duplicates,
                id_policy='Globally unique IDs required by loader; APK hash identity only when source IDs really are verified APK hashes',
                duplicate_policy='Report only; equal vectors do not prove identical APKs',
                source_verification='Labels and observation dates remain source declarations')


def fit_preprocessing(data, fit, max_features=None):
    """Select nonconstant features and scale using fit rows only; never center CSR."""
    from scipy import sparse
    train = data.X[fit]
    if sparse.issparse(train):
        mean = np.asarray(train.mean(axis=0)).ravel()
        variance = np.asarray(train.multiply(train).mean(axis=0)).ravel() - mean**2
    else:
        mean = train.mean(axis=0)
        variance = train.var(axis=0)
    columns = np.flatnonzero(variance > 1e-12)
    if max_features is not None:
        if not isinstance(max_features, int) or max_features < 1:
            raise ValueError('max_features must be a positive integer')
        columns = columns[np.argsort(-variance[columns], kind='stable')[:max_features]]
        columns.sort()
    if not len(columns):
        raise ValueError('No nonconstant training features')
    scale = np.sqrt(np.maximum(variance[columns], 0))
    scale[scale < 0.1] = 1
    center = np.zeros(len(columns)) if sparse.issparse(train) else mean[columns]
    state = dict(columns=columns.tolist(), mean=center.tolist(), scale=scale.tolist(),
                 fit_ids=data.ids[fit].tolist(), feature_names=[data.features[i] for i in columns])
    selected = data.X[:, columns]
    X = selected.multiply(1 / scale).tocsr() if sparse.issparse(selected) else (selected - center) / scale
    return X, state


def save_protocol(output, data, groups):
    rows = split_manifest(data, groups)
    result = audit(data, groups)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'splits.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'month', 'split'])
        writer.writeheader(); writer.writerows(rows)
    (out / 'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
