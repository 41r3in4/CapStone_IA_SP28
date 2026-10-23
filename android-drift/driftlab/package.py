"""Export and verify self-contained model artifacts with declared provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import csv
import numpy as np
from .apk import SCHEMA, vectorize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_package(run_dir, output, declaration):
    root = Path(run_dir)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if not summary.get('completed'):
        raise ValueError('Only completed runs may be exported')
    required = {'model_id','dataset_source','training_data_kind','threshold','limitations'}
    if not required.issubset(declaration) or declaration['training_data_kind'] not in ('synthetic','real_android'):
        raise ValueError('Model declaration requires provenance, kind, threshold and limitations')
    if not declaration['model_id'] or not declaration['dataset_source'] or not 0 < declaration['threshold'] < 1:
        raise ValueError('Invalid model declaration')
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    if config.get('training_data_kind') != declaration['training_data_kind']:
        raise ValueError('Model declaration kind must match recorded run kind')
    with (root/'splits.csv').open(encoding='utf-8',newline='') as stream:
        fit_months=[r['month'] for r in csv.DictReader(stream) if r['split']=='fit']
    training_period=dict(initial_fit_start=min(fit_months),initial_fit_end=max(fit_months))
    source = root/f"model_{summary['completed_month']}.npz"
    with np.load(source, allow_pickle=False) as checkpoint:
        names = checkpoint['feature_names'].tolist()
        vectorize([], names)
        dim = len(names); hidden = checkpoint['W1'].shape[1]
        architecture = dict(type='MLP', input=dim, hidden=hidden)
        shapes = dict(W1=(dim,hidden), b1=(hidden,), W2=(hidden,), b2=(1,), mean=(dim,), scale=(dim,))
        for name, shape in shapes.items():
            if checkpoint[name].shape != shape or not np.isfinite(checkpoint[name]).all():
                raise ValueError('Invalid checkpoint architecture')
        if np.any(checkpoint['scale'] <= 0):
            raise ValueError('Invalid preprocessing scale')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, out/'checkpoint.npz')
    for name in ('config.json','preprocessing.json','data_audit.json','splits.csv'):
        shutil.copyfile(root/name, out/name)
    (out/'schema.json').write_text(json.dumps(dict(feature_schema=SCHEMA, feature_names=names,
        architecture=architecture), indent=2), encoding='utf-8')
    sources = summary['source_sha256']
    provenance = dict(dataset_sha256=summary['data_sha256'], source_sha256=sources,
                      extractor_version=SCHEMA, source_run=str(root.resolve()),
                      train_period=config.get('temporal', {}).get('train_through', 'initial_months='+str(config['initial_months'])),
                      last_observed_update_month=summary['completed_month'],
                      python=summary['python'], numpy=summary['numpy'])
    provenance['dependencies'] = summary['dependencies']
    if (root/'queries.csv').exists():
        with (root/'queries.csv').open(encoding='utf-8',newline='') as stream:
            queries=list(csv.DictReader(stream))
        provenance['last_observed_update_month'] = max((r['month'] for r in queries),default=None)
    training_period['last_observed_update_month']=provenance.get('last_observed_update_month')
    provenance['training_period']=training_period
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    files = ['checkpoint.npz','schema.json','preprocessing.json','config.json','provenance.json','data_audit.json','splits.csv']
    card = dict(declaration, package_version=1, feature_schema=SCHEMA, architecture=architecture,
                training_period=training_period,extractor_version=SCHEMA,
                extractor_source_sha256=sources['apk.py'],
                checkpoint_sha256=digest(out/'checkpoint.npz'), schema_sha256=digest(out/'schema.json'),
                label_mapping={'0':'benign','1':'malware'}, calibrated_probability=False,
                provenance_notice='Author declarations do not establish model quality',
                artifacts={name:digest(out/name) for name in files})
    (out/'model_card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    verify_package(out)
    return card


def verify_package(root):
    root = Path(root)
    card = json.loads((root/'model_card.json').read_text(encoding='utf-8'))
    required = {'checkpoint.npz','schema.json','preprocessing.json','config.json','provenance.json','data_audit.json','splits.csv'}
    if card.get('package_version') != 1 or set(card.get('artifacts', {})) != required:
        raise ValueError('Invalid package artifacts')
    if card.get('checkpoint_sha256') != digest(root/'checkpoint.npz'):
        raise ValueError('Checkpoint hash mismatch')
    if card.get('feature_schema') != SCHEMA or card.get('label_mapping') != {'0':'benign','1':'malware'}:
        raise ValueError('Package schema/label mapping mismatch')
    for name, expected in card['artifacts'].items():
        if digest(root/name) != expected:
            raise ValueError('Package artifact hash mismatch: '+name)
    schema = json.loads((root/'schema.json').read_text(encoding='utf-8'))
    pre = json.loads((root/'preprocessing.json').read_text(encoding='utf-8'))
    if card['schema_sha256'] != digest(root/'schema.json') or schema['feature_schema'] != SCHEMA:
        raise ValueError('Schema hash/version mismatch')
    with np.load(root/'checkpoint.npz', allow_pickle=False) as saved:
        names = saved['feature_names'].tolist()
        if names != schema['feature_names'] or names != pre['feature_names']:
            raise ValueError('Feature order mismatch')
        vectorize([], names)
        if not np.array_equal(saved['mean'], pre['mean']) or not np.array_equal(saved['scale'], pre['scale']):
            raise ValueError('Preprocessing mismatch')
        if schema['architecture'] != card['architecture'] or saved['W1'].shape != (len(names),card['architecture']['hidden']):
            raise ValueError('Architecture mismatch')
    return card


def predict_package(report, root, fixture=False):
    from .apk import predict
    card = verify_package(root)
    if card['training_data_kind'] != 'real_android' and not fixture:
        report['notes'].append('Synthetic package: inspection only; no real APK verdict.')
        return report
    return predict(report, Path(root)/'checkpoint.npz', Path(root)/'model_card.json', fixture=fixture)
