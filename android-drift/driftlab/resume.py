"""Numeric NPZ weights and JSON state. No pickle; month snapshots are immutable."""
import hashlib
import json
from pathlib import Path


CONTROL = {'output', 'resume_from', 'stop_after_month'}


def fingerprint(config):
    c = {k: v for k, v in config.items() if k not in CONTROL}
    if c.get('query_manifest'):
        c['query_manifest_sha256'] = hashlib.sha256(Path(c['query_manifest']).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def save_state(path, config, data, preprocessing, model_path, month, memory, observed, rngs,
               logs, queries, predictions, initial_seconds):
    state = dict(version=1, config_sha256=fingerprint(config), data_sha256=data.sha256,
                 preprocessing=preprocessing, checkpoint=Path(model_path).name,
                 checkpoint_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                 month=month, memory=memory.indices, reservoir_seen=memory.seen,
                 observed={str(k): v for k, v in observed.items()},
                 rngs=[r.bit_generator.state for r in rngs], logs=logs, queries=queries,
                 predictions=predictions, initial_seconds=initial_seconds,
                 ewc_policy='Recompute diagonal Fisher from restored previous model and buffer each update')
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2, allow_nan=False)


def load_state(path, config, data, preprocessing):
    path = Path(path)
    state = json.loads(path.read_text(encoding='utf-8'))
    if (state['version'] != 1 or state['config_sha256'] != fingerprint(config)
            or state['data_sha256'] != data.sha256 or state['preprocessing'] != preprocessing):
        raise ValueError('Resume dataset/schema/config mismatch')
    checkpoint = path.parent / state['checkpoint']
    if checkpoint.parent != path.parent or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != state['checkpoint_sha256']:
        raise ValueError('Resume checkpoint hash mismatch')
    observed={int(k):int(v) for k,v in state['observed'].items()}
    by_id={str(v):i for i,v in enumerate(data.ids)}
    query_ids=[q['sample_id'] for q in state['queries']]
    fit_ids=preprocessing['fit_ids']
    if (len(query_ids)!=len(set(query_ids)) or set(query_ids).intersection(fit_ids)
            or any(q['sample_id'] not in by_id or q['month'] != data.months[by_id[q['sample_id']]]
                   or q['month']>state['month'] or q['label'] != int(data.y[by_id[q['sample_id']]]) for q in state['queries'])
            or set(observed)!={by_id[i] for i in fit_ids+query_ids}):
        raise ValueError('Resume labels must come only from initial fit and recorded oracle queries')
    if (len(state['rngs']) != 5 or len(set(state['memory'])) != len(state['memory'])
            or state['reservoir_seen'] != len(observed) or len(state['memory'])>config['memory_size']
            or not set(state['memory']).issubset(observed)
            or any(i<0 or i>=len(data.ids) or data.months[i]>state['month'] or data.y[i]!=v for i,v in observed.items())
            or len(state['logs']) != len({r['month'] for r in state['logs']})
            or not state['logs'] or state['logs'][-1]['month'] != state['month']):
        raise ValueError('Invalid resume training/log state')
    return state, checkpoint
