"""Configurable sequential fair-comparison matrix; large real runs require a lock."""
import hashlib
import json
from pathlib import Path
from .experiment import run, write_csv
from .statistics import aggregate


def methods(mode):
    if mode=='baselines':
        return [('frozen',dict(selector='random',updater='frozen')),('full_oracle',dict(selector='random',updater='full_oracle'))]
    if mode=='selectors':
        return [(s,dict(selector=s,updater='replay')) for s in ('random','uncertainty','conformal','distance','hybrid')]
    if mode=='updaters':
        return [(u,dict(selector='random',updater=u)) for u in ('retrain','finetune','ewc','replay','replay_ewc','replay_ewc_reg')]
    if mode=='closed_loop':
        return [(s+'_'+u,dict(selector=s,updater=u)) for s in ('random','hybrid') for u in ('finetune','replay','replay_ewc','replay_ewc_reg')]
    if mode=='ablation':
        return [('full',dict(selector='hybrid',updater='replay_ewc_reg')),
                ('no_conformal',dict(selector='hybrid',updater='replay_ewc_reg',weights=[0.,.5,.5])),
                ('no_diversity',dict(selector='hybrid',updater='replay_ewc_reg',diversity=False)),
                ('no_replay',dict(selector='hybrid',updater='ewc_reg')),
                ('no_ewc',dict(selector='hybrid',updater='replay_reg')),
                ('no_regression_loss',dict(selector='hybrid',updater='replay_ewc'))]
    raise ValueError('Unknown matrix mode')


def run_matrix(base, output, mode, budgets=(.01,.02,.05), seeds=range(5), query_manifests=None):
    if base.get('training_data_kind') != 'synthetic':
        from .data import load_dataset
        lock=json.loads(Path(base['protocol_lock']).read_text(encoding='utf-8'))
        from .resume import fingerprint
        if (lock.get('data_sha256') != load_dataset(base['data']).sha256
                or lock.get('config_sha256') != fingerprint({k:v for k,v in base.items() if k!='protocol_lock'})
                or not lock.get('audit_reviewed') or not lock.get('locked_before_test')):
            raise ValueError('Real experiments require reviewed audit and compatible locked protocol')
        current_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
        if lock.get('source_sha256') != current_sources:
            raise ValueError('Code changed after protocol lock; review audit and lock a new protocol')
    out=Path(output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    results=[]
    for beta in budgets:
        for seed in seeds:
            manifest=None
            if mode=='updaters':
                key=f'{beta}:{seed}'
                if not query_manifests or key not in query_manifests:
                    raise ValueError('Updater isolation requires one fixed manifest per budget/seed')
                manifest=query_manifests[key]
            for name,options in methods(mode):
                c=dict(base,**options); c.update(budget=beta,seed=seed,output=str(out/f'{name}_b{beta}_s{seed}'))
                c.pop('query_manifest',None)
                if manifest: c['query_manifest']=manifest
                _,summary=run(c)
                results.append(dict(method=name,budget=beta,seed=seed,aut_f1=summary['aut_f1'],
                                    run_dir=c['output'],
                                    query_mode=summary['query_mode'],queried_labels=summary['queried_labels']))
                write_csv(out/'comparison.csv',results)
    (out/'aggregate.json').write_text(json.dumps(aggregate(results),indent=2),encoding='utf-8')
    return results
