import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
from .data import load_dataset, initial_split
from .model import MLP
from .selection import select
from .metrics import metrics, regression, aut

UPDATERS = {"frozen", "full_oracle", "retrain", "finetune", "ewc", "ewc_reg", "replay", "replay_reg", "replay_ewc", "replay_ewc_reg"}
SELECTORS = {"random", "uncertainty", "conformal", "distance", "hybrid"}


class Reservoir:
    """Uniform reservoir over observed, training-eligible labels only."""
    def __init__(self, capacity, rng):
        self.capacity = capacity; self.rng = rng; self.indices = []; self.seen = 0

    def add(self, indices):
        for idx in indices:
            self.seen += 1
            if len(self.indices) < self.capacity:
                self.indices.append(int(idx))
            else:
                j = int(self.rng.integers(self.seen))
                if j < self.capacity:
                    self.indices[j] = int(idx)


class Oracle:
    def __init__(self, labels):
        self._labels = labels

    def reveal(self, indices, allowed, budget):
        if len(indices) != len(set(indices)) or len(indices) > budget:
            raise ValueError("Duplicate query or exceeded label budget")
        if not set(indices).issubset(set(allowed)):
            raise ValueError("Query outside current month")
        return self._labels[indices].copy()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def validate(c):
    if not 0 < c["budget"] <= 1:
        raise ValueError("budget must be in (0,1]")
    if c["selector"] not in SELECTORS or c["updater"] not in UPDATERS:
        raise ValueError("Unknown selector/updater")
    for key in ("epochs", "update_epochs", "batch_size", "hidden", "memory_size", "fisher_samples"):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["learning_rate"] <= 0 or min(c["ewc_lambda"], c["reg_lambda"]) < 0:
        raise ValueError("Invalid optimization parameters")
    if len(c["weights"]) != 3 or min(c["weights"]) < 0 or not np.isclose(sum(c["weights"]),1):
        raise ValueError("weights must be 3 nonnegative values summing to 1")


def run(config):
    run_start = time.perf_counter()
    c = dict(config); validate(c)
    out = Path(c["output"])
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new --output")
    d = load_dataset(c["data"])
    validation = np.array([], dtype=int)
    if c.get('temporal'):
        from .protocol import temporal_split
        groups = temporal_split(d, **c['temporal'])
        fit, cal, monitor = [groups[k] for k in ('fit', 'calibration', 'regression_monitor')]
        validation = groups['validation']
        future_months = sorted(set(d.months[groups['future_test']]))
        if c.get('validation_candidates'):
            from .validation import temporal_validation
            validation_result = temporal_validation(d, groups, c['validation_candidates'], c['seed'])
            # Candidate settings are restricted to initial training hyperparameters.
            selected = validation_result['selected']
            if set(selected) != {'hidden', 'epochs', 'learning_rate', 'batch_size'}:
                raise ValueError('Invalid validation candidate keys')
            c.update(selected)
            validate(c)
    else:
        (fit, cal, monitor), future_months = initial_split(d, c['initial_months'], c['seed'])
        groups = dict(fit=fit, calibration=cal, regression_monitor=monitor,
                      future_test=np.flatnonzero(np.isin(d.months, future_months)))
    # Fit normalization on eligible initial training examples only.
    from .protocol import fit_preprocessing, audit, split_manifest
    provenance_path = (Path(c['data']) if Path(c['data']).is_dir() else Path(c['data']).parent) / 'provenance.json'
    if c.get('temporal') and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        vocabulary_ids = provenance.get('vocabulary_fit_ids')
        if vocabulary_ids is None or not set(vocabulary_ids).issubset(set(d.ids[fit])):
            raise ValueError('Vocabulary provenance must fit only actual train IDs; rebuild dataset')
    X, preprocessing = fit_preprocessing(d, fit, c.get('max_features'))
    mean = np.array(preprocessing['mean']); scale = np.array(preprocessing['scale'])
    feature_names = preprocessing['feature_names']
    audit_result = audit(d, groups)
    if provenance_path.exists():
        audit_result['feature_provenance'] = json.loads(provenance_path.read_text(encoding='utf-8'))
    else:
        audit_result['feature_provenance'] = 'External feature vocabulary provenance unavailable; train-only column selection cannot prove upstream extraction did not use future data'
    query_replay = None
    if c.get('query_manifest'):
        from .queries import read_queries
        query_replay = read_queries(c['query_manifest'], d, future_months, c['budget'], c['updater'])
    restored = None
    if c.get('resume_from'):
        from .resume import load_state
        restored = load_state(c['resume_from'], c, d, preprocessing)
    init_rng, selection_rng, train_rng, memory_rng, fisher_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(c["seed"]).spawn(5)]
    model = MLP(X.shape[1], c["hidden"], c["seed"])
    start = time.perf_counter()
    if restored is None:
        model.fit(X[fit],d.y[fit],c["epochs"],c["learning_rate"],c["batch_size"],init_rng)
    initial_seconds = time.perf_counter()-start
    memory = Reservoir(c["memory_size"],memory_rng); memory.add(fit)
    observed = {int(i): int(d.y[i]) for i in fit}
    oracle = Oracle(d.y)
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(c,indent=2),encoding="utf-8")
    manifest = split_manifest(d, groups)
    write_csv(out/'splits.csv',manifest)
    (out/'data_audit.json').write_text(json.dumps(audit_result, indent=2), encoding='utf-8')
    (out/'preprocessing.json').write_text(json.dumps(preprocessing, indent=2), encoding='utf-8')
    if c.get('temporal') and c.get('validation_candidates'):
        (out/'validation.json').write_text(json.dumps(validation_result, indent=2), encoding='utf-8')
    logs=[]; query_logs=[]; prediction_logs=[]
    rngs = [init_rng, selection_rng, train_rng, memory_rng, fisher_rng]
    completed_month = None
    if restored is not None:
        state, checkpoint = restored
        with np.load(checkpoint, allow_pickle=False) as saved:
            if (saved['feature_names'].tolist()!=feature_names or not np.array_equal(saved['mean'],mean)
                    or not np.array_equal(saved['scale'],scale)):
                raise ValueError('Resume checkpoint preprocessing/schema mismatch')
            for key,value in model.p.items():
                if saved[key].shape!=value.shape or not np.isfinite(saved[key]).all():
                    raise ValueError('Resume checkpoint architecture mismatch')
            model.p = {k: saved[k].copy() for k in model.p}
        memory.indices = state['memory']; memory.seen = state['reservoir_seen']
        observed = {int(k): int(v) for k, v in state['observed'].items()}
        heldout = set(cal) | set(monitor) | set(validation)
        if heldout.intersection(observed) or not set(memory.indices).issubset(observed):
            raise ValueError('Resume state contains ineligible training labels')
        for rng, rng_state in zip(rngs, state['rngs']):
            rng.bit_generator.state = rng_state
        logs = state['logs']; query_logs = state['queries']; prediction_logs = state['predictions']
        initial_seconds = state['initial_seconds']; completed_month = state['month']
        # Carry the selected snapshot for an already-completed resume too.
        import shutil
        shutil.copyfile(checkpoint, out/checkpoint.name)
    for t, month in enumerate(future_months):
        if completed_month is not None and month <= completed_month:
            continue
        month_start=time.perf_counter()
        idx=np.flatnonzero(d.months==month)
        start=time.perf_counter(); probability=model.predict_proba(X[idx]); prediction=(probability>=0.5).astype(int)
        inference_seconds=time.perf_counter()-start
        row=dict(month=month,n_samples=len(idx),**metrics(d.y[idx],prediction))
        prediction_logs += [dict(sample_id=d.ids[i],month=month,label=int(d.y[i]),probability=float(p),prediction=int(q)) for i,p,q in zip(idx,probability,prediction)]
        evaluation_seconds=time.perf_counter()-start
        # Do not query/update the final month: no future evaluation benefits from it.
        budget=0 if t==len(future_months)-1 or c["updater"]=="frozen" else math.ceil(c["budget"]*len(idx))
        if c["updater"]=="full_oracle" and t<len(future_months)-1:
            budget=len(idx)
        evaluation_start=time.perf_counter()
        old_monitor=(model.predict_proba(X[monitor])>=0.5).astype(int)
        evaluation_seconds+=time.perf_counter()-evaluation_start
        selected=np.array([],dtype=int); selection_seconds=0.; update_seconds=0.
        if budget:
            reference=np.array(memory.indices,dtype=int)
            reference_y=np.array([observed[int(i)] for i in reference])
            start=time.perf_counter()
            if c["updater"]=="full_oracle":
                local=np.arange(len(idx)); scores=np.zeros(len(idx))
            else:
                if query_replay is not None:
                    lookup = {int(i): j for j, i in enumerate(idx)}
                    local = np.array([lookup[i] for i in query_replay[month]], dtype=int)
                    scores = np.zeros(len(idx))
                else:
                    local,scores=select(model,X[idx],budget,c["selector"],selection_rng,X[cal],d.y[cal],X[reference],reference_y,c["weights"],c.get('diversity', True))
            selection_seconds=time.perf_counter()-start
            selected=idx[local]
            revealed=oracle.reveal(selected,idx,budget)
            query_logs += [dict(month=month,sample_id=d.ids[i],label=int(y),selection_score=float(s)) for i,y,s in zip(selected,revealed,scores[local])]
            start=time.perf_counter(); anchor=model.snapshot()
            if c["updater"] in {"retrain","full_oracle"}:
                for i,y in zip(selected,revealed): observed[int(i)]=int(y)
                train_indices=np.array(sorted(observed),dtype=int)
                model=MLP(X.shape[1],c["hidden"],c["seed"])
                epochs=c["epochs"]; options={}
            else:
                options={}; epochs=c["update_epochs"]
                train_indices=selected
                if c["updater"].startswith("replay"):
                    train_indices=np.concatenate([reference,selected])
                if "ewc" in c["updater"]:
                    fisher=model.empirical_fisher(X[reference],reference_y,c["fisher_samples"],fisher_rng)
                    options.update(anchor=anchor,fisher=fisher,ewc=c["ewc_lambda"])
                if c["updater"].endswith("_reg"):
                    options.update(teacher=anchor,reg=c["reg_lambda"])
                for i,y in zip(selected,revealed): observed[int(i)]=int(y)
            train_labels=np.array([observed[int(i)] for i in train_indices])
            model.fit(X[train_indices],train_labels,epochs,c["learning_rate"],c["batch_size"],train_rng,**options)
            memory.add(selected)
            update_seconds=time.perf_counter()-start
        evaluation_start=time.perf_counter()
        new_monitor=(model.predict_proba(X[monitor])>=0.5).astype(int)
        row.update(regression(d.y[monitor],old_monitor,new_monitor))
        evaluation_seconds+=time.perf_counter()-evaluation_start
        row.update(queried_labels=len(selected),known_training_labels=len(observed),memory_samples=len(memory.indices),
                   inference_seconds=inference_seconds,selection_seconds=selection_seconds,update_seconds=update_seconds,
                   evaluation_seconds=evaluation_seconds,
                   month_seconds=time.perf_counter()-month_start)
        logs.append(row)
        # Checkpoints are for inference/inspection; resume training not implemented.
        np.savez_compressed(out/f'model_{month}.npz',**model.p,mean=mean,scale=scale,feature_names=np.array(feature_names),
                            memory_indices=np.array(memory.indices),month=np.array(month))
        from .resume import save_state
        save_state(out/f'state_{month}.json', c, d, preprocessing, out/f'model_{month}.npz',
                   month, memory, observed, rngs, logs, query_logs, prediction_logs, initial_seconds)
        print(f'{month}: F1={row["f1"]}, labels={len(selected)}, memory={len(memory.indices)}',flush=True)
        if c.get('stop_after_month') == month:
            break
    write_csv(out/'monthly_metrics.csv',logs); write_csv(out/'queries.csv',query_logs); write_csv(out/'predictions.csv',prediction_logs)
    query_manifest_rows = [dict(run_id=c.get('run_id',out.name),month=month,
              sample_id=r['sample_id'],query_order=order)
              for month in future_months for order,r in enumerate([q for q in query_logs if q['month']==month])]
    with (out/'query_manifest.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','month','sample_id','query_order'])
        writer.writeheader(); writer.writerows(query_manifest_rows)
    fit_ids = set(d.ids[fit]); future_ids = set(d.ids[groups['future_test']])
    duplicate_future = sum(len(set(r['sample_ids']) & future_ids) for r in audit_result['duplicate_vectors'] if set(r['sample_ids']) & fit_ids)
    summary=dict(aut_f1=aut([r['f1'] for r in logs]),queried_labels=sum(r['queried_labels'] for r in logs),
                 initial_fit_labels=len(fit),initial_calibration_labels=len(cal),initial_monitor_labels=len(monitor),
                 initial_validation_labels=len(validation), protocol='temporal' if c.get('temporal') else 'stratified_demo',
                 completed=bool(logs and logs[-1]['month'] == future_months[-1]),
                 completed_month=logs[-1]['month'] if logs else None,
                 total_seconds=time.perf_counter()-run_start,
                 evaluation_seconds=sum(r['evaluation_seconds'] for r in logs),
                 query_mode='fixed_query_replay' if query_replay is not None else (c['updater'] if c['updater'] in {'frozen','full_oracle'} else 'closed_loop_al'),
                 initial_train_seconds=initial_seconds,total_update_seconds=sum(r['update_seconds'] for r in logs),
                 future_vectors_matching_initial_fit=duplicate_future,
                 data_sha256=d.sha256,python=sys.version,numpy=np.__version__,platform=platform.platform(),
                 notes=["Demo results are not Android malware research evidence.",
                        "Memory cap applies to replay buffer, not offline dataset/evaluator RAM.",
                        "Fixed initial calibration may become stale under drift; no validity guarantee under distribution shift.",
                        "Validation is historical and separate when temporal protocol is configured; demo uses stratified splits.",
                        "EWC re-anchors to previous model each update; not multi-task Fisher accumulation.",
                        "Hybrid uses uncertainty, not paper pseudo-loss; distance is not CADE.",
                        "Duplicate vectors are counted, not automatically removed; audit real data before research use."])
    from .resources import peak_rss
    summary['peak_ram'] = peak_rss()
    summary['budget_rule'] = 'ceil(beta * current month samples); final month has no query/update; full_oracle uses all eligible labels'
    summary['total_selection_seconds'] = sum(r['selection_seconds'] for r in logs)
    summary['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    import importlib.metadata
    summary['dependencies'] = {name: importlib.metadata.version(name) for name in ('numpy','scipy')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    from .report import render
    render(out,c,logs,summary)
    return logs,summary
