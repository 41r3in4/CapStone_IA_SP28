import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
from .data import load_csv, initial_split
from .model import MLP
from .selection import select
from .metrics import metrics, regression, aut

UPDATERS = {"frozen", "full_oracle", "retrain", "finetune", "replay", "replay_ewc", "replay_ewc_reg"}
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
    c = dict(config); validate(c)
    out = Path(c["output"])
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; choose a new --output")
    d = load_csv(c["data"])
    (fit, cal, monitor), future_months = initial_split(d, c["initial_months"], c["seed"])
    # Fit normalization on eligible initial training examples only.
    mean = d.X[fit].mean(axis=0)
    scale = d.X[fit].std(axis=0); scale[scale < 0.1] = 1
    X = (d.X-mean)/scale
    init_rng, selection_rng, train_rng, memory_rng, fisher_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(c["seed"]).spawn(5)]
    model = MLP(X.shape[1], c["hidden"], c["seed"])
    start = time.perf_counter()
    model.fit(X[fit],d.y[fit],c["epochs"],c["learning_rate"],c["batch_size"],init_rng)
    initial_seconds = time.perf_counter()-start
    memory = Reservoir(c["memory_size"],memory_rng); memory.add(fit)
    observed = {int(i): int(d.y[i]) for i in fit}
    oracle = Oracle(d.y)
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(c,indent=2),encoding="utf-8")
    manifest=[]
    for split, indices in (("fit",fit),("calibration",cal),("regression_monitor",monitor)):
        manifest += [dict(sample_id=d.ids[i],month=d.months[i],split=split) for i in indices]
    manifest += [dict(sample_id=d.ids[i],month=d.months[i],split="future_test") for i in np.flatnonzero(np.isin(d.months,future_months))]
    write_csv(out/'splits.csv',manifest)
    logs=[]; query_logs=[]; prediction_logs=[]
    for t, month in enumerate(future_months):
        month_start=time.perf_counter()
        idx=np.flatnonzero(d.months==month)
        start=time.perf_counter(); probability=model.predict_proba(X[idx]); prediction=(probability>=0.5).astype(int)
        inference_seconds=time.perf_counter()-start
        row=dict(month=month,n_samples=len(idx),**metrics(d.y[idx],prediction))
        prediction_logs += [dict(sample_id=d.ids[i],month=month,label=int(d.y[i]),probability=float(p),prediction=int(q)) for i,p,q in zip(idx,probability,prediction)]
        # Do not query/update the final month: no future evaluation benefits from it.
        budget=0 if t==len(future_months)-1 or c["updater"]=="frozen" else math.ceil(c["budget"]*len(idx))
        if c["updater"]=="full_oracle" and t<len(future_months)-1:
            budget=len(idx)
        old_monitor=(model.predict_proba(X[monitor])>=0.5).astype(int)
        selected=np.array([],dtype=int); selection_seconds=0.; update_seconds=0.
        if budget:
            reference=np.array(memory.indices,dtype=int)
            reference_y=np.array([observed[int(i)] for i in reference])
            start=time.perf_counter()
            if c["updater"]=="full_oracle":
                local=np.arange(len(idx)); scores=np.zeros(len(idx))
            else:
                local,scores=select(model,X[idx],budget,c["selector"],selection_rng,X[cal],d.y[cal],X[reference],reference_y,c["weights"])
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
        new_monitor=(model.predict_proba(X[monitor])>=0.5).astype(int)
        row.update(regression(d.y[monitor],old_monitor,new_monitor))
        row.update(queried_labels=len(selected),known_training_labels=len(observed),memory_samples=len(memory.indices),
                   inference_seconds=inference_seconds,selection_seconds=selection_seconds,update_seconds=update_seconds,
                   month_seconds=time.perf_counter()-month_start)
        logs.append(row)
        # Checkpoints are for inference/inspection; resume training not implemented.
        np.savez_compressed(out/f'model_{month}.npz',**model.p,mean=mean,scale=scale,feature_names=np.array(d.features),
                            memory_indices=np.array(memory.indices),month=np.array(month))
        print(f'{month}: F1={row["f1"]}, labels={len(selected)}, memory={len(memory.indices)}',flush=True)
    write_csv(out/'monthly_metrics.csv',logs); write_csv(out/'queries.csv',query_logs); write_csv(out/'predictions.csv',prediction_logs)
    initial_hashes=[hashlib.sha256(np.ascontiguousarray(d.X[i]).tobytes()).hexdigest() for i in fit]
    fit_hashes=set(initial_hashes)
    duplicate_future=sum(hashlib.sha256(np.ascontiguousarray(d.X[i]).tobytes()).hexdigest() in fit_hashes for i in np.flatnonzero(np.isin(d.months,future_months)))
    summary=dict(aut_f1=aut([r['f1'] for r in logs]),queried_labels=sum(r['queried_labels'] for r in logs),
                 initial_fit_labels=len(fit),initial_calibration_labels=len(cal),initial_monitor_labels=len(monitor),
                 initial_train_seconds=initial_seconds,total_update_seconds=sum(r['update_seconds'] for r in logs),
                 future_vectors_matching_initial_fit=duplicate_future,
                 data_sha256=d.sha256,python=sys.version,numpy=np.__version__,platform=platform.platform(),
                 notes=["Demo results are not Android malware research evidence.",
                        "Memory cap applies to replay buffer, not offline dataset/evaluator RAM.",
                        "Fixed initial calibration may become stale under drift; no validity guarantee under distribution shift.",
                        "Initial split is stratified within initial period; no hyperparameter tuning or temporal validation implemented.",
                        "EWC re-anchors to previous model each update; not multi-task Fisher accumulation.",
                        "Hybrid uses uncertainty, not paper pseudo-loss; distance is not CADE.",
                        "Duplicate vectors are counted, not automatically removed; audit real data before research use."])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    from .report import render
    render(out,c,logs,summary)
    return logs,summary
