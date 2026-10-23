"""Frozen linear SVM baseline; margin is not a calibrated probability."""
from .data import load_dataset
from .protocol import temporal_split, fit_preprocessing, save_protocol
from .metrics import metrics, aut


def run_svm(config):
    from sklearn.svm import LinearSVC
    from .experiment import write_csv
    from pathlib import Path
    import json
    import time
    import numpy as np
    data=load_dataset(config['data']); groups=temporal_split(data,**config['temporal'])
    X,state=fit_preprocessing(data,groups['fit'],config.get('max_features'))
    model=LinearSVC(C=config.get('svm_C',1.),random_state=config['seed'],dual='auto')
    start=time.perf_counter(); model.fit(X[groups['fit']],data.y[groups['fit']]); train_seconds=time.perf_counter()-start
    rows=[]
    for month in sorted(set(data.months[groups['future_test']])):
        idx=np.flatnonzero(data.months==month); prediction=model.predict(X[idx])
        rows.append(dict(month=month,**metrics(data.y[idx],prediction)))
    save_protocol(config['output'],data,groups); out=Path(config['output'])
    write_csv(out/'monthly_metrics.csv',rows)
    summary=dict(aut_f1=aut([r['f1'] for r in rows]),model='LinearSVC frozen',queried_labels=0,
                 initial_train_seconds=train_seconds,score='Uncalibrated margin; no APK package export',data_sha256=data.sha256)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'preprocessing.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    return rows,summary
