"""Historical validation without calibration, monitor or future access."""
import numpy as np
from .model import MLP
from .metrics import metrics
from .protocol import fit_preprocessing


def temporal_validation(data, groups, candidates, seed=42):
    if not candidates:
        raise ValueError('Need validation candidates')
    X, preprocessing = fit_preprocessing(data, groups['fit'])
    results = []
    for candidate in candidates:
        model = MLP(X.shape[1], candidate['hidden'], seed)
        model.fit(X[groups['fit']], data.y[groups['fit']], candidate['epochs'],
                  candidate['learning_rate'], candidate['batch_size'], np.random.default_rng(seed))
        result = metrics(data.y[groups['validation']],
                         (model.predict_proba(X[groups['validation']]) >= .5).astype(int))
        if result['f1'] is None:
            raise ValueError('Validation F1 unavailable')
        results.append(dict(config=dict(candidate), metrics=result))
    best = max(range(len(results)), key=lambda i: results[i]['metrics']['f1'])
    return dict(selected=results[best]['config'], candidates=results, preprocessing=preprocessing,
                selection_rule='Highest validation F1; first candidate breaks ties', monitor_used=False)
