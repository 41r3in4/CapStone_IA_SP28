import numpy as np


def rank01(x):
    # Equal values receive equal midranks, avoiding ID-order tie bias.
    values, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(values) <= 1:
        return np.zeros(len(x))
    rank = np.cumsum(counts) - (counts + 1) / 2
    return rank[inv] / max(1, len(x) - 1)


def conformal_credibility(model, X, calibration_X, calibration_y):
    predicted = model.predict_proba(X)
    calibration_prob = model.predict_proba(calibration_X)
    labels = (predicted >= 0.5).astype(int)
    cred = np.zeros(len(X))
    for label in (0, 1):
        reference = calibration_prob[calibration_y == label]
        if not len(reference):
            raise ValueError("Calibration must contain both classes")
        reference = 1 - reference if label else reference
        reference.sort()
        idx = np.flatnonzero(labels == label)
        scores = 1 - predicted[idx] if label else predicted[idx]
        count = len(reference) - np.searchsorted(reference, scores, side="left")
        cred[idx] = (1 + count) / (1 + len(reference))
    return cred


def latent_distance(model, X, reference_X, reference_y):
    # A class-centroid heuristic, NOT a reproduction of CADE.
    z = model.encode(X)
    ref = model.encode(reference_X)
    distances = []
    for label in (0, 1):
        group = ref[reference_y == label]
        if not len(group):
            continue
        center = np.median(group, axis=0)
        scale = np.median(np.abs(group-center), axis=0) + 0.1
        distances.append(np.sqrt(np.mean(((z-center)/scale)**2, axis=1)))
    return np.min(distances, axis=0)


def select(model, X, budget, strategy, rng, calibration_X, calibration_y,
           reference_X, reference_y, weights):
    if not 0 <= budget <= len(X):
        raise ValueError("Invalid selection budget")
    if budget == 0:
        return np.array([], dtype=int), np.zeros(len(X))
    if strategy == "random":
        return rng.choice(len(X), budget, replace=False), np.zeros(len(X))
    p = model.predict_proba(X)
    uncertainty = 1 - 2 * np.abs(p - 0.5)
    if strategy == "uncertainty":
        score = uncertainty
    elif strategy == "conformal":
        score = 1 - conformal_credibility(model, X, calibration_X, calibration_y)
    elif strategy in {"distance", "hybrid"}:
        distance = latent_distance(model, X, reference_X, reference_y)
        score = distance
        if strategy == "hybrid":
            cred = conformal_credibility(model, X, calibration_X, calibration_y)
            score = weights[0]*rank01(1-cred) + weights[1]*rank01(uncertainty) + weights[2]*rank01(distance)
    else:
        raise ValueError(f"Unknown selector {strategy}")
    # Random tie-breaking is reproducible with a dedicated RNG.
    ordered = np.lexsort((rng.random(len(X)), -score))
    if strategy != "hybrid":
        return ordered[:budget], score
    candidates = ordered[:min(2*budget, len(X))]
    z = model.encode(X[candidates])
    chosen = [0]
    distance_to_chosen = np.sum((z-z[0])**2, axis=1)
    while len(chosen) < budget:
        distance_to_chosen[chosen] = -1
        nxt = int(np.argmax(distance_to_chosen)); chosen.append(nxt)
        distance_to_chosen = np.minimum(distance_to_chosen, np.sum((z-z[nxt])**2, axis=1))
    return candidates[chosen], score
