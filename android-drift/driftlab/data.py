import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Dataset:
    ids: np.ndarray
    months: np.ndarray
    X: np.ndarray
    y: np.ndarray
    features: list
    sha256: str


def load_csv(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
        if not {"sample_id", "month", "label"}.issubset(names):
            raise ValueError("CSV requires sample_id, month, label, plus f_* columns")
        features = [k for k in names if k.startswith("f_")]
        if not features:
            raise ValueError("No f_* feature columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty dataset")
    ids = np.array([r["sample_id"] for r in rows])
    months = np.array([r["month"] for r in rows])
    if any(not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be nonempty and globally unique")
    if any(not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", m) for m in months):
        raise ValueError("Invalid month; expected YYYY-MM")
    if any(r["label"] not in {"0", "1"} for r in rows):
        raise ValueError("Labels must be exactly 0 or 1")
    X = np.array([[float(r[k]) for k in features] for r in rows], dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity")
    y = np.array([int(r["label"]) for r in rows])
    return Dataset(ids, months, X, y, features, hashlib.sha256(path.read_bytes()).hexdigest())


def initial_split(data, initial_months, seed):
    months = sorted(set(data.months))
    if initial_months < 1 or len(months) <= initial_months:
        raise ValueError("Need initial_months >= 1 and at least one future month")
    # Contiguous monthly periods required for the normalized AUT implementation.
    serial = [int(m[:4]) * 12 + int(m[5:]) - 1 for m in months]
    if any(b - a != 1 for a, b in zip(serial, serial[1:])):
        raise ValueError("Missing calendar month: fix dataset or evaluation protocol")
    initial = np.flatnonzero(np.isin(data.months, months[:initial_months]))
    rng = np.random.default_rng(seed)
    splits = [[], [], []]  # fit, calibration, regression monitor
    for label in (0, 1):
        idx = rng.permutation(initial[data.y[initial] == label])
        if len(idx) < 12:
            raise ValueError("Need >=12 initial examples per class for disjoint splits")
        n = max(2, int(len(idx) * 0.15))
        splits[1].extend(idx[:n]); splits[2].extend(idx[n:2*n]); splits[0].extend(idx[2*n:])
    return tuple(np.array(s, dtype=int) for s in splits), months[initial_months:]


def generate_demo(path, seed=7, months=20, samples=180, features=24):
    if months < 13 or samples < 80 or features < 6:
        raise ValueError("Demo needs >=13 months, >=80 samples/month, >=6 features")
    rng = np.random.default_rng(seed)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "month", "label"] + [f"f_{j:04}" for j in range(features)])
        base = np.zeros(features); base[:6] = 0.18
        for t in range(months):
            month = f"{2020+t//12:04}-{t%12+1:02}"
            y = rng.binomial(1, 0.15, samples)
            strength = max(0, (t - 10) / max(1, months - 11))
            effect = base.copy(); effect[:3] *= 1 - 1.8 * strength
            effect[6:12] = 0.22 * strength
            probabilities = np.clip(0.12 + y[:, None] * effect, 0.02, 0.95)
            X = rng.binomial(1, probabilities)
            for i in range(samples):
                w.writerow([f"synthetic-{t:02}-{i:05}", month, int(y[i]), *X[i]])
