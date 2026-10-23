import numpy as np


def ratio(a, b):
    return float(a / b) if b else None


def metrics(y, predicted):
    tp = int(np.sum((y == 1) & (predicted == 1)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, f1=ratio(2*tp, 2*tp+fp+fn), fnr=ratio(fn,tp+fn), fpr=ratio(fp,tn+fp))


def regression(y, old, new):
    eligible = (y == 1) & (old == 1)
    denominator = int(eligible.sum())
    numerator = int(np.sum(eligible & (new == 0)))
    return dict(regression_count=numerator, regression_eligible=denominator, regression=ratio(numerator,denominator))


def aut(values):
    if len(values) < 2 or any(v is None for v in values):
        return None
    return float(np.mean((np.array(values[:-1])+np.array(values[1:]))/2))
