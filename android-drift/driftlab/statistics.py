"""Statistics over explicit paired units; no fabricated missing observations."""
import numpy as np
from scipy import stats
from .metrics import aut


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Need finite observations')
    mean = float(x.mean())
    if len(x) < 2:
        return dict(n=len(x), mean=mean, std=None, ci95=None, reason='Need >=2 independent units')
    std = float(x.std(ddof=1))
    half = float(stats.t.ppf(.975,len(x)-1)*std/np.sqrt(len(x)))
    return dict(n=len(x), mean=mean, std=std, ci95=[mean-half,mean+half],
                assumption='Student-t CI over independent seed/run units; not monthly samples')


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p-values')
    order=np.argsort(p); adjusted=np.empty(len(p)); previous=0.
    for rank, idx in enumerate(order):
        previous=max(previous,min(1.,float(p[idx]*(len(p)-rank))))
        adjusted[idx]=previous
    return adjusted.tolist()


def paired(a, b, unit='matched seed AUT'):
    if set(a) != set(b):
        raise ValueError('Paired units must match exactly')
    keys=sorted(a); x=np.array([a[k] for k in keys]); y=np.array([b[k] for k in keys])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite paired values')
    if len(keys)<5:
        return dict(unit=unit,n=len(keys),pvalue=None,reason='Need >=5 paired units; small samples have low power')
    delta=x-y
    return dict(unit=unit,n=len(keys),pvalue=1. if np.all(delta==0) else float(stats.wilcoxon(delta,method='auto').pvalue),
                mean_difference=float(delta.mean()),assumptions='Independent pairs; symmetric differences for signed-rank inference')


def friedman(methods):
    names=sorted(methods)
    if len(names)<3 or len({frozenset(v) for v in methods.values()})!=1:
        raise ValueError('Friedman needs >=3 methods on identical paired units')
    keys=sorted(methods[names[0]])
    if len(keys)<5:
        return dict(pvalue=None,reason='Need >=5 matched units')
    arrays=[np.array([methods[n][k] for k in keys],dtype=float) for n in names]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite Friedman values')
    # Permute method assignments within blocks rather than relying on a small-n
    # chi-square approximation (the default design has just five seeds).
    ranks=stats.rankdata(np.stack(arrays),axis=0)
    statistic=float(np.sum(ranks.sum(axis=1)**2))
    rng=np.random.default_rng(42); exceed=0; repetitions=4999
    for _ in range(repetitions):
        permuted=np.column_stack([rng.permutation(ranks[:,i]) for i in range(len(keys))])
        exceed+=float(np.sum(permuted.sum(axis=1)**2))>=statistic-1e-12
    p=float((exceed+1)/(repetitions+1))
    pairs=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pairs.append(dict(a=a,b=b,**paired(methods[a],methods[b])))
    for result, adjusted in zip(pairs,holm([r['pvalue'] for r in pairs])):
        result['holm_pvalue']=adjusted
    return dict(pvalue=p,posthoc=pairs,unit='matched seed AUT conditional on this dataset',
                test='Friedman rank statistic with within-block Monte Carlo permutation',
                permutations=repetitions,permutation_seed=42,
                assumptions='Independent blocks and exchangeable method assignments under null; posthoc Wilcoxon/Holm; five seeds give low power')


def block_bootstrap_aut(values, block_length, repetitions=2000, seed=42):
    x=np.asarray(values,dtype=float)
    if len(x)<4 or not np.isfinite(x).all() or not 1<=block_length<len(x) or repetitions<100:
        raise ValueError('Need >=4 finite ordered months, valid block length and >=100 replicates')
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(repetitions):
        starts=rng.integers(0,len(x)-block_length+1,size=int(np.ceil(len(x)/block_length)))
        draw=np.concatenate([x[s:s+block_length] for s in starts])[:len(x)]
        samples.append(aut(draw))
    return dict(aut=aut(x),ci95=np.quantile(samples,[.025,.975]).tolist(),block_length=block_length,
                repetitions=repetitions,unit='ordered monthly F1 sequence',
                assumptions='Moving contiguous blocks preserve local dependence; approximate stationarity required; drift may violate it')


def aggregate(rows):
    groups={}
    for row in rows:
        key=(row['method'],float(row['budget']))
        group=groups.setdefault(key,{})
        if row['seed'] in group:
            raise ValueError('Duplicate method/budget/seed')
        if row['aut_f1'] is None:
            raise ValueError('AUT unavailable; do not silently impute')
        group[row['seed']]=float(row['aut_f1'])
    return [dict(method=m,budget=b,**describe(list(v.values()))) for (m,b),v in sorted(groups.items())]
