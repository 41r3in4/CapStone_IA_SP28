"""Summarize saved runs with explicit experimental units and dependence notes."""
import csv
import json
from pathlib import Path
from .statistics import aggregate, paired, friedman, block_bootstrap_aut


def analyze(comparison, output, block_length=None):
    with Path(comparison).open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        row['budget']=float(row['budget']); row['seed']=int(row['seed'])
        row['aut_f1']=float(row['aut_f1']) if row['aut_f1'] else None
    result=dict(aggregate=aggregate(rows),tests=[],block_bootstrap=[],
                unit='Matched seed AUT within each budget; monthly observations are not independent replicates')
    for budget in sorted({r['budget'] for r in rows}):
        methods={}
        for row in rows:
            if row['budget']==budget: methods.setdefault(row['method'],{})[row['seed']]=row['aut_f1']
        if len(methods)>=3:
            test=friedman(methods)
        elif len(methods)==2:
            names=sorted(methods); test=dict(a=names[0],b=names[1],**paired(methods[names[0]],methods[names[1]]))
        else:
            test=dict(pvalue=None,reason='Need >=2 comparable methods')
        result['tests'].append(dict(budget=budget,**test))
    if block_length is not None:
        for row in rows:
            if 'run_dir' not in row:
                raise ValueError('Block bootstrap requires run_dir in comparison CSV')
            with (Path(row['run_dir'])/'monthly_metrics.csv').open(encoding='utf-8',newline='') as stream:
                months=list(csv.DictReader(stream))
            if [r['month'] for r in months] != sorted({r['month'] for r in months}):
                raise ValueError('Monthly metrics must be uniquely ordered')
            result['block_bootstrap'].append(dict(method=row['method'],budget=row['budget'],seed=row['seed'],
                **block_bootstrap_aut([float(r['f1']) for r in months],block_length)))
    with Path(output).open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    return result
