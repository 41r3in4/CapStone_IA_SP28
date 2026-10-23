"""Run a small ablation; optional full budget/seed grid. Sequential by design."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from driftlab.experiment import run,write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/demo.json')
    p.add_argument('--output',default='runs/suite');p.add_argument('--full-grid',action='store_true')
    a=p.parse_args();base=json.loads(Path(a.config).read_text(encoding='utf-8'))
    target=Path(a.output)
    if target.exists(): raise FileExistsError(target)
    target.mkdir(parents=True)
    methods=[('random','frozen'),('random','full_oracle'),('random','retrain'),('random','replay'),
             ('hybrid','replay'),('hybrid','replay_ewc'),('hybrid','replay_ewc_reg')]
    results=[]
    for beta in ([.01,.02,.05] if a.full_grid else [base['budget']]):
        for seed in (range(5) if a.full_grid else [base['seed']]):
            for selector,updater in methods:
                # Frozen/full oracle do not depend on label budget; avoid redundant grid runs.
                if a.full_grid and beta!=.01 and updater in {'frozen','full_oracle'}:continue
                c=dict(base,selector=selector,updater=updater,budget=beta,seed=seed,
                       output=str(target/f'{selector}_{updater}_b{beta}_s{seed}'))
                _,summary=run(c)
                results.append(dict(selector=selector,updater=updater,budget=beta,seed=seed,
                                    aut_f1=summary['aut_f1'],queried_labels=summary['queried_labels'],
                                    total_update_seconds=summary['total_update_seconds']))
                write_csv(target/'comparison.csv',results)
    print(target/'comparison.csv')


if __name__=='__main__': main()
