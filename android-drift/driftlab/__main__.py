import argparse
import json
from .data import generate_demo
from .experiment import run, SELECTORS, UPDATERS


def main():
    p=argparse.ArgumentParser(description="Android Drift research starter")
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo-data'); demo.add_argument('--output',default='data/demo.csv')
    demo.add_argument('--seed',type=int,default=7)
    r=sub.add_parser('run'); r.add_argument('--config',default='configs/demo.json')
    r.add_argument('--data'); r.add_argument('--output'); r.add_argument('--budget',type=float)
    r.add_argument('--seed',type=int); r.add_argument('--selector',choices=sorted(SELECTORS)); r.add_argument('--updater',choices=sorted(UPDATERS))
    a=p.parse_args()
    if a.command=='demo-data':
        generate_demo(a.output,a.seed); print(a.output); return
    c=json.loads(open(a.config,encoding='utf-8').read())
    for k in ('data','output','budget','seed','selector','updater'):
        if getattr(a,k) is not None: c[k]=getattr(a,k)
    run(c)


if __name__=='__main__': main()
