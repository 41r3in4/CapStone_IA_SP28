"""Audit both original datasets without pickle or whole-dataset dense loading."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from driftlab.drebin import audit_directory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',nargs='+',default=['data/gen_apigraph_drebin','data/gen_androzoo_drebin'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--headers-only',action='store_true',help='Explicitly incomplete feature-content audit')
    args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    results=[]
    for source in args.input:
        report=audit_directory(source,scan_features=not args.headers_only)
        name=Path(source).name
        (out/f'{name}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        rows=[dict(file=e['file'],period_start=e.get('period_start'),period_end=e.get('period_end'),
                   kind=e.get('kind'),samples=e.get('samples'),features=e.get('features'),
                   benign=e.get('benign'),malware=e.get('malware'),issues='; '.join(e['issues'])) for e in report['files']]
        if rows:
            with (out/f'{name}_period_counts.csv').open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        results.append(dict(dataset=name,structural_pass=report['structural_pass'],npz_count=report['npz_count'],
                            feature_scan=report['feature_scan'],issues=report['issues']))
    (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
    return 0 if all(r['structural_pass'] for r in results) else 1


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nAudit interrupted (Ctrl+C). Audit is incomplete; do not use it for smoke. Original data is unchanged. Rerun with a new output directory.',file=sys.stderr,flush=True)
        sys.exit(130)
