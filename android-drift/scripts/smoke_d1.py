"""Run only Frozen then Random+replay 1% on audited D1 2012 -> Jan-Mar 2013."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from driftlab.drebin import SCHEMA, sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); audit_root=Path(args.audit)
    report=json.loads((audit_root/'gen_apigraph_drebin.json').read_text(encoding='utf-8'))
    if not report['structural_pass'] or not report['feature_scan']:
        raise ValueError('D1 requires successful content audit before smoke')
    files=['2012-01to2012-12_selected.npz']+[f'2013-{m:02}_selected.npz' for m in range(1,4)]
    by_name={e['file']:e for e in report['files']}
    if not set(files).issubset(by_name): raise ValueError('Missing requested initial/evaluation periods')
    root=Path(report['root'])
    for filename in files:
        entry=by_name[filename]
        if entry['issues'] or not entry['benign'] or not entry['malware'] or sha256(root/filename)!=entry['sha256']:
            raise ValueError('Source mismatch or insufficient classes: '+filename)
    vocab='2012-01to2012-12_selected_training_features.json'
    vocab_entry=next(v for v in report['vocabularies'] if v['file']==vocab)
    if sha256(root/vocab)!=vocab_entry['sha256'] or sha256(root/'family_idx.json')!=report['family_idx_sha256']:
        raise ValueError('Audited schema changed')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(audit_root,out/'audit')
    spec=dict(feature_schema=SCHEMA,source_root=str(root),files=files,vocabulary_file=vocab,
        vocabulary_sha256=vocab_entry['sha256'],family_idx_sha256=report['family_idx_sha256'],
        source_sha256={f:by_name[f]['sha256'] for f in files},chunk_bytes=8*1024*1024,
        max_csr_bytes=64*1024*1024,
        purpose='Pipeline smoke; original data untouched; no final research claim')
    (out/'dataset.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    base=json.loads(Path('configs/demo.json').read_text(encoding='utf-8'))
    base.update(data=str(out/'dataset.json'),training_data_kind='real_android',feature_schema=SCHEMA,
        initial_range='2012-01to2012-12',selector='random',budget=.01,seed=42,
        epochs=3,update_epochs=2,batch_size=128,hidden=16,max_features=512,memory_size=512,
        fisher_samples=16,blas_threads=1,source_audit=str(out/'audit'/'gen_apigraph_drebin.json'),
        purpose='Pipeline smoke only; no tuning on 2013 test; no final research conclusion')
    commands=[]
    for name,updater in [('frozen','frozen'),('random_replay_b001','replay')]:
        config=dict(base,updater=updater,output=str(out/name))
        path=out/(name+'.config.json'); path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        command=[sys.executable,'-m','driftlab','run','--config',str(path)]
        print('Running '+name,flush=True)
        with (out/(name+'.console.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        commands.append(dict(command=command,exit_code=result.returncode))
        if result.returncode: raise RuntimeError('Smoke failed; see '+str(out/(name+'.console.log')))
    rows=[]
    for name in ('frozen','random_replay_b001'):
        summary=json.loads((out/name/'summary.json').read_text(encoding='utf-8'))
        rows.append(dict(method=name,aut_f1=summary['aut_f1'],queried_labels=summary['queried_labels'],
            initial_fit_labels=summary['initial_fit_labels'],peak_ram=summary['peak_ram'],
            total_seconds=summary['total_seconds'],feature_schema=summary['feature_schema']))
    unchanged=all(sha256(root/name)==digest for name,digest in spec['source_sha256'].items())
    unchanged=unchanged and sha256(root/vocab)==spec['vocabulary_sha256'] and sha256(root/'family_idx.json')==spec['family_idx_sha256']
    with (out/'verification.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(commands=commands,source_files_unchanged=unchanged,results=rows,
            purpose='Pipeline smoke only; 2012 pooled initial fit; Jan-Mar 2013 pre-update evaluation',
            limitations=report['limitations'],audit_warnings=report['warnings']),stream,indent=2)
    if not unchanged: raise ValueError('Source changed during smoke')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
