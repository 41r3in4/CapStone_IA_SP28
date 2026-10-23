import argparse
import json
from pathlib import Path


def main():
    # Keep APK inspection usable with Python's standard library alone.
    p=argparse.ArgumentParser(description="Android Drift research starter")
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo-data'); demo.add_argument('--output',default='data/demo.csv')
    demo.add_argument('--seed',type=int,default=7)
    r=sub.add_parser('run'); r.add_argument('--config',default='configs/demo.json')
    r.add_argument('--data'); r.add_argument('--output'); r.add_argument('--budget',type=float)
    r.add_argument('--seed',type=int); r.add_argument('--selector'); r.add_argument('--updater')
    r.add_argument('--resume-from'); r.add_argument('--stop-after-month'); r.add_argument('--query-manifest')
    export = sub.add_parser('export-model')
    export.add_argument('--run',required=True); export.add_argument('--output',required=True)
    export.add_argument('--declaration',required=True)
    package = sub.add_parser('predict-package')
    package.add_argument('--apk',required=True); package.add_argument('--package',required=True)
    package.add_argument('--output',required=True)
    svm = sub.add_parser('svm-run'); svm.add_argument('--config',required=True)
    analysis = sub.add_parser('analyze-runs'); analysis.add_argument('--comparison',required=True)
    analysis.add_argument('--output',required=True); analysis.add_argument('--block-length',type=int)
    lock = sub.add_parser('lock-protocol'); lock.add_argument('--config',required=True)
    lock.add_argument('--output',required=True); lock.add_argument('--audit-reviewed',action='store_true')
    lock.add_argument('--locked-before-test',action='store_true')
    for name in ('inspect-apk', 'predict-apk'):
        apk = sub.add_parser(name)
        apk.add_argument('--apk', required=True); apk.add_argument('--output', required=True)
        if name == 'predict-apk':
            apk.add_argument('--checkpoint', required=True); apk.add_argument('--model-card', required=True)
    batch = sub.add_parser('extract-batch')
    batch.add_argument('--input', default='data/apks', help='APK directory (default: data/apks; includes subdirectories)')
    batch.add_argument('--output', required=True)
    metadata = sub.add_parser('validate-metadata')
    metadata.add_argument('--metadata', required=True)
    build = sub.add_parser('build-dataset')
    build.add_argument('--extractions', required=True); build.add_argument('--metadata', required=True)
    build.add_argument('--output', required=True); build.add_argument('--train-through', required=True)
    build.add_argument('--max-elements', type=int, default=5_000_000)
    build.add_argument('--fit-ids',help='JSON array of exact train IDs for vocabulary fitting')
    protocol = sub.add_parser('audit-data')
    protocol.add_argument('--data', required=True); protocol.add_argument('--output', required=True)
    for boundary in ('train-through', 'validation-through', 'calibration-through', 'monitor-through'):
        protocol.add_argument('--' + boundary, required=True)
    a=p.parse_args()
    if a.command == 'lock-protocol':
        from .locking import lock_protocol
        lock_protocol(json.loads(Path(a.config).read_text(encoding='utf-8')),a.output,a.audit_reviewed,a.locked_before_test)
        return
    if a.command == 'analyze-runs':
        from .analysis import analyze
        analyze(a.comparison,a.output,a.block_length)
        return
    if a.command == 'svm-run':
        from .svm import run_svm
        run_svm(json.loads(Path(a.config).read_text(encoding='utf-8')))
        return
    if a.command == 'export-model':
        from .package import export_package
        export_package(a.run,a.output,json.loads(Path(a.declaration).read_text(encoding='utf-8')))
        return
    if a.command == 'predict-package':
        from .package import predict_package
        from .apk import inspect_apk, save_report
        save_report(predict_package(inspect_apk(a.apk),a.package),a.output)
        return
    if a.command == 'audit-data':
        from .data import load_dataset
        from .protocol import temporal_split, save_protocol
        data = load_dataset(a.data)
        groups = temporal_split(data, a.train_through, a.validation_through, a.calibration_through, a.monitor_through)
        save_protocol(a.output, data, groups)
        print(a.output)
        return
    if a.command in ('extract-batch', 'validate-metadata', 'build-dataset'):
        from .preparation import batch_extract, validate_metadata, build_dataset
        try:
            if a.command == 'extract-batch':
                result = batch_extract(a.input, a.output)
            elif a.command == 'validate-metadata':
                result = {'valid': True, 'samples': len(validate_metadata(a.metadata))}
            else:
                fit_ids = json.loads(Path(a.fit_ids).read_text(encoding='utf-8')) if a.fit_ids else None
                info = build_dataset(a.extractions, a.metadata, a.output, a.train_through, a.max_elements, fit_ids)
                result = {'samples': info['samples'], 'features': len(info['feature_names']), 'output': a.output}
        except (ValueError, OSError, KeyError, TypeError) as e:
            p.exit(2, 'Data preparation failed: '+str(e)+'\n')
        print(json.dumps(result, indent=2))
        if a.command == 'extract-batch' and result['counts']['error']:
            p.exit(1)
        return
    if a.command in ('inspect-apk', 'predict-apk'):
        from .apk import inspect_apk, predict, save_report
        try:
            report = inspect_apk(a.apk)
            if a.command == 'predict-apk':
                report = predict(report, a.checkpoint, a.model_card)
            save_report(report, a.output)
        except (ValueError, OSError, KeyError, ImportError) as e:
            p.exit(2, 'APK check failed: '+str(e)+'\n')
        print(json.dumps({'status': report['status'], 'package': report['package'],
            'verdict': report['verdict'], 'sha256': report['sha256'], 'output': a.output}, indent=2))
        return
    if a.command=='demo-data':
        from .data import generate_demo
        generate_demo(a.output,a.seed); print(a.output); return
    from .experiment import run
    c=json.loads(Path(a.config).read_text(encoding='utf-8'))
    for k in ('data','output','budget','seed','selector','updater','resume_from','stop_after_month','query_manifest'):
        if getattr(a,k) is not None: c[k]=getattr(a,k)
    run(c)


if __name__=='__main__': main()
