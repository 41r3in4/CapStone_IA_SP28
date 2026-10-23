"""Run local checks and write measured verification artifacts, never APK execution."""
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    os.chdir(ROOT)
    warnings.simplefilter('error',ResourceWarning)
    start=time.perf_counter()
    log=ROOT/'docs'/'verification-tests.log'
    with log.open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
    repository=ROOT.parent
    command=['git','-c','safe.directory='+repository.as_posix(),'diff','--check']
    diff=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    sources=[*ROOT.glob('driftlab/*.py'),*ROOT.glob('tests/*.py'),*ROOT.glob('scripts/*.py')]
    whitespace=[]
    for path in sources:
        for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            if line.rstrip()!=line: whitespace.append(f'{path.relative_to(ROOT)}:{number}')
    verification=dict(date=datetime.now(timezone(timedelta(hours=7))).date().isoformat(),timezone='Asia/Bangkok',python=sys.version,
        platform=platform.platform(),dependencies={name:importlib.metadata.version(name) for name in
            ('numpy','scipy','scikit-learn','joblib','threadpoolctl','narwhals','cloudpickle')},
        tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            skipped=len(result.skipped),successful=result.wasSuccessful(),log=str(log.relative_to(ROOT)),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),
        git_diff_check=dict(command=command,exit_code=diff.returncode,stdout=diff.stdout,stderr=diff.stderr),
        untracked_source_whitespace=whitespace,elapsed_seconds=time.perf_counter()-start,
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Generated synthetic fixtures and software contracts only',
        not_run=['Real Android research grid','Real APK malware accuracy','Real model calibration','GPU workload'])
    (ROOT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(dict(tests=result.testsRun,successful=result.wasSuccessful(),
                         git_diff_check=diff.returncode,source_whitespace_errors=len(whitespace)),indent=2))
    return 0 if result.wasSuccessful() and diff.returncode==0 and not whitespace else 1


if __name__=='__main__': sys.exit(main())
