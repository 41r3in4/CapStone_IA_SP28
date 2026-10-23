"""Record a reviewed, historical protocol before running a real suite."""
import hashlib
import json
from pathlib import Path
from .data import load_dataset
from .protocol import temporal_split, audit
from .resume import fingerprint


def lock_protocol(config, output, audit_reviewed=False, locked_before_test=False):
    if not audit_reviewed or not locked_before_test:
        raise ValueError('Require explicit audit-reviewed and locked-before-test declarations')
    data=load_dataset(config['data']); groups=temporal_split(data,**config['temporal'])
    report=audit(data,groups)
    result=dict(data_sha256=data.sha256,
                config_sha256=fingerprint({k:v for k,v in config.items() if k!='protocol_lock'}),
                audit_reviewed=True,locked_before_test=True,audit=report,
                declarations='Review and timing are user declarations, not independently verified',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    with Path(output).open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    return result
