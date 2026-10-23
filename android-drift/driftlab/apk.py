"""Offline manifest extraction, without installing or executing APKs."""
import hashlib
import html
import json
import struct
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

SCHEMA = 'android-manifest-v1'
ANDROID = '{http://schemas.android.com/apk/res/android}'
LIMIT = 16 * 1024 * 1024


def u32(b, p):
    return struct.unpack_from('<I', b, p)[0]


def length(b, p, utf8):
    if utf8:
        n = b[p]; p += 1
        if n & 128:
            n = ((n & 127) << 8) | b[p]; p += 1
    else:
        n = struct.unpack_from('<H', b, p)[0]; p += 2
        if n & 32768:
            n = ((n & 32767) << 16) | struct.unpack_from('<H', b, p)[0]; p += 2
    return n, p


def strings(chunk):
    header = struct.unpack_from('<H', chunk, 2)[0]
    count, _, flags, start, _ = struct.unpack_from('<5I', chunk, 8)
    if header < 28 or count > (len(chunk)-header)//4:
        raise ValueError('Invalid string pool')
    result = []
    for i in range(count):
        p = start + u32(chunk, header+4*i)
        if flags & 256:
            _, p = length(chunk, p, True)
            n, p = length(chunk, p, True); end = p+n
            encoding = 'utf-8'
        else:
            n, p = length(chunk, p, False); end = p+2*n
            encoding = 'utf-16le'
        if end > len(chunk): raise ValueError('Truncated string')
        result.append(chunk[p:end].decode(encoding))
    return result


def manifest_elements(raw):
    if raw.lstrip().startswith(b'<'):
        root = ET.fromstring(raw)
        return [(e.tag.split('}')[-1], dict(e.attrib)) for e in root.iter()]
    if len(raw) < 8 or struct.unpack_from('<H', raw)[0] != 3:
        raise ValueError('Unsupported AndroidManifest.xml encoding')
    total = u32(raw, 4)
    if total != len(raw): raise ValueError('Truncated binary XML')
    p = struct.unpack_from('<H', raw, 2)[0]
    if p < 8: raise ValueError('Invalid XML header')
    pool = []; elements = []
    def s(index):
        if index == 0xffffffff: return ''
        if index >= len(pool): raise ValueError('Invalid string index')
        return pool[index]
    while p < total:
        if p+8 > total: raise ValueError('Truncated XML chunk')
        kind, header, size = struct.unpack_from('<HHI', raw, p)
        if size < header or header < 8 or p+size > total or not size:
            raise ValueError('Invalid XML chunk')
        chunk = raw[p:p+size]
        if kind == 1: pool = strings(chunk)
        elif kind == 0x102:
            if header < 16 or header+20 > size: raise ValueError('Invalid element')
            ext = header; tag = s(u32(chunk, ext+4))
            offset, stride, count = struct.unpack_from('<HHH', chunk, ext+8)
            if offset < 20 or stride < 20 or ext+offset+stride*count > size:
                raise ValueError('Invalid attributes')
            attrs = {}
            for i in range(count):
                a = ext+offset+i*stride
                ns, name, value = struct.unpack_from('<III', chunk, a)
                dtype = chunk[a+15]; typed = u32(chunk, a+16)
                key = ('{'+s(ns)+'}' if ns != 0xffffffff else '')+s(name)
                if value != 0xffffffff: v = s(value)
                elif dtype == 3: v = s(typed)
                elif dtype == 18: v = 'true' if typed else 'false'
                elif dtype in (16, 17): v = str(typed)
                else: v = '@resource:'+hex(typed)
                attrs[key] = v
            elements.append((tag, attrs))
        p += size
    if not any(t == 'manifest' for t, _ in elements): raise ValueError('Manifest root missing')
    return elements


def inspect_apk(path):
    path = Path(path); digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    try:
        with zipfile.ZipFile(path) as z:
            matches = [x for x in z.infolist() if x.filename == 'AndroidManifest.xml']
            if len(matches) != 1: raise ValueError('APK must contain exactly one AndroidManifest.xml')
            info = matches[0]
            if info.file_size > LIMIT: raise ValueError('Manifest exceeds 16 MiB limit')
            with z.open(info) as f: raw = f.read(LIMIT+1)
            if len(raw) > LIMIT: raise ValueError('Manifest exceeds size limit')
            elements = manifest_elements(raw)
            dex = [x.filename for x in z.infolist() if x.filename.startswith('classes') and x.filename.endswith('.dex')]
    except (zipfile.BadZipFile, struct.error, IndexError, UnicodeError, ET.ParseError) as e:
        raise ValueError('Invalid APK/manifest: '+str(e)) from e
    root = next((a for t, a in elements if t == 'manifest'), None)
    if root is None: raise ValueError('Manifest root missing')
    app = next((a for t, a in elements if t == 'application'), {})
    permissions = sorted({a[ANDROID+'name'] for t, a in elements
        if t in ('uses-permission', 'uses-permission-sdk-23') and ANDROID+'name' in a})
    components = {t: sorted({a.get(ANDROID+'name', '') for tag, a in elements if tag == t})
        for t in ('activity', 'activity-alias', 'service', 'receiver', 'provider')}
    sdk = next((a for t, a in elements if t == 'uses-sdk'), {})
    tokens = ['permission:'+p for p in permissions]
    tokens += ['component:'+t for t, names in components.items() if names]
    tokens += ['application:'+k for k in ('debuggable', 'usesCleartextTraffic') if app.get(ANDROID+k) == 'true']
    return dict(file=str(path.resolve()), size_bytes=path.stat().st_size, sha256=digest.hexdigest(),
        feature_schema=SCHEMA, package=root.get('package'), version_name=root.get(ANDROID+'versionName'),
        version_code=root.get(ANDROID+'versionCode'), min_sdk=sdk.get(ANDROID+'minSdkVersion'),
        target_sdk=sdk.get(ANDROID+'targetSdkVersion'), permissions=permissions, components=components,
        dex_files=dex, feature_tokens=sorted(tokens), status='inspection_only', verdict=None,
        notes=['Static manifest inspection only; APK was not installed or executed.',
        'No DEX/API extraction, signature verification or malware verdict from permissions.',
        'Missing manifest flags are not interpreted as false; platform defaults may differ.'])


def vectorize(tokens, names):
    if not names or len(names) != len(set(names)): raise ValueError('Feature names must be nonempty and unique')
    names = [n[2:] if isinstance(n, str) and n.startswith('f_') else n for n in names]
    if len(names) != len(set(names)): raise ValueError('Duplicate canonical feature names')
    valid_flags = {'application:debuggable', 'application:usesCleartextTraffic'}
    valid_components = {'component:'+x for x in ('activity', 'activity-alias', 'service', 'receiver', 'provider')}
    if any(not isinstance(n, str) or not (n.startswith('permission:') and len(n) > 11 or n in valid_flags or n in valid_components) for n in names):
        raise ValueError('Features do not match manifest-v1; anonymous f_* features are unsupported')
    present = set(tokens)
    return [int(n in present) for n in names]


def predict(report, checkpoint, card_path, fixture=False):
    import numpy as np
    from .model import MLP
    card = json.loads(Path(card_path).read_text(encoding='utf-8-sig'))
    if report.get('feature_schema', SCHEMA) != SCHEMA:
        raise ValueError('Extractor report schema mismatch')
    if card.get('package_version'):
        from .package import verify_package
        verify_package(Path(card_path).parent)
        if Path(checkpoint).resolve() != (Path(card_path).parent/'checkpoint.npz').resolve():
            raise ValueError('Package checkpoint path mismatch')
    if card.get('feature_schema') != SCHEMA or (card.get('training_data_kind') != 'real_android' and not (fixture and card.get('training_data_kind') == 'synthetic')):
        raise ValueError('Model card must declare real_android training with android-manifest-v1 features')
    if not card.get('dataset_source') or not card.get('model_id'):
        raise ValueError('Model card requires model_id and dataset_source')
    h = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if card.get('checkpoint_sha256') != h: raise ValueError('Model card checkpoint SHA256 mismatch')
    threshold = float(card.get('threshold', 0.5))
    if not 0 < threshold < 1: raise ValueError('Threshold must be in (0,1)')
    with np.load(checkpoint, allow_pickle=False) as c:
        names = c['feature_names'].tolist()
        x = np.array([vectorize(report['feature_tokens'], names)], dtype=float)
        dim = len(names)
        if c['W1'].ndim != 2: raise ValueError('Invalid W1 shape')
        hidden = c['W1'].shape[1]
        shapes = {'W1':(dim, hidden), 'b1':(hidden,), 'W2':(hidden,), 'b2':(1,), 'mean':(dim,), 'scale':(dim,)}
        for key, shape in shapes.items():
            if c[key].shape != shape or not np.isfinite(c[key]).all(): raise ValueError('Invalid checkpoint: '+key)
        if np.any(c['scale'] <= 0): raise ValueError('Checkpoint scale must be positive')
        model = MLP(dim, hidden); model.p = {key:c[key].copy() for key in model.p}
        score = float(model.predict_proba((x-c['mean'])/c['scale'])[0])
    report.update(status='fixture_prediction' if fixture else 'model_prediction', verdict=None if fixture else ('malware' if score >= threshold else 'benign'),
        malware_score=score, threshold=threshold, model_id=card['model_id'], checkpoint_sha256=h,
        matched_present_features=sum(vectorize(report['feature_tokens'], names)))
    report['notes'].append('Model prediction, not a guarantee of safety. Provenance is declared by the model author.')
    report['notes'].append('Score is not a verified calibrated probability.' if not fixture else 'Synthetic fixture plumbing test only; no Android accuracy claim.')
    return report


def save_report(report, output):
    out = Path(output)
    if out.exists(): raise FileExistsError('Choose a new output directory: '+str(out))
    out.mkdir(parents=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (out/'apk_report.json').write_text(text, encoding='utf-8')
    (out/'features.json').write_text(json.dumps({'feature_schema':SCHEMA, 'sample_id':report['sha256'],
        'feature_tokens':report['feature_tokens']}, ensure_ascii=False, indent=2), encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><title>APK static report</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:20px}</style><h1>APK static report</h1><p>Status: '+html.escape(report['status'])+'</p><p>Verdict: '+html.escape(str(report['verdict']))+'</p><pre>'+html.escape(text)+'</pre>'
    (out/'apk_report.html').write_text(page, encoding='utf-8')
