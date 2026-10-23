import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
from driftlab.apk import inspect_apk, manifest_elements, vectorize, predict, save_report, SCHEMA

XML = b'''<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="test.app"><uses-sdk android:minSdkVersion="21"/><uses-permission android:name="android.permission.INTERNET"/><application android:debuggable="true"><service android:name=".Svc"/></application></manifest>'''


class APKTests(unittest.TestCase):
    def test_extract_and_report(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            p = Path(tmp)/'test.apk'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('AndroidManifest.xml', XML); z.writestr('classes.dex', b'not executed')
            r = inspect_apk(p)
            self.assertEqual(r['package'], 'test.app')
            self.assertEqual(r['sha256'], hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertIn('permission:android.permission.INTERNET', r['feature_tokens'])
            self.assertIn('application:debuggable', r['feature_tokens'])
            self.assertIsNone(r['verdict'])
            save_report(r, Path(tmp)/'out')
            self.assertTrue((Path(tmp)/'out/apk_report.html').exists())
            with self.assertRaises(FileExistsError): save_report(r, Path(tmp)/'out')

    def test_binary_xml(self):
        pool = ['manifest', 'package', 'test.binary']
        payload = b''; offsets = []
        for s in pool:
            b = s.encode(); offsets.append(len(payload)); payload += bytes([len(s), len(b)])+b+b'\0'
        payload += b'\0' * (-len(payload) % 4)
        chunk = struct.pack('<HHI5I', 1, 28, 28+4*len(pool)+len(payload), len(pool), 0, 256, 28+4*len(pool), 0)
        chunk += struct.pack('<3I', *offsets)+payload
        element = struct.pack('<HHIII', 0x102, 16, 56, 1, 0xffffffff)
        element += struct.pack('<II6H', 0xffffffff, 0, 20, 20, 1, 0, 0, 0)
        element += struct.pack('<IIIHBBI', 0xffffffff, 1, 2, 8, 0, 3, 2)
        raw = struct.pack('<HHI', 3, 8, 8+len(chunk)+len(element))+chunk+element
        self.assertEqual(manifest_elements(raw), [('manifest', {'package':'test.binary'})])
        with self.assertRaises(ValueError): manifest_elements(raw[:-1])

    def test_invalid_archive(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            p = Path(tmp)/'bad.apk'; p.write_bytes(b'invalid')
            with self.assertRaises(ValueError): inspect_apk(p)
            with zipfile.ZipFile(p, 'w') as z: z.writestr('other', b'')
            with self.assertRaises(ValueError): inspect_apk(p)

    def test_feature_contract(self):
        self.assertEqual(vectorize(['permission:A'], ['f_permission:A', 'component:service']), [1,0])
        for names in (['f_0001'], ['component:unknown'], ['permission:A','f_permission:A']):
            with self.assertRaises(ValueError): vectorize([], names)

    def test_checkpoint_contract_and_math(self):
        try: import numpy as np
        except ImportError: self.skipTest('NumPy required for inference')
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            p = Path(tmp)/'model.npz'; card_path = Path(tmp)/'card.json'
            # Fabricated weights validate plumbing only, not malware accuracy.
            np.savez(p, W1=np.array([[1.]]), b1=np.array([0.]), W2=np.array([2.]),
                b2=np.array([0.]), mean=np.array([0.]), scale=np.array([2.]),
                feature_names=np.array(['f_permission:A']))
            card = dict(feature_schema=SCHEMA, training_data_kind='synthetic', model_id='TEST_ONLY',
                dataset_source='TEST_FIXTURE_NOT_RESEARCH', checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            card_path.write_text(json.dumps(card))
            with self.assertRaises(ValueError): predict({'feature_tokens':['permission:A']}, p, card_path)
            card['training_data_kind'] = 'real_android'; card_path.write_text(json.dumps(card))
            result = predict({'feature_tokens':['permission:A'], 'notes':[]}, p, card_path)
            self.assertAlmostEqual(result['malware_score'], 1/(1+np.exp(-1)))
            card['checkpoint_sha256'] = 'wrong'; card_path.write_text(json.dumps(card))
            with self.assertRaises(ValueError): predict({'feature_tokens':[]}, p, card_path)


if __name__ == '__main__': unittest.main()

