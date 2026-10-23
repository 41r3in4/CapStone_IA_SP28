import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from driftlab.preparation import batch_extract, validate_metadata, build_dataset
from driftlab.data import load_csv


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        apks = root / 'apks'
        apks.mkdir()
        for name, permission in [('a', 'PAST'), ('b', 'FUTURE')]:
            with zipfile.ZipFile(apks / (name + '.apk'), 'w') as archive:
                archive.writestr('AndroidManifest.xml', '<manifest xmlns:android="http://schemas.android.com/apk/res/android"><uses-permission android:name="' + permission + '"/></manifest>')
        (apks / 'copy.apk').write_bytes((apks / 'a.apk').read_bytes())
        (apks / 'bad.apk').write_bytes(b'invalid')
        batch = root / 'batch'
        summary = batch_extract(apks, batch)
        records = [json.loads(s) for s in (batch / 'extractions.jsonl').read_text().splitlines()]
        rows = []
        for record in records:
            if record['status'] == 'ok':
                rows.append(dict(sample_id=record['sample_id'], month='2024-01' if record['file'] == 'a.apk' else '2024-02', label='0' if record['file'] == 'a.apk' else '1', source='synthetic fixture'))
        metadata = root / 'metadata.csv'
        self.write_metadata(metadata, rows)
        return batch, metadata, rows, summary

    def write_metadata(self, path, rows):
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_id', 'month', 'label', 'source'])
            writer.writeheader()
            writer.writerows(rows)

    def test_end_to_end_no_future_vocabulary_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            batch, metadata, rows, summary = self.fixture(root)
            self.assertEqual(summary['counts'], dict(ok=2, duplicate=1, error=1))
            out = root / 'dataset'
            info = build_dataset(batch / 'extractions.jsonl', metadata, out, '2024-01')
            data = load_csv(out / 'dataset.csv')
            self.assertEqual(data.features, ['f_permission:PAST'])
            self.assertEqual(data.X.tolist(), [[1.0], [0.0]])
            self.assertEqual(info['future_only_tokens'], ['permission:FUTURE'])
            self.assertEqual(info['dataset_sha256'], data.sha256)
            with self.assertRaises(FileExistsError):
                build_dataset(batch / 'extractions.jsonl', metadata, out, '2024-01')
            with self.assertRaises(FileExistsError):
                batch_extract(root / 'apks', batch)

    def test_invalid_metadata(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            batch, metadata, rows, _ = self.fixture(root)
            for key, value in [('label', '2'), ('month', '2024-13'), ('sample_id', 'bad'), ('source', ' ')]:
                changed = [dict(r) for r in rows]
                changed[0][key] = value
                self.write_metadata(metadata, changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_metadata(metadata)
            self.write_metadata(metadata, rows + [rows[0]])
            with self.assertRaises(ValueError):
                validate_metadata(metadata)

    def test_exact_vocabulary_fit_ids(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); batch,metadata,rows,_=self.fixture(root)
            fit_ids=[next(r['sample_id'] for r in rows if r['month']=='2024-01')]
            info=build_dataset(batch/'extractions.jsonl',metadata,root/'dataset','2024-02',fit_ids=fit_ids)
            self.assertEqual(info['vocabulary_fit_ids'],fit_ids)
            self.assertEqual(info['feature_names'],['f_permission:PAST'])
            with self.assertRaises(ValueError):
                build_dataset(batch/'extractions.jsonl',metadata,root/'bad','2024-01',fit_ids=['0'*64])

    def test_missing_extraction_limit_and_conflict(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            batch, metadata, rows, _ = self.fixture(root)
            extraction = batch / 'extractions.jsonl'
            for kwargs in [dict(train_through='2023-12'), dict(train_through='2024-01', max_elements=1)]:
                with self.assertRaises(ValueError):
                    build_dataset(extraction, metadata, root / 'out', **kwargs)
                self.assertFalse((root / 'out').exists())
            original = extraction.read_text()
            extraction.write_text(original + json.dumps(dict(status='ok', sample_id=rows[0]['sample_id'], feature_schema='android-manifest-v1', feature_tokens=['permission:CONFLICT'])) + '\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                build_dataset(extraction, metadata, root / 'out', '2024-01')
            extraction.write_text(original)
            rows[0]['sample_id'] = '0' * 64
            self.write_metadata(metadata, rows)
            with self.assertRaisesRegex(ValueError, 'missing successful extraction'):
                build_dataset(extraction, metadata, root / 'out', '2024-01')


if __name__ == '__main__':
    unittest.main()
