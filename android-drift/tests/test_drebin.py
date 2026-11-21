import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from driftlab.drebin import SCHEMA, audit_directory, load_spec, numeric_blocks, sha256
from driftlab.data import load_dataset
from driftlab.protocol import range_initial_split
from driftlab.experiment import run
from driftlab.package import export_package


class DrebinTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'original'; source.mkdir()
        (source/'family_idx.json').write_text(json.dumps({'benign':0,'test_family':2}))
        vocab='2012-01to2012-12_selected_training_features.json'
        (source/vocab).write_text(json.dumps({'real_permission_feature':0,'real_api_feature':1,'train_constant':2}))
        files=['2012-01to2012-12_selected.npz']+[f'2013-{m:02}_selected.npz' for m in range(1,4)]
        for filename in files:
            y=np.tile([0.,2.],10); X=np.column_stack([y!=0,np.arange(20)%3,np.zeros(20)]).astype('i8')
            np.savez_compressed(source/filename,X_train=X,y_train=y,y_mal_family=np.array(['test_family']*10))
        spec=dict(feature_schema=SCHEMA,source_root=str(source),files=files,vocabulary_file=vocab,
            source_sha256={name:sha256(source/name) for name in files},vocabulary_sha256=sha256(source/vocab),
            family_idx_sha256=sha256(source/'family_idx.json'),chunk_bytes=128,max_csr_bytes=1024*1024)
        path=root/'dataset.json'; path.write_text(json.dumps(spec))
        return source,path,spec

    def test_audit_streaming_schema_and_labels(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); source,path,spec=self.fixture(root)
            before={name:sha256(source/name) for name in spec['files']}
            report=audit_directory(source)
            self.assertTrue(report['structural_pass']); self.assertFalse(report['research_ready'])
            self.assertTrue(report['content_pass'])
            self.assertEqual(report['overlapping_file_periods'],{})
            self.assertEqual(report['missing_periods_between_observed_bounds'],[])
            self.assertEqual(report['files'][0]['malware'],10)
            with patch.object(sparse.csr_matrix,'toarray',side_effect=AssertionError('No densify')):
                data=load_dataset(path)
            self.assertEqual(data.X.shape,(80,3)); self.assertEqual(data.feature_schema,SCHEMA)
            self.assertEqual(data.features,['real_permission_feature','real_api_feature','train_constant'])
            self.assertEqual(data.months[0],'2012-01to2012-12')
            self.assertEqual(data.provenance['records'][1]['family_id'],2)
            self.assertIsNone(data.provenance['records'][0]['apk_sha256'])
            self.assertEqual(before,{name:sha256(source/name) for name in spec['files']})
            groups,months=range_initial_split(data,'2012-01to2012-12')
            self.assertEqual(len(groups['fit']),20); self.assertEqual(months,['2013-01','2013-02','2013-03'])

    def test_overlap_hash_pickle_memory_and_family_mismatch(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); source,path,spec=self.fixture(root)
            original=json.loads(path.read_text())
            spec['files'].append(spec['files'][0]); path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError): load_spec(path)
            spec=dict(original,max_csr_bytes=1); path.write_text(json.dumps(spec))
            with self.assertRaises(MemoryError): load_spec(path)
            spec=dict(original,vocabulary_sha256='wrong'); path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'schema hash'): load_spec(path)
            np.savez_compressed(root/'object.npz',X_train=np.array([{'unsafe':'never loaded'}],dtype=object))
            with self.assertRaisesRegex(ValueError,'pickle'): list(numeric_blocks(root/'object.npz'))
            filename=original['files'][0]
            with np.load(source/filename,allow_pickle=False) as archive:
                X=archive['X_train']; y=archive['y_train']
            np.savez_compressed(source/filename,X_train=X,y_train=y,y_mal_family=np.array(['wrong']*10))
            original['source_sha256'][filename]=sha256(source/filename); path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'align'): load_spec(path)

    def test_header_only_and_overlapping_period_audit(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            source,path,spec=self.fixture(Path(tmp))
            import shutil
            shutil.copyfile(source/spec['files'][0],source/'2012-06_selected.npz')
            report=audit_directory(source,scan_features=False)
            self.assertFalse(report['content_pass'])
            self.assertEqual(len(report['overlapping_file_periods']['2012-06']),2)
            self.assertEqual(report['missing_periods_between_observed_bounds'],[])
            spec['files'].append('2012-06_selected.npz')
            spec['source_sha256']['2012-06_selected.npz']=sha256(source/'2012-06_selected.npz')
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'Overlapping'): load_spec(path)

    def test_frozen_replay_smoke_schema_and_label_budget(self):
        import csv
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); source,path,spec=self.fixture(root)
            c=json.loads(Path('configs/demo.json').read_text())
            c.update(data=str(path),output=str(root/'frozen'),initial_range='2012-01to2012-12',
                training_data_kind='real_android',selector='random',updater='frozen',budget=.01,
                epochs=2,update_epochs=1,hidden=3,blas_threads=1)
            frozen,fsummary=run(c)
            self.assertEqual(fsummary['initial_fit_labels'],20); self.assertEqual(fsummary['queried_labels'],0)
            self.assertTrue((root/'frozen'/'queries.csv').exists())
            c.update(output=str(root/'replay'),updater='replay'); replay,summary=run(c)
            self.assertEqual([r['queried_labels'] for r in replay],[1,1,0])
            self.assertEqual(summary['queried_labels'],2)
            self.assertEqual(frozen[0]['f1'],replay[0]['f1'])
            self.assertTrue(all(r['regression'] is None for r in replay))
            self.assertEqual(summary['initial_monitor_labels'],0)
            with np.load(root/'replay'/'model_2013-03.npz',allow_pickle=False) as saved:
                self.assertEqual(str(saved['feature_schema']),SCHEMA)
            with self.assertRaisesRegex(ValueError,'Drebin schema'):
                export_package(root/'replay',root/'package',dict(model_id='TEST',dataset_source='fixture',
                    training_data_kind='real_android',threshold=.5,limitations=['fixture only']))
