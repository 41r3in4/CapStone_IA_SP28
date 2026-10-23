import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from scipy import sparse
from driftlab.data import generate_demo
from driftlab.experiment import run
from driftlab.statistics import describe, holm, paired, friedman, block_bootstrap_aut, aggregate
from driftlab.metrics import metrics, regression
from driftlab.package import export_package, verify_package, predict_package
from driftlab.preparation import batch_extract, build_dataset
from driftlab.apk import inspect_apk
from driftlab.suite import methods, run_matrix


class ExtensionTests(unittest.TestCase):
    def config(self,root):
        p=root/'data.csv'; generate_demo(p,months=15,samples=80,features=8)
        c=json.loads(Path('configs/demo.json').read_text())
        c.update(data=str(p),output=str(root/'continuous'),epochs=2,update_epochs=1,hidden=4,
                 memory_size=30,fisher_samples=3)
        return c

    def test_resume_equivalence_and_mismatch(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); c=self.config(root); run(c)
            c.update(output=str(root/'partial'),stop_after_month='2021-01'); _,partial=run(c)
            self.assertFalse(partial['completed'])
            c.pop('stop_after_month'); c.update(output=str(root/'resumed'),resume_from=str(root/'partial'/'state_2021-01.json'))
            run(c)
            for name in ('queries.csv','predictions.csv','query_manifest.csv'):
                a=(root/'continuous'/name).read_text(); b=(root/'resumed'/name).read_text()
                if name!='query_manifest.csv': self.assertEqual(a,b)
            with np.load(root/'continuous'/'model_2021-03.npz') as a, np.load(root/'resumed'/'model_2021-03.npz') as b:
                for name in a.files: np.testing.assert_array_equal(a[name],b[name])
            a=json.loads((root/'continuous'/'state_2021-03.json').read_text())
            b=json.loads((root/'resumed'/'state_2021-03.json').read_text())
            for name in ('memory','reservoir_seen','observed','rngs'): self.assertEqual(a[name],b[name])
            # A forged state may not add an unqueried label even from a past month.
            state_path=root/'partial'/'state_2021-01.json'
            state=json.loads(state_path.read_text())
            unqueried=next(i for i in range(12*80,13*80) if str(i) not in state['observed'])
            state['observed'][str(unqueried)]=0
            forged=root/'partial'/'forged.json'; forged.write_text(json.dumps(state))
            forged_config=dict(c,output=str(root/'forged_run'),resume_from=str(forged))
            with self.assertRaisesRegex(ValueError,'oracle queries'): run(forged_config)
            c.update(output=str(root/'bad'),budget=.02)
            with self.assertRaisesRegex(ValueError,'mismatch'): run(c)

    def test_statistics_and_metric_edges(self):
        self.assertIsNone(describe([.5])['ci95'])
        np.testing.assert_allclose(holm([.01,.03,.04]),[.03,.06,.06])
        self.assertIsNone(paired({0:.1},{0:.2})['pvalue'])
        with self.assertRaises(ValueError): paired({0:.1},{1:.2})
        a={i:i/10 for i in range(6)}; b={i:i/10+.1 for i in range(6)}
        self.assertTrue(0<=paired(a,b)['pvalue']<=1)
        self.assertTrue(0<=friedman(dict(a=a,b=b,c={i:v+.2 for i,v in a.items()}))['pvalue']<=1)
        result=block_bootstrap_aut([.1,.2,.3,.4,.5,.6],2,200)
        self.assertEqual(result,block_bootstrap_aut([.1,.2,.3,.4,.5,.6],2,200))
        self.assertEqual(result['block_length'],2)
        self.assertIsNone(metrics(np.array([0]),np.array([0]))['fnr'])
        self.assertIsNone(regression(np.array([0]),np.array([0]),np.array([1]))['regression'])
        with self.assertRaises(ValueError): aggregate([dict(method='a',budget=.01,seed=0,aut_f1=None)])

    def test_fixture_build_train_export_predict_and_mismatch(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); apks=root/'apks'; apks.mkdir(); metadata=[]
            # Manifest archives are generated fixtures; never installed or executed.
            for month in range(1,7):
                for i in range(8):
                    label=i%2; path=apks/f'{month}_{i}.apk'
                    xml=f'<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="fixture.m{month}i{i}"><uses-permission android:name="P{label}"/></manifest>'
                    with zipfile.ZipFile(path,'w') as z: z.writestr('AndroidManifest.xml',xml)
                    metadata.append([hashlib.sha256(path.read_bytes()).hexdigest(),f'2024-{month:02}',label,'SYNTHETIC_FIXTURE'])
            with (root/'metadata.csv').open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['sample_id','month','label','source']); w.writerows(metadata)
            batch_extract(apks,root/'batch')
            build_dataset(root/'batch'/'extractions.jsonl',root/'metadata.csv',root/'dataset','2024-01')
            c=json.loads(Path('configs/demo.json').read_text()); c.update(data=str(root/'dataset'/'dataset.csv'),
                output=str(root/'run'),epochs=2,update_epochs=1,hidden=3,fisher_samples=2,
                temporal=dict(train_through='2024-01',validation_through='2024-02',calibration_through='2024-03',monitor_through='2024-04'))
            run(c)
            export_package(root/'run',root/'package',dict(model_id='FIXTURE_ONLY',dataset_source='GENERATED_ARCHIVES',
                           training_data_kind='synthetic',threshold=.5,limitations=['No Android accuracy evidence']))
            report=inspect_apk(apks/'1_0.apk')
            self.assertEqual(predict_package(report,root/'package')['status'],'inspection_only')
            fixture=predict_package(inspect_apk(apks/'1_0.apk'),root/'package',fixture=True)
            self.assertEqual(fixture['status'],'fixture_prediction'); self.assertIsNone(fixture['verdict'])
            card_path=root/'package'/'model_card.json'; original_card=card_path.read_text()
            card=json.loads(original_card)
            schema=root/'package'/'schema.json'; original_schema=schema.read_text()
            changed=json.loads(original_schema); changed['feature_names'].reverse()
            schema.write_text(json.dumps(changed))
            card['artifacts']['schema.json']=hashlib.sha256(schema.read_bytes()).hexdigest()
            card['schema_sha256']=card['artifacts']['schema.json']; card_path.write_text(json.dumps(card))
            with self.assertRaisesRegex(ValueError,'Feature order'): verify_package(root/'package')
            schema.write_text(original_schema); card_path.write_text(original_card)
            pre=root/'package'/'preprocessing.json'; original_pre=pre.read_text()
            changed=json.loads(original_pre); changed['mean'][0]+=1; pre.write_text(json.dumps(changed))
            card=json.loads(original_card); card['artifacts']['preprocessing.json']=hashlib.sha256(pre.read_bytes()).hexdigest()
            card_path.write_text(json.dumps(card))
            with self.assertRaisesRegex(ValueError,'Preprocessing'): verify_package(root/'package')
            pre.write_text(original_pre); card_path.write_text(original_card)
            card=json.loads(original_card); card['architecture']['hidden']+=1; card_path.write_text(json.dumps(card))
            with self.assertRaisesRegex(ValueError,'Architecture'): verify_package(root/'package')
            card_path.write_text(original_card)
            schema=root/'package'/'schema.json'; schema.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): verify_package(root/'package')

    def test_cli_audit_run_resume_and_lock(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); c=self.config(root)
            c['temporal']=dict(train_through='2020-08',validation_through='2020-09',calibration_through='2020-10',monitor_through='2020-12')
            c['epochs']=1; c['output']=str(root/'cli-run'); config=root/'config.json'
            config.write_text(json.dumps(c))
            def cli(*args):
                result=subprocess.run([sys.executable,'-W','error::ResourceWarning','-m','driftlab',*args],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            cli('audit-data','--data',c['data'],'--output',str(root/'audit'),
                '--train-through','2020-08','--validation-through','2020-09',
                '--calibration-through','2020-10','--monitor-through','2020-12')
            cli('lock-protocol','--config',str(config),'--output',str(root/'lock.json'),'--audit-reviewed','--locked-before-test')
            cli('run','--config',str(config),'--stop-after-month','2021-01')
            cli('run','--config',str(config),'--resume-from',str(root/'cli-run'/'state_2021-01.json'),
                '--output',str(root/'cli-resume'))
            self.assertTrue(json.loads((root/'cli-resume'/'summary.json').read_text())['completed'])

    def test_matrix_and_svm(self):
        for mode in ('baselines','selectors','updaters','closed_loop','ablation'):
            self.assertTrue(methods(mode))
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); c=self.config(root)
            c['temporal']=dict(train_through='2020-08',validation_through='2020-09',calibration_through='2020-10',monitor_through='2020-12')
            from driftlab.svm import run_svm
            c['output']=str(root/'svm'); rows,summary=run_svm(c)
            self.assertEqual(len(rows),3); self.assertEqual(summary['queried_labels'],0)
            with self.assertRaises(ValueError): run_matrix(c,root/'matrix','updaters',budgets=(.01,),seeds=(0,))

    def test_sparse_run_without_densification(self):
        from unittest.mock import patch
        from driftlab.data import load_csv
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); c=self.config(root); data=load_csv(c['data'])
            folder=root/'csr'; folder.mkdir()
            sparse.save_npz(folder/'matrix.npz',sparse.csr_matrix(data.X))
            (folder/'features.json').write_text(json.dumps(data.features))
            with (folder/'metadata.csv').open('w',newline='') as stream:
                writer=csv.writer(stream); writer.writerow(['sample_id','month','label'])
                writer.writerows(zip(data.ids,data.months,data.y))
            c.update(data=str(folder),output=str(root/'sparse_run'))
            with patch.object(sparse.csr_matrix,'toarray',side_effect=AssertionError('No densify')):
                rows,summary=run(c)
            self.assertEqual(len(rows),3); self.assertTrue(summary['completed'])

    def test_monitor_labels_never_train_or_select(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); c=self.config(root)
            c['temporal']=dict(train_through='2020-08',validation_through='2020-09',calibration_through='2020-10',monitor_through='2020-12')
            run(c)
            path=Path(c['data'])
            with path.open(newline='') as stream:
                reader=csv.DictReader(stream); names=reader.fieldnames; rows=list(reader)
            for row in rows:
                if '2020-10'<row['month']<='2020-12': row['label']=str(1-int(row['label']))
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=names); writer.writeheader(); writer.writerows(rows)
            c['output']=str(root/'changed_monitor'); run(c)
            self.assertEqual((root/'continuous'/'queries.csv').read_text(),(root/'changed_monitor'/'queries.csv').read_text())
            self.assertEqual((root/'continuous'/'predictions.csv').read_text(),(root/'changed_monitor'/'predictions.csv').read_text())

    def test_small_matrix_analysis_and_lock_guard(self):
        from driftlab.analysis import analyze
        from driftlab.locking import lock_protocol
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); c=self.config(root)
            rows=run_matrix(c,root/'suite','ablation',budgets=(.01,),seeds=(0,))
            self.assertEqual(len(rows),6)
            result=analyze(root/'suite'/'comparison.csv',root/'analysis.json')
            self.assertIsNone(result['tests'][0]['pvalue'])
            c['training_data_kind']='real_android'
            with self.assertRaises((KeyError,FileNotFoundError)): run_matrix(c,root/'real','baselines',budgets=(.01,),seeds=(0,))
            with self.assertRaises(ValueError): lock_protocol(c,root/'lock.json')
