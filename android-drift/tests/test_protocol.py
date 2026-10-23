import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from driftlab.data import Dataset, load_sparse, generate_demo
from driftlab.protocol import temporal_split, audit, fit_preprocessing, split_manifest
from driftlab.validation import temporal_validation
from driftlab.queries import read_queries
from driftlab.experiment import run


class ProtocolTests(unittest.TestCase):
    def data(self, csr=False):
        months = np.repeat(['2024-01','2024-02','2024-03','2024-04','2024-05','2024-06'],4)
        y = np.tile([0,1,0,1],6)
        X = np.column_stack([y, np.arange(24)%3, np.r_[np.zeros(4),np.ones(20)]])
        return Dataset(np.array(['id'+str(i) for i in range(24)]), months,
                       sparse.csr_matrix(X) if csr else X, y, ['f_a','f_b','f_future'], 'fixture')

    def groups(self, d):
        return temporal_split(d,'2024-01','2024-02','2024-03','2024-04')

    def test_temporal_and_preprocessing_no_future_access(self):
        for csr in (False, True):
            d = self.data(csr); groups = self.groups(d)
            X, state = fit_preprocessing(d, groups['fit'])
            self.assertEqual(state['columns'], [0,1])
            self.assertEqual(sparse.issparse(X), csr)
            d.y[groups['regression_monitor']] = 1
            first = temporal_validation(d, groups, [dict(hidden=3,epochs=2,learning_rate=.01,batch_size=4)])
            d.y[groups['regression_monitor']] = 0
            second = temporal_validation(d, groups, [dict(hidden=3,epochs=2,learning_rate=.01,batch_size=4)])
            self.assertEqual(first, second)
            self.assertEqual(len(split_manifest(d,groups)),24)
            self.assertEqual(len(audit(d,groups)['monthly']),6)
            broken = dict(groups); broken['fit'] = np.r_[groups['fit'],groups['validation'][0]]
            with self.assertRaises(ValueError): split_manifest(d, broken)

    def test_sparse_loader(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp); d = self.data(True)
            sparse.save_npz(root/'matrix.npz', d.X)
            (root/'features.json').write_text(json.dumps(d.features))
            with (root/'metadata.csv').open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['sample_id','month','label']); w.writerows(zip(d.ids,d.months,d.y))
            loaded = load_sparse(root)
            self.assertEqual((loaded.X != d.X).nnz,0)
            (root/'features.json').write_text('["f_a"]')
            with self.assertRaises(ValueError): load_sparse(root)

    def test_replay_validation(self):
        d=self.data(); future=['2024-05','2024-06']
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            p=Path(tmp)/'queries.csv'
            def write(rows):
                with p.open('w',newline='') as f:
                    w=csv.writer(f); w.writerow(['run_id','month','sample_id','query_order']); w.writerows(rows)
            write([['r','2024-05','id16',0]])
            self.assertEqual(read_queries(p,d,future,.01,'ewc')['2024-05'],[16])
            for rows in ([['r','2024-05','id1',0]], [['r','2024-06','id20',0]],
                         [['r','2024-05','id16',0],['r','2024-05','id16',1]],
                         [['r','2024-05','id16',0],['r','2024-05','id17',1]]):
                write(rows)
                with self.assertRaises(ValueError): read_queries(p,d,future,.01,'ewc')

    def test_temporal_run_and_fixed_query_ewc(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root=Path(tmp); generate_demo(root/'demo.csv',months=14,samples=80,features=8)
            c=json.loads(Path('configs/demo.json').read_text())
            c.update(data=str(root/'demo.csv'), output=str(root/'first'), epochs=2, update_epochs=1,
                     hidden=4, fisher_samples=3, temporal=dict(train_through='2020-08',
                     validation_through='2020-09',calibration_through='2020-10',monitor_through='2020-12'))
            run(c)
            c.update(output=str(root/'second'),updater='ewc',query_manifest=str(root/'first'/'query_manifest.csv'))
            _, summary=run(c)
            self.assertEqual(summary['query_mode'],'fixed_query_replay')
            with (root/'first'/'queries.csv').open() as stream: a=list(csv.DictReader(stream))
            with (root/'second'/'queries.csv').open() as stream: b=list(csv.DictReader(stream))
            self.assertEqual([r['sample_id'] for r in a],[r['sample_id'] for r in b])
