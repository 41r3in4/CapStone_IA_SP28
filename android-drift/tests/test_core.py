import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from driftlab.model import MLP
from driftlab.metrics import metrics,aut,regression
from driftlab.selection import select,rank01
from driftlab.experiment import Oracle,Reservoir,run
from driftlab.data import generate_demo,load_csv,initial_split


class CoreTests(unittest.TestCase):
    def test_gradient(self):
        m=MLP(3,4,2); X=np.array([[.2,.3,-.4],[-.2,.1,.8]]);y=np.array([1,0])
        anchor={k:v-.01 for k,v in m.snapshot().items()}; fisher={k:np.ones_like(v)*.2 for k,v in m.p.items()}
        teacher=m.snapshot();options=dict(anchor=anchor,fisher=fisher,ewc=.3,teacher=teacher,reg=.2)
        _,g=m.loss_grad(X,y,**options)
        for k in m.p:
            for idx in np.ndindex(m.p[k].shape):
                old=m.p[k][idx];eps=1e-6
                m.p[k][idx]=old+eps;plus,_=m.loss_grad(X,y,**options)
                m.p[k][idx]=old-eps;minus,_=m.loss_grad(X,y,**options)
                m.p[k][idx]=old
                self.assertAlmostEqual(g[k][idx],(plus-minus)/(2*eps),places=5)

    def test_fisher(self):
        m=MLP(2,3);X=np.array([[1.,0],[0,1.]]);y=np.array([1,0])
        f=m.empirical_fisher(X,y,2,np.random.default_rng(0))
        for k in f:
            expected=sum(m.loss_grad(X[i:i+1],y[i:i+1])[1][k]**2 for i in range(2))/2
            np.testing.assert_allclose(f[k],expected)

    def test_metrics(self):
        r=metrics(np.array([1,1,0,0]),np.array([1,0,1,0]))
        self.assertEqual(r['f1'],.5);self.assertEqual(aut([0.,1.,1.]),.75)
        self.assertIsNone(regression(np.array([1]),np.array([0]),np.array([0]))['regression'])

    def test_oracle_guard(self):
        o=Oracle(np.array([0,1,0]))
        for indices,budget in [([0,0],2),([0,1],1),([2],1)]:
            with self.assertRaises(ValueError): o.reveal(indices,[0,1],budget)

    def test_memory(self):
        r=Reservoir(5,np.random.default_rng(1));r.add(range(100))
        self.assertEqual(len(r.indices),5);self.assertEqual(len(set(r.indices)),5)
        self.assertEqual(r.seen,100)

    def test_selectors(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(30,4));y=np.arange(30)%2;m=MLP(4)
        for strategy in ('random','uncertainty','conformal','distance','hybrid'):
            idx,score=select(m,X,4,strategy,np.random.default_rng(3),X,y,X,y,[.5,.25,.25])
            self.assertEqual(len(set(idx)),4);self.assertTrue(np.isfinite(score).all())
        np.testing.assert_allclose(rank01(np.ones(3)),0)

    def test_split_disjoint(self):
        with tempfile.TemporaryDirectory(dir='.') as td:
            p=Path(td)/'d.csv';generate_demo(p,months=13,samples=80,features=8)
            d=load_csv(p); (fit,cal,monitor),months=initial_split(d,12,42)
            self.assertFalse(set(fit)&set(cal));self.assertFalse(set(fit)&set(monitor));self.assertFalse(set(cal)&set(monitor))
            self.assertLess(max(d.months[fit]),months[0])

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory(dir='.') as td:
            td=Path(td);p=td/'d.csv';generate_demo(p,months=14,samples=80,features=8)
            base=json.loads(Path('configs/demo.json').read_text())
            base.update(data=str(p),epochs=2,update_epochs=1,hidden=5,memory_size=30,fisher_samples=4)
            for updater in ('frozen','full_oracle','retrain','finetune','replay','replay_ewc','replay_ewc_reg'):
                c=dict(base,updater=updater,output=str(td/updater))
                rows,summary=run(c)
                self.assertEqual(len(rows),2)
                self.assertEqual(rows[-1]['queried_labels'],0)
                self.assertTrue((td/updater/'report.html').exists())
                self.assertLessEqual(max(r['memory_samples'] for r in rows),30)
                expected=0 if updater=='frozen' else 80 if updater=='full_oracle' else 4
                self.assertEqual(summary['queried_labels'],expected)
                # Calibration and monitor never enter training or query pool.
                self.assertEqual(rows[0]['known_training_labels'],summary['initial_fit_labels']+expected)
                if updater=='frozen':
                    with np.load(td/updater/'model_2021-01.npz') as a, np.load(td/updater/'model_2021-02.npz') as b:
                        np.testing.assert_array_equal(a['W1'],b['W1'])


if __name__=='__main__': unittest.main()
