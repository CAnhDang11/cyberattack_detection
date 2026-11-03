import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from unsw.data import LABELS,NAMES,CASES,encode,decode,partition,Preprocessor,load
from unsw.model import model
from unsw.evaluation import metrics


class PipelineTests(unittest.TestCase):
    def test_labels_model_loss(self):
        self.assertEqual(len(LABELS),10); self.assertIn('Normal',LABELS)
        self.assertEqual(set(NAMES),{'Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms'})
        np.testing.assert_array_equal(decode(encode(NAMES)),NAMES)
        with self.assertRaises(ValueError): encode(['unknown'])
        with self.assertRaises(ValueError): encode([None])
        logits=model(3)(torch.ones(10,3)); self.assertEqual(logits.shape,(10,10))
        self.assertTrue(torch.isfinite(torch.nn.CrossEntropyLoss()(logits,torch.arange(10))))

    def test_expanded_architecture(self):
        from unsw.architecture import hidden_sizes
        from unsw.run import parser
        self.assertEqual(hidden_sizes(), (512,256,128,64))
        net = model(42)
        linear = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([layer.out_features for layer in linear], [512,256,128,64,10])
        old = model(42,128,2)
        self.assertGreater(sum(p.numel() for p in net.parameters()),
                           10 * sum(p.numel() for p in old.parameters()))
        loss = net(torch.randn(20,42)).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        self.assertEqual(hidden_sizes(16,2), (16,8))
        args = parser().parse_args(['centralized'])
        self.assertEqual((args.hidden,args.hidden_layers,args.n_estimators),(512,4,500))
        for width,depth in [(1,4),(16,0)]:
            with self.assertRaises(ValueError): hidden_sizes(width,depth)

    def test_partitions(self):
        y=np.repeat(np.arange(10),200)
        for case in CASES:
            a=partition(y,4,case,.1,42); b=partition(y,4,case,.1,42)
            for x,z in zip(a,b): np.testing.assert_array_equal(x,z)
            np.testing.assert_array_equal(np.sort(np.concatenate(a)),np.arange(len(y)))
            counts=[len(p) for p in a]
            if case.startswith('equal_'): self.assertLessEqual(max(counts)-min(counts),1)
            else: self.assertGreater(max(counts),2*min(counts))
            ratios=np.array([np.bincount(y[p],minlength=10)/len(p) for p in a])
            if 'non_iid' in case: self.assertGreater(ratios.std(axis=0).mean(),.04)
            else: self.assertLess(ratios.std(axis=0).mean(),.04)

    def test_preprocessing_and_evaluation(self):
        tr=pd.DataFrame({'x':[0.,1.,2.],'proto':['a','b','a'],'label':[0,1,1],'attack_cat':['Normal','DoS','Worms']})
        pp=Preprocessor().fit(tr); before=pp.mean.copy()
        pp.transform(pd.DataFrame({'x':[999.],'proto':['unseen']}))
        np.testing.assert_array_equal(before,pp.mean)
        self.assertEqual(pp.transform(tr).shape[1],2)
        r=metrics(np.arange(10),np.eye(10)*.99+.001)
        self.assertEqual(r['class_names'],NAMES); self.assertEqual(len(r['per_class']),10)
        self.assertEqual(np.asarray(r['confusion_matrix']).shape,(10,10))
        self.assertEqual(r['accuracy'],1)
        missing=metrics(np.array([0,1]),np.full((2,10),.1))
        self.assertIsNone(missing['roc_auc_ovr_macro'])
        self.assertTrue(missing['roc_auc_unavailable_reason'])

    def test_csv_preparation_has_no_test_rows_in_partitions(self):
        from unsw.data import prepare, features
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            train=pd.DataFrame({'id':np.arange(300),'x':np.arange(300),
                                'proto':['tcp']*300,'attack_cat':NAMES*30,'label':[999]*300})
            test=pd.DataFrame({'id':np.arange(30),'x':np.arange(1000,1030),
                               'proto':['unseen']*30,'attack_cat':NAMES*3,'label':[-999]*30})
            train.to_csv(p/'train.csv',index=False); test.to_csv(p/'test.csv',index=False)
            d=prepare(p/'train.csv',p/'test.csv',p/'prepared')
            self.assertFalse(set(d['train_ids']) & set(d['test_ids']))
            self.assertFalse(set(d['train_ids']) & set(d['val_ids']))
            self.assertEqual(len(d['y_test']),30)
            np.testing.assert_array_equal(d['y_test'],encode(test.attack_cat))
            self.assertEqual(load(p/'prepared/data.npz')['X_train'].shape[1],1)
            for case in CASES:
                parts=partition(d['y_train'],4,case)
                union=np.concatenate(parts)
                self.assertEqual(len(np.unique(union)),len(d['y_train']))
                self.assertFalse(set(d['train_ids'][union]) & set(d['test_ids']))

    def smoke(self,method,ddp=False,baseline_model='logistic'):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); rng=np.random.default_rng(7)
            arrays={}
            for split,n in [('train',200),('val',40),('test',40)]:
                arrays['X_'+split]=rng.normal(size=(n,4)).astype('float32')
                arrays['y_'+split]=np.arange(n)%10
            arrays['train_ids']=np.array([f'train:{i}' for i in range(200)])
            arrays['test_ids']=np.array([f'test:{i}' for i in range(40)])
            self.assertFalse(set(arrays['train_ids']) & set(arrays['test_ids']))
            np.savez(p/'data.npz',**arrays); (p/'labels.json').write_text(json.dumps(LABELS))
            cmd=[sys.executable,'-m','unsw.run',method,'--data',str(p/'data.npz'),'--output',str(p/'run'),'--epochs','1','--rounds','1','--clients','2','--hidden','16']
            if method=='baseline':
                cmd+=['--baseline-model',baseline_model,'--lr','0.002','--l2','0.01',
                      '--n-estimators','3','--max-depth','2','--tree-lr','0.2']
            if ddp: cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m',*cmd[2:]]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            result=json.loads((p/'run/results.json').read_text())
            log=(p/'run/training.log').read_text()
            self.assertIn('FINAL TEST RESULTS',log)
            self.assertIn('COMPLETED',log)
            self.assertIn('macro_f1-score',log)
            self.assertEqual(result['test_samples'],40)
            self.assertEqual(result['class_names'],NAMES)
            if method=='baseline':
                params=json.loads((p/'run/estimator_params.json').read_text())
                self.assertEqual(result['model'],baseline_model)
                if baseline_model in ('rf','xgboost'):
                    self.assertEqual(params['n_estimators'],3)
                    self.assertEqual(params['max_depth'],2)
                    history=pd.read_csv(p/'run/history.csv')
                    self.assertEqual(history.iloc[0]['step'],3)
                    if baseline_model=='xgboost': self.assertEqual(params['learning_rate'],0.2)
                else:
                    self.assertEqual(params['alpha'],0.01)
                    self.assertEqual(params['eta0' if baseline_model=='logistic' else 'learning_rate_init'],0.002)
                    if baseline_model=='mlp':
                        self.assertEqual(params['solver'],'adam')
                        self.assertEqual(params['hidden_layer_sizes'],[16,8,4,2])
    def test_centralized_cpu(self): self.smoke('centralized')
    def test_classical_cpu(self): self.smoke('baseline')
    def test_sklearn_mlp_cpu(self): self.smoke('baseline',baseline_model='mlp')
    def test_rf_cpu(self): self.smoke('baseline',baseline_model='rf')
    @unittest.skipUnless(importlib.util.find_spec('xgboost'),'Optional XGBoost dependency unavailable')
    def test_xgboost_cpu(self): self.smoke('baseline',baseline_model='xgboost')
    def test_ddp_cpu(self): self.smoke('ddp')
    @unittest.skipUnless(importlib.util.find_spec('flwr'),'Optional Flower dependency unavailable')
    def test_fedavg_cpu(self): self.smoke('fedavg')
    @unittest.skipUnless(importlib.util.find_spec('flwr'),'Optional Flower dependency unavailable')
    def test_fedprox_cpu(self): self.smoke('fedprox')

if __name__=='__main__': unittest.main()
