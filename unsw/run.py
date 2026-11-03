"""CLI for comparable centralized, DDP, and Flower experiments."""
import argparse
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .architecture import DEFAULT_HIDDEN, DEFAULT_HIDDEN_LAYERS
from .data import LABELS, CASES, load, partition, save_partitions, prepare
from .evaluation import metrics, save_tables
from .logging_utils import LOGGER_NAME, setup as setup_logging, close as close_logging


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('method',choices=['prepare','centralized','ddp','fedavg','fedprox','baseline'])
    p.add_argument('--data',default='experiments/data/data.npz')
    p.add_argument('--train-csv',default='UNSW_NB15_training-set.csv')
    p.add_argument('--test-csv',default='UNSW_NB15_testing-set.csv')
    p.add_argument('--output',default='experiments/run')
    p.add_argument('--case',choices=CASES,default='equal_iid')
    for name,default in [('clients',4),('seed',42),('epochs',10),('local-epochs',1),('rounds',10),('batch-size',256),('hidden',DEFAULT_HIDDEN),('hidden-layers',DEFAULT_HIDDEN_LAYERS),('limit-per-class',0)]:
        p.add_argument('--'+name,type=int,default=default,
                       help='Preparation only: 0 uses all training CSV rows (default); positive values cap each class.'
                       if name == 'limit-per-class' else None)
    for name,default in [('alpha',.5),('lr',.001),('participation',1.),('mu',.01)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--baseline-model',choices=['logistic','mlp','rf','xgboost'],default='logistic')
    p.add_argument('--n-estimators',type=int,default=500,help='RF trees / XGBoost boosting rounds')
    p.add_argument('--max-depth',type=int,default=None,help='Tree depth (default: RF 25, XGBoost 10)')
    p.add_argument('--tree-lr',type=float,default=0.3,help='XGBoost learning rate; separate from neural/SGD --lr')
    p.add_argument('--l2',type=float,default=0.,help='SGD/MLP L2 regularization; PyTorch Adam weight decay')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.method=='prepare':
        prepare(args.train_csv,args.test_csv,args.output,args.seed,args.limit_per_class); return
    if min(args.epochs,args.rounds,args.local_epochs,args.batch_size)<1 or not 0<args.participation<=1:
        raise ValueError('Positive training counts and participation in (0,1] required')
    if (args.hidden<2 or args.hidden_layers<1 or args.lr<=0 or args.tree_lr<=0 or args.l2<0 or args.n_estimators<1
            or (args.max_depth is not None and args.max_depth<1)):
        raise ValueError('Require hidden >= 2, hidden-layers >= 1, positive learning rates/tree counts/depth, and l2 >= 0')
    np.random.seed(args.seed)
    # Classical estimators do not need PyTorch. Loading both native runtimes can
    # crash XGBoost on macOS, even when each estimator uses just one thread.
    device=args.device
    if args.method!='baseline':
        import torch
        import torch.distributed as dist
        torch.set_num_threads(1); torch.manual_seed(args.seed)
        device=torch.device(args.device)
        if args.device=='cuda':
            device=torch.device('cuda',int(os.getenv('LOCAL_RANK','0'))); torch.cuda.set_device(device)
    rank=int(os.getenv('RANK','0')); world=int(os.getenv('WORLD_SIZE','1')) if args.method=='ddp' else 1
    distributed=args.method=='ddp' and world>1
    if distributed: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    logger = setup_logging(args.output) if rank == 0 else None
    try:
        if logger:
            logger.info('START method=%s case=%s seed=%s world_size=%s device=%s',
                        args.method, args.case, args.seed, world, device)
            logger.info('CONFIG %s', json.dumps(vars(args), sort_keys=True))
        execute(args,rank,world,device,distributed)
    except Exception:
        if logger:
            logger.exception('Training failed')
        raise
    finally:
        if logger:
            close_logging()
        if distributed: dist.destroy_process_group()


def execute(args,rank,world,device,distributed):
    if args.method!='baseline':
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader, DistributedSampler, Subset
        from torch.nn.parallel import DistributedDataParallel
        from .model import model, dataset, train_epoch, predict
    logger = logging.getLogger(LOGGER_NAME)
    data=load(args.data)
    if rank == 0:
        logger.info('DATA %s | fit=%s | validation=%s | test=%s', args.data,
                    len(data['y_train']), len(data['y_val']), len(data['y_test']))
    parts=partition(data['y_train'],args.clients,args.case,args.alpha,args.seed)
    out=Path(args.output)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True); save_partitions(data['y_train'],parts,out)
        (out/'config.json').write_text(json.dumps(vars(args),indent=2))
        (out/'labels.json').write_text(json.dumps(LABELS,indent=2))
        logger.info('CLIENT DISTRIBUTIONS\n%s', (out/'clients.csv').read_text())
    net=model(data['X_train'].shape[1],args.hidden,args.hidden_layers).to(device) if args.method!='baseline' else None
    history=[]; start=time.perf_counter(); communication=None; best=-1.; best_state=None
    def record(step, current, train_loss, volume=None):
        nonlocal best,best_state
        v=metrics(data['y_val'],predict(current,data['X_val'],device))
        history.append({'step':step,'train_loss':train_loss,'val_loss':v['loss'],'val_macro_f1':v['macro_f1-score'],
                        'elapsed_s':time.perf_counter()-start,'communication_bytes':volume})
        logger.info('%s %s | %s', 'ROUND' if args.method in ('fedavg','fedprox') else 'EPOCH',
                    step, json.dumps(history[-1]))
        if v['macro_f1-score']>best:
            best=v['macro_f1-score']; best_state={k:t.detach().cpu().clone() for k,t in current.state_dict().items()}
    estimator=None
    if args.method=='baseline':
        from .baseline import fit
        estimator=fit(data,args,out)
        p=estimator.predict_proba(data['X_val'])
        v=metrics(data['y_val'],p)
        from sklearn.metrics import log_loss
        steps=args.n_estimators if args.baseline_model in ('rf','xgboost') else int(estimator.n_iter_)
        history.append({'step':steps,'train_loss':log_loss(data['y_train'],estimator.predict_proba(data['X_train']),labels=list(range(10))),
                        'val_loss':v['loss'],'val_macro_f1':v['macro_f1-score']})
        logger.info('BASELINE FIT | %s', json.dumps(history[-1]))
    elif args.method in ('fedavg','fedprox'):
        if device.type!='cpu': raise ValueError('Local Flower runner currently supports CPU')
        from .federated import run
        net,communication=run(data,parts,args,net,record)
    else:
        ds=dataset(data['X_train'],data['y_train'])
        # Cases reorder the same union. DDP remains globally shuffled gradient synchronization.
        ds=Subset(ds,np.concatenate(parts).tolist())
        sampler=DistributedSampler(ds,num_replicas=world,rank=rank,seed=args.seed) if args.method=='ddp' else None
        loader=DataLoader(ds,batch_size=args.batch_size,sampler=sampler,shuffle=sampler is None)
        wrapped=DistributedDataParallel(net,device_ids=[device.index] if device.type=='cuda' else None) if distributed else net
        opt=torch.optim.Adam(wrapped.parameters(),lr=args.lr,weight_decay=args.l2)
        for epoch in range(1,args.epochs+1):
            if sampler is not None: sampler.set_epoch(epoch)
            s,n=train_epoch(wrapped,loader,opt,device)
            totals=torch.tensor([s,n],dtype=torch.float64,device=device)
            if distributed: dist.all_reduce(totals)
            if rank==0: record(epoch,net,(totals[0]/totals[1]).item())
            if distributed: dist.barrier()
    train_s=time.perf_counter()-start
    if rank!=0: return
    if estimator is None:
        net.load_state_dict(best_state)
        torch.save({'state_dict':net.state_dict(),'labels':LABELS,'config':vars(args),'input_dim':data['X_train'].shape[1]},out/'best.pt')
    t=time.perf_counter()
    probabilities=estimator.predict_proba(data['X_test']) if estimator is not None else predict(net,data['X_test'],device)
    result=metrics(data['y_test'],probabilities)
    result.update(method=args.method,case=args.case,seed=args.seed,train_s=train_s,evaluation_s=time.perf_counter()-t,
                  model=args.baseline_model if estimator is not None else 'torch_mlp',
                  communication_bytes=communication,train_loss=history[-1]['train_loss'],world_size=world,
                  test_samples=len(data['y_test']),selected_val_macro_f1=best if estimator is None else history[-1]['val_macro_f1'])
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    save_tables(result,out); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,(dict,list))}]).to_csv(out/'results.csv',index=False)
    logger.info('FINAL TEST RESULTS\n%s', json.dumps(result, indent=2, allow_nan=False))
    logger.info('COMPLETED | artifacts=%s', out.resolve())


if __name__=='__main__': main()
