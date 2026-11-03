"""Compare centralized model families in separate native-runtime processes."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='experiments/data/data.npz')
    p.add_argument('--output',default='experiments/centralized_comparison')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--models',nargs='+',choices=['torch_mlp','logistic','mlp','rf','xgboost'],
                   default=['torch_mlp','logistic','mlp','rf','xgboost'])
    args,extra=p.parse_known_args()
    # Prevent pass-through options from replacing experiment identities or splits.
    reserved={'--case','--baseline-model','--seed','--data','--output'}
    if any(arg.split('=')[0] in reserved for arg in extra):
        p.error('case, model, seed, data and output are managed by the suite')
    for seed in args.seeds:
        for name in args.models:
            method='centralized' if name=='torch_mlp' else 'baseline'
            cmd=[sys.executable,'-m','unsw.run',method,*extra,'--data',args.data,
                 '--seed',str(seed),'--case','equal_iid','--output',str(Path(args.output)/f'{name}_{seed}')]
            if method=='baseline': cmd+=['--baseline-model',name]
            subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,'-m','unsw.compare',args.output],check=True)


if __name__=='__main__': main()
