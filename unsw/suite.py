"""Run all cases with explicit output paths; optional dependencies fail visibly."""
import argparse
import subprocess
import sys
from .data import CASES


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='experiments/data/data.npz')
    p.add_argument('--output',default='experiments/suite'); p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--methods',nargs='+',default=['baseline','centralized','ddp','fedavg','fedprox'])
    a,extra=p.parse_known_args()
    for seed in a.seeds:
        for case in CASES:
            for method in a.methods:
                subprocess.run([sys.executable,'-m','unsw.run',method,'--data',a.data,'--case',case,'--seed',str(seed),
                                '--output',f'{a.output}/{method}_{case}_{seed}',*extra],check=True)
    subprocess.run([sys.executable,'-m','unsw.compare',a.output],check=True)

if __name__=='__main__': main()
