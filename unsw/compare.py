"""Collect completed runs; never invent rows for missing approaches."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument('root',nargs='?',default='experiments'); args=p.parse_args()
    root=Path(args.root)
    frames=[]
    for path in sorted(root.rglob('results.csv')):
        frame=pd.read_csv(path); frame['run_directory']=str(path.parent); frames.append(frame)
    if not frames: raise ValueError('No completed results')
    df=pd.concat(frames,ignore_index=True); df.to_csv(root/'comparison.csv',index=False)
    cols=[c for c in ['model','method','seed','case','selected_val_macro_f1','accuracy','macro_f1-score','weighted_f1-score','train_s'] if c in df]
    report='# Measured multiclass results\n\n'+df[cols].to_csv(index=False)
    report+='\nThese are short CPU validation runs, not tuned benchmarks. No improvement claim is warranted. '
    report+='Centralized and DDP train on the union of client partitions; cases affect data order, not data availability. '
    report+='Federated comparisons depend on class skew, client capacity, participation and local steps.\n\n'
    report+='Communication bytes for Flower count uncompressed model arrays sent and returned; DDP traffic is unmeasured.\n\n'
    audit=root/'data/audit.json'
    if audit.exists():
        a=json.loads(audit.read_text()); report+='Full CSV class counts and duplicate audit:\n\n```json\n'+json.dumps(a,indent=2)+'\n```\n'
    report+='\nNormal, Generic and Exploits dominate the supplied data. Worms is rare (130 train, 44 test). '
    report+='Weighted metrics and accuracy emphasize common classes; macro F1 weights each class equally. '
    report+='Inspect per_class.csv and confusion_matrix.csv for minority-class failures. '
    report+='The original CSV split contains duplicate feature rows across train and test; these are reported, retained, and limit generalization claims. '
    report+='Training partitions contain only training row identities, never held-out test identities.\n'
    (root/'REPORT.md').write_text(report)

if __name__=='__main__': main()
