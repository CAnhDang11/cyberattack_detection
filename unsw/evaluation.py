import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
from .data import NAMES


def metrics(y, probabilities, names=NAMES):
    names = list(names); k = len(names); labels = list(range(k))
    p = np.asarray(probabilities)
    if p.shape != (len(y), k) or not np.isfinite(p).all():
        raise ValueError(f'Expected finite N x {k} probabilities')
    report = classification_report(y, p.argmax(1), labels=labels, target_names=names, output_dict=True, zero_division=0)
    result = {'accuracy':accuracy_score(y,p.argmax(1)), 'loss':log_loss(y,p,labels=labels),
              'per_class':{n:report[n] for n in names}, 'confusion_matrix':confusion_matrix(y,p.argmax(1),labels=labels).tolist(),
              'class_names':names}
    for avg in ['macro','weighted']:
        for key in ['precision','recall','f1-score']:
            result[f'{avg}_{key}'] = report[f'{avg} avg'][key]
    try:
        if len(np.unique(y)) != k:
            raise ValueError(f'Multiclass ROC-AUC requires all {k} classes in the evaluation labels')
        result['roc_auc_ovr_macro'] = float(roc_auc_score(y,p,labels=labels,multi_class='ovr',average='macro'))
        result['roc_auc_unavailable_reason'] = None
    except ValueError as e:
        result['roc_auc_ovr_macro'] = None
        result['roc_auc_unavailable_reason'] = str(e)
    return result


def recall_ceiling(x, y, k):
    """Highest recall any classifier can reach per class on (x, y): identical inputs get one prediction,
    so within each group of identical rows only the majority label can be right (ties: lowest index)."""
    rows = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    _, inv = np.unique(rows, return_inverse=True)
    counts = np.zeros((inv.max() + 1, k), dtype=np.int64)
    np.add.at(counts, (inv, y), 1)
    correct = counts.argmax(1)[inv] == y
    return {c: float(correct[y == c].mean()) if (y == c).any() else None for c in range(k)}, float(correct.mean())


def add_ceiling(result, x, y):
    per, overall = recall_ceiling(x, y, len(result['class_names']))
    for i, n in enumerate(result['class_names']):
        result['per_class'][n]['max_recall'] = per[i]
    result['max_accuracy'] = overall
    return result


def original_breakdown(y_orig, pred, orig_names, names):
    """Rows: original attack_cat; columns: predicted class (row-normalised), plus support."""
    t = pd.crosstab(pd.Categorical(np.asarray(orig_names)[y_orig], categories=list(orig_names)),
                    pd.Categorical(np.asarray(names)[pred], categories=list(names)), dropna=False)
    out = t.div(t.sum(1).replace(0, 1), axis=0).round(4)
    out['support'] = t.sum(1)
    out.index.name = 'original/predicted'
    return out


def save_tables(result, out):
    names = result['class_names']
    pd.DataFrame(result['per_class']).T.to_csv(out/'per_class.csv')
    pd.DataFrame(result['confusion_matrix'], index=names, columns=names).to_csv(out/'confusion_matrix.csv',index_label='true/predicted')
