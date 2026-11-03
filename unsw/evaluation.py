import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
from .data import NAMES


def metrics(y, probabilities):
    p = np.asarray(probabilities)
    if p.shape != (len(y), 10) or not np.isfinite(p).all():
        raise ValueError('Expected finite N x 10 probabilities')
    report = classification_report(y, p.argmax(1), labels=list(range(10)), target_names=NAMES, output_dict=True, zero_division=0)
    result = {'accuracy':accuracy_score(y,p.argmax(1)), 'loss':log_loss(y,p,labels=list(range(10))),
              'per_class':{k:report[k] for k in NAMES}, 'confusion_matrix':confusion_matrix(y,p.argmax(1),labels=list(range(10))).tolist(),
              'class_names':NAMES}
    for avg in ['macro','weighted']:
        for key in ['precision','recall','f1-score']:
            result[f'{avg}_{key}'] = report[f'{avg} avg'][key]
    try:
        if len(np.unique(y)) != 10:
            raise ValueError('Multiclass ROC-AUC requires all ten classes in the evaluation labels')
        result['roc_auc_ovr_macro'] = float(roc_auc_score(y,p,labels=list(range(10)),multi_class='ovr',average='macro'))
        result['roc_auc_unavailable_reason'] = None
    except ValueError as e:
        result['roc_auc_ovr_macro'] = None
        result['roc_auc_unavailable_reason'] = str(e)
    return result


def save_tables(result, out):
    pd.DataFrame(result['per_class']).T.to_csv(out/'per_class.csv')
    pd.DataFrame(result['confusion_matrix'], index=NAMES, columns=NAMES).to_csv(out/'confusion_matrix.csv',index_label='true/predicted')
