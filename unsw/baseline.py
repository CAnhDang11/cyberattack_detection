"""Multiclass adaptation of the notebook's SGD, MLP, RF and XGBoost families."""
import joblib
import json
from .architecture import hidden_sizes
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def fit(data,args,out):
    if set(data['y_train']) != set(range(10)):
        raise ValueError('Classical baseline requires all ten training classes')
    if args.baseline_model=='logistic':
        net=SGDClassifier(loss='log_loss',learning_rate='constant',eta0=args.lr,alpha=args.l2,max_iter=args.epochs,tol=None,random_state=args.seed)
    elif args.baseline_model=='mlp':
        net=MLPClassifier(hidden_layer_sizes=hidden_sizes(args.hidden,args.hidden_layers),solver='adam',alpha=args.l2,
                          learning_rate_init=args.lr,batch_size=args.batch_size,max_iter=args.epochs,
                          early_stopping=False,tol=0,n_iter_no_change=args.epochs+1,random_state=args.seed)
    elif args.baseline_model=='rf':
        net=RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth or 25,random_state=args.seed,n_jobs=1)
    else:
        from xgboost import XGBClassifier
        net=XGBClassifier(objective='multi:softprob',num_class=10,eval_metric='mlogloss',n_estimators=args.n_estimators,
                          max_depth=args.max_depth or 10,learning_rate=args.tree_lr,random_state=args.seed,n_jobs=1)
    (out/'estimator_params.json').write_text(json.dumps(net.get_params(),indent=2))
    net.fit(data['X_train'],data['y_train'])
    joblib.dump({'model':net,'labels':__import__('unsw.data',fromlist=['LABELS']).LABELS,'config':vars(args)},out/'baseline.joblib')
    return net
