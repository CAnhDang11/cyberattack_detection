"""Multiclass adaptation of the notebook's SGD, MLP, RF and XGBoost families."""
import joblib
import json
from .architecture import hidden_sizes
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def fit(data,args,out):
    k=len(data['labels'])
    if set(data['y_train']) != set(range(k)):
        raise ValueError(f'Classical baseline requires all {k} training classes')
    if args.baseline_model=='logistic':
        net=SGDClassifier(loss='log_loss',learning_rate='constant',eta0=args.lr,alpha=args.l2,max_iter=args.epochs,tol=None,random_state=args.seed)
    elif args.baseline_model=='mlp':
        net=MLPClassifier(hidden_layer_sizes=hidden_sizes(args.hidden,args.hidden_layers),solver='adam',alpha=args.l2,
                          learning_rate_init=args.lr,batch_size=args.batch_size,max_iter=args.epochs,
                          early_stopping=False,tol=0,n_iter_no_change=args.epochs+1,random_state=args.seed)
    elif args.baseline_model=='rf':
        net=RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth or 25,random_state=args.seed,n_jobs=args.n_jobs)
    else:
        from xgboost import XGBClassifier
        net=XGBClassifier(objective='multi:softprob',num_class=k,eval_metric='mlogloss',n_estimators=args.n_estimators,tree_method='hist',device=getattr(args,'xgb_device','cpu'),
                          max_depth=args.max_depth or 10,learning_rate=args.tree_lr,random_state=args.seed,n_jobs=args.n_jobs)
    (out/'estimator_params.json').write_text(json.dumps(net.get_params(),indent=2))
    kw={}
    if args.class_weight_alpha:  # same global per-class weights as the neural/federated runs
        from .data import class_weights
        kw['sample_weight']=class_weights(data['y_train'],k,args.class_weight_alpha)[data['y_train']]
    net.fit(data['X_train'],data['y_train'],**kw)
    joblib.dump({'model':net,'labels':data['labels'],'config':vars(args)},out/'baseline.joblib')
    return net
