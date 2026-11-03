"""Sequential local Flower transport: server receives FitRes, never client examples."""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import model, dataset, train_epoch


def run(data, parts, args, initial, callback):
    try:
        import flwr as fl
        from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
    except ImportError as e:
        raise RuntimeError('Flower is required: pip install -r requirements-flower.txt') from e
    strategy_cls = fl.server.strategy.FedProx if args.method == 'fedprox' else fl.server.strategy.FedAvg
    kwargs = {'proximal_mu':args.mu} if args.method == 'fedprox' else {}
    strategy = strategy_cls(fraction_fit=args.participation, min_fit_clients=1, min_available_clients=args.clients, **kwargs)
    keys = list(initial.state_dict())
    def get(net):
        return [v.detach().cpu().numpy().copy() for v in net.state_dict().values()]
    def put(net, arrays):
        net.load_state_dict({k:torch.from_numpy(a.copy()) for k,a in zip(keys,arrays)}, strict=True)
    class Client(fl.client.NumPyClient):
        def __init__(self, indices):
            self.indices = indices
        def fit(self, parameters, config):
            net = model(data['X_train'].shape[1],args.hidden,args.hidden_layers)
            put(net,parameters)
            reference=[p.detach().clone() for p in net.parameters()]
            opt=torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.l2)
            loader=DataLoader(dataset(data['X_train'][self.indices], data['y_train'][self.indices]),batch_size=args.batch_size,shuffle=True)
            total=count=0
            for _ in range(args.local_epochs):
                s,n=train_epoch(net,loader,opt,reference=reference,mu=config.get('proximal_mu',0))
                total+=s; count+=n
            return get(net),len(self.indices),{'train_loss':total/count}
    clients=[Client(p) for p in parts]
    parameters=get(initial); rng=np.random.default_rng(args.seed); volume=0
    for rnd in range(1,args.rounds+1):
        selected=rng.choice(len(clients),max(1,math.ceil(args.participation*len(clients))),replace=False)
        responses=[]; losses=[]
        for cid in selected:
            arrays,n,m=clients[cid].fit(parameters,{'proximal_mu':args.mu if args.method=='fedprox' else 0.})
            volume+=sum(a.nbytes for a in parameters)+sum(a.nbytes for a in arrays)
            responses.append((None,FitRes(status=Status(Code.OK,''), parameters=ndarrays_to_parameters(arrays),num_examples=n,metrics=m)))
            losses.append((n,m['train_loss']))
        aggregated,_=strategy.aggregate_fit(rnd,responses,[])
        parameters=parameters_to_ndarrays(aggregated)
        put(initial,parameters)
        callback(rnd,initial,sum(n*l for n,l in losses)/sum(n for n,l in losses),volume)
    return initial,volume
