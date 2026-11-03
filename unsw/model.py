import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .architecture import DEFAULT_HIDDEN, DEFAULT_HIDDEN_LAYERS, hidden_sizes


def model(input_dim, hidden=DEFAULT_HIDDEN, hidden_layers=DEFAULT_HIDDEN_LAYERS):
    layers = []
    previous = input_dim
    for width in hidden_sizes(hidden, hidden_layers):
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, 10))
    return nn.Sequential(*layers)


def dataset(x,y):
    return TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).long())


def train_epoch(net, loader, optimizer, device='cpu', reference=None, mu=0):
    net.train(); loss_sum = count = 0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        ce=nn.functional.cross_entropy(net(x),y)
        loss=ce
        if reference is not None and mu:
            loss=loss+mu/2*sum((p-r).square().sum() for p,r in zip(net.parameters(),reference))
        loss.backward(); optimizer.step()
        loss_sum+=ce.item()*len(y); count+=len(y)
    return loss_sum,count


@torch.no_grad()
def predict(net,x,device='cpu'):
    net.eval()
    return torch.cat([net(b.to(device)).softmax(1).cpu() for (b,) in DataLoader(TensorDataset(torch.from_numpy(x).float()),batch_size=2048)]).numpy()
