import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .architecture import DEFAULT_HIDDEN, DEFAULT_HIDDEN_LAYERS, hidden_sizes


def model(input_dim, hidden=DEFAULT_HIDDEN, hidden_layers=DEFAULT_HIDDEN_LAYERS, n_classes=10):
    layers = []
    previous = input_dim
    for width in hidden_sizes(hidden, hidden_layers):
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, n_classes))
    return nn.Sequential(*layers)


def dataset(x,y):
    return TensorDataset(torch.from_numpy(x).float(),torch.from_numpy(y).long())


class TensorLoader:
    """Shuffled mini-batches by slicing whole tensors (optionally resident on `device`).

    Yields exactly the batches of DataLoader(dataset(x[indices], y[indices]), batch_size, shuffle=True)
    for the same global torch seed: per epoch it draws the DataLoader's base seed, then the
    RandomSampler seed for randperm. DataLoader collates sample by sample, which starves a GPU.
    """
    def __init__(self, x, y, batch_size, shuffle=True, device='cpu', indices=None):
        if indices is not None: x, y = x[indices], y[indices]
        self.x = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long().to(device)
        self.batch_size, self.shuffle, self.device = batch_size, shuffle, self.x.device

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        n = len(self.y)
        if self.shuffle:
            torch.empty((), dtype=torch.int64).random_()  # DataLoader iterator's _base_seed draw
            g = torch.Generator(); g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            order = torch.randperm(n, generator=g).to(self.device)
        else:
            order = torch.arange(n, device=self.device)
        for i in range(0, n, self.batch_size):
            b = order[i:i + self.batch_size]
            yield self.x[b], self.y[b]


def train_epoch(net, loader, optimizer, device='cpu', reference=None, mu=0, weight=None):
    net.train(); loss_sum = count = 0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        ce=nn.functional.cross_entropy(net(x),y,weight=None if weight is None else weight.to(device))
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
