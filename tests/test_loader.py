import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import DataLoader, Subset
from unsw.model import TensorLoader, dataset, model, train_epoch


def _data(n=1003, d=7, k=4, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, d)).astype(np.float32), r.integers(0, k, n)


@pytest.mark.parametrize('subset', [False, True])
def test_tensor_loader_matches_dataloader_batches(subset):
    x, y = _data()
    idx = np.random.default_rng(1).permutation(len(y))[:700] if subset else None
    def ref():
        ds = dataset(x, y)
        if subset: ds = Subset(ds, idx.tolist())
        return DataLoader(ds, batch_size=64, shuffle=True)
    for make in (ref, lambda: TensorLoader(x, y, 64, indices=idx)):
        torch.manual_seed(42); loader = make()
        batches = [(bx.clone(), by.clone()) for _ in range(3) for bx, by in loader]  # 3 epochs: reshuffles each time
        if make is ref: expected = batches
    assert len(batches) == len(expected) == 3 * len(loader)
    for (a, b), (c, d) in zip(batches, expected):
        assert torch.equal(a, c) and torch.equal(b, d)


def test_training_identical_to_dataloader():
    x, y = _data(n=600)
    def train(make):
        torch.manual_seed(7); net = model(x.shape[1], 16, 2, 4)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2); loader = make()
        losses = [train_epoch(net, loader, opt, weight=torch.tensor([1., 2., 1., .5]))[0] for _ in range(2)]
        return losses, [p.detach().clone() for p in net.parameters()]
    l1, p1 = train(lambda: DataLoader(dataset(x, y), batch_size=32, shuffle=True))
    l2, p2 = train(lambda: TensorLoader(x, y, 32))
    assert l1 == l2 and all(torch.equal(a, b) for a, b in zip(p1, p2))
