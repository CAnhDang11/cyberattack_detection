"""Training-only preprocessing and capacity-constrained client partitions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold

LABELS = json.loads(Path(__file__).with_name('labels.json').read_text())
NAMES = list(LABELS)
CASES = ('equal_iid', 'unequal_iid', 'equal_non_iid', 'unequal_non_iid')


def encode(values):
    s = pd.Series(values)
    if s.isna().any() or not s.isin(LABELS).all():
        raise ValueError(f'Missing/unexpected attack_cat labels: {s[~s.isin(LABELS)].unique()}')
    return s.map(LABELS).to_numpy(dtype=np.int64)


def decode(indices):
    a = np.asarray(indices)
    if not np.isin(a, range(10)).all():
        raise ValueError('Prediction index outside saved label mapping')
    return np.asarray(NAMES)[a.astype(int)]


def partition(y, clients=4, case='equal_iid', alpha=.5, seed=42):
    if case not in CASES or alpha <= 0 or clients < 1 or clients > len(y):
        raise ValueError('Invalid case, alpha, or client count')
    rng = np.random.default_rng(seed)
    n = len(y)
    weights = np.ones(clients) if case.startswith('equal_') else np.arange(1, clients + 1)
    capacity = np.ones(clients, dtype=int)
    extra = (n-clients)*weights/weights.sum()
    capacity += np.floor(extra).astype(int)
    for k in np.argsort(-(extra-np.floor(extra)))[:n-capacity.sum()]:
        capacity[k] += 1
    remaining = capacity.copy()
    preference = (rng.dirichlet(np.full(clients, alpha), 10)
                  if 'non_iid' in case else np.ones((10, clients)))
    buckets = [[] for _ in range(clients)]
    # Random interleaving prevents class-order exhaustion; exact capacities are enforced.
    for idx in rng.permutation(n):
        p = preference[int(y[idx])] * remaining
        if p.sum() == 0:
            p = remaining.astype(float)
        p /= p.sum()
        k = rng.choice(clients, p=p)
        buckets[k].append(int(idx))
        remaining[k] -= 1
    return [np.asarray(b, dtype=np.int64) for b in buckets]


def features(df):
    return df.drop(columns=['id', 'attack_cat', 'label'], errors='ignore').rename(columns={'ackdat':'ackdata'})


class Preprocessor:
    def fit(self, df):
        x = features(df)
        self.numeric = list(x.select_dtypes(include='number').columns)
        self.categorical = [c for c in x if c not in self.numeric]
        self.medians = x[self.numeric].median().fillna(0)
        a = self.numbers(x)
        self.mean, self.std = a.mean(0), a.std(0)
        self.std[self.std == 0] = 1
        self.vocab = {c: sorted(x[c].fillna('__MISSING__').astype(str).unique()) for c in self.categorical}
        return self

    def numbers(self, x):
        a = x[self.numeric].fillna(self.medians).to_numpy(dtype=float)
        if (a < 0).any() or not np.isfinite(a).all():
            raise ValueError('Numeric features must be finite and nonnegative for log1p')
        return np.log1p(a)

    def transform(self, df):
        x = features(df)
        blocks = [(self.numbers(x)-self.mean)/self.std]
        for c, vocab in self.vocab.items():
            s = x[c].fillna('__MISSING__').astype(str).to_numpy()
            blocks += [(s == v).astype(float)[:, None] for v in vocab[1:]]
        return np.concatenate(blocks, axis=1).astype('float32')


def prepare(train_path, test_path, output, seed=42, limit=0):
    if limit < 0:
        raise ValueError('limit_per_class must be 0 (all training rows) or positive')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(train_path), pd.read_csv(test_path)
    yt, ye = encode(tr.attack_cat), encode(te.attack_cat)
    # Row identities remain separate; document pre-existing feature duplicates across CSVs.
    ht = pd.util.hash_pandas_object(features(tr), index=False).to_numpy()
    he = pd.util.hash_pandas_object(features(te), index=False).to_numpy()
    audit = {'train_counts':tr.attack_cat.value_counts().to_dict(), 'test_counts':te.attack_cat.value_counts().to_dict(),
             'test_rows_with_train_feature_duplicate':int(np.isin(he, ht).sum()),
             'seed':seed, 'limit_per_class':limit, 'train_csv':str(train_path), 'test_csv':str(test_path)}
    if limit:
        # Per-class cap, selected only from the supplied training CSV; test remains unchanged.
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.permutation(np.flatnonzero(yt == c))[:limit] for c in range(10)])
        tr, yt, ht = tr.iloc[idx], yt[idx], ht[idx]
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, yt, ht))
    audit.update(selected_training_csv_rows=len(tr), fit_rows=len(a), validation_rows=len(b), test_rows=len(te))
    pp = Preprocessor().fit(tr.iloc[a])
    arrays = dict(X_train=pp.transform(tr.iloc[a]), y_train=yt[a], X_val=pp.transform(tr.iloc[b]), y_val=yt[b],
                  X_test=pp.transform(te), y_test=ye,
                  train_ids=np.asarray([f'train:{i}' for i in tr.index[a]]),
                  test_ids=np.asarray([f'test:{i}' for i in te.index]),
                  val_ids=np.asarray([f'train:{i}' for i in tr.index[b]]))
    np.savez_compressed(out/'data.npz', **arrays)
    joblib.dump(pp, out/'preprocess.joblib')
    (out/'labels.json').write_text(json.dumps(LABELS, indent=2))
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    return arrays


def load(path):
    p = Path(path)
    if json.loads((p.parent/'labels.json').read_text()) != LABELS:
        raise ValueError('Data label mapping differs from canonical mapping')
    return dict(np.load(p, allow_pickle=False))


def save_partitions(y, parts, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'partitions.npz', **{f'client_{i}':p for i,p in enumerate(parts)})
    pd.DataFrame([dict(client=i, count=len(p), **{name:int((y[p] == c).sum()) for name,c in LABELS.items()})
                  for i,p in enumerate(parts)]).to_csv(out/'clients.csv', index=False)
