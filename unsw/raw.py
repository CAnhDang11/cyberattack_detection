"""Raw UNSW-NB15 (dataset/UNSW-NB15_1..4.csv): loading, pre-split analysis and leakage-safe re-splitting.

Usage:
    python -m unsw.raw analyze --raw-dir dataset --output experiments/raw_analysis
    python -m unsw.raw prepare --raw-dir dataset --output experiments/data_raw
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import LABELS, NAMES, SCHEMES, Preprocessor, encode, scheme_labels

# Column order from NUSW-NB15_features.csv; the raw files have no header row.
RAW_COLUMNS = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
               'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
               'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
               'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
               'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
               'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
# Align names with the official training/testing-set CSVs so Preprocessor and reports stay comparable.
RENAME = {'smeansz': 'smean', 'dmeansz': 'dmean', 'res_bdy_len': 'response_body_len',
          'Sintpkt': 'sinpkt', 'Dintpkt': 'dinpkt', 'Label': 'label'}
# Host/port/timestamp identify the testbed (attacks come from 175.45.176.0/24), not traffic behaviour;
# the official split also excludes them, so they are kept only as metadata.
METADATA = ['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime']
CATEGORICAL = ['proto', 'state', 'service']
# Blank in raw means "no HTTP/FTP activity"; the official CSVs encode it as 0.
BLANK_IS_ZERO = ['ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd']
CATEGORY_FIX = {'': 'Normal', 'Backdoors': 'Backdoor'}


def raw_files(raw_dir):
    files = sorted(Path(raw_dir).glob('UNSW-NB15_[0-9].csv'))
    if not files:
        raise FileNotFoundError(f'No UNSW-NB15_<n>.csv files in {raw_dir}')
    return files


def raw_columns(raw_dir):
    """Column names from NUSW-NB15_features.csv (raw dir or repo root), else the documented fallback."""
    for p in (Path(raw_dir)/'NUSW-NB15_features.csv', Path('NUSW-NB15_features.csv')):
        if p.exists():
            names = pd.read_csv(p, encoding='latin-1').iloc[:, 1].str.replace(' ', '', regex=False).tolist()
            if [n.lower() for n in names] != [n.lower() for n in RAW_COLUMNS]:
                raise ValueError(f'{p} column list differs from the expected 49 raw columns')
            return names, str(p)
    return RAW_COLUMNS, None


def load_raw(raw_dir):
    """Return (frame, info). Frame columns: metadata, features, attack_cat, label, file, row_id."""
    names, source = raw_columns(raw_dir)
    frames = []
    for f in raw_files(raw_dir):
        d = pd.read_csv(f, header=None, names=names, encoding='utf-8-sig', low_memory=False,
                        dtype={'sport': str, 'dsport': str, 'srcip': str, 'dstip': str, 'attack_cat': str})
        d['file'] = f.name
        d['row_id'] = [f'{f.stem}:{i}' for i in range(len(d))]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    df.columns = [c if c in ('file', 'row_id') else c.lower() for c in df.columns]
    raw_cat = df.attack_cat.fillna('').str.strip()
    info = {'column_names_from': source, 'raw_attack_cat_spellings': {str(k): int(v) for k, v in df.attack_cat.fillna('<blank>').value_counts().items()}}
    df['attack_cat'] = raw_cat.replace(CATEGORY_FIX)
    blanks = {}
    for c in BLANK_IS_ZERO:
        s = pd.to_numeric(df[c].astype(str).str.strip().replace('', np.nan), errors='coerce')
        blanks[c] = int(s.isna().sum())
        df[c] = s.fillna(0)
    info['blank_filled_with_zero'] = blanks
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    return df, info


TTL_FEATURES = ['sttl', 'dttl', 'ct_state_ttl']  # proxies for the sending host (see CLAUDE.md, F3)


def add_derived(df):
    """Per-row ratio features (F4). Uses only the row itself: no labels, no other rows, no identifiers."""
    eps = 1e-6
    out = pd.DataFrame(index=df.index)
    out['byte_ratio'] = df.dbytes / (df.sbytes + 1)
    out['pkt_ratio'] = df.dpkts / (df.spkts + 1)
    out['mean_ratio'] = df.dmean / (df.smean + 1)
    out['payload_est'] = (df.sbytes - 40 * df.spkts).clip(lower=0)
    out['load_ratio'] = df.dload / (df.sload + 1)
    out['jit_ratio'] = df.djit / (df.sjit + eps)
    out['intpkt_ratio'] = df.dinpkt / (df.sinpkt + eps)
    out['srv_src_share'] = df.ct_srv_src / (df.ct_src_ltm + 1)
    out['srv_dst_share'] = df.ct_srv_dst / (df.ct_dst_ltm + 1)
    return out


DERIVED = ['byte_ratio', 'pkt_ratio', 'mean_ratio', 'payload_est', 'load_ratio', 'jit_ratio', 'intpkt_ratio',
           'srv_src_share', 'srv_dst_share']


def feature_columns(df):
    return [c for c in df.columns if c not in METADATA + ['attack_cat', 'label', 'file', 'row_id']]


def feature_hash(df):
    return pd.util.hash_pandas_object(df[feature_columns(df)], index=False).to_numpy()


# ----------------------------------------------------------------------------------------------- analysis
def conflict_matrix(df, h):
    """M[A, B] = rows labelled A whose exact feature vector also occurs labelled B (A != B)."""
    counts = pd.crosstab(h, df.attack_cat).reindex(columns=NAMES, fill_value=0)
    present = counts.to_numpy() > 0
    rows = counts.to_numpy()
    m = (rows.T @ present).astype(np.int64)  # m[a, b] = sum over groups containing b of rows of a
    np.fill_diagonal(m, 0)
    return pd.DataFrame(m, index=NAMES, columns=NAMES), counts


def analyze(raw_dir, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    df, info = load_raw(raw_dir)
    feats = feature_columns(df)
    h = feature_hash(df)
    df['_h'] = h
    numeric = [c for c in feats if c not in CATEGORICAL]

    # 1. labels
    bad_label = int(((df.attack_cat == 'Normal') != (df.label == 0)).sum())
    unexpected = sorted(set(df.attack_cat) - set(NAMES))
    by_file = pd.crosstab(df.attack_cat, df.file).reindex(NAMES)

    # 2. duplicates and label conflicts
    conflict, counts = conflict_matrix(df, h)
    n_labels = (counts.to_numpy() > 0).sum(1)
    conflicting_groups = set(counts.index[n_labels > 1])
    df['_conflict'] = df._h.isin(conflicting_groups)
    exact_dup = df.duplicated(['_h', 'attack_cat'])
    g = df.groupby('attack_cat')
    cls = pd.DataFrame({
        'rows': g.size(),
        'unique_feature_rows': g._h.nunique(),
        'exact_duplicate_rows': exact_dup.groupby(df.attack_cat).sum(),
        'rows_in_conflicting_groups': g._conflict.sum(),
    }).reindex(NAMES)
    cls['share_of_all_rows'] = cls.rows / len(df)
    cls['duplicate_ratio'] = cls.exact_duplicate_rows / cls.rows
    cls['conflict_ratio'] = cls.rows_in_conflicting_groups / cls.rows
    cls['main_conflict_with'] = [conflict.loc[c].idxmax() if conflict.loc[c].max() else '' for c in NAMES]
    # Best achievable recall: a classifier gives one answer per feature vector, so only the group's majority wins.
    for scheme in SCHEMES:
        lab = df.attack_cat.map(SCHEMES[scheme])
        win = pd.crosstab(h, lab).idxmax(axis=1)
        ok = (lab == df._h.map(win)).groupby(lab).mean()
        cls[f'max_recall_{scheme}'] = [ok[SCHEMES[scheme][c]] for c in NAMES]
    largest = df.groupby(['_h', 'attack_cat']).size().sort_values(ascending=False).head(10)

    # 3. data quality
    missing = df[feats].isna().sum()
    negative = (df[numeric] < 0).sum()
    infinite = pd.Series({c: int(np.isinf(df[c].to_numpy(dtype=float)).sum()) for c in numeric})
    constant = [c for c in feats if df[c].nunique(dropna=False) <= 1]
    vocab = {c: int(df[c].nunique()) for c in CATEGORICAL}

    # 4. identifiers that would leak if used as features
    attacker_subnet = df.srcip.str.startswith('175.45.176.')
    subnet = pd.crosstab(df.attack_cat == 'Normal', attacker_subnet).rename(index={True: 'Normal', False: 'Attack'},
                                                                           columns={True: 'from_175.45.176.x', False: 'other_src'})

    summary = dict(info, rows=len(df), files={k: int(v) for k, v in df.file.value_counts().sort_index().items()},
                   feature_count=len(feats), features=feats, excluded_metadata=METADATA,
                   unique_feature_rows=int(df._h.nunique()), exact_duplicate_rows=int(exact_dup.sum()),
                   conflicting_feature_groups=len(conflicting_groups), rows_in_conflicting_groups=int(df._conflict.sum()),
                   label_attack_cat_mismatch=bad_label, unexpected_categories=unexpected,
                   columns_with_missing={k: int(v) for k, v in missing[missing > 0].items()},
                   columns_with_negative={k: int(v) for k, v in negative[negative > 0].items()},
                   columns_with_infinite={k: int(v) for k, v in infinite[infinite > 0].items()},
                   constant_columns=constant, categorical_vocab_size=vocab)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    cls.to_csv(out/'class_stats.csv')
    conflict.to_csv(out/'conflict_matrix.csv')
    by_file.to_csv(out/'class_by_file.csv')
    subnet.to_csv(out/'source_subnet.csv')

    pct = lambda x: f'{100*x:.1f}%'
    lines = ['# Raw UNSW-NB15 analysis', '',
             f'- Rows: **{len(df):,}** from {len(summary["files"])} files; {len(feats)} model features '
             f'(excluded metadata: {", ".join(METADATA)}).',
             f'- Unique feature vectors: **{summary["unique_feature_rows"]:,}** '
             f'({pct(summary["unique_feature_rows"]/len(df))} of rows); exact duplicate rows (same features + label): '
             f'{summary["exact_duplicate_rows"]:,}.',
             f'- Feature vectors carrying more than one attack_cat: **{len(conflicting_groups):,}**, covering '
             f'{summary["rows_in_conflicting_groups"]:,} rows ({pct(summary["rows_in_conflicting_groups"]/len(df))}).',
             f'- label vs attack_cat mismatches: {bad_label}; unexpected categories after cleaning: {unexpected or "none"}.',
             f'- Raw attack_cat spellings were normalised (strip, blank→Normal, Backdoors→Backdoor): '
             f'`{json.dumps(info["raw_attack_cat_spellings"])}`',
             f'- Blank values filled with 0: `{json.dumps(info["blank_filled_with_zero"])}`; other missing: `{json.dumps(summary["columns_with_missing"])}`;'
             f' negative: `{json.dumps(summary["columns_with_negative"])}`; constant columns: {constant or "none"}.',
             '', '## Per class', '', cls.to_markdown(floatfmt='.3f'), '',
             '## Label conflicts', '', 'Row = true class, column = another class that has the *identical* feature vector '
             '(count of rows of the row class).', '', conflict.to_markdown(), '',
             '## Largest identical-row groups', '', largest.rename('rows').reset_index().drop(columns='_h').to_markdown(index=False), '',
             '## Class by raw file (files are chronological capture slices)', '', by_file.to_markdown(), '',
             '## Source subnet (why IP/port/time are excluded)', '', subnet.to_markdown(), '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    return summary


# ------------------------------------------------------------------------------------------------- split
def group_split(y, h, test_size, val_size, seed):
    """Split rows so that identical feature vectors never cross splits; stratified on each group's majority label."""
    groups, inverse = np.unique(h, return_inverse=True)
    counts = np.zeros((len(groups), int(y.max()) + 1), dtype=np.int64)
    np.add.at(counts, (inverse, y), 1)
    major = counts.argmax(1)
    gidx = np.arange(len(groups))
    rest, test = train_test_split(gidx, test_size=test_size, stratify=major, random_state=seed)
    fit, val = train_test_split(rest, test_size=val_size/(1-test_size), stratify=major[rest], random_state=seed)
    which = np.empty(len(groups), dtype=np.int8); which[fit], which[val], which[test] = 0, 1, 2
    side = which[inverse]
    return [np.flatnonzero(side == k) for k in range(3)]


def prepare_raw(raw_dir, output, seed=42, test_size=.2, val_size=.1, limit=0, scheme='original',
                drop=(), derived=()):
    """drop/derived change model inputs only; the split is always computed on the 41 base features,
    so every feature variant shares identical train/val/test rows."""
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1) or limit < 0:
        raise ValueError('Need 0 < test_size, val_size and test_size + val_size < 1, limit >= 0')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    labels = scheme_labels(scheme)
    df, info = load_raw(raw_dir)
    o = encode(df.attack_cat)                    # original attack_cat, always kept
    y = encode(df.attack_cat, labels, scheme)    # training target under the chosen scheme
    h = feature_hash(df)
    # Stratify on the original classes, so every scheme gets the identical split (comparable test rows).
    a, b, t = group_split(o, h, test_size, val_size, seed)
    if limit:  # cap fit rows per class only; validation and test keep their natural distribution
        rng = np.random.default_rng(seed)
        a = np.sort(np.concatenate([rng.permutation(a[y[a] == c])[:limit] for c in range(len(labels))]))
    assert not (set(h[a]) & set(h[t])) and not (set(h[a]) & set(h[b])) and not (set(h[b]) & set(h[t]))
    cols = feature_columns(df)
    unknown = (set(drop) - set(cols)) | (set(derived) - set(DERIVED))
    if unknown:
        raise ValueError(f'Unknown features: {sorted(unknown)}')
    x = df[[c for c in cols if c not in set(drop)]]
    if derived:
        x = pd.concat([x, add_derived(df)[list(derived)]], axis=1)
    cols = list(x.columns)
    pp = Preprocessor().fit(x.iloc[a])
    ids = df.row_id.to_numpy(dtype=str)  # unicode, not object: load() uses allow_pickle=False
    arrays = dict(X_train=pp.transform(x.iloc[a]), y_train=y[a], X_val=pp.transform(x.iloc[b]), y_val=y[b],
                  X_test=pp.transform(x.iloc[t]), y_test=y[t], train_ids=ids[a], val_ids=ids[b], test_ids=ids[t],
                  y_orig_train=o[a], y_orig_val=o[b], y_orig_test=o[t], orig_names=np.asarray(NAMES))
    np.savez_compressed(out/'data.npz', **arrays)
    joblib.dump(pp, out/'preprocess.joblib')
    (out/'labels.json').write_text(json.dumps(labels, indent=2))
    names = list(labels)
    counts = lambda idx: {names[k]: int(v) for k, v in zip(*np.unique(y[idx], return_counts=True))}
    audit = dict(info, source='raw', raw_dir=str(raw_dir), files=[f.name for f in raw_files(raw_dir)],
                 seed=seed, test_size=test_size, val_size=val_size, limit_per_class=limit,
                 label_scheme=scheme, label_mapping=SCHEMES[scheme],
                 test_counts_original={NAMES[k]: int(v) for k, v in zip(*np.unique(o[t], return_counts=True))},
                 split='group-stratified on exact feature vector (majority label per group)',
                 features=cols, excluded_metadata=METADATA, dropped_features=list(drop), derived_features=list(derived), rows_loaded=len(df),
                 fit_rows=len(a), validation_rows=len(b), test_rows=len(t),
                 fit_counts=counts(a), validation_counts=counts(b), test_counts=counts(t),
                 input_dim=int(arrays['X_train'].shape[1]), shared_feature_vectors_between_splits=0)
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ('rows_loaded', 'fit_rows', 'validation_rows',
                                            'test_rows', 'input_dim', 'test_counts')}, indent=2))
    return arrays


def parse_feature_args(drop, derived):
    split = lambda s: tuple(x.strip() for x in s.split(',') if x.strip())
    return split(drop), (tuple(DERIVED) if derived.strip() == 'all' else split(derived))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['analyze', 'prepare'])
    p.add_argument('--raw-dir', default='dataset')
    p.add_argument('--output')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=float, default=.2)
    p.add_argument('--val-size', type=float, default=.1)
    p.add_argument('--limit-per-class', type=int, default=0, help='Cap fit rows per class (0 = all)')
    p.add_argument('--label-scheme', choices=list(SCHEMES), default='original')
    p.add_argument('--drop-features', default='', help='Comma list of base features to exclude, e.g. sttl,dttl,ct_state_ttl')
    p.add_argument('--derived-features', default='', help=f'Comma list from {DERIVED} or "all"')
    a = p.parse_args(argv)
    if a.command == 'analyze':
        analyze(a.raw_dir, a.output or 'experiments/raw_analysis')
    else:
        prepare_raw(a.raw_dir, a.output or 'experiments/data_raw', a.seed, a.test_size, a.val_size,
                    a.limit_per_class, a.label_scheme, *parse_feature_args(a.drop_features, a.derived_features))


if __name__ == '__main__':
    main()
