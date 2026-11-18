import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from unsw.data import NAMES, load
from unsw.raw import RAW_COLUMNS, load_raw, feature_hash, prepare_raw, analyze

SPELLINGS = {'Normal': '', 'Backdoor': 'Backdoors', 'Fuzzers': ' Fuzzers ', 'Reconnaissance': ' Reconnaissance'}


def raw_rows(n, rng):
    cats = [NAMES[i % 10] for i in range(n)]
    d = {c: rng.integers(0, 50, n) for c in RAW_COLUMNS}
    d.update(srcip=['175.45.176.1'] * n, sport=['0x000b'] * n, dstip=['149.171.126.1'] * n, dsport=['-'] * n,
             proto=rng.choice(['tcp', 'udp'], n), state=['CON'] * n, service=['-'] * n,
             ct_flw_http_mthd=[''] * n, attack_cat=[SPELLINGS.get(c, c) for c in cats],
             Label=[int(c != 'Normal') for c in cats])
    return pd.DataFrame(d)[RAW_COLUMNS]


class RawTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); p = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        a, b = raw_rows(600, rng), raw_rows(400, rng)
        b.iloc[:50] = a.iloc[:50].to_numpy()          # duplicates across files
        b.iloc[50, -2], b.iloc[50, -1] = 'DoS', 1      # make a label conflict
        b.iloc[50, :-2] = a.iloc[54, :-2].to_numpy()  # a[54] is Exploits
        a.to_csv(p/'UNSW-NB15_1.csv', header=False, index=False)
        b.to_csv(p/'UNSW-NB15_2.csv', header=False, index=False)
        self.dir = p

    def tearDown(self):
        self.tmp.cleanup()

    def test_cleaning(self):
        df, info = load_raw(self.dir)
        self.assertEqual(len(df), 1000)
        self.assertTrue(set(df.attack_cat) <= set(NAMES))
        self.assertEqual(info['blank_filled_with_zero']['ct_flw_http_mthd'], 1000)
        self.assertEqual(df.ct_flw_http_mthd.sum(), 0)

    def test_split_never_shares_feature_vectors(self):
        d = prepare_raw(self.dir, self.dir/'out', test_size=.2, val_size=.1)
        df, _ = load_raw(self.dir)
        h = dict(zip(df.row_id, feature_hash(df)))
        sets = [{h[i] for i in d[k]} for k in ('train_ids', 'val_ids', 'test_ids')]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(d['y_train']) + len(d['y_val']) + len(d['y_test']), 1000)
        self.assertEqual(load(self.dir/'out/data.npz')['X_train'].shape[1], d['X_test'].shape[1])
        audit = json.loads((self.dir/'out/audit.json').read_text())
        self.assertNotIn('srcip', audit['features'])

    def test_analysis(self):
        s = analyze(self.dir, self.dir/'analysis')
        self.assertEqual(s['conflicting_feature_groups'], 1)
        m = pd.read_csv(self.dir/'analysis/conflict_matrix.csv', index_col=0)
        self.assertEqual(m.loc['DoS', 'Exploits'], 1)
        c = pd.read_csv(self.dir/'analysis/class_stats.csv', index_col=0)
        # one DoS and one Exploits row share a feature vector (a tie): one of them is unreachable...
        self.assertLess(c.loc[['DoS', 'Exploits'], 'max_recall_original'].min(), 1)
        # ...until both are Exploit-like
        self.assertEqual(c.loc[['DoS', 'Exploits'], 'max_recall_merged'].min(), 1)

    def test_merged_scheme_keeps_split_and_original_labels(self):
        a = prepare_raw(self.dir, self.dir/'o')
        b = prepare_raw(self.dir, self.dir/'m', scheme='merged')
        np.testing.assert_array_equal(a['test_ids'], b['test_ids'])     # identical split across schemes
        np.testing.assert_array_equal(a['y_orig_test'], b['y_orig_test'])
        np.testing.assert_array_equal(a['y_test'], a['y_orig_test'])
        self.assertEqual(int(b['y_test'].max()), 6)
        self.assertEqual(json.loads((self.dir/'m/labels.json').read_text())['Exploit-like'], 1)

    def test_feature_variants_keep_split(self):
        from unsw.raw import TTL_FEATURES, DERIVED
        a = prepare_raw(self.dir, self.dir/'base', scheme='merged')
        b = prepare_raw(self.dir, self.dir/'nottl', scheme='merged', drop=TTL_FEATURES, derived=DERIVED)
        np.testing.assert_array_equal(a['test_ids'], b['test_ids'])
        np.testing.assert_array_equal(a['train_ids'], b['train_ids'])
        audit = json.loads((self.dir/'nottl/audit.json').read_text())
        self.assertFalse(set(TTL_FEATURES) & set(audit['features']))
        self.assertTrue(set(DERIVED) <= set(audit['features']))
        self.assertTrue(np.isfinite(b['X_train']).all())
        with self.assertRaises(ValueError):
            prepare_raw(self.dir, self.dir/'bad', drop=('nope',))


if __name__ == '__main__':
    unittest.main()
