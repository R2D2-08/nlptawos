"""Reproducible random 80:20 split and 5-fold CV ONLY within dev data."""
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from .common import save_json


def make_splits(n, seed=42, n_folds=5, test_size=.20):
    if n < 30:
        raise ValueError('Need at least 30 clean labeled issues')
    dev, test = train_test_split(np.arange(n), test_size=test_size, random_state=seed, shuffle=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_index, val_index in kf.split(dev):
        train = dev[train_index]
        val = dev[val_index]
        assert len(set(train) & set(val)) == 0
        assert len(set(test) & (set(train) | set(val))) == 0
        folds.append({'train': train.tolist(), 'val': val.tolist()})
    assert sorted(dev.tolist() + test.tolist()) == list(range(n))
    assert sorted(v for f in folds for v in f['val']) == sorted(dev.tolist())
    return {'seed': seed, 'n': n, 'test_size': test_size,
            'dev': dev.tolist(), 'test': test.tolist(), 'folds': folds}


def save_splits(n, seed, output, folds=5, test_size=.20):
    result = make_splits(n, seed, folds, test_size)
    save_json(output, result)
    return result
