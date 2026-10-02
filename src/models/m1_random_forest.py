"""Model 1 of 4 -- Random Forest over handcrafted wavelet and statistical
features: the classical machine-learning baseline of the comparison.

Owner: Tanishq Sharma
Architecture family: classical machine learning (bagged decision trees)

Each beat becomes a fixed-length feature vector:
  * wavelet morphology   db4 discrete wavelet transform, 4 levels: the level-4
                         approximation coefficients (coarse beat shape, 0-11 Hz)
                         plus energy, std and max |coef| of each detail band
  * statistics           mean, std, min, max, median, IQR, energy, skewness,
                         kurtosis and zero-crossing rate of the beat
  * rhythm               the four shared RR-interval features
The forest uses class_weight='balanced_subsample' so each tree sees the
minority classes re-weighted to equal total importance.

Usage
    python src/models/m1_random_forest.py          # 3 seeds, full DS1-train
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SEEDS
from evaluate import aggregate, report

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results')

WAVELET, LEVEL = 'db4', 4
CONFIG = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 2,
          'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}
RR_NAMES = ['pre_RR', 'post_RR', 'local_RR', 'RR_ratio']


def featurise(X, F):
    """Return (features, names) for beats X (n, 360) and RR features F (n, 4)."""
    coeffs = pywt.wavedec(X, WAVELET, level=LEVEL, axis=1)
    approx, details = coeffs[0], coeffs[1:]            # cA4, cD4 .. cD1
    cols, names = [approx], [f'cA{LEVEL}_{i}' for i in range(approx.shape[1])]
    for lvl, d in zip(range(LEVEL, 0, -1), details):
        cols.append(np.column_stack([np.mean(d * d, axis=1), d.std(axis=1),
                                     np.abs(d).max(axis=1)]))
        names += [f'cD{lvl}_energy', f'cD{lvl}_std', f'cD{lvl}_maxabs']
    q75, q25 = np.percentile(X, [75, 25], axis=1)
    cols.append(np.column_stack([
        X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1),
        np.median(X, axis=1), q75 - q25, np.mean(X * X, axis=1),
        skew(X, axis=1), kurtosis(X, axis=1),
        np.mean(np.diff(np.signbit(X).astype(np.int8), axis=1) != 0, axis=1)]))
    names += ['mean', 'std', 'min', 'max', 'median', 'iqr', 'energy',
              'skewness', 'kurtosis', 'zero_cross_rate']
    cols.append(F)
    names += RR_NAMES
    return np.hstack(cols).astype(np.float32), names


def load_features():
    d = np.load(os.path.join(ROOT, 'data', 'processed', 'beats.npz'))
    out = {}
    for sp in ['train', 'val', 'test']:
        m = d['split'] == sp
        A, names = featurise(d['X'][m], d['F'][m])
        out[sp] = (A, d['y'][m].astype(np.int64))
    return out, names


def build(seed, n_jobs=-1):
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **CONFIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()

    data, names = load_features()
    (Atr, ytr), (Ava, yva), (Ate, yte) = data['train'], data['val'], data['test']
    print(f'{Atr.shape[1]} features per beat, {len(ytr)} training beats')
    runs, importances = [], []
    for seed in SEEDS:
        t0 = time.time()
        model = build(seed, args.jobs).fit(Atr, ytr)
        secs = time.time() - t0
        val = report(f'M1 Random Forest [val, seed {seed}]', yva, model.predict(Ava))
        extra = {'seed': seed, 'config': CONFIG, 'train_seconds': round(secs, 1),
                 'n_features': int(Atr.shape[1]),
                 'val_macro_f1_nsvf': val['macro_f1_nsvf']}
        runs.append(report(f'M1 Random Forest [DS2 test, seed {seed}]', yte,
                           model.predict(Ate), extra=extra,
                           save_to=os.path.join(RESULTS, 'runs',
                                                f'm1_random_forest_seed{seed}.json')))
        importances.append(model.feature_importances_)
    aggregate('M1 Random Forest', runs,
              save_to=os.path.join(RESULTS, 'm1_random_forest.json'))
    imp = np.mean(importances, axis=0)
    order = np.argsort(imp)[::-1]
    with open(os.path.join(RESULTS, 'm1_feature_importance.json'), 'w') as fh:
        json.dump([{'feature': names[i], 'importance': float(imp[i])} for i in order],
                  fh, indent=2)
    print('top features:', [names[i] for i in order[:10]])


if __name__ == '__main__':
    main()
