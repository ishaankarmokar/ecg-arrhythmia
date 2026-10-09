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
    python src/models/m1_random_forest.py               # interim: 3 seeds, DS1-train
    python src/models/m1_random_forest.py --mode cv     # final phase: grouped-CV ablations
    python src/models/m1_random_forest.py --mode final  # final phase: all of DS1, 3 seeds

Final-phase options (chosen by grouped CV, see src/cv.py):
  * rel_rr  add the 4 patient-relative RR features (RR divided by the patient's
            own mean RR over the previous 5 minutes), so a beat counts as
            "early" relative to that patient's rhythm, not the training patients'
  * smote   oversample S, V and F in each training set with SMOTE (Chawla et
            al., 2002); held-out and test beats are never resampled
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
from config import CLASSES, SEEDS, SMOKE
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


# ---------------------------------------------------------------- final phase
TAG = 'm1_random_forest'
REL_NAMES = ['pre_RR_rel', 'post_RR_rel', 'local_RR_rel', 'RR_ratio_rel']


def all_features(rel_rr=False):
    d = np.load(os.path.join(ROOT, 'data', 'processed', 'beats.npz'))
    A, names = featurise(d['X'], d['F'])
    if rel_rr:
        Fr = np.load(os.path.join(ROOT, 'data', 'processed', 'context.npz'))['Frel']
        A, names = np.hstack([A, Fr]).astype(np.float32), names + REL_NAMES
    data = {'A': A, 'y': d['y'].astype(np.int64), 'rec': d['rec'], 'split': d['split']}
    if SMOKE:
        rng = np.random.default_rng(0)
        keep = (data['y'] != 0) | (rng.random(len(data['y'])) < 0.04)
        data = {k: v[keep] for k, v in data.items()}
    return data, names


def smote(A, y, seed, target=3000):
    """Raise S, V and F to at least `target` training beats each. Classes with
    fewer than 6 beats in a training set (Q) are left as they are."""
    from imblearn.over_sampling import SMOTE
    counts = np.bincount(y, minlength=len(CLASSES))
    strategy = {c: target for c in (1, 2, 3) if 6 <= counts[c] < target}
    if not strategy:
        return A, y
    k = int(min(5, min(counts[c] for c in strategy) - 1))
    return SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed).fit_resample(A, y)


def fit_rf(A, y, cfg, seed, n_jobs=-1):
    if cfg.get('smote'):
        A, y = smote(A, y, seed)
    return build(seed, n_jobs).fit(A, y)


def proba5(model, A):
    """predict_proba with one column per AAMI class, even if a class was absent."""
    out = np.zeros((len(A), len(CLASSES)), dtype=np.float32)
    out[:, model.classes_] = model.predict_proba(A)
    return out


def cv_rf(cfg, data, seed=SEEDS[0], n_jobs=-1):
    from cv import fold_splits
    from calibrate import fast_f1
    A, y = data['A'], data['y']
    ds1 = np.flatnonzero(data['split'] != 'test')
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    fold_of = np.full(len(y), -1)
    t0 = time.time()
    for k, tr, va in fold_splits(data):
        oof[va] = proba5(fit_rf(A[tr], y[tr], cfg, seed, n_jobs), A[va])
        fold_of[va] = k
        print(f'    fold {k + 1} done', flush=True)
    pred = oof[ds1].argmax(1)
    return {'cv_macro_f1_nsvf': float(fast_f1(y[ds1], pred)[:4].mean()), 'best_epoch': None,
            'fold_macro_f1_nsvf': [float(fast_f1(y[ds1][fold_of[ds1] == k],
                                                 pred[fold_of[ds1] == k])[:4].mean())
                                   for k in range(4)],
            'per_class_f1_oof': fast_f1(y[ds1], pred).tolist(),
            'seconds': round(time.time() - t0, 1), '_oof': oof[ds1], '_y': y[ds1]}


def cv_mode(n_jobs):
    from cv import cached_cv, greedy_ablation, select
    cache = {}

    def evaluate(variant, cfg):
        rel = bool(cfg.get('rel_rr'))
        if rel not in cache:
            cache[rel] = all_features(rel)[0]
        return cached_cv(TAG, variant, cfg, lambda: cv_rf(cfg, cache[rel], n_jobs=n_jobs))

    steps = [('relRR', {'rel_rr': True}), ('smote', {'smote': True})]
    cfg, variant, best, log = greedy_ablation(TAG, {'rel_rr': False, 'smote': False}, steps, evaluate)
    select(TAG, cfg, variant, best, log)


def final_mode(n_jobs):
    from cv import FINAL, FINAL_SEEDS, _read, finish_run, summarise
    sel = _read(os.path.join(FINAL, TAG, 'selected.json'))
    cfg, offsets = sel['config'], np.array(sel['offsets'])
    data, names = all_features(bool(cfg.get('rel_rr')))
    A, y = data['A'], data['y']
    tr, te = data['split'] != 'test', data['split'] == 'test'
    importances = []
    for seed in FINAL_SEEDS:
        if os.path.exists(os.path.join(FINAL, 'runs', f'{TAG}_seed{seed}.json')):
            continue
        t0 = time.time()
        model = fit_rf(A[tr], y[tr], cfg, seed, n_jobs)
        p = proba5(model, A[te])
        os.makedirs(os.path.join(FINAL, 'runs'), exist_ok=True)
        np.savez_compressed(os.path.join(FINAL, 'runs', f'{TAG}_seed{seed}_probs.npz'), p=p)
        finish_run('M1 Random Forest', TAG, seed, y[te], p, offsets,
                   {'config': cfg, 'train_seconds': round(time.time() - t0, 1),
                    'n_features': int(A.shape[1])})
        importances.append(model.feature_importances_)
    if importances:
        imp = np.mean(importances, axis=0)
        with open(os.path.join(FINAL, TAG, 'feature_importance.json'), 'w') as fh:
            json.dump([{'feature': names[i], 'importance': float(imp[i])}
                       for i in np.argsort(imp)[::-1]], fh, indent=2)
    summarise('M1 Random Forest', TAG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--mode', choices=['interim', 'cv', 'final'], default='interim')
    args = ap.parse_args()
    if args.mode == 'cv':
        return cv_mode(args.jobs)
    if args.mode == 'final':
        return final_mode(args.jobs)

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
