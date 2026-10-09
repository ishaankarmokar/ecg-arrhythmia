"""Grouped (leave-patients-out) cross-validation on DS1 for the final phase.

Owner: Parth Upadhyay

Every model decision in the final phase is made here, on DS1 only: which
ablations to keep, the CNN's DE/PSO search, how many epochs to train and the
decision-calibration offsets. DS2 is touched once, by run_final(), after all
of that is fixed.

How a configuration is scored:
  * the 22 DS1 records are split into the 4 folds of config.CV_FOLDS; each
    fold is held out once while the model trains on the other three
  * the model trains for max_epochs and, after every epoch, predicts the
    held-out fold; these out-of-fold (OOF) probabilities are pooled over the
    four folds into one prediction per DS1 beat, per epoch
  * the CV score of an epoch is macro-F1 over N/S/V/F of the pooled OOF
    predictions; the best epoch E* and its score describe the configuration
  * E* is also the number of epochs used when the final model is trained on
    all of DS1, and the OOF probabilities at E* are what calibrate.py fits on

Every finished evaluation is written to results/final/ immediately and reused
if the script is restarted, so a Colab disconnect only loses the current run.
"""
import json
import os
import time

import numpy as np

from config import CLASSES, CV_FOLDS, RANDOM_SEED, SEEDS, SMOKE
from calibrate import apply as apply_offsets, fast_f1, fit_offsets
from evaluate import aggregate, report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, 'results', '_smoke' if SMOKE else 'final')
FINAL_WEIGHTS = os.path.join(ROOT, 'saved_models', '_smoke' if SMOKE else 'final')
CV_EPOCHS = 2 if SMOKE else 20
FINAL_SEEDS = [RANDOM_SEED] if SMOKE else SEEDS
MIN_GAIN = 0.01        # an ablation is kept only if CV macro-F1 rises by this much


def fold_splits(data):
    """Yield (fold, train_idx, heldout_idx) over DS1. Raises if any DS2 beat
    could enter cross-validation."""
    ds1 = np.flatnonzero(data['split'] != 'test')
    assert not np.isin(data['split'][ds1], ['test']).any()
    for k, held in enumerate(CV_FOLDS):
        is_held = np.isin(data['rec'][ds1], held)
        yield k, ds1[~is_held], ds1[is_held]


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2)


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def cv_deep(build_fn, cfg, data, seed=RANDOM_SEED, max_epochs=CV_EPOCHS, verbose=False):
    """Grouped CV of one deep-model configuration (see module docstring)."""
    from deep_common import scaled_rr, train_fixed
    X, F, y = data['X'], data['F'], data['y']
    ds1 = np.flatnonzero(data['split'] != 'test')
    oof = np.zeros((max_epochs, len(y), len(CLASSES)), dtype=np.float32)
    fold_of = np.full(len(y), -1)
    t0 = time.time()
    for k, tr, va in fold_splits(data):
        Ftr, Fva = scaled_rr(F, tr, va)
        _, probs = train_fixed(build_fn, cfg, X[tr], Ftr, y[tr], max_epochs, seed,
                               eval_sets=[(X[va], Fva)], verbose=verbose)
        for e, p in enumerate(probs[0]):
            oof[e, va] = p
        fold_of[va] = k
        print(f'    fold {k + 1}/{len(CV_FOLDS)} done', flush=True)
    yd = y[ds1]
    curve = [float(fast_f1(yd, oof[e, ds1].argmax(1))[:4].mean()) for e in range(max_epochs)]
    best = int(np.argmax(curve))
    pred = oof[best, ds1].argmax(1)
    folds = [float(fast_f1(yd[fold_of[ds1] == k], pred[fold_of[ds1] == k])[:4].mean())
             for k in range(len(CV_FOLDS))]
    return {'cv_macro_f1_nsvf': curve[best], 'best_epoch': best + 1, 'curve': curve,
            'fold_macro_f1_nsvf': folds,
            'per_class_f1_oof': fast_f1(yd, pred).tolist(),
            'seconds': round(time.time() - t0, 1),
            '_oof': oof[best, ds1], '_y': yd}


def cached_cv(tag, variant, cfg, fn):
    """Run fn() once per (model, variant) and keep its result + OOF on disk."""
    path = os.path.join(FINAL, tag, f'cv_{variant}.json')
    oof_path = os.path.join(FINAL, tag, f'oof_{variant}.npz')
    if os.path.exists(path) and os.path.exists(oof_path):
        res = _read(path)
        print(f'  [cached] {tag}/{variant}: CV macro-F1 {res["cv_macro_f1_nsvf"]:.4f}', flush=True)
        return res
    print(f'  [run] {tag}/{variant}  cfg {cfg}', flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res = fn()
    np.savez_compressed(oof_path, oof=res.pop('_oof'), y=res.pop('_y'))
    res.update({'variant': variant, 'config': cfg})
    _write(path, res)
    print(f'  -> CV macro-F1 {res["cv_macro_f1_nsvf"]:.4f} (best epoch {res.get("best_epoch")})',
          flush=True)
    return res


def greedy_ablation(tag, base_cfg, steps, evaluate):
    """Start from base_cfg and try each (name, update) in order; keep an update
    only if CV macro-F1 improves by at least MIN_GAIN. evaluate(variant, cfg)
    must return a cached_cv result."""
    cfg, variant = dict(base_cfg), 'baseline'
    best = evaluate(variant, cfg)
    log = [{'step': 'baseline', 'cv_macro_f1_nsvf': best['cv_macro_f1_nsvf'], 'kept': True}]
    for name, update in steps:
        trial_cfg = {**cfg, **update}
        trial_variant = f'{variant}+{name}'
        res = evaluate(trial_variant, trial_cfg)
        kept = res['cv_macro_f1_nsvf'] >= best['cv_macro_f1_nsvf'] + MIN_GAIN
        log.append({'step': name, 'variant': trial_variant,
                    'cv_macro_f1_nsvf': res['cv_macro_f1_nsvf'], 'kept': bool(kept)})
        if kept:
            cfg, variant, best = trial_cfg, trial_variant, res
    print(f'  {tag}: selected {variant}  CV macro-F1 {best["cv_macro_f1_nsvf"]:.4f}', flush=True)
    return cfg, variant, best, log


def select(tag, cfg, variant, cv_result, log, extra=None):
    """Fit the calibration offsets on the OOF probabilities of the selected
    configuration and store everything the final run needs."""
    z = np.load(os.path.join(FINAL, tag, f'oof_{variant}.npz'))
    offsets, before, after = fit_offsets(z['oof'], z['y'].astype(np.int64))
    sel = {'tag': tag, 'variant': variant, 'config': cfg,
           'epochs': cv_result.get('best_epoch'),
           'cv_macro_f1_nsvf': cv_result['cv_macro_f1_nsvf'],
           'offsets': offsets.tolist(),
           'cv_macro_f1_nsvf_calibrated_oof': after,
           'ablation_log': log, **(extra or {})}
    _write(os.path.join(FINAL, tag, 'selected.json'), sel)
    print(f'  {tag}: offsets {np.round(offsets, 2).tolist()}  OOF macro-F1 {before:.4f} -> {after:.4f}',
          flush=True)
    return sel


def run_final_deep(name, tag, build_fn, seeds=FINAL_SEEDS):
    """Train the selected configuration on all 22 DS1 records for the CV-chosen
    number of epochs, once per seed, and score DS2 once (raw and calibrated)."""
    from deep_common import load_arrays, predict_proba, scaled_rr, train_fixed
    sel = _read(os.path.join(FINAL, tag, 'selected.json'))
    cfg, epochs, offsets = sel['config'], sel['epochs'], np.array(sel['offsets'])
    data = load_arrays(cfg.get('input', 'beat'))
    X, F, y = data['X'], data['F'], data['y']
    tr = np.flatnonzero(data['split'] != 'test')
    te = np.flatnonzero(data['split'] == 'test')
    Ftr, Fte = scaled_rr(F, tr, te)
    os.makedirs(FINAL_WEIGHTS, exist_ok=True)
    for seed in seeds:
        out = os.path.join(FINAL, 'runs', f'{tag}_seed{seed}.json')
        if os.path.exists(out):
            print(f'  [cached] {tag} seed {seed}', flush=True)
            continue
        print(f'\n--- {name} final, seed {seed}, {epochs} epochs, cfg {cfg}', flush=True)
        t0 = time.time()
        model, _ = train_fixed(build_fn, cfg, X[tr], Ftr, y[tr], epochs, seed, verbose=False)
        model.save_weights(os.path.join(FINAL_WEIGHTS, f'{tag}_seed{seed}.weights.h5'))
        p = predict_proba(model, X[te], Fte)
        np.savez_compressed(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}_probs.npz'), p=p)
        finish_run(name, tag, seed, y[te], p, offsets,
                   {'config': cfg, 'epochs': epochs, 'train_seconds': round(time.time() - t0, 1),
                    'params': int(model.count_params())})
    return summarise(name, tag, seeds)


def finish_run(name, tag, seed, y_true, probs, offsets, extra):
    raw = report(f'{name} [DS2 raw, seed {seed}]', y_true, probs.argmax(1))
    cal = report(f'{name} [DS2 calibrated, seed {seed}]', y_true, apply_offsets(probs, offsets))
    _write(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}.json'),
           {'model': name, 'seed': seed, 'offsets': list(map(float, offsets)), **extra,
            'raw': raw, 'calibrated': cal})


def summarise(name, tag, seeds=FINAL_SEEDS):
    paths = [os.path.join(FINAL, 'runs', f'{tag}_seed{s}.json') for s in seeds]
    if not all(os.path.exists(p) for p in paths):
        return None
    runs = [_read(p) for p in paths]
    out = {}
    for kind in ['raw', 'calibrated']:
        rs = [dict(r[kind], seed=r['seed']) for r in runs]
        out[kind] = aggregate(f'{name} ({kind})', rs)
    _write(os.path.join(FINAL, f'{tag}.json'), out)
    return out


ABLATION_STEPS = [
    ('context', {'input': 'context'}),     # the beat plus its neighbours (3 s)
    ('augment', {'augment': True}),
    ('focal', {'loss': 'focal'}),
]


def deep_evaluator(tag, build_fn, max_epochs=CV_EPOCHS):
    """evaluate(variant, cfg) for greedy_ablation, loading each input kind once."""
    from deep_common import load_arrays
    cache = {}

    def evaluate(variant, cfg):
        kind = cfg.get('input', 'beat')
        if kind not in cache:
            cache[kind] = load_arrays(kind)
        return cached_cv(tag, variant, cfg,
                         lambda: cv_deep(build_fn, cfg, cache[kind], max_epochs=max_epochs))
    return evaluate


def deep_cv_select(tag, build_fn, base_cfg, steps=ABLATION_STEPS):
    """Baseline + greedy ablation under grouped CV, then calibration offsets."""
    cfg, variant, best, log = greedy_ablation(tag, base_cfg, steps, deep_evaluator(tag, build_fn))
    return select(tag, cfg, variant, best, log)
