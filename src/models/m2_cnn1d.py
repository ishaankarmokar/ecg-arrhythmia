"""Model 2 of 4 -- 1D CNN over the raw ECG beat, trained with RMSprop and
tuned with Differential Evolution / Particle Swarm Optimisation.

Owner: Parth Upadhyay
Architecture family: convolutional neural network

Usage (interim phase, 6 validation patients)
    python src/models/m2_cnn1d.py --mode default          # untuned baseline, 3 seeds
    python src/models/m2_cnn1d.py --mode search --opt de  # DE search on validation
    python src/models/m2_cnn1d.py --mode search --opt pso # PSO search on validation
    python src/models/m2_cnn1d.py --mode final            # best DE/PSO config, 3 seeds

Usage (final phase, grouped CV over all 22 DS1 patients; see src/cv.py)
    python src/models/m2_cnn1d.py --mode cv-ablation             # context / augment / focal
    python src/models/m2_cnn1d.py --mode cv-search --opt de      # DE, objective = CV macro-F1
    python src/models/m2_cnn1d.py --mode cv-search --opt pso     # PSO, same budget
    python src/models/m2_cnn1d.py --mode cv-select               # pick config, fit offsets
    python src/models/m2_cnn1d.py --mode cv-final                # all of DS1, 3 seeds, DS2 once
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import tensorflow as tf

from config import CLASSES, RANDOM_SEED, WINDOW
from evaluate import search_f1
from search import HyperSpace, OPTIMIZERS
from deep_common import (RESULTS, compile_model, limit_threads, load_data,
                         maybe_augment, predict, run_seeds, train)

# Box searched by DE and PSO. Bounds are kept modest so one candidate trains
# in a few minutes on a laptop CPU.
SPACE = HyperSpace({
    'lr':      ('log',    2e-4, 3e-3),
    'filters': ('choice', [16, 24, 32, 48]),
    'kernel':  ('choice', [5, 7, 9, 11, 15]),
    'blocks':  ('int',    2, 4),
    'dropout': ('float',  0.1, 0.5),
    'dense':   ('choice', [32, 64, 128]),
})

DEFAULT = {'lr': 1e-3, 'filters': 32, 'kernel': 7, 'blocks': 3,
           'dropout': 0.3, 'dense': 64}

# Per-candidate budget during the search. Candidates train on every S/V/F/Q
# beat plus a fixed random quarter of the N beats, for at most SEARCH_EPOCHS
# epochs; this ranks configurations at ~1/5 of the cost. The chosen config is
# then retrained from scratch on the full DS1-train split (--mode final).
SEARCH_EPOCHS = 6
SEARCH_N_FRACTION = 1 / 4


def build(cfg):
    """Conv blocks (Conv1D -> BatchNorm -> ReLU -> MaxPool) with filters
    doubling per block, global average pooling, then the RR features join
    before the dense classifier."""
    beat = tf.keras.Input(shape=(WINDOW, 1), name='beat')
    rr = tf.keras.Input(shape=(4,), name='rr')
    x = maybe_augment(beat, cfg)
    for b in range(int(cfg['blocks'])):
        x = tf.keras.layers.Conv1D(int(cfg['filters']) * 2 ** b, int(cfg['kernel']),
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, rr])
    x = tf.keras.layers.Dense(int(cfg['dense']), activation='relu')(x)
    x = tf.keras.layers.Dropout(float(cfg['dropout']))(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    return compile_model(tf.keras.Model([beat, rr], out, name='cnn1d'), cfg['lr'],
                         cfg.get('loss', 'ce'))


def search_subset(data, seed=RANDOM_SEED):
    X, F, y = data['train']
    rng = np.random.default_rng(seed)
    n_idx = np.flatnonzero(y == CLASSES.index('N'))
    keep = np.sort(np.concatenate([
        np.flatnonzero(y != CLASSES.index('N')),
        rng.choice(n_idx, int(len(n_idx) * SEARCH_N_FRACTION), replace=False)]))
    return dict(data, train=[X[keep], F[keep], y[keep]])


def search(opt, agents, iters):
    data = search_subset(load_data())

    def objective(cfg):
        c = dict(cfg, epochs=SEARCH_EPOCHS)
        model, _ = train(build, c, data, seed=RANDOM_SEED, verbose=False)
        Xva, Fva, yva = data['val']
        return 1.0 - search_f1(yva, predict(model, Xva, Fva))

    cfg, best, history, log = OPTIMIZERS[opt](
        objective, SPACE, n_agents=agents, n_iter=iters, seed=RANDOM_SEED)
    print(f'{opt.upper()} best config {cfg}  val macro-F1(NSVF) {1 - best:.4f}')
    out = {'optimizer': opt, 'agents': agents, 'iterations': iters,
           'budget_evals': len(log), 'search_epochs': SEARCH_EPOCHS,
           'search_train_beats': int(len(data['train'][2])),
           'best_config': cfg, 'best_val_macro_f1_nsvf': 1 - best,
           'history_best_objective': history, 'evaluations': log}
    with open(os.path.join(RESULTS, f'm2_cnn1d_search_{opt}.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


def best_searched_config():
    """Pick whichever of DE / PSO found the better validation score."""
    found = []
    for opt in OPTIMIZERS:
        p = os.path.join(RESULTS, f'm2_cnn1d_search_{opt}.json')
        if os.path.exists(p):
            with open(p) as fh:
                found.append(json.load(fh))
    if not found:
        raise SystemExit('run --mode search --opt de / pso first')
    best = max(found, key=lambda r: r['best_val_macro_f1_nsvf'])
    print(f"using {best['optimizer'].upper()} config {best['best_config']}")
    return dict(best['best_config'], tuned_by=best['optimizer'])


# ---------------------------------------------------------------- final phase
TAG = 'm2_cnn1d'
SEARCH_EPOCHS_CV = 1 if os.environ.get('ECG_SMOKE') == '1' else 12
FLAG_KEYS = ('input', 'augment', 'loss')


def _final_path(name):
    from cv import FINAL
    return os.path.join(FINAL, TAG, name)


def cv_ablation():
    from cv import ABLATION_STEPS, _write, deep_evaluator, greedy_ablation
    cfg, variant, best, log = greedy_ablation(TAG, DEFAULT, ABLATION_STEPS,
                                              deep_evaluator(TAG, build))
    _write(_final_path('ablation.json'), {'config': cfg, 'variant': variant,
                                          'cv_macro_f1_nsvf': best['cv_macro_f1_nsvf'],
                                          'log': log})


def _ablation_flags():
    from cv import _read
    cfg = _read(_final_path('ablation.json'))['config']
    return {k: cfg[k] for k in FLAG_KEYS if k in cfg}


def cv_search(opt, agents, iters):
    """DE or PSO over the architecture box, keeping the input/augment/loss flags
    the ablation chose. Each candidate is scored by grouped-CV macro-F1, and
    every evaluation is cached on disk so a restart resumes where it stopped."""
    from cv import _read, _write, cv_deep
    from deep_common import load_arrays
    flags = _ablation_flags()
    data = load_arrays(flags.get('input', 'beat'))
    cache_path = _final_path(f'search_{opt}_evals.json')
    cache = _read(cache_path) if os.path.exists(cache_path) else {}

    def objective(cfg):
        key = json.dumps(sorted(cfg.items()))
        if key not in cache:
            r = cv_deep(build, {**cfg, **flags}, data, max_epochs=SEARCH_EPOCHS_CV)
            cache[key] = {'config': cfg, 'cv_macro_f1_nsvf': r['cv_macro_f1_nsvf'],
                          'best_epoch': r['best_epoch']}
            _write(cache_path, cache)
        return 1.0 - cache[key]['cv_macro_f1_nsvf']

    cfg, best, history, log = OPTIMIZERS[opt](objective, SPACE, n_agents=agents,
                                              n_iter=iters, seed=RANDOM_SEED)
    _write(_final_path(f'search_{opt}.json'),
           {'optimizer': opt, 'agents': agents, 'iterations': iters, 'flags': flags,
            'budget_evals': len(log), 'search_epochs': SEARCH_EPOCHS_CV,
            'best_config': cfg, 'best_cv_macro_f1_nsvf': 1 - best,
            'history_best_objective': history, 'evaluations': log})
    print(f'{opt.upper()} best {cfg}  CV macro-F1 {1 - best:.4f}')


def cv_select():
    """Re-run the better DE/PSO configuration with the full CV epoch budget and
    keep it only if it beats the ablation-selected default under the same CV;
    then fit the calibration offsets."""
    from cv import _read, cached_cv, cv_deep, select
    from deep_common import load_arrays
    flags = _ablation_flags()
    searches = [_read(_final_path(f'search_{o}.json')) for o in OPTIMIZERS]
    win = max(searches, key=lambda r: r['best_cv_macro_f1_nsvf'])
    tuned_cfg = {**win['best_config'], **flags}
    data = load_arrays(flags.get('input', 'beat'))
    tuned = cached_cv(TAG, 'tuned', tuned_cfg, lambda: cv_deep(build, tuned_cfg, data))
    abl = _read(_final_path('ablation.json'))
    default_res = _read(_final_path(f"cv_{abl['variant']}.json"))
    log = abl['log'] + [{'step': f"{win['optimizer']}-tuned", 'variant': 'tuned',
                         'cv_macro_f1_nsvf': tuned['cv_macro_f1_nsvf'],
                         'kept': tuned['cv_macro_f1_nsvf'] >= default_res['cv_macro_f1_nsvf']}]
    if log[-1]['kept']:
        cfg, variant, res = dict(tuned_cfg, tuned_by=win['optimizer']), 'tuned', tuned
    else:
        cfg, variant, res = abl['config'], abl['variant'], default_res
    select(TAG, cfg, variant, res, log,
           extra={'search': {s['optimizer']: s['best_cv_macro_f1_nsvf'] for s in searches}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['default', 'search', 'final', 'cv-ablation', 'cv-search',
                                       'cv-select', 'cv-final'], default='default')
    ap.add_argument('--opt', choices=list(OPTIMIZERS), default='de')
    ap.add_argument('--agents', type=int, default=5)
    ap.add_argument('--iters', type=int, default=2)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    limit_threads(args.threads)

    if args.mode == 'cv-ablation':
        cv_ablation()
    elif args.mode == 'cv-search':
        cv_search(args.opt, args.agents, args.iters)
    elif args.mode == 'cv-select':
        cv_select()
    elif args.mode == 'cv-final':
        from cv import run_final_deep
        run_final_deep('M2 1D CNN', TAG, build)
    elif args.mode == 'search':
        search(args.opt, args.agents, args.iters)
    elif args.mode == 'default':
        run_seeds('M2 1D CNN (default)', 'm2_cnn1d_default', build, DEFAULT)
    else:
        run_seeds('M2 1D CNN (DE/PSO-tuned)', 'm2_cnn1d_tuned', build,
                  best_searched_config())


if __name__ == '__main__':
    main()
