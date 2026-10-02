"""Model 2 of 4 -- 1D CNN over the raw ECG beat, trained with RMSprop and
tuned with Differential Evolution / Particle Swarm Optimisation.

Owner: Parth Upadhyay
Architecture family: convolutional neural network

Usage
    python src/models/m2_cnn1d.py --mode default          # untuned baseline, 3 seeds
    python src/models/m2_cnn1d.py --mode search --opt de  # DE search on validation
    python src/models/m2_cnn1d.py --mode search --opt pso # PSO search on validation
    python src/models/m2_cnn1d.py --mode final            # best DE/PSO config, 3 seeds
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
                         predict, run_seeds, train)

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
    x = beat
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
    return compile_model(tf.keras.Model([beat, rr], out, name='cnn1d'), cfg['lr'])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['default', 'search', 'final'], default='default')
    ap.add_argument('--opt', choices=list(OPTIMIZERS), default='de')
    ap.add_argument('--agents', type=int, default=5)
    ap.add_argument('--iters', type=int, default=2)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    limit_threads(args.threads)

    if args.mode == 'search':
        search(args.opt, args.agents, args.iters)
    elif args.mode == 'default':
        run_seeds('M2 1D CNN (default)', 'm2_cnn1d_default', build, DEFAULT)
    else:
        run_seeds('M2 1D CNN (DE/PSO-tuned)', 'm2_cnn1d_tuned', build,
                  best_searched_config())


if __name__ == '__main__':
    main()
