"""Shared training loop for the three deep models (M2 CNN, M3 BiGRU, M4
Transformer), so that any difference between them comes from the architecture
and not from how each one was trained.

Owner: Riddhi Puneyani

Every deep model gets exactly the same treatment:
  * inputs     the 360-sample cleaned beat + 4 standardised RR features
  * optimiser  RMSprop (Adam and SGD are not used in this project)
  * loss       categorical cross-entropy with inverse-sqrt class weights
  * selection  the epoch with the best validation macro-F1 over N/S/V/F
               is kept; training stops after PATIENCE epochs without gain
  * seeds      config.SEEDS, each run fully deterministic on CPU
DS2 (test) is scored once, after training, with the restored best weights.

Final-phase additions (all off by default, so interim models rebuild exactly):
  * cfg['input']   'beat' (1 s at 360 Hz) or 'context' (3 s at 120 Hz: the beat
                   and its neighbours); both are 360 samples long
  * cfg['augment'] training-only augmentation: amplitude scaling, Gaussian
                   noise, baseline offset, +/-40 ms time shift
  * cfg['loss']    'ce' (class-weighted cross-entropy) or 'focal' (the same
                   weights times (1 - p)^2, which down-weights easy beats)
  * train_fixed()  trains for a fixed number of epochs (chosen by grouped CV)
                   and can score held-out sets after every epoch
"""
import json
import os
import time

os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from config import CLASSES, SEEDS, SMOKE
from evaluate import aggregate, report, search_f1

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results')
WEIGHTS = os.path.join(ROOT, 'saved_models')

MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 128
MAX_CLASS_WEIGHT = 20.0


def set_reproducible(seed):
    tf.keras.utils.set_random_seed(seed)          # python, numpy and tf seeds
    # Bit-exact determinism is guaranteed on CPU. On a GPU (Colab) some kernels
    # have no deterministic version, so seeds are fixed but runs may differ in
    # the last digits; results are reported as mean +/- std over seeds anyway.
    if not tf.config.list_physical_devices('GPU'):
        tf.config.experimental.enable_op_determinism()


def limit_threads(n):
    """Lets two searches share the CPU without fighting over cores."""
    if n:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(2)


def load_data():
    """Train / val / test arrays with RR features standardised on train only."""
    d = np.load(os.path.join(ROOT, 'data', 'processed', 'beats.npz'))
    out = {}
    for sp in ['train', 'val', 'test']:
        m = d['split'] == sp
        out[sp] = [d['X'][m].astype(np.float32), d['F'][m].astype(np.float32),
                   d['y'][m].astype(np.int64)]
    scaler = StandardScaler().fit(out['train'][1])
    for sp in out:
        out[sp][1] = scaler.transform(out[sp][1]).astype(np.float32)
    return out


def class_weights(y):
    """w_c = sqrt(n / (K * n_c)), capped. Full inverse-frequency weights would
    make one Q beat worth ~1000 N beats and destabilise training; the square
    root still lifts S/V/F well above N."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = {}
    for i, n in enumerate(counts):
        if n > 0:
            w[i] = float(min(MAX_CLASS_WEIGHT,
                             np.sqrt(counts.sum() / (len(CLASSES) * n))))
    return w


def focal_loss(gamma=2.0):
    """Sparse categorical focal loss (Lin et al., 2017). Class weights are still
    applied through Keras' class_weight, so this only adds the (1 - p)^gamma
    factor that shrinks the loss of beats the model already gets right."""
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        p = tf.clip_by_value(tf.gather(y_pred, y_true, axis=1, batch_dims=1), 1e-7, 1.0)
        return -tf.pow(1.0 - p, gamma) * tf.math.log(p)
    return loss


def compile_model(model, lr, loss='ce'):
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9),
                  loss=focal_loss() if loss == 'focal' else 'sparse_categorical_crossentropy')
    return model


class BeatAugment(tf.keras.layers.Layer):
    """Random, label-preserving changes applied only while training (Keras
    passes training=False at prediction time, so evaluation is untouched).
    Inputs are z-scored, so noise and offset are in units of one std."""

    def __init__(self, max_shift=14, scale=0.1, noise=0.05, offset=0.1, **kw):
        super().__init__(**kw)
        self.max_shift, self.scale, self.noise, self.offset = max_shift, scale, noise, offset

    def call(self, x, training=None):
        if not training:
            return x
        b, n = tf.shape(x)[0], tf.shape(x)[1]
        x = x * tf.random.uniform([b, 1, 1], 1 - self.scale, 1 + self.scale)
        x = x + tf.random.uniform([b, 1, 1], -self.offset, self.offset)
        x = x + tf.random.normal(tf.shape(x), stddev=self.noise)
        if self.max_shift:
            k = tf.random.uniform([b], -self.max_shift, self.max_shift + 1, dtype=tf.int32)
            idx = tf.clip_by_value(tf.range(n)[None, :] - k[:, None], 0, n - 1)
            x = tf.gather(x, idx, axis=1, batch_dims=1)
        return x

    def get_config(self):
        return {**super().get_config(), 'max_shift': self.max_shift, 'scale': self.scale,
                'noise': self.noise, 'offset': self.offset}


def maybe_augment(x, cfg):
    """Insert BeatAugment when cfg['augment'] is set. 40 ms of time shift is
    14 samples at 360 Hz (beat input) or 5 samples at 120 Hz (context)."""
    if not cfg.get('augment'):
        return x
    return BeatAugment(max_shift=5 if cfg.get('input') == 'context' else 14)(x)


def predict_proba(model, X, F):
    return model.predict([X[..., None], F], batch_size=1024, verbose=0)


def predict(model, X, F):
    p = model.predict([X[..., None], F], batch_size=1024, verbose=0)
    return np.argmax(p, axis=1)


class BestValMacroF1(tf.keras.callbacks.Callback):
    """Early stopping on the metric we actually report, not on val loss.
    Class-weighted loss and macro-F1 often disagree on the best epoch."""

    def __init__(self, Xva, Fva, yva, patience=PATIENCE, verbose=True):
        super().__init__()
        self.Xva, self.Fva, self.yva = Xva, Fva, yva
        self.patience, self.verbose = patience, verbose
        self.best, self.best_epoch, self.best_weights = -1.0, -1, None
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        f1 = search_f1(self.yva, predict(self.model, self.Xva, self.Fva))
        self.history.append({'epoch': epoch + 1, 'loss': float(logs['loss']),
                             'val_macro_f1_nsvf': f1})
        if f1 > self.best:
            self.best, self.best_epoch = f1, epoch + 1
            self.best_weights = self.model.get_weights()
        if self.verbose:
            print(f'    epoch {epoch+1:2d}  loss {logs["loss"]:.4f}  '
                  f'val macro-F1(NSVF) {f1:.4f}  best {self.best:.4f}@{self.best_epoch}',
                  flush=True)
        if epoch + 1 - self.best_epoch >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train(build_fn, cfg, data, seed, max_epochs=MAX_EPOCHS, verbose=True):
    """Train one model with one seed. Returns (model, epoch history)."""
    set_reproducible(seed)
    Xtr, Ftr, ytr = data['train']
    Xva, Fva, yva = data['val']
    model = build_fn(cfg)
    cb = BestValMacroF1(Xva, Fva, yva, verbose=verbose)
    model.fit([Xtr[..., None], Ftr], ytr,
              epochs=int(cfg.get('epochs', max_epochs)),
              batch_size=int(cfg.get('batch_size', BATCH_SIZE)),
              class_weight=class_weights(ytr), callbacks=[cb],
              shuffle=True, verbose=0)
    return model, cb


def run_seeds(name, tag, build_fn, cfg, seeds=SEEDS):
    """Train once per seed, score val and DS2, save per-seed JSON, weights and
    learning curves, then the mean/std summary used in the comparison table."""
    data = load_data()
    os.makedirs(WEIGHTS, exist_ok=True)
    runs = []
    for seed in seeds:
        print(f'\n--- {name}  seed {seed}  cfg {cfg}', flush=True)
        t0 = time.time()
        model, cb = train(build_fn, cfg, data, seed)
        secs = time.time() - t0
        model.save_weights(os.path.join(WEIGHTS, f'{tag}_seed{seed}.weights.h5'))
        Xva, Fva, yva = data['val']
        Xte, Fte, yte = data['test']
        val = report(f'{name} [val, seed {seed}]', yva, predict(model, Xva, Fva))
        extra = {'seed': seed, 'config': cfg, 'best_epoch': cb.best_epoch,
                 'train_seconds': round(secs, 1),
                 'params': int(model.count_params()),
                 'val_macro_f1_nsvf': val['macro_f1_nsvf'],
                 'history': cb.history}
        report(f'{name} [DS2 test, seed {seed}]', yte, predict(model, Xte, Fte),
               extra=extra, save_to=os.path.join(RESULTS, 'runs', f'{tag}_seed{seed}.json'))
    # Seeds may be trained in separate processes (--seeds); the summary is
    # built from the per-seed files once every seed in config.SEEDS exists.
    paths = [os.path.join(RESULTS, 'runs', f'{tag}_seed{s}.json') for s in SEEDS]
    if not all(os.path.exists(p) for p in paths):
        print('summary deferred: not all seeds in config.SEEDS have finished')
        return None
    return aggregate(name, [read_json(p) for p in paths],
                     save_to=os.path.join(RESULTS, f'{tag}.json'))


# --------------------------------------------------------------------------
# Final phase: arrays for grouped CV on DS1 and fixed-epoch training
# --------------------------------------------------------------------------
def load_arrays(input_kind='beat'):
    """All beats with record ids and split labels. RR features are returned
    raw; callers standardise them on their own training rows only."""
    d = np.load(os.path.join(ROOT, 'data', 'processed', 'beats.npz'))
    if input_kind == 'context':
        X = np.load(os.path.join(ROOT, 'data', 'processed', 'context.npz'))['Xc'].astype(np.float32)
    else:
        X = d['X'].astype(np.float32)
    out = {'X': X, 'F': d['F'].astype(np.float32), 'y': d['y'].astype(np.int64),
           'rec': d['rec'], 'split': d['split']}
    return smoke_subset(out) if SMOKE else out


def smoke_subset(data, keep_n=0.04):
    """Smoke tests only: every non-N beat plus 4% of N beats."""
    rng = np.random.default_rng(0)
    keep = (data['y'] != 0) | (rng.random(len(data['y'])) < keep_n)
    return {k: v[keep] for k, v in data.items()}


def scaled_rr(F, train_idx, *other_idx):
    scaler = StandardScaler().fit(F[train_idx])
    return [scaler.transform(F[i]).astype(np.float32) for i in (train_idx, *other_idx)]


def train_fixed(build_fn, cfg, X, F, y, epochs, seed, eval_sets=(), verbose=True):
    """Train for exactly `epochs` epochs (no early stopping). After every epoch
    the class probabilities of each (X, F) in eval_sets are stored, so the
    caller can pick the best epoch across CV folds. Returns (model, probs)
    where probs[k][e] are the probabilities of eval set k after epoch e+1."""
    set_reproducible(seed)
    model = build_fn(cfg)
    probs = [[] for _ in eval_sets]

    class _Collect(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for k, (Xe, Fe) in enumerate(eval_sets):
                probs[k].append(predict_proba(self.model, Xe, Fe).astype(np.float32))
            if verbose:
                print(f'    epoch {epoch+1:2d}  loss {logs["loss"]:.4f}', flush=True)

    model.fit([X[..., None], F], y, epochs=int(epochs),
              batch_size=int(cfg.get('batch_size', BATCH_SIZE)),
              class_weight=class_weights(y), callbacks=[_Collect()],
              shuffle=True, verbose=0)
    return model, probs


def load_trained(build_fn, cfg, tag, seed):
    """Rebuild a model and load its saved weights (used by the notebooks)."""
    model = build_fn(cfg)
    model.load_weights(os.path.join(WEIGHTS, f'{tag}_seed{seed}.weights.h5'))
    return model


def read_json(path):
    with open(path) as fh:
        return json.load(fh)
