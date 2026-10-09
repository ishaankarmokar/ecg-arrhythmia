"""Patient-level bootstrap confidence intervals for the DS2 results.

Owner: Tanishq Sharma

Why patients and not beats: DS2 beats are not independent. 75% of the S beats
come from record 232 and 93% of the F beats from record 213, so resampling
individual beats would make the intervals look far tighter than they are.
Here each bootstrap sample draws the 22 DS2 *records* with replacement, and
the macro-F1 of each seed's predictions is averaged over the seeds.

Step 1 regenerates the DS2 predictions of every trained model and seed
(deep models from saved_models/, the Random Forest by refitting with its
fixed seed) into results/predictions/. Step 2 bootstraps them.

    python src/bootstrap_ci.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from config import RANDOM_SEED, SEEDS
from evaluate import score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, 'results', 'predictions')
N_BOOT = 2000

MODELS = ['m1_random_forest', 'm2_cnn1d_tuned', 'm3_bigru', 'm4_transformer',
          'm2_cnn1d_default']


def export_predictions():
    os.makedirs(PRED, exist_ok=True)
    d = np.load(os.path.join(ROOT, 'data', 'processed', 'beats.npz'))
    m = d['split'] == 'test'
    np.savez_compressed(os.path.join(PRED, 'ds2_truth.npz'), y=d['y'][m], rec=d['rec'][m])

    todo = [t for t in MODELS
            if not all(os.path.exists(os.path.join(PRED, f'{t}_seed{s}.npz')) for s in SEEDS)]
    if 'm1_random_forest' in todo:
        from m1_random_forest import build, load_features
        data, _ = load_features()
        (Atr, ytr), (Ate, _) = data['train'], data['test']
        for s in SEEDS:
            np.savez_compressed(os.path.join(PRED, f'm1_random_forest_seed{s}.npz'),
                                y_pred=build(s).fit(Atr, ytr).predict(Ate).astype(np.int8))
    deep = [t for t in todo if t != 'm1_random_forest']
    if deep:
        from deep_common import load_data, load_trained, predict
        import m2_cnn1d, m3_bigru, m4_transformer
        data = load_data()
        Xte, Fte, _ = data['test']
        for tag in deep:
            for s in SEEDS:
                with open(os.path.join(ROOT, 'results', 'runs', f'{tag}_seed{s}.json')) as fh:
                    cfg = json.load(fh)['config']
                build = {'m2': m2_cnn1d.build, 'm3': m3_bigru.build, 'm4': m4_transformer.build}[tag[:2]]
                model = load_trained(build, cfg, tag, s)
                np.savez_compressed(os.path.join(PRED, f'{tag}_seed{s}.npz'),
                                    y_pred=predict(model, Xte, Fte).astype(np.int8))


def fast_f1(y, p, k=5):
    """Macro-F1 over all 5 classes and over N/S/V/F, from one confusion matrix
    (same definition as evaluate.score, zero_division=0)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return {'macro_f1': f1.mean(), 'macro_f1_nsvf': f1[:4].mean()}


def bootstrap():
    t = np.load(os.path.join(PRED, 'ds2_truth.npz'))
    y, rec = t['y'].astype(np.int64), t['rec']
    records = np.unique(rec)
    idx_of = {r: np.flatnonzero(rec == r) for r in records}
    rng = np.random.default_rng(RANDOM_SEED)
    draws = [np.concatenate([idx_of[r] for r in rng.choice(records, len(records))])
             for _ in range(N_BOOT)]
    out = {'n_boot': N_BOOT, 'unit': 'DS2 record (patient)', 'n_records': int(len(records)),
           'models': {}}
    for tag in MODELS:
        preds = [np.load(os.path.join(PRED, f'{tag}_seed{s}.npz'))['y_pred'].astype(np.int64) for s in SEEDS]
        point = {k: float(np.mean([score(y, p)[k] for p in preds]))
                 for k in ['macro_f1', 'macro_f1_nsvf']}
        assert abs(point['macro_f1'] - np.mean([fast_f1(y, p)['macro_f1'] for p in preds])) < 1e-9
        boots = {'macro_f1': [], 'macro_f1_nsvf': []}
        for ix in draws:
            ms = [fast_f1(y[ix], p[ix]) for p in preds]
            for k in boots:
                boots[k].append(np.mean([m[k] for m in ms]))
        out['models'][tag] = {k: {'point': point[k],
                                  'ci95': [float(np.percentile(boots[k], 2.5)),
                                           float(np.percentile(boots[k], 97.5))]}
                              for k in boots}
        r = out['models'][tag]['macro_f1']
        print(f"{tag:18s} macro-F1 {r['point']:.3f}  95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]",
              flush=True)
    with open(os.path.join(ROOT, 'results', 'bootstrap_ci.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


def bootstrap_final():
    """Same patient-level bootstrap for the final-phase models, using the CPU
    re-scored DS2 probabilities (results/final/runs/*_probs_cpu.npz), both raw
    (argmax) and with the calibration offsets fitted on DS1 cross-validation."""
    from calibrate import apply as apply_offsets
    final = os.path.join(ROOT, 'results', 'final')
    t = np.load(os.path.join(PRED, 'ds2_truth.npz'))
    y, rec = t['y'].astype(np.int64), t['rec']
    records = np.unique(rec)
    idx_of = {r: np.flatnonzero(rec == r) for r in records}
    rng = np.random.default_rng(RANDOM_SEED)
    draws = [np.concatenate([idx_of[r] for r in rng.choice(records, len(records))])
             for _ in range(N_BOOT)]
    out = {'n_boot': N_BOOT, 'unit': 'DS2 record (patient)', 'models': {}}
    for tag in ['m1_random_forest', 'm2_cnn1d', 'm3_bigru', 'm4_transformer']:
        with open(os.path.join(final, tag, 'selected.json')) as fh:
            offsets = json.load(fh)['offsets']
        probs = [np.load(os.path.join(final, 'runs', f'{tag}_seed{s}_probs_cpu.npz'))['p']
                 for s in SEEDS]
        out['models'][tag] = {}
        for kind in ['raw', 'calibrated']:
            preds = [p.argmax(1) if kind == 'raw' else apply_offsets(p, offsets) for p in probs]
            point = {k: float(np.mean([fast_f1(y, q)[k] for q in preds]))
                     for k in ['macro_f1', 'macro_f1_nsvf']}
            boots = {'macro_f1': [], 'macro_f1_nsvf': []}
            for ix in draws:
                ms = [fast_f1(y[ix], q[ix]) for q in preds]
                for k in boots:
                    boots[k].append(np.mean([m[k] for m in ms]))
            out['models'][tag][kind] = {k: {'point': point[k],
                                            'ci95': [float(np.percentile(boots[k], 2.5)),
                                                     float(np.percentile(boots[k], 97.5))]}
                                        for k in boots}
            r = out['models'][tag][kind]['macro_f1']
            print(f"{tag:18s} {kind:10s} macro-F1 {r['point']:.3f}  95% CI "
                  f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]", flush=True)
    with open(os.path.join(final, 'bootstrap_ci_final.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    if '--final' in sys.argv:
        bootstrap_final()
    else:
        export_predictions()
        bootstrap()
