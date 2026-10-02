"""One evaluation function, used by all four models.

If each member writes their own metric code the comparison table cannot be
trusted. Import `score()` and `report()` from here, nothing else.
"""
import os, sys, json
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSES, SEARCH_CLASSES


def score(y_true, y_pred) -> dict:
    """Common metric set. Macro-averaged, because ~89% of beats are class N
    and plain accuracy therefore rewards a model that never predicts S or F."""
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), average='macro', zero_division=0)
    return {
        'accuracy':        float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(mp),
        'macro_recall':    float(mr),
        'macro_f1':        float(mf1),
        # Most inter-patient papers drop Q, so this is the number that can be
        # set beside the literature. The five-class macro_f1 stays primary.
        'macro_f1_nsvf':   search_f1(y_true, y_pred),
        'per_class_f1':    {c: float(v) for c, v in zip(CLASSES, f1)},
        'per_class_recall':{c: float(v) for c, v in zip(CLASSES, r)},
        'support':         {c: int(v) for c, v in zip(CLASSES, sup)},
        'confusion_matrix': confusion_matrix(
            y_true, y_pred, labels=range(len(CLASSES))).tolist(),
    }


def search_f1(y_true, y_pred) -> float:
    """Macro-F1 over N/S/V/F only. Used for model selection on the validation
    split (early stopping, and the CNN's DE/PSO objective as 1 - this),
    because validation contains no Q beats. Final reporting covers all five
    classes."""
    idx = [CLASSES.index(c) for c in SEARCH_CLASSES]
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average='macro', zero_division=0)
    return float(f1)


def report(name, y_true, y_pred, save_to=None, extra=None) -> dict:
    m = score(y_true, y_pred)
    if extra:
        m.update(extra)
    print(f'\n=== {name} ===')
    print(f"accuracy {m['accuracy']:.4f}   macro-F1 {m['macro_f1']:.4f}   "
          f"macro-recall {m['macro_recall']:.4f}")
    print(classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0, digits=4))
    cm = np.array(m['confusion_matrix'])
    print('confusion matrix (rows = true, cols = predicted)')
    print('        ' + ''.join(f'{c:>8s}' for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f'  {c:5s} ' + ''.join(f'{v:8d}' for v in cm[i]))
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, 'w') as fh:
            json.dump({'model': name, 'classes': CLASSES, **m}, fh, indent=2)
        print(f'-> {save_to}')
    return m


def aggregate(name, runs, save_to=None) -> dict:
    """Mean and standard deviation over seeds of every scalar and per-class
    metric. One seed is an anecdote; the comparison table uses these."""
    def ms(vals):
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    out = {'model': name, 'classes': CLASSES, 'n_runs': len(runs),
           'seeds': [r.get('seed') for r in runs]}
    for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
              'macro_f1_nsvf']:
        out[k] = ms([r[k] for r in runs])
    for k in ['per_class_f1', 'per_class_recall']:
        out[k] = {c: ms([r[k][c] for r in runs]) for c in CLASSES}
    out['support'] = runs[0]['support']
    out['confusion_matrix_sum'] = np.sum(
        [r['confusion_matrix'] for r in runs], axis=0).tolist()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, 'w') as fh:
            json.dump(out, fh, indent=2)
    print(f"\n### {name}: macro-F1 {out['macro_f1']['mean']:.4f} "
          f"+/- {out['macro_f1']['std']:.4f} over {len(runs)} seeds")
    return out


def load(split=None):
    """Load the shared beat dataset. `split` in {'train','val','test'}."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = np.load(os.path.join(here, 'data', 'processed', 'beats.npz'))
    X, y, rec, sp = d['X'], d['y'], d['rec'], d['split']
    if split is None:
        return X, y, rec, sp
    m = sp == split
    return X[m], y[m], rec[m]
