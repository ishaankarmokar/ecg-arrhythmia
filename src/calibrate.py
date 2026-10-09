"""Decision calibration: per-class score offsets fitted on cross-validation.

Owner: Tanishq Sharma

A classifier trained on imbalanced data puts its decision boundaries where
they favour the majority class, so S and F beats are often given a fair
probability but still lose the argmax to N. Instead of argmax(p) we predict

    argmax_c ( log p_c + b_c )

with one offset b_c per class. The offsets are fitted to maximise macro-F1
over N/S/V/F on *out-of-fold* probabilities from grouped cross-validation on
DS1, and are then applied unchanged to DS2. They are never fitted on DS2.
b_N is fixed at 0 (only differences matter) and b_Q at 0, because DS1 holds
too few Q beats (8 in total) to fit it.
"""
import numpy as np

TUNED = (1, 2, 3)          # S, V, F
GRID = np.round(np.arange(-2.0, 4.01, 0.25), 2)


def fast_f1(y, pred, k=5):
    """Per-class F1 from one confusion matrix (zero_division = 0)."""
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)


def apply(probs, offsets):
    return np.argmax(np.log(np.clip(probs, 1e-9, 1.0)) + np.asarray(offsets)[None, :], axis=1)


def fit_offsets(probs, y, rounds=3):
    """Coordinate search over GRID for b_S, b_V, b_F; objective = macro-F1 over
    N/S/V/F. Returns (offsets, objective before, objective after)."""
    y = np.asarray(y, dtype=np.int64)
    b = np.zeros(probs.shape[1])
    obj = lambda off: fast_f1(y, apply(probs, off))[:4].mean()
    before = best = obj(b)
    for _ in range(rounds):
        changed = False
        for c in TUNED:
            for v in GRID:
                trial = b.copy()
                trial[c] = v
                score = obj(trial)
                if score > best + 1e-9:
                    best, b, changed = score, trial, True
        if not changed:
            break
    return b, float(before), float(best)
