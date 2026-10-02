"""Clean and segment MIT-BIH records into labelled single beats.

Pipeline per record:
    1. read the MLII lead and the cardiologist beat annotations (wfdb)
    2. remove baseline wander (cascaded 200 ms / 600 ms median filters)
    3. zero-phase 35 Hz low-pass (4th-order Butterworth, filtfilt)
    4. map annotation symbols to the five AAMI classes, drop non-beat symbols
    5. cut a 1 s window centred on each R peak, z-score it per beat
    6. compute four RR-interval features from the annotated R peaks

Produces data/processed/beats.npz with:
    X     (n, 360) float32  -- 1 s of MLII centred on the R peak, z-scored
    F     (n, 4)   float32  -- RR-interval features (see below)
    y     (n,)     int8     -- 0=N 1=S 2=V 3=F 4=Q
    rec   (n,)     <U3      -- source record, for patient-wise grouping
    split (n,)     <U5      -- 'train' | 'val' | 'test'

RR features matter: a supraventricular (S) beat is defined largely by its
*timing*, not its shape, so a morphology-only model cannot separate S from N.
    F[:,0] pre-RR     seconds from the previous beat
    F[:,1] post-RR    seconds to the next beat
    F[:,2] local-RR   mean of the 10 preceding RR intervals
    F[:,3] ratio      pre-RR / local-RR   (<1 means the beat came early)

Run once. Every model loads this file; nobody re-segments.
"""
import os, sys, collections
import numpy as np
import wfdb
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(__file__))
from config import (DS1, DS2, DS1_VAL, DS1_TRAIN, AAMI, CLASSES, CLASS_TO_IDX,
                    LEAD, PRE, POST, FS, BASELINE_MED_MS, LOWPASS_HZ,
                    LOWPASS_ORDER)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, 'data', 'raw')
OUT = os.path.join(HERE, 'data', 'processed', 'beats.npz')


def lead_index(header):
    """MIT-BIH is not consistent about channel order (record 114 swaps leads)."""
    names = [n.strip() for n in header.sig_name]
    if LEAD in names:
        return names.index(LEAD)
    print(f'    ! {LEAD} absent, channels={names} -> using channel 0')
    return 0


def clean(x, fs=FS):
    """Baseline removal + low-pass, as in de Chazal et al. (2004).

    The median filters track the slow baseline (respiration, electrode
    movement) without smearing the QRS complex; subtracting their output
    leaves the beat morphology on a flat baseline. filtfilt runs the
    Butterworth filter forwards and backwards so R peaks do not shift.
    """
    base = x
    for ms in BASELINE_MED_MS:
        k = int(round(ms / 1000 * fs)) | 1          # odd kernel length
        base = median_filter(base, size=k, mode='nearest')
    y = x - base
    b, a = butter(LOWPASS_ORDER, LOWPASS_HZ / (fs / 2), btype='low')
    return filtfilt(b, a, y).astype(np.float32)


def split_of(rec):
    if rec in DS1_TRAIN: return 'train'
    if rec in DS1_VAL:   return 'val'
    if rec in DS2:       return 'test'
    raise ValueError(rec)


def rr_features(samples, i):
    """RR features for beat i, computed over ALL annotated beats in the record
    (not just the AAMI-mappable ones) so the rhythm context stays correct."""
    pre  = (samples[i] - samples[i - 1]) / FS if i > 0 else np.nan
    post = (samples[i + 1] - samples[i]) / FS if i < len(samples) - 1 else np.nan
    lo = max(0, i - 10)
    local = np.diff(samples[lo:i + 1]).mean() / FS if i - lo >= 2 else pre
    ratio = pre / local if (local and np.isfinite(pre) and local > 0) else np.nan
    return [pre, post, local, ratio]


def main():
    X, F, y, R, S = [], [], [], [], []
    skipped_edge = 0
    skipped_lbl = collections.Counter()

    for rec in DS1 + DS2:
        path = os.path.join(RAW, rec)
        sig = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')
        ch = lead_index(sig)
        x = clean(sig.p_signal[:, ch].astype(np.float64))
        n = len(x)
        sp = split_of(rec)
        samples, symbols = ann.sample, ann.symbol
        kept = 0

        for i, (pos, sym) in enumerate(zip(samples, symbols)):
            cls = AAMI.get(sym)
            if cls is None or cls not in CLASS_TO_IDX:
                skipped_lbl[sym] += 1
                continue
            a, b = pos - PRE, pos + POST
            if a < 0 or b > n:
                skipped_edge += 1
                continue
            feats = rr_features(samples, i)
            if not np.all(np.isfinite(feats)):
                skipped_edge += 1
                continue
            beat = x[a:b]
            sd = beat.std()
            beat = (beat - beat.mean()) / (sd if sd > 1e-6 else 1.0)
            X.append(beat); F.append(feats); y.append(CLASS_TO_IDX[cls])
            R.append(rec); S.append(sp)
            kept += 1
        print(f'  {rec} [{sp:5s}] ch={ch} kept {kept:5d} beats', flush=True)

    X = np.asarray(X, dtype=np.float32)
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    R = np.asarray(R); S = np.asarray(S)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, F=F, y=y, rec=R, split=S)

    print(f'\nX {X.shape}  F {F.shape}  ->  {OUT}')
    print(f'dropped {skipped_edge} beats at record boundaries / without RR context')
    print(f'dropped non-beat annotation symbols: {dict(skipped_lbl)}')
    hdr = '  '.join(f'{c:>8s}' for c in CLASSES)
    print(f'\n{"split":6s} {"total":>8s}  {hdr}')
    for sp in ['train', 'val', 'test']:
        m = S == sp
        row = [int((y[m] == i).sum()) for i in range(len(CLASSES))]
        pct = [100 * (y[m] == i).mean() for i in range(len(CLASSES))]
        print(f'{sp:6s} {m.sum():8d}  ' + '  '.join(f'{v:8d}' for v in row))
        print(f'{"":6s} {"":8s}  ' + '  '.join(f'{v:7.2f}%' for v in pct))


if __name__ == '__main__':
    main()
