"""Model 3 of 4 -- Bidirectional GRU over the ECG beat sequence.

Owner: Ishaan Karmokar
Architecture family: recurrent neural network

The beat is read as a time series in both directions, so the state at the
R peak already knows what comes before it (P wave, PR segment) and after it
(ST segment, T wave). A 3x average-pool first brings the 360-sample beat to
120 steps: after the 35 Hz low-pass, 120 Hz still keeps everything below the
new 60 Hz Nyquist limit, and the GRU unrolls over 3x fewer steps, which
shortens backpropagation through time and its vanishing-gradient path.

Final phase: the context input (3 s at 120 Hz) is already at 120 Hz, so it is
read without pooling: 360 steps covering the beat and its neighbours, i.e.
"patterns across the heartbeat sequence" as the synopsis describes.

Usage
    python src/models/m3_bigru.py                 # interim: fixed config, 3 seeds
    python src/models/m3_bigru.py --mode cv       # final phase: grouped-CV ablations
    python src/models/m3_bigru.py --mode final    # final phase: all of DS1, 3 seeds
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import tensorflow as tf

from config import CLASSES, SEEDS, WINDOW
from deep_common import compile_model, limit_threads, maybe_augment, run_seeds

# Fixed in advance (the synopsis does not commit the BiGRU to a search).
CONFIG = {'lr': 1e-3, 'pool': 3, 'units': 32, 'dropout': 0.3, 'dense': 64}


def build(cfg):
    beat = tf.keras.Input(shape=(WINDOW, 1), name='beat')
    rr = tf.keras.Input(shape=(4,), name='rr')
    x = maybe_augment(beat, cfg)
    if int(cfg['pool']) > 1:
        x = tf.keras.layers.AveragePooling1D(int(cfg['pool']))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(int(cfg['units']), return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(cfg['units'])))(x)
    x = tf.keras.layers.Concatenate()([x, rr])
    x = tf.keras.layers.Dense(int(cfg['dense']), activation='relu')(x)
    x = tf.keras.layers.Dropout(float(cfg['dropout']))(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    return compile_model(tf.keras.Model([beat, rr], out, name='bigru'), cfg['lr'],
                         cfg.get('loss', 'ce'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--mode', choices=['interim', 'cv', 'final'], default='interim')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                    help='train only these seeds (e.g. to run seeds in parallel)')
    args = ap.parse_args()
    limit_threads(args.threads)
    if args.mode == 'cv':
        from cv import ABLATION_STEPS, deep_cv_select
        # context is already 120 Hz, so the 3x pooling of the beat input is dropped
        steps = [(n, {**u, 'pool': 1} if n == 'context' else u) for n, u in ABLATION_STEPS]
        deep_cv_select('m3_bigru', build, CONFIG, steps)
    elif args.mode == 'final':
        from cv import run_final_deep
        run_final_deep('M3 BiGRU', 'm3_bigru', build, seeds=args.seeds)
    else:
        run_seeds('M3 BiGRU', 'm3_bigru', build, CONFIG, seeds=args.seeds)


if __name__ == '__main__':
    main()
