"""Model 3 of 4 -- Bidirectional GRU over the ECG beat sequence.

Owner: Ishaan Karmokar
Architecture family: recurrent neural network

The beat is read as a time series in both directions, so the state at the
R peak already knows what comes before it (P wave, PR segment) and after it
(ST segment, T wave). A 3x average-pool first brings the 360-sample beat to
120 steps: after the 35 Hz low-pass, 120 Hz still keeps everything below the
new 60 Hz Nyquist limit, and the GRU unrolls over 3x fewer steps, which
shortens backpropagation through time and its vanishing-gradient path.

Usage
    python src/models/m3_bigru.py            # fixed config, trained with 3 seeds
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import tensorflow as tf

from config import CLASSES, SEEDS, WINDOW
from deep_common import compile_model, limit_threads, run_seeds

# Fixed in advance (the synopsis does not commit the BiGRU to a search).
CONFIG = {'lr': 1e-3, 'pool': 3, 'units': 32, 'dropout': 0.3, 'dense': 64}


def build(cfg):
    beat = tf.keras.Input(shape=(WINDOW, 1), name='beat')
    rr = tf.keras.Input(shape=(4,), name='rr')
    x = tf.keras.layers.AveragePooling1D(int(cfg['pool']))(beat)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(int(cfg['units']), return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(cfg['units'])))(x)
    x = tf.keras.layers.Concatenate()([x, rr])
    x = tf.keras.layers.Dense(int(cfg['dense']), activation='relu')(x)
    x = tf.keras.layers.Dropout(float(cfg['dropout']))(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    return compile_model(tf.keras.Model([beat, rr], out, name='bigru'), cfg['lr'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                    help='train only these seeds (e.g. to run seeds in parallel)')
    args = ap.parse_args()
    limit_threads(args.threads)
    run_seeds('M3 BiGRU', 'm3_bigru', build, CONFIG, seeds=args.seeds)


if __name__ == '__main__':
    main()
