"""Model 4 of 4 -- Transformer encoder with multi-head self-attention.

Owner: Riddhi Puneyani
Architecture family: attention

A strided convolution cuts the 360-sample beat into 90 overlapping patches
(about 11 ms each) and embeds each one as a d_model vector; a learned
positional embedding tells the encoder where in the beat each patch sits.
Two pre-norm encoder blocks then let every patch attend to every other, so
the QRS complex can be related directly to the P and T waves regardless of
their distance. Mean pooling over patches gives the beat representation,
which is joined with the RR features before the classifier.

Usage
    python src/models/m4_transformer.py               # interim: fixed config, 3 seeds
    python src/models/m4_transformer.py --mode cv     # final phase: grouped-CV ablations
    python src/models/m4_transformer.py --mode final  # final phase: all of DS1, 3 seeds
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import tensorflow as tf

from config import CLASSES, SEEDS, WINDOW
from deep_common import compile_model, limit_threads, maybe_augment, run_seeds

# Fixed in advance (the synopsis does not commit the Transformer to a search).
CONFIG = {'lr': 5e-4, 'd_model': 32, 'heads': 4, 'ff': 64, 'layers': 2,
          'patch': 4, 'dropout': 0.1, 'dense': 64}


class PositionalEmbedding(tf.keras.layers.Layer):
    """Adds a trainable vector per token position."""

    def __init__(self, length, d_model, **kw):
        super().__init__(**kw)
        self.length, self.d_model = length, d_model
        self.pos = tf.keras.layers.Embedding(length, d_model)

    def call(self, x):
        return x + self.pos(tf.range(self.length))

    def get_config(self):
        return {**super().get_config(), 'length': self.length, 'd_model': self.d_model}


def encoder_block(x, cfg):
    """Pre-norm block: x + MHA(LN(x)), then x + FFN(LN(x))."""
    d, h = int(cfg['d_model']), int(cfg['heads'])
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=d // h,
                                           dropout=float(cfg['dropout']))(y, y)
    x = x + tf.keras.layers.Dropout(float(cfg['dropout']))(y)
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Dense(int(cfg['ff']), activation='gelu')(y)
    y = tf.keras.layers.Dense(d)(y)
    return x + tf.keras.layers.Dropout(float(cfg['dropout']))(y)


def build(cfg):
    beat = tf.keras.Input(shape=(WINDOW, 1), name='beat')
    rr = tf.keras.Input(shape=(4,), name='rr')
    patch, d = int(cfg['patch']), int(cfg['d_model'])
    x = maybe_augment(beat, cfg)
    x = tf.keras.layers.Conv1D(d, 2 * patch + 1, strides=patch, padding='same')(x)
    x = PositionalEmbedding(WINDOW // patch, d)(x)
    for _ in range(int(cfg['layers'])):
        x = encoder_block(x, cfg)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, rr])
    x = tf.keras.layers.Dense(int(cfg['dense']), activation='relu')(x)
    x = tf.keras.layers.Dropout(float(cfg['dropout']))(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    return compile_model(tf.keras.Model([beat, rr], out, name='transformer'), cfg['lr'],
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
        from cv import deep_cv_select
        deep_cv_select('m4_transformer', build, CONFIG)
    elif args.mode == 'final':
        from cv import run_final_deep
        run_final_deep('M4 Transformer Encoder', 'm4_transformer', build, seeds=args.seeds)
    else:
        run_seeds('M4 Transformer Encoder', 'm4_transformer', build, CONFIG, seeds=args.seeds)


if __name__ == '__main__':
    main()
