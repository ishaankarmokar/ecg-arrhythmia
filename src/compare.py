"""Builds the single comparison table and the figures used in the report from
the per-model result files. Nothing here trains or re-scores a model: every
number is read from results/*.json, which evaluate.aggregate() wrote.

Owner: Tanishq Sharma

Outputs
    results/comparison.md            comparison tables (mean +/- std over seeds)
    figures/per_class_f1.png         per-class F1, all models
    figures/confusion_matrices.png   row-normalised confusion matrices (DS2)
    figures/cnn_search.png           DE vs PSO convergence for the CNN
    figures/learning_curves.png      validation macro-F1 per epoch (deep models)
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'figures')

# (result file, label in tables and figures)
MODELS = [
    ('m1_random_forest', 'M1 Random Forest'),
    ('m2_cnn1d_tuned', 'M2 1D CNN (DE/PSO-tuned)'),
    ('m3_bigru', 'M3 BiGRU'),
    ('m4_transformer', 'M4 Transformer'),
]
EXTRA = [('m2_cnn1d_default', 'M2 1D CNN (untuned default)')]

# Categorical slots 1-4 (validated palette), fixed order: colour follows model.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def load(tag):
    p = os.path.join(RESULTS, f'{tag}.json')
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def pm(d):
    return f"{d['mean']:.3f} ± {d['std']:.3f}"


def tables(res):
    lines = ['# Model comparison on DS2 (inter-patient test set)', '',
             'Mean ± standard deviation over seeds 42, 43, 44. Macro averages over '
             'the five AAMI classes unless marked N/S/V/F.', '',
             '| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | '
             'Macro-F1 (N/S/V/F) |', '|---|---|---|---|---|---|']
    for tag, label in MODELS + EXTRA:
        r = res.get(tag)
        if r:
            lines.append(f"| {label} | {pm(r['accuracy'])} | {pm(r['macro_precision'])} | "
                         f"{pm(r['macro_recall'])} | {pm(r['macro_f1'])} | "
                         f"{pm(r['macro_f1_nsvf'])} |")
    for key, title in [('per_class_recall', 'Per-class recall (sensitivity)'),
                       ('per_class_f1', 'Per-class F1')]:
        lines += ['', f'## {title}', '', '| Model | ' + ' | '.join(CLASSES) + ' |',
                  '|---|' + '---|' * len(CLASSES)]
        for tag, label in MODELS + EXTRA:
            r = res.get(tag)
            if r:
                lines.append(f'| {label} | ' + ' | '.join(
                    f"{r[key][c]['mean']:.3f}" for c in CLASSES) + ' |')
    some = next(r for r in res.values() if r)
    lines += ['', 'DS2 support: ' + ', '.join(f'{c} {n}' for c, n in some['support'].items())]
    with open(os.path.join(RESULTS, 'comparison.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def per_class_f1(res):
    present = [(t, l, c) for (t, l), c in zip(MODELS, COLORS) if res.get(t)]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    width = 0.8 / len(present)
    x = np.arange(len(CLASSES))
    for i, (tag, label, color) in enumerate(present):
        r = res[tag]['per_class_f1']
        means = [r[c]['mean'] for c in CLASSES]
        stds = [r[c]['std'] for c in CLASSES]
        ax.bar(x + (i - (len(present) - 1) / 2) * width, means, width * 0.92,
               yerr=stds, color=color, label=label, error_kw={'lw': 0.8, 'ecolor': INK2},
               edgecolor='white', linewidth=0.6)
    ax.set_xticks(x, [f'{c}\n(n={res[present[0][0]]["support"][c]})' for c in CLASSES])
    ax.set_ylabel('F1 on DS2')
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, fontsize=7.5, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    fig.savefig(os.path.join(FIGS, 'per_class_f1.png'))
    plt.close(fig)


def confusion(res):
    present = [(t, l) for t, l in MODELS if res.get(t)]
    fig, axes = plt.subplots(1, len(present), figsize=(1.75 * len(present), 2.1))
    axes = np.atleast_1d(axes)
    for ax, (tag, label) in zip(axes, present):
        cm = np.array(res[tag]['confusion_matrix_sum'], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_title(label.replace(' (DE/PSO-tuned)', ''), fontsize=8)
        ax.set_xlabel('predicted', fontsize=7)
        ax.tick_params(length=0, labelsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel('true', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'confusion_matrices.png'))
    plt.close(fig)


def cnn_search():
    runs = []
    for opt, color in [('de', COLORS[0]), ('pso', COLORS[1])]:
        p = os.path.join(RESULTS, f'm2_cnn1d_search_{opt}.json')
        if os.path.exists(p):
            with open(p) as fh:
                runs.append((opt, color, json.load(fh)))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for opt, color, r in runs:
        best = np.maximum.accumulate([1 - e['objective'] for e in r['evaluations']])
        ax.plot(np.arange(1, len(best) + 1), best, color=color, lw=2,
                label=f"{opt.upper()} (best {r['best_val_macro_f1_nsvf']:.3f})")
        ax.scatter(np.arange(1, len(best) + 1), [1 - e['objective'] for e in r['evaluations']],
                   s=10, color=color, alpha=0.45, linewidths=0)
    ax.set_xlabel('candidate evaluations')
    ax.set_ylabel('val macro-F1 (N/S/V/F)')
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIGS, 'cnn_search.png'))
    plt.close(fig)


def learning_curves():
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for (tag, label), color in zip(MODELS[1:], COLORS[1:]):
        p = os.path.join(RESULTS, 'runs', f'{tag}_seed42.json')
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            h = json.load(fh)['history']
        ax.plot([e['epoch'] for e in h], [e['val_macro_f1_nsvf'] for e in h],
                color=color, lw=2, label=label.split(' (')[0])
    ax.set_xlabel('epoch')
    ax.set_ylabel('val macro-F1 (N/S/V/F)')
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIGS, 'learning_curves.png'))
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    res = {tag: load(tag) for tag, _ in MODELS + EXTRA}
    missing = [t for t, r in res.items() if r is None]
    if missing:
        print('not yet available:', missing)
    tables(res)
    per_class_f1(res)
    confusion(res)
    cnn_search()
    learning_curves()
    print(f'figures written to {FIGS}')


if __name__ == '__main__':
    main()
