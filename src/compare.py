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


if __name__ == '__main__' and '--final' not in sys.argv:
    main()


# ======================================================================
# Final phase: interim vs final, per-record error analysis, figures
# ======================================================================
FINAL_MODELS = [('m1_random_forest', 'm1_random_forest', 'M1 Random Forest'),
                ('m2_cnn1d', 'm2_cnn1d_tuned', 'M2 1D CNN'),
                ('m3_bigru', 'm3_bigru', 'M3 BiGRU'),
                ('m4_transformer', 'm4_transformer', 'M4 Transformer')]


def _final_preds(tag, kind):
    """Per-seed DS2 predictions of a final model ('raw' argmax or 'calibrated')."""
    from calibrate import apply as apply_offsets
    with open(os.path.join(RESULTS, 'final', tag, 'selected.json')) as fh:
        offsets = json.load(fh)['offsets']
    out = []
    for s in (42, 43, 44):
        p = np.load(os.path.join(RESULTS, 'final', 'runs', f'{tag}_seed{s}_probs_cpu.npz'))['p']
        out.append(p.argmax(1) if kind == 'raw' else apply_offsets(p, offsets))
    return out


def per_record(kind='calibrated', min_beats=30):
    """Recall of S and V per DS2 record (patient), mean over seeds."""
    t = np.load(os.path.join(RESULTS, 'predictions', 'ds2_truth.npz'))
    y, rec = t['y'].astype(int), t['rec']
    preds = {tag: _final_preds(tag, kind) for tag, _, _ in FINAL_MODELS}
    rows = []
    for r in np.unique(rec):
        m = rec == r
        row = {'record': str(r), 'beats': int(m.sum())}
        for c, ci in [('S', 1), ('V', 2)]:
            mc = m & (y == ci)
            row[f'n_{c}'] = int(mc.sum())
            for tag, _, _ in FINAL_MODELS:
                row[f'{tag}_{c}_recall'] = (float(np.mean([(q[mc] == ci).mean() for q in preds[tag]]))
                                            if mc.sum() else None)
        rows.append(row)
    # S recall with and without the single dominant S record (232)
    summary = {}
    for tag, _, _ in FINAL_MODELS:
        s_all = y == 1
        s_wo = s_all & (rec != '232')
        summary[tag] = {'S_recall_all': float(np.mean([(q[s_all] == 1).mean() for q in preds[tag]])),
                        'S_recall_without_232': float(np.mean([(q[s_wo] == 1).mean() for q in preds[tag]])),
                        'S_recall_232_only': float(np.mean([(q[rec == '232'][y[rec == '232'] == 1] == 1).mean()
                                                            for q in preds[tag]]))}
    out = {'kind': kind, 'records': rows, 'S_record_232_effect': summary}
    with open(os.path.join(RESULTS, 'final', f'per_record_{kind}.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    shown = [r for r in rows if r['n_S'] >= min_beats or r['n_V'] >= min_beats]
    return shown, summary


def final_tables():
    fin = {t: json.load(open(os.path.join(RESULTS, 'final', f'{t}.json'))) for t, _, _ in FINAL_MODELS}
    itr = {t: json.load(open(os.path.join(RESULTS, f'{it}.json'))) for t, it, _ in FINAL_MODELS}
    ci_f = json.load(open(os.path.join(RESULTS, 'final', 'bootstrap_ci_final.json')))['models']
    ci_i = json.load(open(os.path.join(RESULTS, 'bootstrap_ci.json')))['models']
    sel = {t: json.load(open(os.path.join(RESULTS, 'final', t, 'selected.json'))) for t, _, _ in FINAL_MODELS}
    L = ['# Final-phase results on DS2', '',
         'Final models: trained on all 22 DS1 patients with the configuration and epoch count chosen by '
         '4-fold grouped cross-validation on DS1; calibration offsets fitted on the out-of-fold DS1 '
         'predictions. DS2 scored once. Mean ± std over seeds 42/43/44; CI = patient-level bootstrap.', '',
         '## Interim vs final (macro-F1 over 5 classes)', '',
         '| Model | Interim | Interim 95% CI | Final raw | Final raw 95% CI | Final calibrated | Final cal. 95% CI |',
         '|---|---|---|---|---|---|---|']
    for t, it, lbl in FINAL_MODELS:
        ci = ci_i[it]['macro_f1']['ci95']
        fr, fc = ci_f[t]['raw']['macro_f1']['ci95'], ci_f[t]['calibrated']['macro_f1']['ci95']
        L.append(f"| {lbl} | {pm(itr[t]['macro_f1'])} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                 f"{pm(fin[t]['raw']['macro_f1'])} | [{fr[0]:.3f}, {fr[1]:.3f}] | "
                 f"{pm(fin[t]['calibrated']['macro_f1'])} | [{fc[0]:.3f}, {fc[1]:.3f}] |")
    for kind in ['raw', 'calibrated']:
        L += ['', f'## Final {kind}: all metrics', '',
              '| Model | Accuracy | Macro-F1 | Macro-F1 (N/S/V/F) | Rec. N | Rec. S | Rec. V | Rec. F | F1 S | F1 V |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for t, _, lbl in FINAL_MODELS:
            r = fin[t][kind]
            L.append(f"| {lbl} | {pm(r['accuracy'])} | {pm(r['macro_f1'])} | {pm(r['macro_f1_nsvf'])} | "
                     + ' | '.join(f"{r['per_class_recall'][c]['mean']:.2f}" for c in 'NSVF') + ' | '
                     + ' | '.join(f"{r['per_class_f1'][c]['mean']:.2f}" for c in 'SV') + ' |')
    L += ['', '## What grouped cross-validation chose (CV macro-F1 over N/S/V/F on DS1)', '',
          '| Model | Steps tried (CV macro-F1, kept/rejected) | Selected | Epochs | Offsets (N,S,V,F,Q) |',
          '|---|---|---|---|---|']
    for t, _, lbl in FINAL_MODELS:
        s = sel[t]
        steps = '; '.join(f"{l['step']} {l['cv_macro_f1_nsvf']:.3f} ({'kept' if l['kept'] else 'rejected'})"
                          for l in s['ablation_log'])
        L.append(f"| {lbl} | {steps} | {s['variant']} | {s['epochs'] or '-'} | "
                 f"{', '.join(f'{v:+.2f}' for v in s['offsets'])} |")
    shown, summ = per_record('calibrated')
    L += ['', '## Per-record recall (calibrated, DS2 records with at least 30 S or 30 V beats)', '',
          '| Record | S beats | ' + ' | '.join(f'S rec. {l}' for _, _, l in FINAL_MODELS) + ' | V beats | '
          + ' | '.join(f'V rec. {l}' for _, _, l in FINAL_MODELS) + ' |',
          '|---|---|' + '---|' * 4 + '---|' + '---|' * 4]
    fmt = lambda v: '-' if v is None else f'{v:.2f}'
    for r in shown:
        L.append(f"| {r['record']} | {r['n_S']} | " + ' | '.join(fmt(r[f'{t}_S_recall']) for t, _, _ in FINAL_MODELS)
                 + f" | {r['n_V']} | " + ' | '.join(fmt(r[f'{t}_V_recall']) for t, _, _ in FINAL_MODELS) + ' |')
    L += ['', '## Effect of record 232 on S recall (calibrated)', '',
          '| Model | S recall, all DS2 | record 232 only | all other records |', '|---|---|---|---|']
    for t, _, lbl in FINAL_MODELS:
        v = summ[t]
        L.append(f"| {lbl} | {v['S_recall_all']:.3f} | {v['S_recall_232_only']:.3f} | {v['S_recall_without_232']:.3f} |")
    with open(os.path.join(RESULTS, 'final', 'comparison_final.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return fin, itr, ci_f, ci_i, sel


def final_figures(fin, itr, ci_f, ci_i, sel):
    labels = [l for _, _, l in FINAL_MODELS]
    # 1. interim vs final macro-F1 with patient-bootstrap CIs
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    for k, (t, it, lbl) in enumerate(FINAL_MODELS):
        for dx, (val, ci, color, name) in zip(
                (-0.18, 0.0, 0.18),
                [(itr[t]['macro_f1']['mean'], ci_i[it]['macro_f1']['ci95'], INK2, 'interim'),
                 (fin[t]['raw']['macro_f1']['mean'], ci_f[t]['raw']['macro_f1']['ci95'], COLORS[0], 'final raw'),
                 (fin[t]['calibrated']['macro_f1']['mean'], ci_f[t]['calibrated']['macro_f1']['ci95'],
                  COLORS[1], 'final calibrated')]):
            ax.errorbar(k + dx, val, yerr=[[val - ci[0]], [ci[1] - val]], fmt='o', ms=5, color=color,
                        ecolor=color, elinewidth=1.4, capsize=3, label=name if k == 0 else None)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel('DS2 macro-F1 (5 classes)')
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(FIGS, 'final_vs_interim.png'))
    plt.close(fig)
    # 2. CV ablation steps
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    for ax, (t, _, lbl), color in zip(axes, FINAL_MODELS, COLORS):
        log = sel[t]['ablation_log']
        vals = [l['cv_macro_f1_nsvf'] for l in log]
        ax.bar(range(len(log)), vals, color=[color if l['kept'] else '#c9c8c3' for l in log],
               edgecolor='white', linewidth=0.6)
        ax.set_xticks(range(len(log)), [l['step'].replace('baseline', 'base') for l in log],
                      rotation=45, ha='right', fontsize=7)
        ax.set_title(lbl, fontsize=8)
        ax.set_ylim(0.38, 0.55)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('CV macro-F1 (N/S/V/F)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'final_cv_ablation.png'))
    plt.close(fig)
    # 3. confusion matrices of the final calibrated models
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    for ax, (t, _, lbl) in zip(axes, FINAL_MODELS):
        cm = np.array(fin[t]['calibrated']['confusion_matrix_sum'], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_title(lbl, fontsize=8)
        ax.set_xlabel('predicted', fontsize=7)
        ax.tick_params(length=0, labelsize=7)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axes[0].set_ylabel('true', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'final_confusion_matrices.png'))
    plt.close(fig)


if __name__ == '__main__' and '--final' in sys.argv:
    os.makedirs(FIGS, exist_ok=True)
    final_figures(*final_tables())
    print('final figures written')
