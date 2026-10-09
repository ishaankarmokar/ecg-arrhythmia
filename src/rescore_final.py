"""Re-score the final-phase models on CPU so every stored number can be
reproduced from the saved weights on any machine.

Owner: Tanishq Sharma

The final models were trained on a Colab GPU, where some kernels are not
bit-deterministic. This script rebuilds each deep model from its selected
configuration, loads saved_models/final/<tag>_seed<s>.weights.h5, predicts
DS2 on CPU and rewrites results/final/runs/*.json and the per-model summaries
from those CPU predictions. The Random Forest has no GPU step and is simply
refitted (scikit-learn is deterministic for a fixed seed). For each run it
reports how often the CPU predictions agree with the GPU predictions saved
during training.

    python src/rescore_final.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from cv import FINAL, FINAL_SEEDS, FINAL_WEIGHTS, _read, _write, finish_run, summarise

NAMES = {'m1_random_forest': 'M1 Random Forest', 'm2_cnn1d': 'M2 1D CNN',
         'm3_bigru': 'M3 BiGRU', 'm4_transformer': 'M4 Transformer Encoder'}


def rescore_deep(tag, build_fn):
    from deep_common import load_arrays, predict_proba, scaled_rr
    sel = _read(os.path.join(FINAL, tag, 'selected.json'))
    cfg, offsets = sel['config'], np.array(sel['offsets'])
    data = load_arrays(cfg.get('input', 'beat'))
    tr = np.flatnonzero(data['split'] != 'test')
    te = np.flatnonzero(data['split'] == 'test')
    _, Fte = scaled_rr(data['F'], tr, te)
    agree = {}
    for seed in FINAL_SEEDS:
        run_path = os.path.join(FINAL, 'runs', f'{tag}_seed{seed}.json')
        gpu = _read(run_path)
        model = build_fn(cfg)
        model.load_weights(os.path.join(FINAL_WEIGHTS, f'{tag}_seed{seed}.weights.h5'))
        p = predict_proba(model, data['X'][te], Fte)
        gpu_p = np.load(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}_probs.npz'))['p']
        agree[seed] = float((p.argmax(1) == gpu_p.argmax(1)).mean())
        np.savez_compressed(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}_probs_cpu.npz'), p=p)
        extra = {k: gpu[k] for k in ('config', 'epochs', 'train_seconds', 'params') if k in gpu}
        finish_run(NAMES[tag], tag, seed, data['y'][te], p, offsets,
                   {**extra, 'scored_on': 'cpu', 'gpu_cpu_argmax_agreement': agree[seed]})
    return agree


def rescore_rf():
    from m1_random_forest import all_features, fit_rf, proba5
    tag = 'm1_random_forest'
    sel = _read(os.path.join(FINAL, tag, 'selected.json'))
    cfg, offsets = sel['config'], np.array(sel['offsets'])
    data, _ = all_features(bool(cfg.get('rel_rr')))
    tr, te = data['split'] != 'test', data['split'] == 'test'
    agree = {}
    for seed in FINAL_SEEDS:
        prev = _read(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}.json'))
        p = proba5(fit_rf(data['A'][tr], data['y'][tr], cfg, seed), data['A'][te])
        colab_p = np.load(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}_probs.npz'))['p']
        agree[seed] = float((p.argmax(1) == colab_p.argmax(1)).mean())
        np.savez_compressed(os.path.join(FINAL, 'runs', f'{tag}_seed{seed}_probs_cpu.npz'), p=p)
        finish_run(NAMES[tag], tag, seed, data['y'][te], p, offsets,
                   {'config': cfg, 'n_features': prev.get('n_features'), 'scored_on': 'cpu',
                    'gpu_cpu_argmax_agreement': agree[seed]})
    return agree


def main():
    import m2_cnn1d, m3_bigru, m4_transformer
    report = {'m1_random_forest': rescore_rf()}
    for tag, mod in [('m2_cnn1d', m2_cnn1d), ('m3_bigru', m3_bigru), ('m4_transformer', m4_transformer)]:
        report[tag] = rescore_deep(tag, mod.build)
    for tag in NAMES:
        summarise(NAMES[tag], tag)
    _write(os.path.join(FINAL, 'cpu_rescore_agreement.json'), report)
    for tag, a in report.items():
        print(f'{tag:18s} CPU vs Colab argmax agreement per seed: '
              + ', '.join(f'{s}: {v:.5f}' for s, v in a.items()))


if __name__ == '__main__':
    main()
