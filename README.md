# ECG Arrhythmia Detection and Classification on MIT-BIH

ICT-4442 Deep Learning mini project, School of Computer Engineering, MIT Manipal.
Four model families are compared on one fixed inter-patient protocol, as
committed in the synopsis: Random Forest, 1D CNN (RMSprop, tuned with DE/PSO),
BiGRU and Transformer encoder.

| ID | Model | Family | Owner |
|---|---|---|---|
| M1 | Random Forest on wavelet + statistical + RR features | classical ML | Tanishq Sharma |
| M2 | 1D CNN, RMSprop, hyperparameters by Differential Evolution / PSO | convolutional | Parth Upadhyay |
| M3 | Bidirectional GRU | recurrent | Ishaan Karmokar |
| M4 | Transformer encoder (multi-head self-attention) | attention | Riddhi Puneyani |

## Protocol (fixed in `src/config.py`)

* **Data:** MIT-BIH Arrhythmia Database (PhysioNet), MLII lead, 44 non-paced records.
* **Cleaning:** 200/600 ms median-filter baseline removal, 35 Hz zero-phase low-pass.
* **Beats:** 360-sample (1 s) window centred on each annotated R peak, z-scored per beat,
  plus 4 RR-interval features.
* **Classes:** the five AAMI classes N, S, V, F, Q.
* **Split:** de Chazal DS1/DS2 inter-patient division. Six DS1 records are held
  out by patient for validation; DS2 is touched once per trained model.
* **Metrics:** macro-F1 and per-class recall are primary; accuracy is secondary.
  Model selection uses validation macro-F1 over N/S/V/F (validation has no Q beats).
* **Training:** RMSprop for every deep model (no Adam / SGD), inverse-sqrt class
  weights, early stopping on validation macro-F1. Every model is run with seeds
  42, 43 and 44 and reported as mean ± std. Runs are deterministic per seed.

| split | beats | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|---:|
| train | 38,855 | 34,853 | 716 | 2,889 | 389 | 8 |
| val | 12,130 | 10,979 | 227 | 899 | 25 | 0 |
| test (DS2) | 49,679 | 44,228 | 1,836 | 3,220 | 388 | 7 |

## Layout

```
src/config.py              shared protocol (split, classes, filters, seeds)
src/download_data.py       fetch the 44 records from PhysioNet
src/build_dataset.py       clean + segment + label -> data/processed/beats.npz
src/evaluate.py            the one metric implementation every model uses
src/search.py              Differential Evolution and Particle Swarm Optimisation
src/compare.py             comparison tables and figures from results/*.json
src/bootstrap_ci.py        patient-level bootstrap 95% CIs for DS2 macro-F1
src/cv.py                  final phase: grouped CV, ablations, final training
src/calibrate.py           final phase: per-class decision offsets fitted on CV
src/rescore_final.py       final phase: CPU re-scoring of the GPU-trained models
src/models/m1_random_forest.py
src/models/m2_cnn1d.py
src/models/m3_bigru.py
src/models/m4_transformer.py
src/models/deep_common.py  shared RMSprop training loop for M2-M4
notebooks/                 00 data, 01-04 one per model, 05 comparison
results/                   per-model summaries; results/runs/ per seed
saved_models/              trained weights for every deep model and seed
figures/                   figures used in the report
```

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` downloads the records if needed, rebuilds the dataset and trains
every model. On an 8-core laptop CPU the full run takes about 2.5 hours; the
notebooks re-score the saved weights in a few minutes.

Interim DS2 results are in `results/comparison.md`.

## Final phase

Every decision is made on DS1 with 4-fold grouped (leave-patients-out)
cross-validation over its 22 patients (`src/cv.py`, folds in `src/config.py`):
which improvements to keep, the CNN's DE/PSO search, the number of epochs and
the decision-calibration offsets (`src/calibrate.py`). Final models are then
trained on all 22 DS1 patients (3 seeds) and DS2 is scored once.

Improvements tested, each kept only if CV macro-F1 rose by at least 0.01:

* multi-beat context input: 3 s around each beat at 120 Hz (`data/processed/context.npz`)
* training-time augmentation (amplitude, noise, baseline offset, +/-40 ms shift)
* focal loss instead of plain class-weighted cross-entropy
* Random Forest: patient-relative RR features and SMOTE

Training ran on a Colab GPU (`notebooks/colab_runner.ipynb`); `src/rescore_final.py`
re-scores every saved model on CPU so all stored numbers reproduce locally.
Results: `results/final/comparison_final.md` (interim vs final, patient-level
bootstrap CIs, per-record error analysis) and `figures/final_*.png`.

```bash
FINAL_PHASE=1 ./run_all.sh     # full final phase (GPU strongly recommended)
```
