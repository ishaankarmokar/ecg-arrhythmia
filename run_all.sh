#!/usr/bin/env bash
# Reproduces every result in results/ from scratch (downloads the PhysioNet
# records first if they are missing).
# Two lanes run side by side on an 8-core CPU (4 threads each).
set -e
cd "$(dirname "$0")"
mkdir -p logs
python3 src/download_data.py                                    > logs/download.log 2>&1
python3 src/build_dataset.py                                   > logs/build_dataset.log 2>&1
python3 src/models/m1_random_forest.py                          > logs/m1_random_forest.log 2>&1

lane_a() {
  python3 src/models/m2_cnn1d.py --mode search --opt de  --threads 4 > logs/m2_search_de.log  2>&1 &
  python3 src/models/m2_cnn1d.py --mode search --opt pso --threads 4 > logs/m2_search_pso.log 2>&1
  wait
  python3 src/models/m2_cnn1d.py --mode final            --threads 4 > logs/m2_final.log      2>&1
}
lane_b() {
  python3 src/models/m2_cnn1d.py --mode default --threads 4 > logs/m2_default.log    2>&1
  python3 src/models/m3_bigru.py               --threads 4 > logs/m3_bigru.log       2>&1
  python3 src/models/m4_transformer.py         --threads 4 > logs/m4_transformer.log 2>&1
}
lane_a & lane_b & wait
python3 src/compare.py      > logs/compare.log 2>&1
python3 src/bootstrap_ci.py > logs/bootstrap_ci.log 2>&1

# ---------------------------------------------------------------- final phase
# Grouped CV on DS1, ablations, CNN DE/PSO search and final 3-seed training.
# About 3-4 GPU hours (we ran this part on Colab with notebooks/colab_runner.ipynb);
# on a laptop CPU expect well over 12 hours. Finished runs are cached in
# results/final/ and skipped on restart.
if [ "${FINAL_PHASE:-0}" = "1" ]; then
  python3 src/models/m1_random_forest.py --mode cv            > logs/final_m1_cv.log 2>&1
  python3 src/models/m2_cnn1d.py --mode cv-ablation           > logs/final_m2_ablation.log 2>&1
  python3 src/models/m2_cnn1d.py --mode cv-search --opt de    > logs/final_m2_de.log 2>&1
  python3 src/models/m2_cnn1d.py --mode cv-search --opt pso   > logs/final_m2_pso.log 2>&1
  python3 src/models/m2_cnn1d.py --mode cv-select             > logs/final_m2_select.log 2>&1
  python3 src/models/m3_bigru.py --mode cv                    > logs/final_m3_cv.log 2>&1
  python3 src/models/m4_transformer.py --mode cv              > logs/final_m4_cv.log 2>&1
  python3 src/models/m1_random_forest.py --mode final         > logs/final_m1.log 2>&1
  python3 src/models/m2_cnn1d.py --mode cv-final              > logs/final_m2.log 2>&1
  python3 src/models/m3_bigru.py --mode final                 > logs/final_m3.log 2>&1
  python3 src/models/m4_transformer.py --mode final           > logs/final_m4.log 2>&1
fi
# Analysis of the final models (CPU, minutes): re-score from saved weights,
# patient-level bootstrap CIs, comparison tables and figures.
if [ -f results/final/m4_transformer.json ]; then
  python3 src/rescore_final.py          > logs/rescore_final.log 2>&1
  python3 src/bootstrap_ci.py --final   > logs/bootstrap_final.log 2>&1
  python3 src/compare.py --final        > logs/compare_final.log 2>&1
fi
