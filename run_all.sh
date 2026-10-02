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
