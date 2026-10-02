# Code ownership and commit history

Every file in this repository has one owner, who is responsible for it and
defends it in the viva. For the interim submission all commits were pushed
from Ishaan Karmokar's GitHub account; each commit message names the member
who owns that part.

| Commit | Owner | Files |
|---|---|---|
| Shared protocol, PhysioNet download and preprocessing pipeline | Ishaan Karmokar | `src/config.py`, `src/download_data.py`, `src/build_dataset.py`, `notebooks/00_data_pipeline.ipynb`, `figures/example_beats.png`, `figures/preprocessing.png`, `README.md`, `requirements.txt`, `run_all.sh`, `.gitignore`, `literature/README.md` |
| Bidirectional GRU (M3) | Ishaan Karmokar | `src/models/m3_bigru.py`, `results/m3_bigru.json`, `results/runs/m3_bigru_seed*.json`, `saved_models/m3_bigru_seed*.weights.h5`, `notebooks/03_m3_bigru.ipynb` |
| Evaluation module and Random Forest (M1) | Tanishq Sharma | `src/evaluate.py`, `src/models/m1_random_forest.py`, `results/m1_*.json`, `results/runs/m1_random_forest_seed*.json`, `notebooks/01_m1_random_forest.ipynb` |
| RMSprop training loop and Transformer encoder (M4) | Riddhi Puneyani | `src/models/deep_common.py`, `src/models/m4_transformer.py`, `results/m4_transformer.json`, `results/runs/m4_transformer_seed*.json`, `saved_models/m4_transformer_seed*.weights.h5`, `notebooks/04_m4_transformer.ipynb` |
| DE/PSO search and 1D CNN (M2) | Parth Upadhyay | `src/search.py`, `src/models/m2_cnn1d.py`, `results/m2_cnn1d_*.json`, `results/runs/m2_cnn1d_*_seed*.json`, `saved_models/m2_cnn1d_*_seed*.weights.h5`, `notebooks/02_m2_cnn1d.ipynb` |
| Comparison, figures and bootstrap CIs | Tanishq Sharma | `src/compare.py`, `src/bootstrap_ci.py`, `results/comparison.md`, `results/bootstrap_ci.json`, `results/predictions/`, `notebooks/05_comparison.ipynb`, remaining `figures/*.png` |

## From the final phase onwards

Members should commit their own changes from their own GitHub accounts, so
that the history shows individual work directly:

1. Ishaan: repository Settings → Collaborators → invite each member.
2. Each member, once:

```bash
git clone https://github.com/ishaankarmokar/ecg-arrhythmia.git
git config --global user.name "Your Name"
git config --global user.email "the-email-on-your-github-account"
```

3. Then for every change to files you own: `git pull`, edit, `git add <your files>`,
   `git commit -m "..."`, `git push`.
