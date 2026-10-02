# Commit plan: each member commits their own work

The rubric checks individual contribution through GitHub commit history. Each
member commits **only the files they own**, from **their own GitHub account**.
Nobody commits on someone else's behalf.

Commit in the order below, because later files import earlier ones.

## 0. One-time setup (Ishaan)

1. Create an empty GitHub repository `ishaankarmokar/ecg-arrhythmia` (no README, no .gitignore).
2. Repository → Settings → Collaborators → invite Parth, Riddhi and Tanishq by their GitHub usernames.
3. Share the project folder (zip of `ecg-arrhythmia/`) with the team.

## How each teammate runs their step (Parth, Riddhi, Tanishq)

On **your own laptop**, logged in to **your own** GitHub account:

```bash
git clone https://github.com/ishaankarmokar/ecg-arrhythmia.git
cd ecg-arrhythmia
# copy YOUR files from the shared zip into this folder, keeping the same paths,
# then run the git add / commit / push commands of your step below
```

Steps must be done in order (1 → 5); run `git pull` before each one.

## 1. Ishaan Karmokar: shared protocol, data pipeline and BiGRU (two commits)

Run inside this project folder on Ishaan's laptop (the repository is already
initialised here). Both of Ishaan's commits are made first, in one sitting.

```bash
cd ecg-arrhythmia
git remote add origin https://github.com/ishaankarmokar/ecg-arrhythmia.git
git add .gitignore requirements.txt README.md run_all.sh COMMIT_PLAN.md literature/README.md \
        src/config.py src/download_data.py src/build_dataset.py \
        notebooks/00_data_pipeline.ipynb figures/example_beats.png figures/preprocessing.png
git commit -m "Add shared protocol, PhysioNet download and preprocessing pipeline"
git add src/models/m3_bigru.py results/m3_bigru.json results/runs/m3_bigru_seed*.json \
        saved_models/m3_bigru_seed*.weights.h5 notebooks/03_m3_bigru.ipynb
git commit -m "Add bidirectional GRU model"
git branch -M main && git push -u origin main
```

After this, Ishaan should do any later work in a fresh `git clone`, because
this folder still holds untracked copies of the teammates' files.

## 2. Tanishq Sharma: evaluation module and Random Forest

```bash
git pull
git add src/evaluate.py src/models/m1_random_forest.py \
        results/m1_random_forest.json results/m1_feature_importance.json \
        results/runs/m1_random_forest_seed*.json notebooks/01_m1_random_forest.ipynb
git commit -m "Add common evaluation module and Random Forest baseline"
git push
```

## 3. Riddhi Puneyani: shared deep-model trainer and Transformer

```bash
git pull
git add src/models/deep_common.py src/models/m4_transformer.py \
        results/m4_transformer.json results/runs/m4_transformer_seed*.json \
        saved_models/m4_transformer_seed*.weights.h5 notebooks/04_m4_transformer.ipynb
git commit -m "Add RMSprop training loop and Transformer encoder model"
git push
```

## 4. Parth Upadhyay: DE/PSO search and 1D CNN

```bash
git pull
git add src/search.py src/models/m2_cnn1d.py \
        results/m2_cnn1d_*.json results/runs/m2_cnn1d_*_seed*.json \
        saved_models/m2_cnn1d_*_seed*.weights.h5 notebooks/02_m2_cnn1d.ipynb
git commit -m "Add DE/PSO hyperparameter search and 1D CNN model"
git push
```

## 5. Tanishq Sharma: comparison and confidence intervals

```bash
git pull
git add src/compare.py src/bootstrap_ci.py results/comparison.md results/bootstrap_ci.json \
        results/predictions notebooks/05_comparison.ipynb \
        figures/per_class_f1.png figures/confusion_matrices.png \
        figures/cnn_search.png figures/learning_curves.png
git commit -m "Add model comparison, figures and patient-level bootstrap CIs"
git push
```

Check with `git status` after step 5: nothing should be left untracked, apart
from the ignored data, logs and PDFs.

Before your first commit, set your own identity once on your own machine:

```bash
git config --global user.name "Your Name"
git config --global user.email "the-email-on-your-github-account"
```
