# Leaf Population Mass and WLACP for Deployed Gradient-Boosted Trees

This repository contains the code, data splits and analysis scripts for the paper

> M. Lee and D. Ryu, "Leaf Population Mass in Deployed Gradient-Boosted Trees: Error Ranking and Locally Adaptive Conformal Prediction Without Retraining" (manuscript under review).

It implements the two methods of the paper and every experiment, table and figure reported in it.

- **Leaf population mass (LPM).** A reliability score for a fitted gradient-boosted decision tree (GBDT) ensemble. For a query, LPM averages over the trees the fraction of training points that share the query's leaf. It needs no refitting and no labels, and it can be computed from the node counts that scikit-learn, XGBoost and LightGBM store in their model files.
- **Weighted leaf agreement conformal prediction (WLACP).** A localized split conformal procedure. It weights every calibration point by its per-tree normalized leaf agreement with the query and adjusts the level as in localized conformal prediction (Guan, *Biometrika*, 2023), which guarantees finite-sample marginal coverage.

## Installation

```bash
pip install -r requirements.txt
```

The 19 regression tasks with numerical features come from the tabular benchmark of Grinsztajn et al. (NeurIPS 2022). They are downloaded on first use from the Hugging Face dataset `inria-soda/tabular-benchmark` (folder `reg_num`) and cached in `data/`.

## Quick start

A small run on two datasets with reduced split sizes takes a few minutes. It checks that the installation works.

```bash
QUICK=1 N_JOBS=4 bash run_all.sh       # results go to results_quick/
python -m tests.test_core              # unit checks of the core identities
```

The unit checks verify four things:

- LPM equals the mean leaf agreement with the training points.
- Without row subsampling, the node counts stored in the model match the counts obtained by passing the training set through it.
- WLACP with a constant kernel reduces to split conformal prediction.
- Mondrian conformal prediction and CV+ return intervals of the expected shape.

## Reproducing the paper

```bash
# 1. All experiments on the nine core datasets, five random splits each
N_JOBS=16 bash run_all.sh

# 2. Error ranking, intervals and the cross-library study on all 19 tasks
for e in e01_ranking e02_conformal e03_extended; do
  python -m experiments.$e --datasets all --n_jobs 16 --out results_all
done
python -m analysis.a01_tables --results results_all
python -m analysis.a02_stats  --results results_all

# 3. Numbers, figures and tables of the paper
python -m analysis.a03_key_numbers   --results results --results_all results_all
python -m analysis.a04_paper_numbers --results results --results_all results_all
python -m analysis.f01_figures       --results results --results_all results_all --out figures
python -m analysis.a05_latex_tables  --results results --results_all results_all --out tables
```

Step 1 writes the summary tables of the core datasets to `results/tables/`, and step 3 writes the remaining output:

- `results/tables/paper_numbers.json`: every number quoted in the text that is not already in a table.
- `figures/`: each figure as a 600-dpi PNG, with its caption in a `.txt` file of the same name.
- `tables/`: one LaTeX file per table.

Every number in the paper can be traced to the stored result records.

### Running single experiments

Each experiment script splits its work into one job per dataset and random split and accepts the same options.

```bash
python -m experiments.e02_conformal --datasets main --n_jobs 16 --out results
```

- `--datasets` is `main` (the nine core datasets), `extra` (the other ten tasks), `all` (all 19), or a comma-separated list of dataset names.
- `--seeds` sets the random splits. The default is `42,7,13,99,2024`, which is used throughout the paper.
- Each job writes `results/<experiment>/<dataset>_s<seed>.json`, and `.npz` per-point arrays where needed.
- Jobs whose output already exists are skipped, so an interrupted run resumes where it stopped. Use `--overwrite` to recompute.
- A failed job leaves its traceback in a `.err` file next to the expected output.
- Each job runs on one thread, so `N_JOBS` can be set to the number of physical cores.
- `results/env.json` records the software versions used.

## Experiments

| Script | Content | In the paper |
|---|---|---|
| `e01_ranking` | Error ranking of LPM and all risk scores. Reports the Spearman correlation, the prediction rejection ratio, rejection curves, MAE per LPM quartile, model accuracy, the active-feature fraction, LPM from stored node counts and tree-weighted variants. | Section VI-A and VI-B, Tables 3 and 4, Figs. 1 and 2 |
| `e02_conformal` | Interval methods that keep the deployed model: split conformal prediction, the weighted quantile, WLACP with four kernels and a kernel chosen on the validation split, Mondrian conformal prediction, LoBoost, localized conformal prediction with an RBF kernel, normalized conformal prediction and IBUG. Reports marginal, per-quartile, kNN-subgroup and worst-slab coverage, widths, the interval score and effective sample sizes. | Section VI-E, Tables 8 and 9, Fig. 6 |
| `e03_extended` | Methods that refit the model (CQR, CV+ and NGBoost with 100 trees or early stopping), and LPM under scikit-learn, XGBoost, LightGBM with leaf-wise growth and LightGBM limited to depth 4. | Sections VI-C and VI-E, Tables 1, 5 and 8, Fig. 3(c) |
| `e04_sensitivity` | LPM when one hyperparameter of the default model changes, when the number of LightGBM leaves grows, and for models tuned on the validation split. | Section VI-C, Fig. 3(a)(b) |
| `e05_shift` | Detection of synthetic extrapolation along four directions and over shift magnitudes from 0.5 to 300 standard deviations. Compares LPM, LPM with a range check and the ambient-space scores. | Section VI-D, Table 6, Figs. 4 and 5 |
| `e06_calib_size` | Coverage and width as the calibration set grows from 100 to 2000 points. | Section VI-F, Fig. 7 |
| `e07_noise` | Target-noise experiments on synthetic data and on Bike Sharing. | Section VI-G, Fig. 8 |
| `e08_ring` | Two-dimensional ring benchmark with a hollow interior. | Section VI-G, Fig. 9 |
| `e09_timing` | Inference time of the risk scores and of the interval methods. | Section VI-H, Table 10, Fig. 10 |
| `e10_coverage_check` | Monte Carlo check of the finite-sample marginal coverage of WLACP. | Section VI-E, Table 7, Fig. 6(c) |

Analysis scripts:

| Script | Content |
|---|---|
| `a01_tables` | Averages the records over random splits and datasets and writes CSV and Markdown tables. |
| `a02_stats` | Dataset-level statistics: Wilcoxon signed-rank tests with Holm adjustment, hierarchical bootstrap intervals, a linear mixed model, and per-dataset bootstrap intervals of the LPM correlation. |
| `a03_key_numbers` | A summary of the main results in Markdown. |
| `a04_paper_numbers` | The numbers quoted in the text, and the dataset-level comparison of WLACP with every interval method. |
| `f01_figures` | All figures of the paper. |
| `a05_latex_tables` | All result tables of the paper in LaTeX. |
| `a00_timing` | Run time of each experiment, read from the stored records. |

## Repository layout

```
core/          data loading and splits, models, leaf structure, risk scores,
               conformal procedures, metrics, shifts, job runner
experiments/   one script per experiment (e01 to e10)
analysis/      aggregation, statistics, figures and tables
tests/         unit checks
run_all.sh     runs every experiment and the aggregation for one dataset group
```

Key files:

- `core/leaves.py`: leaf counts, LPM and the leaf agreement kernels.
- `core/conformal.py`: WLACP (`wlacp_lcp`) and all baseline interval methods.
- `core/config.py`: the split sizes, model settings and hyperparameter grids of the paper.

## Protocol

- **Splits.** Each dataset is split at random into training (60%), validation (10%), calibration (15%) and test (15%) sets, capped at 6000, 1000, 1000 and 800 points. Experiments are repeated over five random splits.
- **Validation split.** It fixes everything that would otherwise depend on the calibration or test data: the LPM quartile thresholds, the hyperparameters of the baselines, the WLACP kernel choice, the early stopping of NGBoost, the tuned models, and the source points of the extrapolation sets.
- **Default model.** The scikit-learn gradient boosting regressor with 100 trees of depth 4, a minimum leaf size of 20, a learning rate of 0.05 and a row subsampling rate of 0.8.
- **Other libraries.** XGBoost uses the same depth and a minimum child weight of 20. LightGBM grows 16 leaves per tree.
- **Coverage level.** Intervals target 90% coverage.

## Names in the code and result files

| In the code and result files | In the paper |
|---|---|
| `LPDS` | LPM (leaf population mass) |
| `LPDS_stored` | LPM from the node counts stored in the model |
| `LPDS+Range` | LPM + range |
| `WLACP_LCP_la_norm_g1` | WLACP with the per-tree normalized kernel (the default) |
| `WLACP_LCP_la_g1`, `_la_g2`, `_la_g4` | WLACP with the raw leaf agreement kernel and its sharpened versions |
| `WLACP_LCP_auto` | WLACP with the kernel chosen on the validation split |
| `WLACP_submitted` | the weighted quantile |
| `WLACP_testmass_*` | the weighted quantile with the test-point mass |
| `Mondrian_LPDS4` | Mondrian conformal prediction on LPM quartiles |
| `NCP_LPDS`, `NCP_ErrModel`, `NCP_IBUG` | normalized conformal prediction with an LPM, error-model or IBUG scale |
| `LCP_RBF` | localized conformal prediction with an RBF kernel |
| `CVplus` | CV+ |
| `cov_q1` | coverage on the lowest LPM quartile, with thresholds fixed on the validation split |
| `cov_low_testq` | coverage on the 25% of test points with the lowest LPM |

## Compute

The full run on the nine core datasets takes about 7 CPU core-hours. Most of this time goes to the hyperparameter tuning of `e04_sensitivity`; all other experiments together take about 1.2 core-hours.

- With 16 parallel jobs, the full run finishes in about one hour. The longest single job is the tuning on Superconduct, at about 25 minutes.
- Running `e01` to `e03` on the ten additional tasks takes about the same time as on the core datasets.
- Fewer tuning trials shorten `e04_sensitivity` by about half, at the cost of a less thorough search. Pass `--trials_sklearn 5 --trials_xgb 15 --trials_lgbm 15` to do this.
- Run `e09_timing` alone on an otherwise idle machine, since it measures inference time.

## Implementation notes

- IBUG and LoBoost are implemented from the descriptions in their papers (Brophy and Lowd, NeurIPS 2022; Santos et al., 2026). They use the same leaf assignments as LPM.
- The ambient-space scores (Isolation Forest, Mahalanobis distance, kNN distance and LOF) use features standardized with the training split. The axis-aligned range check uses the original units.
- The generators of the ring benchmark and of the noise experiments are specified in the docstrings of `experiments/e08_ring.py` and `experiments/e07_noise.py`.

## Citation

If you use this code, please cite the paper above. A BibTeX entry will be added on publication.
