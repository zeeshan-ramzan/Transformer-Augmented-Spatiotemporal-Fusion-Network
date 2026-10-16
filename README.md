# TASTFN — code for all algorithms and experiments in the article

*TASTFN: a transformer-augmented spatiotemporal fusion network for farm-level tea yield estimation from Landsat-8 NDVI and agrometeorological data*

This package has Python code for the proposed TASTFN, the sixteen baseline regressors and every experiment in the manuscript. Each algorithm is in its own file. Shared code (data loading, stratified 10-fold cross-validation, standardisation within each fold, metrics, inference-speed timing) is in `common.py`.

## 1. Installation

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt     # for a CUDA GPU, install torch from pytorch.org first
```

You need Python 3.9 or newer. Everything runs on a CPU; a GPU only speeds up the TASTFN experiments.

## 2. Data

Put the NTHRI data in `data/nthri_tea_yield.csv`, with one row per monthly yield observation of one plot:

| column | meaning |
|---|---|
| `farm_id` | plot identifier |
| `date` | month of the observation (`YYYY-MM`) |
| `min_temp`, `max_temp`, `mean_temp` | air temperature (°C) |
| `humidity` | relative humidity (%) |
| `rainfall` | rainfall (mm) |
| `solar_radiation` | solar radiation (W m⁻²) |
| `ndvi_mean`, `ndvi_std`, `ndvi_min`, `ndvi_max` | Landsat-8 NDVI statistics of the plot |
| `yield` | fresh leaf yield (kg ha⁻¹) |
| `l7_ndvi_mean`, `l7_ndvi_std`, `l7_ndvi_min`, `l7_ndvi_max` | optional, Landsat-7 NDVI (only used by `ndvi_source_comparison.py`) |

The NTHRI data are not public. To test the code without them, run `python make_synthetic_dataset.py`, which writes a synthetic file with the same columns. Its results are meaningless and must not be reported.

## 3. Quick check (about 3 minutes on a CPU)

```bash
bash smoke_test.sh      # every script, small synthetic data, 3 folds, 3 epochs
```

## 4. Files

### Proposed method
| File | Algorithm |
|---|---|
| `tastfn.py` | **TASTFN**: dual-stream projection (6 weather + 4 NDVI features → d_model = 64), phenological positional encoding, 2-layer pre-norm multi-head self-attention encoder (8 heads, d_ff = 256), NDVI feed-forward branch, cross-modal attention gating (CMAG), residual decoder [128, 64, 32] with dropout 0.30. Training: AdamW (lr 1e-3, weight decay 1e-4), cosine annealing with warm restarts (T₀ = 50, η_min = 1e-5), MSE loss, batch 32, ≤ 500 epochs, early stopping (patience 30). Window: 6 consecutive months, zero-padded and masked. |

### Baselines (same 10 folds, features standardised inside each fold)
| File | Model |
|---|---|
| `mlr.py` | Multiple linear regression |
| `mlr_interactions.py` | MLR with all pairwise interactions |
| `stepwise_mlr.py` | Bidirectional stepwise MLR (F-test, p-enter 0.05, p-remove 0.10) |
| `decision_tree.py` | Regression tree |
| `svr_linear.py`, `svr_quadratic.py`, `svr_cubic.py`, `svr_gaussian.py` | Support vector regression with linear, quadratic, cubic and Gaussian kernels |
| `gpr_rational_quadratic.py`, `gpr_squared_exponential.py`, `gpr_matern52.py`, `gpr_exponential.py` | Gaussian process regression with rational quadratic, squared exponential, Matérn 5/2 and exponential kernels |
| `random_forest.py` | Random forest |
| `gradient_boosting.py` | Gradient boosting |
| `xgboost_model.py` | XGBoost |
| `nas_dnn.py` | NAS-DNN benchmark: published architecture (188-29-77, ReLU, L-BFGS), or `--search` for the Bayesian-optimisation neural architecture search (objective log(1 + CV MSE)) |

### Experiments
| File | Manuscript item |
|---|---|
| `dataset_statistics.py` | Table 2 (descriptive statistics, coefficient of variation) |
| `correlation_analysis.py` | Pearson correlation analysis |
| `relieff_feature_selection.py` | RReliefF ranking + sequential forward selection |
| `summarize_results.py` | Table 4 (all models) |
| `ablation_study.py` | Table 5 (cumulative ablation, 7 configurations) |
| `seasonal_analysis.py` | Table 9 (performance by season) |
| `ndvi_source_comparison.py` | Table 10 (NDVI source) |
| `hyperparameter_sensitivity.py` | Table 11 (T, heads, layers, d_model) |
| `permutation_importance.py` | Fig. 9 (permutation importance, 50 repeats) |
| `complexity_benchmark.py` | Table 8 (parameters, FLOPs, training time, OPS). Every model script records its own training time for one 10-fold run, so running all models on one machine gives directly comparable training times and OPS for Table 8. |
| `make_figures.py` | Figs. 1–10 |
| `run_all.sh` / `run_all.bat` | Full pipeline in the order of the manuscript |

## 5. Typical use

```bash
python tastfn.py --data data/nthri_tea_yield.csv                 # proposed model
python xgboost_model.py --data data/nthri_tea_yield.csv          # one baseline
bash run_all.sh data/nthri_tea_yield.csv results                 # everything (Windows: run_all.bat)
```

Each model writes `results/<model>/metrics.json` (mean and per-fold R², RMSE, MSE, MAE, MAPE, OPS, training time) and `oof_predictions.csv` (the out-of-fold prediction for every observation). TASTFN also writes `learning_curves.csv`, `attention_layer2.npy`, `gates_by_month.csv` and `complexity.json`.

Common options: `--folds`, `--seed`, `--out-dir`. TASTFN scripts also take `--epochs`, `--patience`, `--batch-size`, `--device` and `--early-stop-on-test`.

## 6. Notes

* **Cross-validation.** Folds are stratified on yield deciles (seed 42). Features and the target are standardised with the statistics of the training rows of each fold only.
* **Early stopping.** By default TASTFN keeps 10% of the training rows as an inner validation set for early stopping, so the held-out fold stays untouched. `--early-stop-on-test` uses the held-out fold instead, as described in the manuscript; this makes the held-out score slightly optimistic.
* **Model size.** With the hyperparameters in the paper, TASTFN has 167,105 trainable parameters and about 1.50 million FLOPs per observation (`python tastfn.py` prints both). The published NAS-DNN (10 inputs, 188-29-77) has 9,937 parameters and 19,284 FLOPs per observation.
* **NAS-DNN regularisation.** MATLAB's `Lambda` is converted to scikit-learn's `alpha` as `alpha = Lambda × n_train`.
* **Reproducibility.** Seeds are fixed, but GPU kernels are not fully deterministic, so results vary slightly from run to run.
