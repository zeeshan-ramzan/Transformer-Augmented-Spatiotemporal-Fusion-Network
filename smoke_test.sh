#!/usr/bin/env bash
# Quick end-to-end check of every script on a small synthetic dataset (a few minutes on a CPU).
# The numbers it prints are meaningless; it only proves that everything runs.
set -e
python make_synthetic_dataset.py --out smoke/data.csv --n 400 --plots 4
D="--data smoke/data.csv --out-dir smoke/results --folds 3"
E="--epochs 3 --patience 2"
python dataset_statistics.py $D
python correlation_analysis.py $D
python relieff_feature_selection.py $D --wrapper mlr
for m in mlr mlr_interactions stepwise_mlr decision_tree svr_linear svr_quadratic svr_cubic svr_gaussian \
         random_forest gradient_boosting xgboost_model; do python $m.py $D; done
for m in gpr_rational_quadratic gpr_squared_exponential gpr_exponential; do python $m.py $D --restarts 0; done
python nas_dnn.py $D --search --evals 12
python tastfn.py $D $E
python ablation_study.py $D $E
python hyperparameter_sensitivity.py $D $E --only n_layers
python ndvi_source_comparison.py $D $E
python permutation_importance.py $D $E --repeats 2
python seasonal_analysis.py --pred smoke/results/tastfn/oof_predictions.csv --out smoke/results/seasonal_analysis.csv
python complexity_benchmark.py --results smoke/results
python make_figures.py --results smoke/results --out smoke/results/figures
python summarize_results.py --results smoke/results
echo "SMOKE TEST PASSED"
