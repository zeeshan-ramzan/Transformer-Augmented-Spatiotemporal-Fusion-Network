#!/usr/bin/env bash
# Runs every algorithm and experiment of the paper in order.
# Usage: bash run_all.sh [data.csv] [results_dir]
set -e
DATA=${1:-data/nthri_tea_yield.csv}
OUT=${2:-results}
A="--data $DATA --out-dir $OUT"
python dataset_statistics.py $A
python correlation_analysis.py $A
python relieff_feature_selection.py $A
for m in mlr mlr_interactions stepwise_mlr decision_tree svr_linear svr_quadratic svr_cubic svr_gaussian \
         gpr_rational_quadratic gpr_squared_exponential gpr_matern52 gpr_exponential random_forest gradient_boosting xgboost_model nas_dnn; do
  python $m.py $A
done
python tastfn.py $A
python ablation_study.py $A
python hyperparameter_sensitivity.py $A
python ndvi_source_comparison.py $A
python permutation_importance.py $A
python seasonal_analysis.py --pred $OUT/tastfn/oof_predictions.csv --out $OUT/seasonal_analysis.csv
python complexity_benchmark.py --results $OUT
python make_figures.py --results $OUT --out $OUT/figures
python summarize_results.py --results $OUT
