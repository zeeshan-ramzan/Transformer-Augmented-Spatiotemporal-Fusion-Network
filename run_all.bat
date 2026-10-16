@echo off
REM Runs every algorithm and experiment of the paper in order.
REM Usage: run_all.bat [data.csv] [results_dir]
set DATA=%1
if "%DATA%"=="" set DATA=data\nthri_tea_yield.csv
set OUT=%2
if "%OUT%"=="" set OUT=results
set A=--data %DATA% --out-dir %OUT%
python dataset_statistics.py %A% || exit /b 1
python correlation_analysis.py %A% || exit /b 1
python relieff_feature_selection.py %A% || exit /b 1
for %%m in (mlr mlr_interactions stepwise_mlr decision_tree svr_linear svr_quadratic svr_cubic svr_gaussian gpr_rational_quadratic gpr_squared_exponential gpr_matern52 gpr_exponential random_forest gradient_boosting xgboost_model nas_dnn) do (
  python %%m.py %A% || exit /b 1
)
python tastfn.py %A% || exit /b 1
python ablation_study.py %A% || exit /b 1
python hyperparameter_sensitivity.py %A% || exit /b 1
python ndvi_source_comparison.py %A% || exit /b 1
python permutation_importance.py %A% || exit /b 1
python seasonal_analysis.py --pred %OUT%\tastfn\oof_predictions.csv --out %OUT%\seasonal_analysis.csv || exit /b 1
python complexity_benchmark.py --results %OUT% || exit /b 1
python make_figures.py --results %OUT% --out %OUT%\figures || exit /b 1
python summarize_results.py --results %OUT% || exit /b 1
