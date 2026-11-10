# How to run and inspect this project

Open D:\comp9417_project_kit in VS Code. Use the project environment, not another Python.

## One command

From PowerShell in this folder:

    .\.venv\Scripts\python.exe run_all.py

This rebuilds the experiments, plots, final models and ZIP files, and runs the original checker on the ZIP. It does not download anything. Generated results are replaced. Deliverables go in deliverables/ by default. The supplied CSV is required under data/.

## Individual steps

1. experiments/baselines.py - data audit and both authoritative baselines.
2. experiments/stage1.py - two families per task; choose settings on 2014; freeze models fitted through 2014. Its selected artifacts are separate from final artifacts.
3. experiments/stage2.py - fixed PCA/k-means, online input/error detection, four strategies, monthly/yearly scores and confidence intervals. Strategies and detector settings are fixed in source before simulation.
4. experiments/feature_analysis.py - 2014 feature ablations, class weighting comparison, descriptive statistics. Does not revise selection.
5. experiments/make_plots.py - figures from saved results.
6. experiments/diagnostics.py - error dependence, alarm/degradation comparisons and background event dates. Never used for training or feature engineering.
7. experiments/liveweek_analysis.py - season-matched frozen-week backtests and empirical uncertainty support.
8. experiments/verify.py - targeted correctness checks.
9. forecast/train.py data/nem_weather_halfhourly_2010_2026.csv - rebuild final models only.
10. experiments/check_reproducibility.py - rebuild final models again and compare exact model-file hashes.
11. experiments/check_inference.py - schema/timing check on 14 days.
12. experiments/build_packages.py - package source/results and final models separately.
13. check_submission.py deliverables/submission.zip - authoritative local checker on the ZIP.

Run a step using `.\.venv\Scripts\python.exe` followed by its path. Step 9 also needs the CSV argument shown above. Always check that a command ends without an error.

## What to inspect

- forecast/model.py: shared causal features and Forecaster.predict.
- forecast/train.py: short, independent final-model rebuild script.
- results/stage1_validation.csv: 2014 model-selection scores.
- results/stage2_period_metrics.csv: yearly/monthly baselines, every frozen family and adapted models, with both confidence intervals for monthly MAPE/accuracy.
- results/adaptation_summary.csv: development, holdout and full simulation scores, plus retrain counts.
- results/retraining_log.csv: exact training cutoffs, decision days and first deployment days.
- results/detector_config_*.json and drift_signals.csv: references, thresholds and alarms.
- results/cluster_*.csv and cluster_config.json: normalization, PCA, number-of-clusters evidence and day-type interpretation data.
- results/alarm_degradation_comparison.csv: before/after changes, with direct bootstrap intervals for differences. These are descriptive, not causal proof.
- results/error_dependence.csv and ci_width_summary.csv: evidence for discussing uncertainty assumptions.
- results/liveweek_forecast_support.json: estimated live-week MAPE/balanced accuracy and empirical scenario ranges. These ranges have no guaranteed coverage.
- results/plots/: figures in PNG and vector PDF formats; PDF plots are not included in either submission ZIP.
- results/verification.json, reproducibility.json, inference_checks.json and submission_check.log: verification evidence.

## Fixed simple design

Ridge and logistic regression provide linear references; LightGBM provides the tree family. No neural network or extra data is needed. Features use completed historical market days, target-day weather and permitted NSW holiday/calendar information. Same-half-hour market lags are 1, 2 and 7 days.

Day-type analysis divides each daily profile by its daily mean (shape, not level), fits PCA retaining 95% variance and chooses k=2..6 by training-only silhouette. PCA and clusters are fitted once on 2010-2013 and never updated. Future day types are NOT prediction features.

Input drift monitors a 28-day mean of absolute seasonal z-scores of five observed daily inputs. Error drift monitors a 28-day mean of seasonally centered daily loss. References/thresholds use only 2010-2014. A signal needs seven confirmations and a 540-day cooldown. Signals observed after day D may replace a model only for D+1.

Each task compares frozen, scheduled 3-year-window retraining, input-triggered retraining and error-triggered retraining. The budget is at most six replacements per task/strategy. Fixed scheduled dates are January 1 of 2016, 2018, 2020, 2022, 2024 and 2026. The window is 1,095 days, not calendar-exact three years. Family/settings are never retuned after 2014.

Policy selection uses 2015-2024 only. 2025-October 2026 is reported as a subsequent historical check; policies are not revised from those scores. Both final models use the released final 1,095 days. Live-week parameters are frozen; history is used only for features. Final training is a Stage 3 refit, not a seventh historical retrain.

For confidence intervals, independent MAPE uses the normal mean approximation; independent accuracy uses the binomial normal approximation (Hoeffding bounds are also saved). Bootstrap intervals resample 1,000 whole days. Neither method removes dependence between days. Monthly intervals target expected performance under conditions similar to that month, not uncertainty in its already-observed score, and overlapping intervals do not test a difference.

## Submission and remaining human work

The full specification allows AI help for code but requires the report to be written by the group (page 10, 'Use of generative AI'). No report has been generated here. Use REQUIREMENTS_CHECKLIST.md to plan your own report and refer to actual saved results. Understand and review all code before submission.

Submit a separate report PDF, experiment_code.zip and submission.zip on Moodle, plus a group pseudonym. Do not put the report inside either ZIP. Check Moodle announcements yourself for any updated instructions. Deadline in the supplied specification: 22 November 2026, 6pm; peer review: 23 November 2026, 6pm. Confirm these on Moodle.

The final ZIP is checked locally on Windows. The Linux Docker marking image and its resource enforcement have not been tested here. Checker scores on released data are in-sample, not a forecast of live-week performance.
