# Requirements checklist and evidence map

This is an implementation checklist, not report text. The group must write the report.

| Requirement | Code / saved evidence | Remaining report work |
|---|---|---|
| Only released CSV plus permitted calendar | forecast/model.py, train.py | Explain forecasting information and idealized weather |
| Correct market-day convention and no leakage | baselines.py, verify.py, verification.json | Explain cutoffs and midnight convention |
| Both demand/price baselines | baseline_metrics.csv; stage2_period_metrics.csv | Compare and explain why yesterday is competitive |
| At least two regression families | stage1_validation.csv; ridge and boost_demand frozen metrics | Justify families, explain errors and degradation |
| At least two classification families | logistic and boost_price validation/frozen metrics | Justify families, explain class imbalance |
| Three-stage chronological protocol | stage1.py, stage2.py, final_strategy.json, train.py | State dates; distinguish selection from holdout |
| Feature justification | feature_ablations.csv, feature_importance.csv | Explain lag/weather/cross-state effects; importance is not causation |
| PCA then clustering, 2010-2013 only | cluster_config.json, cluster_selection.csv, day_type_models.joblib | Justify normalization/k, interpret profiles and annual shares |
| Day types related to degradation | day_type_error_relation.csv; drift_error_correlations.csv; plots/day_types*.png | Discuss association, outliers and fixed-reference limitations |
| Monthly/yearly metrics, all frozen models and adapted models | stage2_period_metrics.csv | Include essential evidence in main report |
| Both 95% interval methods, >=1000 whole-day resamples | stage2.py confidence_intervals; ci_width_summary.csv; error_dependence.csv | Explain estimand, assumptions, width, between-day dependence and overlap limits |
| Online input detector and error detector | detector_config_*.json; drift_signals.csv; drift_alarm_dates.png | State statistic, reference, threshold, calibration, persistence and timing |
| Compare drift and error; limits of concept drift claims | alarm_degradation_comparison.csv; drift_error_correlations.csv | Explain why neither detector establishes a conditional relationship change |
| Plausible events with sources | market_event_comparisons.csv | Check cited sources; proximity is not causal evidence |
| >=3 adaptation strategies including triggered retraining | adaptation_summary.csv; retraining_log.csv | Compare gains, retrain budget and early budget exhaustion |
| <=6 retrains per task/model/strategy | verification.json; retraining_log.csv | Discuss zero-versus-six tradeoff and timing |
| Final strategy justified | final_strategy.json; adaptation_summary.csv | Explain development selection and holdout result |
| Predictions vs observations | plots/prediction_week_*.png | Add captions and discuss selected weeks |
| Written live-week MAPE and balanced-accuracy forecast with intervals | liveweek_forecast_support.json; seasonal_backtests and holdout_weekly_scores | Write your own prediction/reasoning; ranges are empirical, not guaranteed |
| Discussion/limitations/conclusion | results plus implementation notes | Entirely group-written |
| Report <=10 pages, 11pt, 1.5 spacing | Not generated | Introduction, Data Analysis, Methodology, Results, Live-Week Forecast, Discussion, Conclusion; references excluded from limit |
| Figure axes, units, captions and references in text | PNG plots have axes/units | Group adds captions and in-text references; essential findings stay in main pages |
| Algorithm/background references | Course brief, official library docs, event source URLs | Group verifies and cites sources actually used |
| Final Forecaster API, fixed live parameters | forecast/model.py; inference_checks.json; checker log | Review code and explain it |
| Rebuild final artifacts with fixed seeds | forecast/train.py; reproducibility.json | Rebuild from released CSV; no experiments required in train.py |
| 100MB artifacts, 60s load, 20s prediction | build_packages.py; submission_check.log | Local timing passed; Linux marking image still untested |
| Separate source ZIP, model ZIP, report PDF | experiment_code.zip and submission.zip | Group writes PDF and enters pseudonym on Moodle |
| Moodle updates, group details, peer review | User account / course site | Group must complete; not accessed here |

Before submission: read the specification and latest Moodle updates, review all generated results, write your own report, verify every statement against a result, run the checker on the exact ZIP to upload, and ensure all group members can explain the work.
