# Stage 2 window comparison results

Completed 19 November 2026. This is an experiment/review note, not replacement report prose.

## Outcome

The shorter windows did not improve the development-period winners. Retain the original 1,095-day policies: scheduled retraining for demand and input-triggered retraining for price. Submitted models, final-training code, original result files and existing ZIPs were not changed.

Each row below shows the best policy for that window, chosen from frozen, scheduled, input-triggered and error-triggered alternatives using 2015–2024 scores only. Hyperparameters/features were held fixed.

| Window | Demand policy | Development MAPE (%) ↓ | Price policy | Development balanced accuracy (%) ↑ |
|---|---|---:|---|---:|
| 365 days | input_drift_1y | 3.4409 | periodic_1y | 74.8759 |
| 730 days | input_drift_2y | 3.2525 | input_drift_2y | 75.6165 |
| 1095 days | periodic_3y | 3.2282 | input_drift_3y | 75.7697 |

The selected three-year demand policy has subsequent-period (2025–October 2026) MAPE 4.4038%, and the selected three-year price policy has balanced accuracy 81.8601%. These are unchanged from the original analysis. This later period was already examined and is not presented as a newly untouched test.

## Readability and verification

- `simulate_policy` explicitly separates scheduled deployment, forecasting, observation, detector updates and next-day triggered deployment.
- Dataset and output locations are configurable; nondefault windows cannot overwrite authoritative baseline results.
- The complete three-year replay reproduced original summary/period scores, alarm records and retraining logs within numerical tolerance 1e-9.
- All windows respected six replacements per task/strategy, completed-day cutoffs and their exact training-window lengths.
- All six new behavioral tests and the six existing correctness checks passed. A separate read-only whole-change review found no actionable bugs.
- `STAGE2_WORKFLOW.md` gives the execution sequence, commands and interpretation boundaries.
- No final model or experiment ZIP has been rebuilt from these source changes; rebuild the experiment package when the group finalizes the code/report.

## Evidence

- `results/stage2_window_study/all_window_scores.csv` — every candidate, policy and period.
- `results/stage2_window_study/selected_windows.json` — development-only choices.
- `results/stage2_window_study/verification.json` — replay, budgets/chronology and artifact integrity checks.
- `tests/test_stage2.py` — behavioral guards; `experiments/verify.py` — existing correctness checks.
