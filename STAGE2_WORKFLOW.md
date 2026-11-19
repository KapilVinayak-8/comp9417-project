# Stage 2 workflow and window comparison

`experiments/stage2.py` runs the historical deployment experiment. It does not update the final submitted models. Its default remains the original 1,095-day replacement window.

The daily loop is in `simulate_policy` and follows this order:

1. Deploy a replacement scheduled in advance, trained through yesterday only.
2. Build today's causal inputs and predict all 48 intervals.
3. Observe today's outcomes and score predictions at the end of the day.
4. Update input/error monitoring using observations now available.
5. If the relevant detector fires and budget remains, train through today and deploy tomorrow.

The initial 2010–2014 fitted model is unchanged for every candidate. Only the data window used on later replacement occasions varies. Each task/strategy has its own six-replacement budget across the entire 2015–October 2026 simulation. A new year does not reset that budget.

## Reproduce the original experiment

Use Python 3.11 with the course's pinned dependencies. Supply the released CSV path if it is not in `data/`.

```sh
python experiments/stage2.py --data path/to/nem_weather_halfhourly_2010_2026.csv
```

This intentionally writes the original top-level `results/` files, as before. To replay without replacing saved evidence:

```sh
python experiments/stage2.py --data path/to/nem_weather_halfhourly_2010_2026.csv --output-dir results/baseline_replay
```

## Compare 365-, 730- and 1,095-day replacement windows

```sh
python experiments/window_study.py --data path/to/nem_weather_halfhourly_2010_2026.csv
```

Outputs are under `results/stage2_window_study/`:

- `window_1095d/`, `window_365d/`, `window_730d/`: each complete, separate simulation.
- `all_window_scores.csv`: candidate scores, labelled by window, task, policy and period.
- `selected_windows.json`: choices made solely from 2015–2024 development scores.
- `selected_comparison.csv`: original versus selected scores on development, subsequent period and full history.
- `verification.json`: original three-year replay, chronology, budget and protected-file checks.

The three-year replay must reproduce original scores, alarms and replacement dates before the runner proceeds to shorter windows. Nondefault windows cannot write to the authoritative top-level `results/` directory.

The complete chronology is simulated for each candidate so model age and remaining budget carry into 2025. Only development scores enter candidate selection. The later scores have already been examined in this project; they are a descriptive subsequent check, not a newly untouched test. Window labels in years mean fixed multiples of 365 days, not calendar-exact years.

## Final models and report consistency

Candidate results do not automatically change `forecast/train.py`, `forecast/artifacts/`, the report or either submission ZIP. A better historical policy/window is evidence for a separate decision about the final refit; it is not proof that live-week performance improves. Promoting a window requires updating final training, rebuilding/rechecking the exact ZIP, and aligning the report with the resulting experiment evidence.

The experiment packager includes the window-study source, its CSV/JSON/log evidence, and the behavioral tests. It does not regenerate the report.

```sh
python -m unittest discover -s tests -v
python experiments/verify.py
```
