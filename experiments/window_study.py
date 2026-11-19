"""Compare replacement windows, selecting solely on 2015-2024 scores.

The subsequent 2025-October 2026 results have already been examined. They are
reported descriptively and never used by the selection function. This script
does not train or replace submission artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stage2 import select_development_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/stage2_window_study')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == (ROOT/'results').resolve():
        parser.error('The study requires a separate output directory')
    out.mkdir(parents=True, exist_ok=True)
    protected = ['results/adaptation_summary.csv', 'results/retraining_log.csv',
                 'results/drift_signals.csv', 'results/stage2_period_metrics.csv',
                 'results/final_strategy.json', 'forecast/model.py', 'forecast/train.py',
                 'forecast/artifacts/models.joblib', 'deliverables/submission.zip',
                 'deliverables/experiment_code.zip']
    before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    (out/'original_hashes.json').write_text(json.dumps(before, indent=2))
    tables = []
    baseline_checks = {}
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}

    # Baseline first: reject a behavior-changing refactor before testing new windows.
    for window in (1095, 365, 730):
        candidate = out/f'window_{window}d'
        command = [sys.executable, str(ROOT/'experiments/stage2.py'),
                   '--data', str(args.data.resolve()), '--window-days', str(window),
                   '--output-dir', str(candidate)]
        print(f'Running {window}-day window; log: {out/f"window_{window}d.log"}', flush=True)
        with (out/f'window_{window}d.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if window == 1095:
            for name in ['adaptation_summary.csv', 'retraining_log.csv',
                         'drift_signals.csv', 'stage2_period_metrics.csv']:
                original = pd.read_csv(ROOT/'results'/name)
                fresh = pd.read_csv(candidate/name)
                pd.testing.assert_frame_equal(original, fresh, check_exact=False,
                                              check_dtype=False, rtol=1e-9, atol=1e-9)
                baseline_checks[name] = 'passed'
            print('Original three-year scores, alarms and replacement dates reproduced.', flush=True)
        summary = pd.read_csv(candidate/'adaptation_summary.csv')
        summary['window_days'] = window
        tables.append(summary)
        retrains = pd.read_csv(candidate/'retraining_log.csv',
                               parse_dates=['decision_day', 'effective_day', 'train_end'])
        assert (retrains.groupby(['task', 'strategy']).size() <= 6).all()
        assert (retrains.train_end <= retrains.decision_day).all()
        assert (retrains.effective_day == retrains.decision_day+pd.Timedelta(days=1)).all()
        assert (retrains.rows == window*48).all()

    all_scores = pd.concat(tables, ignore_index=True)
    all_scores.to_csv(out/'all_window_scores.csv', index=False)
    selected = select_development_candidates(all_scores)
    selection = {'selection_period': '2015-2024 only', 'selected': selected,
                 'candidate_window_days': [365, 730, 1095],
                 'hyperparameters': 'unchanged, chosen on 2014',
                 'subsequent_period': '2025-October 2026; already examined, descriptive check only',
                 'submission_models_changed': False}
    (out/'selected_windows.json').write_text(json.dumps(selection, indent=2))

    original_policy = json.loads((ROOT/'results/final_strategy.json').read_text())['selected']
    comparison = []
    for task in ('demand', 'price'):
        metric = 'MAPE' if task == 'demand' else 'BalAcc'
        winner = selected[task]
        for split in ('development', 'holdout', 'all'):
            old = all_scores[(all_scores.task == task) & (all_scores.split == split) &
                             (all_scores.window_days == 1095) &
                             (all_scores.strategy == original_policy[task]['strategy'])].iloc[0]
            new = all_scores[(all_scores.task == task) & (all_scores.split == split) &
                             (all_scores.window_days == winner['window_days']) &
                             (all_scores.strategy == winner['strategy'])].iloc[0]
            improvement = old[metric]-new[metric] if task == 'demand' else new[metric]-old[metric]
            comparison.append({'task': task, 'split': split, 'metric': metric,
                               'original_strategy': old.strategy, 'original_score': old[metric],
                               'selected_strategy': new.strategy, 'selected_window_days': winner['window_days'],
                               'selected_score': new[metric], 'improvement_percentage_points': improvement,
                               'selected_retrains_to_end': int(new.retrains_to_end)})
    pd.DataFrame(comparison).to_csv(out/'selected_comparison.csv', index=False)

    after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    assert before == after, 'The study changed protected baseline/submission files'
    evidence = {'baseline_regression_checks': baseline_checks,
                'all_candidate_chronology_budgets_and_window_lengths': 'passed',
                'protected_original_files_unchanged': True,
                'selection_uses_development_only': True}
    (out/'verification.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(selection, indent=2), flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
