"""Test retraining timing, budget limits and window selection."""
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import stage2


class Stage2Tests(unittest.TestCase):
    def simulation(self, policy='input_drift', window_days=2, budget=1, end='2015-02-01'):
        runner = getattr(stage2, 'simulate_policy', None)
        self.assertIsNotNone(runner, 'Stage 2 needs a reusable chronological policy simulation')
        dates = pd.date_range('2014-12-01', end, freq='D')
        index = pd.date_range('2014-12-01 00:30', periods=len(dates)*48, freq='30min')
        days = (index-pd.Timedelta('30min')).normalize()
        x = pd.DataFrame({'feature': np.zeros(len(index))}, index=index)
        target = np.ones(len(index))
        target[days == '2015-01-07'] = 100
        target[days >= '2015-01-08'] = 1000
        target[days == '2016-01-01'] = 9999
        target[days == '2014-12-30'] = 3
        target[days == '2014-12-31'] = 7
        initial = DummyRegressor().fit(x.iloc[:48], np.ones(48))
        points = pd.Series((dates >= '2015-01-01').astype(float), index=dates)
        result = runner(x=x, target=target, days=days, valid=np.ones(len(index), bool),
                        initial_model=initial, model_factory=DummyRegressor,
                        input_points=points, input_threshold=0.,
                        error_reference=(pd.Series(0., index=range(1, 13)), 1., 99999.),
                        task='demand', policy=policy, window_days=window_days, budget=budget)
        return result

    def test_triggered_replacement_uses_completed_days_and_first_predicts_tomorrow(self):
        result = self.simulation()
        pred = pd.Series(result['predictions'], index=result['index'])
        self.assertTrue((pred.loc['2015-01-07 00:30':'2015-01-08 00:00'] == 1).all())
        self.assertTrue((pred.loc['2015-01-08 00:30':'2015-01-09 00:00'] == 50.5).all())
        record = result['retrains'][0]
        self.assertEqual(record['decision_day'], '2015-01-07')
        self.assertEqual(record['effective_day'], '2015-01-08')
        self.assertEqual(record['train_end'], '2015-01-07')
        self.assertEqual(record['rows'], 96)

    def test_scheduled_replacement_cannot_train_on_its_forecast_day(self):
        result = self.simulation(policy='periodic', end='2016-01-02')
        pred = pd.Series(result['predictions'], index=result['index'])
        # Jan 1 2016 replacement uses only Dec 30/31 2015, both set to 1000.
        self.assertTrue((pred.loc['2016-01-01 00:30':'2016-01-02 00:00'] == 1000).all())
        self.assertEqual(result['retrains'][0]['train_end'], '2015-12-31')

    def test_budget_exhaustion_stops_replacements_even_when_alarms_continue(self):
        result = self.simulation(end='2018-01-01')
        input_alarms = [r for r in result['alarms'] if r['detector'] == 'input']
        self.assertGreaterEqual(len(input_alarms), 2)
        self.assertEqual(len(result['retrains']), 1)
        self.assertEqual(result['predictions'][-1], 50.5)

    def test_invalid_training_window_is_rejected(self):
        runner = getattr(stage2, 'simulate_policy', None)
        self.assertIsNotNone(runner, 'Stage 2 needs a reusable chronological policy simulation')
        with self.assertRaises(ValueError):
            self.simulation(window_days=0)

    def test_selection_ignores_better_subsequent_scores(self):
        select = getattr(stage2, 'select_development_candidates', None)
        self.assertIsNotNone(select, 'Window selection must enforce the development cutoff')
        rows = pd.DataFrame([
            {'task':'demand','window_days':730,'strategy':'periodic_2y','split':'development','MAPE':2.},
            {'task':'demand','window_days':1095,'strategy':'periodic_3y','split':'development','MAPE':3.},
            {'task':'demand','window_days':365,'strategy':'periodic_1y','split':'holdout','MAPE':0.01},
            {'task':'price','window_days':365,'strategy':'input_drift_1y','split':'development','BalAcc':80.},
            {'task':'price','window_days':1095,'strategy':'input_drift_3y','split':'development','BalAcc':80.},
        ])
        selected = select(rows)
        self.assertEqual(selected['demand']['window_days'], 730)
        self.assertEqual(selected['price']['window_days'], 1095)

    def test_nondefault_window_cannot_overwrite_baseline_results(self):
        parse = getattr(stage2, 'parse_arguments', None)
        self.assertIsNotNone(parse, 'Candidate CLI needs baseline-output protection')
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse(['--window-days', '365'])


if __name__ == '__main__':
    unittest.main()
