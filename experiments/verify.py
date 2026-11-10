"""Focused checks of chronology, bootstrap, online detectors, and feature causality."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forecast.model import features
from experiments.baselines import prepare
from experiments.stage2 import Detector,confidence_intervals,SEED
from harness import dayahead as h


def main():
    checks={}
    logs=pd.read_csv(ROOT/'results/retraining_log.csv',parse_dates=['decision_day','effective_day','train_end'])
    assert (logs.groupby(['task','strategy']).size()<=6).all()
    assert (logs.train_end<=logs.decision_day).all()
    assert (logs.effective_day==logs.decision_day+pd.Timedelta(days=1)).all()
    checks['budget_and_retraining_chronology']='passed'
    values=np.arange(48*20,dtype=float)/10
    rng=np.random.default_rng(SEED)
    blocks=values.reshape(-1,48)
    canonical=[np.concatenate([blocks[j] for j in rng.integers(0,20,20)]).mean() for _ in range(1000)]
    fast=confidence_intervals(values)
    np.testing.assert_allclose([fast['day_low'],fast['day_high']],np.quantile(canonical,[.025,.975]))
    checks['bootstrap_matches_appendix_algorithm']='passed'
    dates=pd.date_range('2015-01-01',periods=100)
    values=np.r_[np.zeros(30),np.ones(70)*5]
    def run(v):
        detector=Detector(1.)
        return [detector.observe(day,value) for day,value in zip(dates,v)]
    a=run(values)
    b=run(np.r_[values[:60],np.ones(40)*999])
    np.testing.assert_equal(a[:60],b[:60])
    assert any(alarm for _,alarm in a)
    checks['detector_future_invariance_and_signal']='passed'
    df,_,_=prepare()
    for day in ('2014-12-31','2016-02-29','2026-10-31'):
        history,weather=h.make_inputs(df,day)
        real=df.loc[(h.market_day(df.index)>=pd.Timestamp(day)-pd.Timedelta(days=14)) & (h.market_day(df.index)<=day)]
        expected=features(real).iloc[-48:]
        serving=features(pd.concat([history.iloc[-14*48:],weather.reindex(columns=history.columns)])).iloc[-48:]
        pd.testing.assert_frame_equal(expected,serving)
    checks['causal_features_match_inference_on_boundaries']='passed'
    # Every frozen Stage 1 family must have monthly intervals, not just winners.
    metrics=pd.read_csv(ROOT/'results/stage2_period_metrics.csv')
    for family in ('ridge','boost_demand','logistic','boost_price'):
        sub=metrics[(metrics.model==family)&(metrics.frequency=='M')&(metrics.strategy=='frozen')]
        assert len(sub)==142
        assert sub[['iid_low','iid_high','day_low','day_high']].notna().all().all()
    checks['all_frozen_models_have_142_months_and_both_intervals']='passed'
    final=json.loads((ROOT/'results/final_strategy.json').read_text())
    assert final['selection_period']=='2015-2024 only'
    checks['explicit_holdout_policy']='passed'
    (ROOT/'results/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()

