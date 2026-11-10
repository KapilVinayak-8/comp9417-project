"""Post-hoc numerical diagnostics. Never used for training, thresholds or selection."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.baselines import prepare
from harness import dayahead as h


def main():
    df,truth,_=prepare()
    signals=pd.read_csv(ROOT/'results/drift_signals.csv',parse_dates=['day'])
    rows=[]
    correlations=[]
    rng=np.random.default_rng(9417)
    for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
        p=pd.read_parquet(ROOT/f'results/{task}_frozen_predictions.parquet')[col]
        y=truth.loc[p.index,col]
        error=100*np.abs(y-p)/y if task=='demand' else 100*(y!=p).astype(float)
        daily=error.groupby(h.market_day(p.index)).mean()
        for month,vals in error.groupby(h.market_day(p.index).to_period('M')):
            dayvals=vals.groupby(h.market_day(vals.index)).mean()
            correlations.append({'task':task,'month':str(month),'halfhour_lag1_correlation':vals.autocorr(1),
                                 'day_lag1_correlation':dayvals.autocorr(1)})
        for _,event in signals[(signals.task==task)&(signals.strategy=='frozen')].iterrows():
            day=event.day
            before=daily.loc[day-pd.Timedelta(days=28):day-pd.Timedelta(days=1)].to_numpy()
            after=daily.loc[day+pd.Timedelta(days=1):day+pd.Timedelta(days=28)].to_numpy()
            if len(before)!=28 or len(after)!=28: continue
            delta=after.mean()-before.mean()
            boot=after[rng.integers(0,28,(1000,28))].mean(axis=1)-before[rng.integers(0,28,(1000,28))].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append({'task':task,'detector':event.detector,'signal_day':str(day.date()),
                         'mean_loss_before':before.mean(),'mean_loss_after':after.mean(),
                         'loss_change':delta,'day_bootstrap_difference_low':lo,'day_bootstrap_difference_high':hi})
    pd.DataFrame(rows).to_csv(ROOT/'results/alarm_degradation_comparison.csv',index=False)
    pd.DataFrame(correlations).to_csv(ROOT/'results/error_dependence.csv',index=False)
    # Background event dates are used ONLY for offline comparisons, never as features.
    events=[{'date':'2021-10-01','event':'Five-minute settlement commencement',
             'source':'https://www.aemo.com.au/initiatives/major-programs/past-major-programs/five-minute-settlement'},
            {'date':'2022-06-15','event':'NEM spot market suspension',
             'source':'https://www.aemo.com.au/newsroom/media-release/aemo-suspends-nem-wholesale-market'}]
    evidence=[]
    for event in events:
        for task in ('demand','price'):
            for detector in ('input','error'):
                subset=signals[(signals.task==task)&(signals.strategy=='frozen')&(signals.detector==detector)]
                offset=(subset.day-pd.Timestamp(event['date'])).dt.days
                j=offset.abs().idxmin()
                evidence.append({**event,'task':task,'detector':detector,
                                 'nearest_signal':str(subset.loc[j,'day'].date()),'signal_minus_event_days':int(offset.loc[j]),
                                 'interpretation_limit':'temporal proximity only; does not establish cause or conditional concept drift'})
    pd.DataFrame(evidence).to_csv(ROOT/'results/market_event_comparisons.csv',index=False)
    print('Saved dependence, alarm/degradation and event-date diagnostics.')

if __name__=='__main__': main()
