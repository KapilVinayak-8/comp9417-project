"""Numerical support for the group-written live-week forecast section.
No claim of guaranteed coverage: historical ranges are scenario estimates.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.baselines import prepare
from experiments.stage1 import make_model
from forecast.model import features
from harness import dayahead as h


def main():
    df,truth,_=prepare()
    x=features(df)
    days=h.market_day(df.index)
    valid=x.notna().all(axis=1).to_numpy()
    config=json.loads((ROOT/'results/final_strategy.json').read_text())
    seasonal=[]
    # Same calendar week and same 37-day lead from last training day to first
    # forecast day, with parameters frozen for every seven-day evaluation.
    for year in range(2020,2026):
        cutoff=pd.Timestamp(f'{year}-11-01')
        fitmask=valid & (days<cutoff)&(days>=cutoff-pd.Timedelta(days=1095))
        week=(days>=f'{year}-12-07')&(days<f'{year}-12-14')
        record={'year':year,'training_end':f'{year}-10-31','forecast_start':f'{year}-12-07'}
        for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
            selected=config['selected'][task]
            model=make_model(selected['family'],selected['setting'])
            model.fit(x.loc[fitmask],truth.loc[fitmask,col])
            p=model.predict(x.loc[week])
            y=truth.loc[week,col]
            record.update({'MAPE':h.mape(y,p),'MAE':h.mae(y,p)} if task=='demand'
                          else {'BalAcc':h.balanced_accuracy(y,p),'Acc':h.accuracy(y,p)})
        seasonal.append(record)
        print('Seasonal frozen-week check:',record,flush=True)
    seasonal=pd.DataFrame(seasonal)
    seasonal.to_csv(ROOT/'results/liveweek_seasonal_backtests.csv',index=False)
    # Additional spread estimate: every full seven-day holdout window without a
    # model replacement inside it. Overlap means these are not independent trials.
    windows=[]
    predictions={}
    for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
        strategy=config['selected'][task]['strategy']
        predictions[task]=pd.read_parquet(ROOT/f'results/{task}_{strategy}_predictions.parquet')[col]
    retrains=pd.read_csv(ROOT/'results/retraining_log.csv')
    changes=[]
    for task in ('demand','price'):
        strategy=config['selected'][task]['strategy']
        changes.extend(pd.to_datetime(retrains.loc[(retrains.task==task)&(retrains.strategy==strategy),'effective_day']).tolist())
    holdout=days[(days>='2025-01-01')][::48]
    for start in holdout[:-6]:
        end=start+pd.Timedelta(days=7)
        if any(start<d<end for d in changes): continue
        keep=(days>=start)&(days<end)
        record={'start':str(start.date())}
        for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
            y=truth.loc[keep,col]
            p=predictions[task].loc[y.index]
            record.update({'MAPE':h.mape(y,p)} if task=='demand' else {'BalAcc':h.balanced_accuracy(y,p)})
        windows.append(record)
    weeks=pd.DataFrame(windows)
    weeks.to_csv(ROOT/'results/holdout_weekly_scores.csv',index=False)
    forecast={}
    # Predeclared simple summary: recent three matching December weeks for point;
    # envelope of seasonal scores and central 95% holdout-week spread for range.
    for metric in ('MAPE','BalAcc'):
        low,high=weeks[metric].quantile([.025,.975])
        forecast[metric]={'point':float(seasonal.tail(3)[metric].mean()),
                          'empirical_scenario_interval':[float(min(low,seasonal[metric].min())),float(max(high,seasonal[metric].max()))]}
    result={'live_week':'2026-12-07 to 2026-12-13','metrics_percent':forecast,
            'point_basis':'mean of same-week frozen backtests in 2023, 2024 and 2025',
            'range_basis':'envelope of six same-week backtests and 2.5/97.5 percentiles of holdout rolling-seven-day scores',
            'interpretation':'empirical uncertainty/scenario ranges, not a formal 95% confidence or guaranteed prediction interval',
            'limitations':['only six season-matched annual weeks; only three recent weeks for point',
                           'overlapping holdout windows are dependent and not identically distributed',
                           'weather supplied as perfect reanalysis; real forecast errors not represented',
                           'unseen market changes, extreme heat, holidays and price spikes can invalidate ranges',
                           '2020-2024 seasonal checks reuse development history; 2025 is the season-matched holdout',
                           'December 7 has a different weekday across years; previous-day price threshold can change class mix']}
    (ROOT/'results/liveweek_forecast_support.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
