"""Data checks, authoritative baseline verification, and period summaries."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import dayahead as h


def prepare():
    df = h.load_dataset(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    assert not df.index.has_duplicates
    assert not df.isna().any().any()
    assert (df.index.to_series().diff().iloc[1:] == pd.Timedelta('30min')).all()
    assert len(h.complete_days(df))*48 == len(df)
    days = h.market_day(df.index)
    means = df.nsw_price.groupby(days).mean()
    threshold = pd.Series(days.map(means.shift(1)), index=df.index)
    truth = pd.DataFrame({'nsw_demand_mw':df.nsw_demand,
                          'nsw_price_up':(df.nsw_price > threshold).astype(int)})
    predictions = {}
    for lag in (1,7):
        predictions[f'baseline_{lag}d'] = pd.DataFrame({
            'nsw_demand_mw':df.nsw_demand.shift(48*lag),
            'nsw_price_up':(df.nsw_price.shift(48*lag)>threshold).astype(int)})
    return df, truth, predictions


def summaries(truth, pred, name, valid):
    rows=[]
    days = h.market_day(truth.index)
    for frequency in ('Y','M'):
        labels=days.to_period(frequency)
        for period in labels[valid].unique():
            mask=valid & (labels==period)
            rows.append({'model':name,'frequency':frequency,'period':str(period),
                         'n':int(mask.sum()),'positive_pct':100*truth.loc[mask,'nsw_price_up'].mean(),
                         **h.score(truth.loc[mask],pred.loc[mask])})
    return rows


def main():
    df, truth, predictions = prepare()
    # Check vectorised implementation against authoritative helpers across eras,
    # including year boundaries, leap days and the midnight interval convention.
    for date in ('2010-01-08','2014-12-31','2016-02-29','2021-10-01','2026-10-31'):
        history, weather = h.make_inputs(df,date)
        pd.testing.assert_frame_equal(truth.loc[weather.index],h.targets(df,date))
        for lag in (1,7):
            pd.testing.assert_frame_equal(predictions[f'baseline_{lag}d'].loc[weather.index],
                                          h.baseline(history,weather,lag),check_dtype=False)
    valid=h.market_day(df.index)>=pd.Timestamp('2010-01-08')
    rows=[]
    for name,pred in predictions.items():
        rows.extend(summaries(truth,pred,name,valid))
        for label,start,end in [('validation','2014-01-01','2015-01-01'),
                                ('simulation','2015-01-01','2026-11-01')]:
            days=h.market_day(df.index)
            mask=(days>=start)&(days<end)
            print(name,label,h.score(truth.loc[mask],pred.loc[mask]),flush=True)
    out=ROOT/'results'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'baseline_metrics.csv',index=False)
    (out/'data_audit.json').write_text(json.dumps({'rows':len(df),'complete_days':len(df)//48,
        'first_timestamp':str(df.index.min()),'last_timestamp':str(df.index.max()),
        'harness_checks':'passed: targets and both baselines on five selected days'},indent=2))

if __name__=='__main__':
    main()
