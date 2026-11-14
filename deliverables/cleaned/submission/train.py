
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor, LGBMClassifier
from model import features

SEED=9417
WINDOW_DAYS=1095
STRATEGIES={'demand':'periodic_3y','price':'input_drift_3y'}


def main():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python train.py path/to/released.csv')
    started=time.perf_counter()
    df=pd.read_csv(sys.argv[1],parse_dates=['timestamp'],index_col='timestamp').sort_index()
    if df.index.has_duplicates or df.isna().any().any():
        raise ValueError('Expected the complete released dataset without missing values')
    if not (df.index.to_series().diff().iloc[1:]==pd.Timedelta('30min')).all():
        raise ValueError('Expected consecutive half-hours')
    days=(df.index-pd.Timedelta(minutes=30)).normalize()
    if days.max()>pd.Timestamp('2026-10-31'):
        raise ValueError('Training must not use data after 31 October 2026')
    cutoff=days.max()+pd.Timedelta(days=1)
    # Include extra context for lags, then select the three-year fitting window.
    context=df.loc[days>=cutoff-pd.Timedelta(days=WINDOW_DAYS+14)]
    x=features(context)
    fitdays=(x.index-pd.Timedelta(minutes=30)).normalize()
    valid=(fitdays>=cutoff-pd.Timedelta(days=WINDOW_DAYS)) & x.notna().all(axis=1).to_numpy()
    x=x.loc[valid]
    target=context.loc[x.index]
    previous_mean=context.nsw_price.groupby((context.index-pd.Timedelta(minutes=30)).normalize()).mean().shift(1)
    threshold=fitdays[valid].map(previous_mean).to_numpy()
    demand=target.nsw_demand.to_numpy()
    price=(target.nsw_price.to_numpy()>threshold).astype(int)
    common=dict(n_estimators=180,learning_rate=.05,num_leaves=31,min_child_samples=80,
                reg_lambda=2.,n_jobs=2,random_state=SEED,verbosity=-1)
    regression=LGBMRegressor(**common).fit(x,demand)
    classification=LGBMClassifier(**common).fit(x,price)
    artifact=Path(__file__).resolve().parent/'artifacts'
    artifact.mkdir(exist_ok=True)
    joblib.dump({'demand':regression,'price':classification,'columns':list(x.columns)},artifact/'models.joblib')
    metadata={'seed':SEED,'strategy_selection':'2015-2024; holdout 2025-October 2026',
              'historical_strategies':STRATEGIES,'window_days':WINDOW_DAYS,'rows':len(x),
              'training_first_market_day':str(fitdays[valid].min().date()),
              'training_last_market_day':str(fitdays[valid].max().date()),
              'live_prediction':'fixed parameters; no retraining or persisted state'}
    (artifact/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Final models trained on {len(x)} intervals in {time.perf_counter()-started:.2f}s.',flush=True)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
