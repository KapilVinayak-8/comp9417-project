"""Shared day-ahead features and inference. All market features precede target day."""
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import joblib

MARKET = ['nsw_demand','nsw_price','qld_demand','qld_price','vic_demand','vic_price']
WEATHER = [f'{city}_{v}' for city in ('syd','mel','bne') for v in ('temp','rh','cloud','solar')]


def features(frame):
    """Requires complete consecutive market days; preserves interval-end timestamps."""
    if len(frame) % 48 or not frame.index.is_monotonic_increasing:
        raise ValueError('Expected complete ordered days')
    start = frame.index - pd.Timedelta(minutes=30)
    day = start.normalize()
    x = frame[WEATHER].copy()
    slot = start.hour * 2 + start.minute // 30
    x['slot_sin'] = np.sin(2*np.pi*slot/48)
    x['slot_cos'] = np.cos(2*np.pi*slot/48)
    x['weekday'] = start.dayofweek
    x['annual_sin'] = np.sin(2*np.pi*start.dayofyear/365.25)
    x['annual_cos'] = np.cos(2*np.pi*start.dayofyear/365.25)
    calendar = holidays.Australia(subdiv='NSW', years=sorted(set(start.year)))
    x['holiday'] = [int(d.date() in calendar) for d in day]
    for col in MARKET:
        for lag in (1,2,7):
            x[f'{col}_lag{lag}'] = frame[col].shift(48*lag)
        daily = frame[col].groupby(day).agg(['mean','std','min','max'])
        for stat in ('mean','std','min','max'):
            x[f'{col}_previous_{stat}'] = day.map(daily[stat].shift(1)).to_numpy()
        x[f'{col}_week_mean'] = day.map(daily['mean'].shift(1).rolling(7).mean()).to_numpy()
    for col in WEATHER:
        x[f'{col}_change_yesterday'] = frame[col] - frame[col].shift(48)
    return x.astype('float32')


class Forecaster:
    def __init__(self, artifact_dir):
        self.bundle = joblib.load(Path(artifact_dir) / 'models.joblib')

    def predict(self, history, weather_next):
        # Fourteen complete days suffice for every feature; no fitted state changes.
        past = history.iloc[-14*48:]
        frame = pd.concat([past, weather_next.reindex(columns=past.columns)])
        x = features(frame).iloc[-48:][self.bundle['columns']]
        demand = self.bundle['demand'].predict(x)
        up = self.bundle['price'].predict(x)
        return pd.DataFrame({'nsw_demand_mw': np.clip(demand,0,20000),
                             'nsw_price_up': np.asarray(up,dtype=int)}, index=weather_next.index)
