"""Stage 1 only: choose settings on 2014, refit on 2010-2014, freeze.
Later-year outcomes do not enter any model or hyperparameter selection here.
"""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forecast.model import features, Forecaster
from experiments.baselines import prepare, summaries
from harness import dayahead as h


def make_model(family, setting):
    if family=='ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=setting))
    if family=='logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=setting,max_iter=1000,random_state=9417))
    common=dict(n_estimators=180,learning_rate=.05,num_leaves=setting,
                min_child_samples=80,reg_lambda=2.,n_jobs=2,random_state=9417,verbosity=-1)
    if family=='boost_demand':
        return LGBMRegressor(**common)
    return LGBMClassifier(**common)


def main():
    t0=time.time()
    df, truth, baselines=prepare()
    x=features(df)
    days=h.market_day(df.index)
    valid=x.notna().all(axis=1).to_numpy()
    train=valid & (days<'2014-01-01')
    val=valid & (days>='2014-01-01') & (days<'2015-01-01')
    refit=valid & (days<'2015-01-01')
    simulation=valid & (days>='2015-01-01')
    # Data leakage check: changing actual market values on a target day cannot
    # change that day's features; streaming features must equal batch features.
    sample_day=pd.Timestamp('2014-07-15')
    mask=days==sample_day
    short=df.loc[(days>=sample_day-pd.Timedelta(days=14)) & (days<=sample_day)].copy()
    before=features(short).iloc[-48:]
    short.loc[h.market_day(short.index)==sample_day, h.MARKET_COLS]=999999.
    pd.testing.assert_frame_equal(before,features(short).iloc[-48:])
    pd.testing.assert_frame_equal(before,x.loc[mask])
    print('Feature leakage and batch/streaming consistency checks passed.',flush=True)
    validation=[]
    selected={}
    frozen={}
    histories=[]
    families=[('ridge','demand',[10.,1000.]),('boost_demand','demand',[15,31]),
              ('logistic','price',[.01,1.]),('boost_price','price',[15,31])]
    out=ROOT/'results'
    frozen_dir=out/'stage1_frozen'
    frozen_dir.mkdir(exist_ok=True)
    for family,task,settings in families:
        target='nsw_demand_mw' if task=='demand' else 'nsw_price_up'
        candidates=[]
        for setting in settings:
            model=make_model(family,setting)
            model.fit(x.loc[train],truth.loc[train,target])
            pred=model.predict(x.loc[val])
            metrics=({'MAPE':h.mape(truth.loc[val,target],pred),'MAE':h.mae(truth.loc[val,target],pred)}
                     if task=='demand' else {'BalAcc':h.balanced_accuracy(truth.loc[val,target],pred),
                                            'Acc':h.accuracy(truth.loc[val,target],pred)})
            value=metrics['MAPE'] if task=='demand' else -metrics['BalAcc']
            candidates.append((value,setting))
            validation.append({'family':family,'task':task,'setting':setting,**metrics})
            print(family,setting,metrics,flush=True)
        best_value,best_setting=min(candidates)
        model=make_model(family,best_setting)
        model.fit(x.loc[refit],truth.loc[refit,target])
        frozen[family]=model
        joblib.dump(model,frozen_dir/f'{family}.joblib')
        if task not in selected or best_value<selected[task]['selection_value']:
            selected[task]={'family':family,'setting':best_setting,'selection_value':best_value}
        # Store predictions for subsequent drift, confidence intervals and adaptation comparisons.
        prediction=pd.DataFrame(index=df.index[simulation])
        prediction[target]=model.predict(x.loc[simulation])
        prediction.to_parquet(frozen_dir/f'{family}_simulation.parquet')
        for frequency in ('Y','M'):
            labels=days[simulation].to_period(frequency)
            for period in labels.unique():
                idx=prediction.index[labels==period]
                p=prediction.loc[idx,target]
                y=truth.loc[idx,target]
                metrics=({'MAPE':h.mape(y,p),'MAE':h.mae(y,p)} if task=='demand'
                         else {'BalAcc':h.balanced_accuracy(y,p),'Acc':h.accuracy(y,p)})
                histories.append({'model':family,'task':task,'frequency':frequency,'period':str(period),**metrics})
    pd.DataFrame(validation).to_csv(out/'stage1_validation.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'frozen_metrics.csv',index=False)
    config={'stage':'stage1_frozen_not_final_submission','training_end':'2014-12-31',
            'selection_period':'2014','selected':selected,'columns':list(x.columns)}
    (frozen_dir/'config.json').write_text(json.dumps(config,indent=2))
    artifact_dir=out/'stage1_selected'
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump({'demand':frozen[selected['demand']['family']],
                 'price':frozen[selected['price']['family']], 'columns':list(x.columns)},
                artifact_dir/'models.joblib')
    forecaster=Forecaster(artifact_dir)
    history,weather=h.make_inputs(df,sample_day)
    t1=time.time()
    serving=forecaster.predict(history,weather)
    assert serving.index.equals(weather.index) and len(serving)==48
    for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
        expected=frozen[selected[task]['family']].predict(x.loc[weather.index])
        np.testing.assert_allclose(serving[col],expected,rtol=1e-6)
    print('Selected:',selected,flush=True)
    print(f'Inference matches batch; {time.time()-t1:.3f} seconds. Total {time.time()-t0:.1f} seconds.',flush=True)

if __name__=='__main__':
    main()
